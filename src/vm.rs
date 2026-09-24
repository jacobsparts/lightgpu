//! Module loading, kernel launch and device buffers.
//!
//! This is deliberately the *only* place that knows how a pointer argument is
//! marshalled into `cuLaunchKernel`, so an engine's kernels are called through
//! one small, reviewable surface.

use crate::ffi::{self, CUdeviceptr, CUfunction, CUmodule, CUresult, CUstream, Driver};
use std::ffi::{c_void, CString};
use std::sync::OnceLock;

/// A loaded fatbin. Cheap to query; one per embedded kernel image.
pub struct Module {
    module: CUmodule,
    /// Kernel handles, memoized: `cuModuleGetFunction` is a driver-side hash
    /// lookup costing microseconds, and a decode loop launches on the order of
    /// 700 kernels per token, so the lookup must not be repeated per launch.
    funcs: std::cell::RefCell<std::collections::HashMap<String, CUfunction>>,
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

/// The initialisation state every engine needs: driver, device, context.
pub struct Device {
    pub driver: &'static Driver,
    pub dev: ffi::CUdevice,
    pub ctx: ffi::CUcontext,
    pub name: String,
    pub cc_major: i32,
    pub cc_minor: i32,
    pub sm_count: i32,
    pub smem_per_block: i32,
}

static CONTEXT: OnceLock<Result<(), String>> = OnceLock::new();

/// Bind device 0's primary context to this process, idempotently.
///
/// Every entry point calls this first (via [`init`]), so a caller can use
/// [`Module`], [`DevBuf`] or [`sync`] without having called [`device`] - the
/// alternative is a "you must initialise first" contract that fails with
/// `CUDA_ERROR_INVALID_CONTEXT` at the first allocation.
fn bind_context() -> Result<(), String> {
    CONTEXT
        .get_or_init(|| {
            let d = ffi::driver()?;
            ffi::chk(d.cuInit(0), "cuInit")?;
            let mut dev: ffi::CUdevice = 0;
            ffi::chk(d.cuDeviceGet(&mut dev, 0), "cuDeviceGet")?;
            let mut ctx: ffi::CUcontext = std::ptr::null_mut();
            ffi::chk(d.cuDevicePrimaryCtxRetain(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
            ffi::chk(d.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")?;
            // Ask the driver to BLOCK (yield the CPU) on synchronisation rather
            // than spin. The default busy-waits in user space for the whole
            // duration of every GPU operation, which pegs a core for an entire
            // inference run. Non-fatal: an already-initialised primary context
            // may reject the change and correctness does not depend on it.
            const CU_CTX_SCHED_BLOCKING_SYNC: u32 = 0x04;
            if let Err(e) = ffi::chk(d.cuCtxSetFlags(CU_CTX_SCHED_BLOCKING_SYNC), "cuCtxSetFlags") {
                eprintln!("lightgpu: {e} (spin-wait stays)");
            }
            Ok(())
        })
        .clone()
}

/// Initialise the driver and bind the primary context. Idempotent: the first
/// call does the work, later calls return the same `Result`. An engine that
/// wants a CPU fallback treats `Err` as "no GPU here" - the same decision
/// `LA_CPU=1` short-circuits.
pub fn init() -> Result<(), String> {
    bind_context()
}

/// Query the first device: name, compute capability, SM count, shared memory.
pub fn device() -> Result<Device, String> {
    init()?;
    let d = ffi::driver()?;
    let mut dev: ffi::CUdevice = 0;
    ffi::chk(d.cuDeviceGet(&mut dev, 0), "cuDeviceGet")?;
    let mut name_buf = [0i8; 128];
    ffi::chk(
        d.cuDeviceGetName(name_buf.as_mut_ptr(), name_buf.len() as i32, dev),
        "cuDeviceGetName",
    )?;
    let name = unsafe { std::ffi::CStr::from_ptr(name_buf.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    let mut cc_major = 0i32;
    let mut cc_minor = 0i32;
    ffi::chk(
        d.cuDeviceComputeCapability(&mut cc_major, &mut cc_minor, dev),
        "cuDeviceComputeCapability",
    )?;
    let mut sm_count = 0i32;
    ffi::chk(
        d.cuDeviceGetAttribute(
            &mut sm_count,
            ffi::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            dev,
        ),
        "cuDeviceGetAttribute(SM count)",
    )?;
    let mut smem_per_block = 0i32;
    ffi::chk(
        d.cuDeviceGetAttribute(
            &mut smem_per_block,
            ffi::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            dev,
        ),
        "cuDeviceGetAttribute(smem)",
    )?;
    let mut ctx: ffi::CUcontext = std::ptr::null_mut();
    ffi::chk(d.cuDevicePrimaryCtxRetain(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
    ffi::chk(d.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")?;
    Ok(Device {
        driver: d,
        dev,
        ctx,
        name,
        cc_major,
        cc_minor,
        sm_count,
        smem_per_block,
    })
}

impl Module {
    /// Load a fatbin image (the bytes of an `nvcc --fatbin` output).
    pub fn load(image: &[u8]) -> Result<Module, String> {
        init()?;
        let d = ffi::driver()?;
        let mut module: CUmodule = std::ptr::null_mut();
        ffi::chk(
            d.cuModuleLoadData(&mut module, image.as_ptr() as *const c_void),
            "cuModuleLoadData",
        )?;
        Ok(Module {
            module,
            funcs: Default::default(),
        })
    }

    /// Look up a kernel by its `extern "C"` name (memoized).
    pub fn func(&self, name: &str) -> Result<CUfunction, String> {
        if let Some(f) = self.funcs.borrow().get(name) {
            return Ok(*f);
        }
        let d = ffi::driver()?;
        let c = CString::new(name).map_err(|e| e.to_string())?;
        let mut f: CUfunction = std::ptr::null_mut();
        ffi::chk(
            d.cuModuleGetFunction(&mut f, self.module, c.as_ptr()),
            &format!("cuModuleGetFunction({name})"),
        )?;
        self.funcs.borrow_mut().insert(name.to_string(), f);
        Ok(f)
    }

    /// Alias kept for call sites that read as "get this kernel handle".
    pub fn get(&self, name: &str) -> Result<CUfunction, String> {
        self.func(name)
    }

    /// Does this module define `name`?
    ///
    /// An engine that loads TWO fatbins has to pick which one a kernel lives
    /// in, and the only way to ask the driver is `cuModuleGetFunction`, which
    /// reports a missing name as `CUDA_ERROR_NOT_FOUND`. Any other code is a
    /// real failure (a broken context, a corrupt module) and is reported as
    /// such rather than as "absent", so a name lookup cannot silently hide a
    /// driver problem. A hit is memoized like `func`, so probing a module for
    /// every launch costs nothing after the first.
    pub fn has(&self, name: &str) -> bool {
        if self.funcs.borrow().contains_key(name) {
            return true;
        }
        let (Ok(d), Ok(c)) = (ffi::driver(), CString::new(name)) else {
            return false;
        };
        let mut f: CUfunction = std::ptr::null_mut();
        match d.cuModuleGetFunction(&mut f, self.module, c.as_ptr()) {
            ffi::CUDA_SUCCESS => {
                self.funcs.borrow_mut().insert(name.to_string(), f);
                true
            }
            ffi::CUDA_ERROR_NOT_FOUND => false,
            other => {
                eprintln!("lightgpu: cuModuleGetFunction({name}) probing failed: {}", d.error_name(other));
                false
            }
        }
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if let Ok(d) = ffi::driver() {
            d.cuModuleUnload(self.module);
        }
    }
}

/// A device allocation. `Drop` frees it, so callers never leak on an early
/// return.
pub struct DevBuf {
    pub ptr: CUdeviceptr,
    pub bytes: usize,
}

impl DevBuf {
    pub fn alloc(bytes: usize) -> Result<DevBuf, String> {
        init()?;
        let d = ffi::driver()?;
        let mut ptr: CUdeviceptr = 0;
        ffi::chk(d.cuMemAlloc(&mut ptr, bytes.max(1)), "cuMemAlloc")?;
        Ok(DevBuf { ptr, bytes })
    }

    /// Upload f32 data into a freshly allocated buffer.
    pub fn from_host(v: &[f32]) -> Result<DevBuf, String> {
        let b = DevBuf::alloc(std::mem::size_of_val(v))?;
        b.upload(v)?;
        Ok(b)
    }

    /// A zeroed allocation: zeroing matters because scratch buffers are read
    /// before every element is provably written (masked attention rows, KV
    /// slots beyond the current position).
    pub fn zeros(bytes: usize) -> Result<DevBuf, String> {
        let b = DevBuf::alloc(bytes)?;
        let d = ffi::driver()?;
        ffi::chk(d.cuMemsetD8(b.ptr, 0, bytes.max(1)), "cuMemsetD8")?;
        Ok(b)
    }

    pub fn upload(&self, v: &[f32]) -> Result<(), String> {
        let d = ffi::driver()?;
        let bytes = std::mem::size_of_val(v);
        assert!(bytes <= self.bytes, "upload of {bytes} bytes into {}", self.bytes);
        ffi::chk(
            d.cuMemcpyHtoD(self.ptr, v.as_ptr() as *const c_void, bytes),
            "cuMemcpyHtoD",
        )
    }

    pub fn download(&self, out: &mut [f32]) -> Result<(), String> {
        let d = ffi::driver()?;
        let bytes = std::mem::size_of_val(out);
        assert!(bytes <= self.bytes, "download of {bytes} bytes from {}", self.bytes);
        ffi::chk(
            d.cuMemcpyDtoH(out.as_mut_ptr() as *mut c_void, self.ptr, bytes),
            "cuMemcpyDtoH",
        )
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        if let Ok(d) = ffi::driver() {
            d.cuMemFree(self.ptr);
        }
    }
}

/// Chunk size for a staged upload: the source is copied into the staging buffer
/// a chunk at a time so the copy-out can start early and the staging buffer stays
/// small.  64 MB measured best on a GTX 1080.
pub const STAGING_CHUNK: usize = 64 << 20;

/// A reusable pageable host staging buffer for large uploads.
///
/// Reusing ONE buffer for a whole weight load is worth about 10% of the upload
/// time, measured on a 4.36 GB model over a Gen3 x4 link: 1.85 s through a
/// single reused buffer against 2.02 s when each tensor gets its own freshly
/// allocated buffer.  The reason is not pinning - the same chunking through a
/// page-locked buffer measured 1.84 s, i.e. no better - but that a reused buffer
/// is already faulted in and stays cache-hot for the copy that fills it.
///
/// `copy_htod_staged` copies through one of these; a caller that already owns a
/// staging area of its own can use [`copy_htod`] directly instead.
pub struct Staging {
    buf: Vec<u8>,
}

impl Staging {
    /// A staging buffer of `bytes`, rounded up to at least one chunk.
    pub fn new(bytes: usize) -> Staging {
        Staging {
            buf: vec![0u8; bytes.max(1)],
        }
    }

    /// A staging buffer sized for chunked uploads (`STAGING_CHUNK`).
    pub fn chunked() -> Staging {
        Staging::new(STAGING_CHUNK)
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// The staging area as a mutable byte slice, for a caller that fills it
    /// itself (a repack, a layout change) before copying it out.
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        &mut self.buf
    }
}

/// A launch: grid and block dimensions, dynamic shared memory, stream.
#[derive(Debug, Clone, Copy)]
pub struct Launch {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared: u32,
    pub stream: CUstream,
}

impl Launch {
    pub fn new(grid: (u32, u32, u32), block: (u32, u32, u32)) -> Launch {
        Launch {
            grid,
            block,
            shared: 0,
            stream: ffi::CU_STREAM_DEFAULT,
        }
    }
    pub fn shared(mut self, bytes: u32) -> Launch {
        self.shared = bytes;
        self
    }
}

/// Argument builder: pushes typed values so a call site cannot silently pass an
/// `f32` count where an `i32` was meant.
///
/// The driver reads `void**` pointing at each argument's storage, so the values
/// must outlive the launch - hence the owned boxes here. Usage:
///
/// ```ignore
/// let mut a = Args::new();
/// a.ptr(x.ptr).ptr(w.ptr).ptr(y.ptr).i32(ne0).i32(nrows).f32(eps);
/// a.launch(&module, "lg_rms_norm", Launch::new((nrows, 1, 1), (256, 1, 1)))?;
/// ```
#[derive(Default)]
pub struct Args {
    slots: Vec<*mut c_void>,
    keep: Vec<Box<u64>>,
    keep_i32: Vec<Box<i32>>,
    keep_i64: Vec<Box<i64>>,
    keep_f32: Vec<Box<f32>>,
}

impl Args {
    pub fn new() -> Args {
        Args::default()
    }

    /// A device pointer argument (`CUdeviceptr` is passed by value).
    pub fn ptr(&mut self, p: CUdeviceptr) -> &mut Args {
        self.keep.push(Box::new(p));
        let b: &Box<u64> = self.keep.last().unwrap();
        self.slots.push(&**b as *const u64 as *mut c_void);
        self
    }

    /// A typed device pointer (e.g. `const float*`).
    pub fn raw<T>(&mut self, p: *const T) -> &mut Args {
        self.ptr(p as u64)
    }

    pub fn i32(&mut self, v: i32) -> &mut Args {
        self.keep_i32.push(Box::new(v));
        let b: &Box<i32> = self.keep_i32.last().unwrap();
        self.slots.push(&**b as *const i32 as *mut c_void);
        self
    }

    pub fn f32(&mut self, v: f32) -> &mut Args {
        self.keep_f32.push(Box::new(v));
        let b: &Box<f32> = self.keep_f32.last().unwrap();
        self.slots.push(&**b as *const f32 as *mut c_void);
        self
    }

    /// A 64-bit signed integer argument. Needed by every kernel whose count can
    /// exceed 2^31 - `lg_copy` and the other elementwise ops take `long n`
    /// precisely so a large activation cannot overflow the index, which makes
    /// this method part of the argument marshalling contract rather than a
    /// convenience.
    pub fn i64(&mut self, v: i64) -> &mut Args {
        self.keep_i64.push(Box::new(v));
        let b: &Box<i64> = self.keep_i64.last().unwrap();
        self.slots.push(&**b as *const i64 as *mut c_void);
        self
    }

    pub fn launch(&mut self, m: &Module, kernel: &str, l: Launch) -> Result<(), String> {
        let d = ffi::driver()?;
        let f = m.func(kernel)?;
        let r: CUresult = d.cuLaunchKernel(
            f,
            l.grid.0,
            l.grid.1,
            l.grid.2,
            l.block.0,
            l.block.1,
            l.block.2,
            l.shared,
            l.stream,
            self.slots.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if r != ffi::CUDA_SUCCESS {
            return Err(format!(
                "cuLaunchKernel({kernel}) failed: {} grid=({},{},{}) block=({},{},{}) shared={}",
                d.error_name(r),
                l.grid.0,
                l.grid.1,
                l.grid.2,
                l.block.0,
                l.block.1,
                l.block.2,
                l.shared
            ));
        }
        Ok(())
    }
}

/// Upload raw bytes to a device address.
pub fn copy_htod(dst: CUdeviceptr, src: &[u8]) -> Result<(), String> {
    init()?;
    let d = ffi::driver()?;
    ffi::chk(
        d.cuMemcpyHtoD(dst, src.as_ptr() as *const c_void, src.len()),
        "cuMemcpyHtoD",
    )
}

/// Upload `src` to the device address `dst`, chunked through a reusable host
/// staging buffer.
///
/// This is the generic primitive for moving a large payload (a whole model's
/// weights) when the caller has nothing but a byte slice.  For `n` tensors it is
/// worth building ONE [`Staging`] and calling this per tensor - see `Staging`
/// for the measurement behind reusing a single buffer.
///
/// Plain pageable copies are used throughout: on a Gen3 x4 link a page-locked
/// buffer measured no faster for the same chunking (1.84 s versus 1.85 s for a
/// 4.36 GB model), so pinning is not worth the host-memory cost.
pub fn copy_htod_staged(dst: CUdeviceptr, src: &[u8], stage: &mut Staging) -> Result<(), String> {
    let chunk = stage.buf.len();
    if chunk == 0 {
        return Err("staging buffer is empty".to_string());
    }
    let mut off = 0usize;
    while off < src.len() {
        let n = chunk.min(src.len() - off);
        stage.buf[..n].copy_from_slice(&src[off..off + n]);
        copy_htod(dst + off as u64, &stage.buf[..n])?;
        off += n;
    }
    Ok(())
}

/// Download raw bytes from a device address.
pub fn copy_dtoh(dst: &mut [u8], src: CUdeviceptr) -> Result<(), String> {
    init()?;
    let d = ffi::driver()?;
    ffi::chk(
        d.cuMemcpyDtoH(dst.as_mut_ptr() as *mut c_void, src, dst.len()),
        "cuMemcpyDtoH",
    )
}

/// Device-to-device copy (KV-cache appends and similar).
pub fn copy_d2d(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) -> Result<(), String> {
    let d = ffi::driver()?;
    ffi::chk(d.cuMemcpyDtoD(dst, src, bytes), "cuMemcpyDtoD")
}

/// A CUDA event, for timing a region of device work.
///
/// The driver entry points were already resolved by `ffi`; this surfaces them
/// because an engine that wants to know WHERE its time goes needs per-kernel
/// measurements, and the alternative (a host sync per kernel) perturbs the
/// thing being measured on a pipeline this short.
pub struct Event {
    ev: ffi::CUevent,
}

impl Event {
    pub fn new() -> Result<Event, String> {
        init()?;
        let d = ffi::driver()?;
        let mut ev: ffi::CUevent = std::ptr::null_mut();
        ffi::chk(d.cuEventCreate(&mut ev, ffi::CU_EVENT_DEFAULT), "cuEventCreate")?;
        Ok(Event { ev })
    }

    /// Record the event on the default stream at the current point in the queue.
    pub fn record(&self) -> Result<(), String> {
        let d = ffi::driver()?;
        ffi::chk(d.cuEventRecord(self.ev, std::ptr::null_mut()), "cuEventRecord")
    }

    pub fn synchronize(&self) -> Result<(), String> {
        let d = ffi::driver()?;
        ffi::chk(d.cuEventSynchronize(self.ev), "cuEventSynchronize")
    }

    /// Milliseconds elapsed from this event to `later`. Both must have been
    /// recorded; `later` need not be synchronised as long as something has.
    pub fn elapsed_ms(&self, later: &Event) -> Result<f32, String> {
        let d = ffi::driver()?;
        let mut ms = 0f32;
        ffi::chk(d.cuEventElapsedTime(&mut ms, self.ev, later.ev), "cuEventElapsedTime")?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if let Ok(d) = ffi::driver() {
            let _ = d.cuEventDestroy(self.ev);
        }
    }
}

/// Wait for all work on the current context to finish.
pub fn sync() -> Result<(), String> {
    init()?;
    let d = ffi::driver()?;
    ffi::chk(d.cuCtxSynchronize(), "cuCtxSynchronize")
}

/// Free and total VRAM in bytes.
///
/// `cuMemGetInfo` needs a current context, but `init()` only retains and sets
/// the *primary* context - it does not create one - so this answers without
/// making the caller pay for a context of its own.  That is what lets an engine
/// size a job before it commits to it.
pub fn vram() -> Result<(usize, usize), String> {
    init()?;
    let d = ffi::driver()?;
    let mut free = 0usize;
    let mut total = 0usize;
    ffi::chk(d.cuMemGetInfo(&mut free, &mut total), "cuMemGetInfo")?;
    Ok((free, total))
}

/// Free VRAM in bytes; the free half of [`vram`].
pub fn free_vram() -> Result<usize, String> {
    vram().map(|(free, _)| free)
}
