//! Minimal CUDA driver API bindings, resolved at run time with `dlopen`.
//!
//! Hand-written on purpose: `cust`/`cuda-*` would drag in bindgen (and therefore
//! libclang) plus a large dependency tree, and bindgen output is not
//! reproducible across environments. The driver surface an inference engine
//! actually needs is small and stable, so it is declared by hand.
//!
//! Everything here is inert unless the `cuda` feature is enabled and
//! [`driver`] is called, so a CPU-only build links nothing extra.
//!
//! `LA_CUDA_LIB=/path/to/libcuda.so.1` overrides the library. That matters on a
//! machine whose userspace driver does not match the loaded kernel module: the
//! system path then fails `cuInit` with error 100/804 while a matching copy of
//! the library works.

#![allow(non_snake_case, dead_code)]

use std::ffi::c_void;
use std::sync::OnceLock;

pub type CUresult = i32;
pub type CUdevice = i32;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUdeviceptr = u64;

pub const CUDA_SUCCESS: CUresult = 0;
/// `cuModuleGetFunction` on a name the module does not define.
pub const CUDA_ERROR_NOT_FOUND: CUresult = 500;
pub const CU_EVENT_DEFAULT: u32 = 0;
pub const CU_EVENT_DISABLE_TIMING: u32 = 0x2;
pub const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 0x02;
/// `CU_STREAM_DEFAULT`, i.e. a null stream pointer.
pub const CU_STREAM_DEFAULT: CUstream = std::ptr::null_mut();

/// `CU_DEVICE_ATTRIBUTE_*` values used by these engines.
pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: i32 = 1;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: i32 = 8;
pub const CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: i32 = 9;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: i32 = 81;

/// Turn a `CUresult` into a `Result`, naming the failure through
/// [`Driver::error_name`] when a driver is available (otherwise the raw code).
pub fn chk(r: CUresult, what: &str) -> Result<(), String> {
    if r == CUDA_SUCCESS {
        Ok(())
    } else {
        let name = match driver() {
            Ok(d) => d.error_name(r),
            Err(_) => format!("CUresult {r}"),
        };
        Err(format!("{what} failed: {name}"))
    }
}

pub type FnInit = unsafe extern "C" fn(u32) -> CUresult;
pub type FnDeviceGet = unsafe extern "C" fn(*mut CUdevice, i32) -> CUresult;
pub type FnDeviceGetName = unsafe extern "C" fn(*mut libc::c_char, i32, CUdevice) -> CUresult;
pub type FnDeviceComputeCapability =
    unsafe extern "C" fn(*mut i32, *mut i32, CUdevice) -> CUresult;
pub type FnDeviceGetAttribute = unsafe extern "C" fn(*mut i32, i32, CUdevice) -> CUresult;
pub type FnDevicePrimaryCtxRetain = unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult;
pub type FnDevicePrimaryCtxRelease = unsafe extern "C" fn(CUdevice) -> CUresult;
pub type FnCtxSetCurrent = unsafe extern "C" fn(CUcontext) -> CUresult;
pub type FnCtxSetFlags = unsafe extern "C" fn(u32) -> CUresult;
pub type FnCtxCreate = unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult;
pub type FnCtxDestroy = unsafe extern "C" fn(CUcontext) -> CUresult;
pub type FnCtxSynchronize = unsafe extern "C" fn() -> CUresult;
pub type FnMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult;
pub type FnModuleLoadData = unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
pub type FnModuleGetFunction =
    unsafe extern "C" fn(*mut CUfunction, CUmodule, *const libc::c_char) -> CUresult;
pub type FnModuleUnload = unsafe extern "C" fn(CUmodule) -> CUresult;
pub type FnMemAlloc = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;
pub type FnMemFree = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
/// `cuMemAllocHost` / `cuMemFreeHost`: page-locked host staging buffers.
pub type FnMemAllocHost = unsafe extern "C" fn(*mut *mut c_void, usize) -> CUresult;
pub type FnMemFreeHost = unsafe extern "C" fn(*mut c_void) -> CUresult;
pub type FnMemcpyHtoD = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
pub type FnMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
pub type FnMemcpyDtoD = unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult;
pub type FnMemcpyAsync =
    unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult;
pub type FnMemsetD8 = unsafe extern "C" fn(CUdeviceptr, u8, usize) -> CUresult;
pub type FnLaunchKernel = unsafe extern "C" fn(
    CUfunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    CUstream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;
pub type FnGetErrorName = unsafe extern "C" fn(CUresult, *mut *const libc::c_char) -> CUresult;
pub type FnEventCreate = unsafe extern "C" fn(*mut CUevent, u32) -> CUresult;
pub type FnEventRecord = unsafe extern "C" fn(CUevent, CUstream) -> CUresult;
pub type FnEventSynchronize = unsafe extern "C" fn(CUevent) -> CUresult;
pub type FnEventElapsedTime = unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult;
pub type FnEventDestroy = unsafe extern "C" fn(CUevent) -> CUresult;

/// Resolved entry points.
pub struct Driver {
    pub handle: *mut c_void,
    pub cuInit: FnInit,
    pub cuDeviceGet: FnDeviceGet,
    pub cuDeviceGetName: FnDeviceGetName,
    pub cuDeviceComputeCapability: FnDeviceComputeCapability,
    pub cuDeviceGetAttribute: FnDeviceGetAttribute,
    pub cuDevicePrimaryCtxRetain: FnDevicePrimaryCtxRetain,
    pub cuDevicePrimaryCtxRelease: FnDevicePrimaryCtxRelease,
    pub cuCtxSetCurrent: FnCtxSetCurrent,
    pub cuCtxSetFlags: FnCtxSetFlags,
    pub cuCtxCreate: FnCtxCreate,
    pub cuCtxDestroy: FnCtxDestroy,
    pub cuCtxSynchronize: FnCtxSynchronize,
    pub cuMemGetInfo: FnMemGetInfo,
    pub cuModuleLoadData: FnModuleLoadData,
    pub cuModuleGetFunction: FnModuleGetFunction,
    pub cuModuleUnload: FnModuleUnload,
    pub cuMemAlloc: FnMemAlloc,
    pub cuMemFree: FnMemFree,
    pub cuMemAllocHost: FnMemAllocHost,
    pub cuMemFreeHost: FnMemFreeHost,
    pub cuMemcpyHtoD: FnMemcpyHtoD,
    pub cuMemcpyDtoH: FnMemcpyDtoH,
    pub cuMemcpyDtoD: FnMemcpyDtoD,
    pub cuMemcpyAsync: FnMemcpyAsync,
    pub cuMemsetD8: FnMemsetD8,
    pub cuLaunchKernel: FnLaunchKernel,
    pub cuGetErrorName: FnGetErrorName,
    pub cuEventCreate: FnEventCreate,
    pub cuEventRecord: FnEventRecord,
    pub cuEventSynchronize: FnEventSynchronize,
    pub cuEventElapsedTime: FnEventElapsedTime,
    pub cuEventDestroy: FnEventDestroy,
    /// Path the library was loaded from (`LA_CUDA_LIB` or the default soname).
    pub lib_path: String,
}

macro_rules! forward {
    ($($name:ident($($arg:ident : $t:ty),*) -> $r:ty;)*) => {
        impl Driver { $(
            pub fn $name(&self, $($arg: $t),*) -> $r {
                unsafe { (self.$name)($($arg),*) }
            }
        )* }
    };
}

forward! {
    cuInit(flags: u32) -> CUresult;
    cuDeviceGet(d: *mut CUdevice, o: i32) -> CUresult;
    cuDeviceGetName(n: *mut libc::c_char, l: i32, d: CUdevice) -> CUresult;
    cuDeviceComputeCapability(a: *mut i32, b: *mut i32, d: CUdevice) -> CUresult;
    cuDeviceGetAttribute(v: *mut i32, a: i32, d: CUdevice) -> CUresult;
    cuDevicePrimaryCtxRetain(c: *mut CUcontext, d: CUdevice) -> CUresult;
    cuDevicePrimaryCtxRelease(d: CUdevice) -> CUresult;
    cuCtxSetCurrent(c: CUcontext) -> CUresult;
    cuCtxSetFlags(flags: u32) -> CUresult;
    cuCtxCreate(c: *mut CUcontext, f: u32, d: CUdevice) -> CUresult;
    cuCtxDestroy(c: CUcontext) -> CUresult;
    cuCtxSynchronize() -> CUresult;
    cuMemGetInfo(free: *mut usize, total: *mut usize) -> CUresult;
    cuModuleLoadData(m: *mut CUmodule, i: *const c_void) -> CUresult;
    cuModuleGetFunction(f: *mut CUfunction, m: CUmodule, n: *const libc::c_char) -> CUresult;
    cuModuleUnload(m: CUmodule) -> CUresult;
    cuMemAlloc(p: *mut CUdeviceptr, n: usize) -> CUresult;
    cuMemAllocHost(p: *mut *mut c_void, n: usize) -> CUresult;
    cuMemFreeHost(p: *mut c_void) -> CUresult;
    cuMemFree(p: CUdeviceptr) -> CUresult;
    cuMemcpyHtoD(d: CUdeviceptr, s: *const c_void, n: usize) -> CUresult;
    cuMemcpyDtoH(d: *mut c_void, s: CUdeviceptr, n: usize) -> CUresult;
    cuMemcpyDtoD(d: CUdeviceptr, s: CUdeviceptr, n: usize) -> CUresult;
    cuMemcpyAsync(d: CUdeviceptr, s: *const c_void, n: usize, st: CUstream) -> CUresult;
    cuMemsetD8(d: CUdeviceptr, v: u8, n: usize) -> CUresult;
    cuGetErrorName(r: CUresult, n: *mut *const libc::c_char) -> CUresult;
    cuEventCreate(e: *mut CUevent, f: u32) -> CUresult;
    cuEventRecord(e: CUevent, st: CUstream) -> CUresult;
    cuEventSynchronize(e: CUevent) -> CUresult;
    cuEventElapsedTime(ms: *mut f32, a: CUevent, b: CUevent) -> CUresult;
    cuEventDestroy(e: CUevent) -> CUresult;
}

impl Driver {
    #[allow(clippy::too_many_arguments)]
    pub fn cuLaunchKernel(
        &self,
        f: CUfunction,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
        sh: u32,
        st: CUstream,
        p: *mut *mut c_void,
        e: *mut *mut c_void,
    ) -> CUresult {
        unsafe { (self.cuLaunchKernel)(f, gx, gy, gz, bx, by, bz, sh, st, p, e) }
    }

    /// Symbolic name of a `CUresult` ("CUDA_ERROR_OUT_OF_MEMORY", ...).
    pub fn error_name(&self, r: CUresult) -> String {
        let mut p: *const libc::c_char = std::ptr::null();
        unsafe {
            if (self.cuGetErrorName)(r, &mut p) == CUDA_SUCCESS && !p.is_null() {
                return std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned();
            }
        }
        format!("CUresult {r}")
    }
}

unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

static DRIVER: OnceLock<Result<Driver, String>> = OnceLock::new();

/// Load libcuda and resolve entry points. `LA_CUDA_LIB` overrides the path.
pub fn driver() -> Result<&'static Driver, String> {
    let r = DRIVER.get_or_init(|| unsafe {
        let lib = std::env::var("LA_CUDA_LIB").unwrap_or_else(|_| "libcuda.so.1".to_string());
        let cname = std::ffi::CString::new(lib.clone()).map_err(|e| e.to_string())?;
        // RTLD_NOW|RTLD_GLOBAL so the driver's own dependencies (the libnvidia
        // components) resolve against the same search path.
        let h = libc::dlopen(cname.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL);
        if h.is_null() {
            let e = libc::dlerror();
            let msg = if e.is_null() {
                format!("{lib} not found")
            } else {
                std::ffi::CStr::from_ptr(e).to_string_lossy().into_owned()
            };
            return Err(format!("cannot load CUDA driver: {msg} (set LA_CUDA_LIB)"));
        }
        fn sym<T: Copy>(h: *mut c_void, name: &str) -> Result<T, String> {
            let c = std::ffi::CString::new(name).unwrap();
            let p = unsafe { libc::dlsym(h, c.as_ptr()) };
            if p.is_null() {
                return Err(format!("libcuda: missing symbol {name}"));
            }
            Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&p) })
        }
        // Symbols with a `_v2` variant MUST resolve to it: the legacy unversioned
        // entry points have a narrower ABI (an `unsigned int*` size for
        // `cuMemGetInfo`, a `cuCtxCreate` without flags) and calling the plain
        // symbol through a `size_t*` signature misreports as
        // CUDA_ERROR_INVALID_CONTEXT. Prefer `_v2`, fall back to the plain name
        // only for a library too old to export it.
        macro_rules! s {
            ($v:literal, $n:literal, $t:ty) => {{
                match sym::<$t>(h, $v) {
                    Ok(f) => f,
                    Err(_) => sym::<$t>(h, $n)?,
                }
            }};
        }
        let d = Driver {
            handle: h,
            lib_path: lib.clone(),
            cuInit: s!("cuInit", "cuInit", FnInit),
            cuDeviceGet: s!("cuDeviceGet", "cuDeviceGet", FnDeviceGet),
            cuDeviceGetName: s!("cuDeviceGetName", "cuDeviceGetName", FnDeviceGetName),
            cuDeviceComputeCapability: s!(
                "cuDeviceComputeCapability",
                "cuDeviceComputeCapability",
                FnDeviceComputeCapability
            ),
            cuDeviceGetAttribute: s!(
                "cuDeviceGetAttribute",
                "cuDeviceGetAttribute",
                FnDeviceGetAttribute
            ),
            cuDevicePrimaryCtxRetain: s!(
                "cuDevicePrimaryCtxRetain",
                "cuDevicePrimaryCtxRetain",
                FnDevicePrimaryCtxRetain
            ),
            cuDevicePrimaryCtxRelease: s!(
                "cuDevicePrimaryCtxRelease",
                "cuDevicePrimaryCtxRelease",
                FnDevicePrimaryCtxRelease
            ),
            cuCtxSetCurrent: s!("cuCtxSetCurrent", "cuCtxSetCurrent", FnCtxSetCurrent),
            cuCtxSetFlags: s!("cuCtxSetFlags", "cuCtxSetFlags", FnCtxSetFlags),
            cuCtxCreate: s!("cuCtxCreate_v2", "cuCtxCreate", FnCtxCreate),
            cuCtxDestroy: s!("cuCtxDestroy_v2", "cuCtxDestroy", FnCtxDestroy),
            cuCtxSynchronize: s!("cuCtxSynchronize", "cuCtxSynchronize", FnCtxSynchronize),
            cuMemGetInfo: s!("cuMemGetInfo_v2", "cuMemGetInfo", FnMemGetInfo),
            cuModuleLoadData: s!("cuModuleLoadData", "cuModuleLoadData", FnModuleLoadData),
            cuModuleGetFunction: s!(
                "cuModuleGetFunction",
                "cuModuleGetFunction",
                FnModuleGetFunction
            ),
            cuModuleUnload: s!("cuModuleUnload", "cuModuleUnload", FnModuleUnload),
            cuMemAlloc: s!("cuMemAlloc_v2", "cuMemAlloc", FnMemAlloc),
            cuMemFree: s!("cuMemFree_v2", "cuMemFree", FnMemFree),
            cuMemAllocHost: s!("cuMemAllocHost_v2", "cuMemAllocHost", FnMemAllocHost),
            cuMemFreeHost: s!("cuMemFreeHost_v2", "cuMemFreeHost", FnMemFreeHost),
            cuMemcpyHtoD: s!("cuMemcpyHtoD_v2", "cuMemcpyHtoD", FnMemcpyHtoD),
            cuMemcpyDtoH: s!("cuMemcpyDtoH_v2", "cuMemcpyDtoH", FnMemcpyDtoH),
            cuMemcpyDtoD: s!("cuMemcpyDtoD_v2", "cuMemcpyDtoD", FnMemcpyDtoD),
            cuMemcpyAsync: s!("cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync", FnMemcpyAsync),
            cuMemsetD8: s!("cuMemsetD8_v2", "cuMemsetD8", FnMemsetD8),
            cuLaunchKernel: s!("cuLaunchKernel", "cuLaunchKernel", FnLaunchKernel),
            cuGetErrorName: s!("cuGetErrorName", "cuGetErrorName", FnGetErrorName),
            cuEventCreate: s!("cuEventCreate", "cuEventCreate", FnEventCreate),
            cuEventRecord: s!("cuEventRecord", "cuEventRecord", FnEventRecord),
            cuEventSynchronize: s!("cuEventSynchronize", "cuEventSynchronize", FnEventSynchronize),
            cuEventElapsedTime: s!(
                "cuEventElapsedTime",
                "cuEventElapsedTime",
                FnEventElapsedTime
            ),
            cuEventDestroy: s!("cuEventDestroy", "cuEventDestroy", FnEventDestroy),
        };
        Ok(d)
    });
    r.as_ref().map_err(|e| e.clone())
}
