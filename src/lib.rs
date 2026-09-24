//! `lightgpu` - the shared toolkit behind these dependency-light inference engines.
//!
//! Layers, so an engine can take only what it needs:
//!
//! * [`ffi`] - the CUDA driver API, `dlopen`ed at run time. No CUDA toolkit, no
//!   bindgen, no build-time dependency on anything but `libc`.
//! * [`vm`] - a loaded module, a launch helper and a device buffer, on top of
//!   `ffi`. This is the layer an engine's `cuda.rs` would otherwise hand-roll.
//! * [`ops`] - the kernel **name** table. The CUDA implementation lives in
//!   `cuda/kernels.cu` (embedded as a fatbin) and the CPU twin in
//!   `src/ops/cpu.rs`; both expose the same names and the same arithmetic.
//! * [`safetensors`] - a reader for the standard `.safetensors` container:
//!   `mmap` the file, parse the JSON header, hand out borrowed slices. This is
//!   the layer an engine's `weights.rs` would otherwise hand-roll. Only
//!   safetensors is understood here; a container with its own format (for
//!   example a quantized one) keeps its own reader and can still reuse
//!   [`mmap::File`] and [`json`].
//!
//! The conventions every kernel in `cuda/kernels.cu` follows are documented in
//! `cuda/CONVENTIONS.md`; they are the contract that lets kernels be lifted
//! between engines.

pub mod ffi;
pub mod json;
pub mod mmap;
pub mod ops;
pub mod safetensors;
pub mod vm;

pub use ops::{Op, NAMES, OPS};

/// True if `name` is a kernel this toolkit defines.
///
/// A consumer that embeds a SUBSET of the kernels (see
/// [`lightgpu_build::fatbin_entries`]) can check its list against this before the
/// build emits a fatbin that would fail to resolve the name at module load:
///
/// ```ignore
/// // build.rs
/// let want = ["lg_rms_norm", "lg_attn_gqa"];
/// for n in want {
///     assert!(lightgpu::has_kernel(n), "unknown kernel `{n}`");
/// }
/// lightgpu_build::fatbin_entries("cuda/kernels.cu", "la_kernels.fatbin", &want);
/// ```
pub fn has_kernel(name: &str) -> bool {
    NAMES.contains(&name)
}

// The toolkit deliberately does NOT embed a fatbin. Each consumer compiles and
// embeds the SUBSET of kernels it calls (see `lightgpu_build::fatbin_entries`);
// exposing the full set here would link every kernel into every binary through
// include_bytes!, however little of it is used.

/// Name of the fatbin's proving kernel: an engine can load the module and call
/// this to verify the driver path end to end (`gpuinfo` does exactly that).
pub const NOOP_KERNEL: &str = "lg_noop";
