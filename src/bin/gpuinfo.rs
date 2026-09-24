//! `gpuinfo` - prove the toolkit end to end against the local machine.
//!
//! With the `cuda` feature: loads the embedded fatbin, resolves EVERY name in
//! `lightgpu::ops::NAMES`, launches the no-op kernel, and reports the device. A
//! missing symbol or a failed launch fails the command, so this is the smoke
//! test for the whole driver + kernel set combination (and the reason `lg_noop`
//! exists).
//!
//! Without it: touches no GPU at all and just runs the CPU twins' self-test,
//! because `--no-default-features` is a supported build for a machine with no
//! driver, no toolkit and no GPU.

#[cfg(feature = "cuda")]
use lightgpu::{ffi, ops, vm};

/// The FULL kernel set, embedded here and only here: this binary's job is to
/// prove that every kernel the toolkit defines resolves in the loaded module,
/// so it is the one place that wants all of them. Consumers compile and embed
/// their own subset (`lightgpu_build::fatbin_entries`) and never link this.
#[cfg(feature = "cuda")]
static FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/la_kernels.fatbin"));

fn main() {
    if let Err(e) = run() {
        eprintln!("gpuinfo: {e}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "cuda"))]
fn run() -> Result<(), String> {
    println!("driver    : none (built with --no-default-features)");
    lightgpu::ops::cpu::selftest().map_err(|e| format!("CPU twins: {e}"))?;
    println!("cpu twins : ok");
    Ok(())
}

#[cfg(feature = "cuda")]
fn run() -> Result<(), String> {
    let d = vm::device()?;
    println!("driver    : {}", d.driver.lib_path);
    println!("device    : {} (sm_{}{})", d.name, d.cc_major, d.cc_minor);
    println!("SMs       : {}", d.sm_count);
    println!("smem/block: {} bytes", d.smem_per_block);
    println!("free VRAM : {:.2} GiB", vm::free_vram()? as f64 / (1u64 << 30) as f64);

    let m = vm::Module::load(FATBIN)?;
    println!("fatbin    : {} bytes", FATBIN.len());

    // Every advertised kernel must resolve, otherwise the op table and the .cu
    // have drifted apart.
    let mut missing = Vec::new();
    for name in ops::NAMES {
        if let Err(e) = m.func(name) {
            missing.push(format!("{name}: {e}"));
        }
    }
    if !missing.is_empty() {
        return Err(format!(
            "{} of {} kernels did not resolve:\n  {}",
            missing.len(),
            ops::NAMES.len(),
            missing.join("\n  ")
        ));
    }
    println!("kernels   : {} resolved in the module", ops::NAMES.len());

    // Proving the launch path, not just the symbol table.
    let mut a = vm::Args::new();
    a.launch(&m, lightgpu::NOOP_KERNEL, vm::Launch::new((1, 1, 1), (1, 1, 1)))?;
    vm::sync()?;
    println!("launch    : {}(1,1) ok", lightgpu::NOOP_KERNEL);

    ops::cpu::selftest().map_err(|e| format!("CPU twins: {e}"))?;
    println!("cpu twins : ok");
    let _ = ffi::CUDA_SUCCESS;
    Ok(())
}
