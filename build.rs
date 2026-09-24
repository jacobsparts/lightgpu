//! The toolkit's own build script: compiles the shared kernel set, and exports
//! the absolute path of `cuda/kernels.cu` so a consumer's build script can
//! compile its own subset from the same source without guessing at relative
//! paths (the `links = "lightgpu"` key in Cargo.toml turns the `cargo:KERNELS_CU`
//! line into `DEP_LIGHTGPU_KERNELS_CU` for every dependent).

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/kernels.cu");

    let kernels = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
        .join("cuda/kernels.cu");
    // Consumed by dependents' build scripts as DEP_LIGHTGPU_KERNELS_CU.
    println!("cargo:KERNELS_CU={}", kernels.display());

    lightgpu_build::fatbin("cuda/kernels.cu", "la_kernels.fatbin");
}
