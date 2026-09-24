//! Build helper: compile a kernel file into a fatbin and print the embed
//! instructions. Called from a crate's `build.rs` so that the nvcc invocation,
//! the arch list and the numeric flags live in exactly one place.
//!
//! ```ignore
//! // build.rs
//! fn main() {
//!     // every kernel in the file
//!     lightgpu_build::fatbin("cuda/kernels.cu", "la_kernels.fatbin");
//!     // only the ones this consumer calls
//!     lightgpu_build::fatbin_entries("cuda/kernels.cu", "kernels.fatbin", &["lg_rms_norm", "lg_attn_gqa"]);
//!     // or several files, each becoming its own module - which is how a
//!     // project keeps its own kernels out of the toolkit: see `fatbin_modules`
//! }
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

/// Portability targets. SASS for Pascal, Turing and Ampere plus PTX for 8.0, so
/// a newer driver JITs the forward-compatible image.
///
/// `LA_CUDA_ARCH` overrides the list (comma-separated `compute_XY,code=sm_XY`
/// pairs) for a build that only needs one device.
pub const DEFAULT_ARCHES: &[(&str, &str)] = &[
    ("compute_61", "sm_61"),
    ("compute_75", "sm_75"),
    ("compute_80", "sm_80"),
    ("compute_80", "compute_80"),
];

/// Compile every kernel in `src` into `out_name` inside `OUT_DIR`.
///
/// Prefer [`fatbin_entries`]: a binary should carry the kernels it calls, not
/// every kernel the toolkit happens to define.
pub fn fatbin(src: &str, out_name: &str) -> usize {
    compile(src, out_name, None)
}

/// Compile only `entries` (the `--entries` names) into `out_name`.
///
/// nvcc generates no code for any other global entry function, so unselected
/// kernels cost nothing in the embedded image. An EMPTY `entries` list is
/// rejected: it would produce a fatbin that resolves no kernel at all, which
/// fails at module load rather than at build time.
pub fn fatbin_entries(src: &str, out_name: &str, entries: &[&str]) -> usize {
    assert!(
        !entries.is_empty(),
        "fatbin_entries: empty kernel list - pass the kernels the consumer calls, \
         or use `fatbin` to embed the whole file"
    );
    compile(src, out_name, Some(entries))
}

/// One source file in a [`fatbin_modules`] build.
///
/// `out_name` is the fatbin's file name inside `OUT_DIR`; `entries` is that
/// file's own `--entries` list (`None` compiles the whole file). Each source
/// gets its own module, so a project can keep its kernels in its own
/// repository and compile them together with the toolkit's in one build.
pub struct Source<'a> {
    pub path: &'a str,
    pub out_name: &'a str,
    pub entries: Option<&'a [&'a str]>,
}

/// Compile SEVERAL source files, each into its OWN fatbin, and return the
/// paths. The consumer loads one [`lightgpu::vm::Module`] per fatbin.
///
/// This is how a project owns its own kernels. nvcc accepts only one input per
/// `--fatbin`, and a shim translation unit that `#include`s several files
/// DEFEATS `--entries` (it prunes per translation unit, so every included
/// kernel survives) - hence one module per source rather than one merged image.
/// Separate modules are also separate namespaces: a project kernel and a
/// toolkit kernel may share a name without either shadowing the other, and a
/// missing name fails at resolution, which is what makes an eager
/// resolve-every-name-at-load list a valid check.
///
/// Typical build script:
///
/// ```ignore
/// let toolkit = lightgpu_build::toolkit_kernels_cu().expect("toolkit kernels.cu");
/// let paths = lightgpu_build::fatbin_modules(&[
///     lightgpu_build::Source { path: &toolkit.to_string_lossy(), out_name: "toolkit.fatbin",
///                              entries: Some(&["lg_layer_norm", "lg_copy"]) },
///     lightgpu_build::Source { path: "cuda/swin.cu", out_name: "swin.fatbin",
///                              entries: Some(&["rmbg_window_gather", "rmbg_patch_merge"]) },
/// ]);
/// // one `include_bytes!` per path, or hand the paths to a bundler
/// ```
pub fn fatbin_modules(sources: &[Source]) -> Vec<PathBuf> {
    assert!(
        !sources.is_empty(),
        "fatbin_modules: no sources - pass at least one, or use `fatbin`/`fatbin_entries`"
    );
    for s in sources {
        if let Some(e) = s.entries {
            assert!(
                !e.is_empty(),
                "fatbin_modules: `{}` has an empty entries list - it would resolve no kernel",
                s.path
            );
        }
    }
    let mut out = Vec::with_capacity(sources.len());
    for s in sources {
        compile(s.path, s.out_name, s.entries);
        let path = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR")).join(s.out_name);
        println!("cargo:rustc-env=LIGHTGPU_FATBIN_{}={}", env_key(s.out_name), path.display());
        out.push(path);
    }
    out
}

/// A stable env-var suffix for a fatbin file name: `swin.fatbin` -> `SWIN`.
fn env_key(out_name: &str) -> String {
    out_name
        .trim_end_matches(".fatbin")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_uppercase() } else { '_' })
        .collect()
}

/// Convenience for a build script that just wants the default file name.
pub fn default_fatbin(crate_dir: &Path) -> usize {
    fatbin(
        &crate_dir.join("cuda/kernels.cu").to_string_lossy(),
        "la_kernels.fatbin",
    )
}

fn compile(src: &str, out_name: &str, entries: Option<&[&str]>) -> usize {
    println!("cargo:rerun-if-changed={src}");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=LA_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=LA_CUDA_LINEINFO");

    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return 0;
    }

    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR")).join(out_name);
    // nvcc does not reliably overwrite an existing --fatbin output: if one is
    // left over from an earlier build (a wider --entries list, or no list at
    // all) the embedded image silently keeps the old, larger kernel set. Remove
    // it first so what is embedded is always what this invocation produced.
    let _ = std::fs::remove_file(&out);
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());

    let mut cmd = Command::new(&nvcc);
    cmd.arg("-O3").arg("-std=c++17").arg("--fatbin");
    match std::env::var("LA_CUDA_ARCH") {
        Ok(list) if !list.trim().is_empty() => {
            for pair in list.split(',') {
                cmd.arg(format!("--generate-code={}", pair.trim()));
            }
        }
        _ => {
            for (arch, code) in DEFAULT_ARCHES {
                cmd.arg(format!("--generate-code=arch={arch},code={code}"));
            }
        }
    }
    if let Some(names) = entries {
        // One --entries per name, so a name can never be split on a comma.
        for n in names {
            cmd.arg("--entries").arg(n);
        }
    }
    // No fast math, no flush-to-zero: every kernel has an arithmetic twin on the
    // CPU and the two must agree, so the GPU side may not win precision by
    // dropping IEEE behaviour.
    cmd.arg("--ftz=false")
        .arg("--prec-div=true")
        .arg("--prec-sqrt=true")
        .arg("--fmad=true");
    // `-lineinfo` is OPT-IN and off by default. nvcc 12.4's line tables make
    // `--entries` a no-op: a 24-kernel build goes from 415,184 B / no stray
    // entry points to 1,151,688 B with all 56 kernels reachable, so a debug flag
    // silently reintroduces exactly the dead code the entries list removes.
    // Set LA_CUDA_LINEINFO=1 to profile with source attribution.
    if std::env::var_os("LA_CUDA_LINEINFO").is_some() {
        cmd.arg("-lineinfo");
    }
    cmd.arg("-o").arg(&out).arg(src);

    println!(
        "cargo:warning=nvcc {}",
        cmd.get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(" ")
    );
    match cmd.status() {
        Ok(s) if s.success() => {}
        Ok(s) => panic!("nvcc failed with status {s}"),
        Err(e) => panic!(
            "could not run `{nvcc}`: {e} (set NVCC=/path/to/nvcc, or build with \
             --no-default-features for the CPU-only path)"
        ),
    }
    let n = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0) as usize;
    let what = match entries {
        Some(names) => format!("{} of the file's kernels", names.len()),
        None => "all kernels".to_string(),
    };
    println!("cargo:warning={out_name}: {n} bytes ({what})");
    n
}

/// The kernel names the toolkit defines, read from `cuda/kernels.cu` rather than
/// hard-coded here: the file is the single source of truth, and duplicating the
/// list would let a consumer name a kernel that was renamed away.
pub fn toolkit_kernel_names() -> Vec<String> {
    match toolkit_kernels_cu() {
        Some(p) => {
            let src = std::fs::read_to_string(&p).unwrap_or_default();
            kernel_names_in(&src)
        }
        None => Vec::new(),
    }
}

/// True if the toolkit defines `name`.
pub fn known_kernel(name: &str) -> bool {
    toolkit_kernel_names().iter().any(|n| n == name)
}

/// The toolkit's `cuda/kernels.cu`, so a consumer can compile a SUBSET of it
/// without copying the file.
///
/// Resolved in this order: `LA_GPU_DIR` (the toolkit's root), then the obvious
/// relative locations from the consumer's manifest dir (`../lightgpu`,
/// `../../lightgpu`, `third_party/lightgpu`). A path dependency's own
/// `CARGO_MANIFEST_DIR` is not visible to a dependent's build script, so the
/// search is the only way to find the file without an env var.
pub fn toolkit_kernels_cu() -> Option<PathBuf> {
    // Preferred: the path the toolkit exported from its own build script
    // (`links = "lightgpu"` -> `DEP_LIGHTGPU_KERNELS_CU`). This needs no
    // assumption about where the toolkit sits relative to the consumer.
    if let Ok(p) = std::env::var("DEP_LIGHTGPU_KERNELS_CU") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    if let Ok(dir) = std::env::var("LA_GPU_DIR") {
        let p = PathBuf::from(dir).join("cuda/kernels.cu");
        if p.is_file() {
            return Some(p);
        }
    }
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").ok()?);
    for rel in ["../lightgpu", "../../lightgpu", "third_party/lightgpu"] {
        let p = manifest.join(rel).join("cuda/kernels.cu");
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// Pull every `extern "C" __global__ void <name>` out of a kernel source file.
pub fn kernel_names_in(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let marker = "extern \"C\" __global__ void ";
    let mut base = 0usize;
    while let Some(i) = src[base..].find(marker) {
        let i = base + i;
        let mut after = &src[i + marker.len()..];
        // An attribute may sit between `void` and the name - `__launch_bounds__`
        // is the common one. Skipping it matters more than it looks: without
        // this, the parser reports the attribute as the kernel's name and the
        // REAL kernel is silently absent from the list, so a build that lists it
        // fails a check it should pass and a build that forgets it passes a check
        // it should fail.
        //
        // The test for 'is this an attribute' cannot be 'an identifier followed
        // by a parenthesis', because a kernel DECLARATION has exactly that shape
        // too - that mistake made the parser skip `plain_kernel(int n)` as if it
        // were an attribute and return an empty name. Attributes are the
        // implementation's reserved identifiers, so they begin with a double
        // underscore, and that is the test.
        let mut name = &after[..0];
        loop {
            let name_end = after
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(after.len());
            let candidate = &after[..name_end];
            if !candidate.starts_with("__") {
                name = candidate;
                break;
            }
            let tail = &after[name_end..];
            if !tail.starts_with('(') {
                break;
            }
            let mut depth = 0usize;
            let mut k = 0usize;
            for (idx, c) in tail.char_indices() {
                if c == '(' {
                    depth += 1;
                } else if c == ')' {
                    depth -= 1;
                    if depth == 0 {
                        k = idx + 1;
                        break;
                    }
                }
            }
            if k == 0 {
                break;
            }
            after = tail[k..].trim_start();
        }
        out.push(name.to_string());
        // Resume after the consumed name, in absolute terms: `after` is a
        // suffix of `src`, so its start is where the next search begins.
        base = src.len() - after.len() + name.len();
    }
    out
}

#[cfg(test)]
mod kernel_names_tests {
    use super::kernel_names_in;

    #[test]
    fn plain_and_attributed_declarations_both_parse() {
        let src = r#"
extern "C" __global__ void plain_kernel(const float *a, float *b) { }
extern "C" __global__ void __launch_bounds__(256, 2) bounds_first(const float *a, float *b) { }
extern "C" __global__ void another_kernel(int n) { }
"#;
        let names = kernel_names_in(src);
        assert_eq!(names, vec!["plain_kernel", "bounds_first", "another_kernel"]);
    }

    #[test]
    fn nested_parens_in_the_attribute_are_skipped() {
        let src = r#"extern "C" __global__ void __launch_bounds__(256, 2) one(int);"#;
        assert_eq!(kernel_names_in(src), vec!["one"]);
    }
}
