# lightgpu

The toolkit behind the lightgpu inference engines:
[rmbg-rs](https://github.com/jacobsparts/rmbg-rs),
[locate-anything-rs](https://github.com/jacobsparts/locate-anything-rs),
[realesrgan-rs](https://github.com/jacobsparts/realesrgan-rs) and
[lama-inpaint-rs](https://github.com/jacobsparts/lama-inpaint-rs).
[pixeldeck](https://github.com/jacobsparts/pixeldeck) is a local web app for
cleaning up product photos that drives all of these engines.

A dependency-light CUDA toolkit for building small inference engines without
reimplementing the plumbing each time: the driver layer, the kernel set, and the
CPU backend that keeps an engine running where there is no GPU, in one place.

It exists because three engines in the family had grown their own copies of the same four
things - the `dlopen` driver bindings, the nvcc/fatbin build, the launch and
buffer layer, and a duplicate-but-different kernel for the same operation.

## Layers

| layer | what it is | use it if |
| --- | --- | --- |
| `lightgpu` `::ffi` | the CUDA driver API, `dlopen`ed at run time | you want no CUDA toolkit, no bindgen and no build-time dependency beyond `libc` |
| `lightgpu` `::vm` | `Module`, `DevBuf`, `Launch`, `Args` - one place that marshals pointers into `cuLaunchKernel` | you are writing an engine's `cuda.rs` |
| `lightgpu` `::ops` | the kernel name table and each op's layout contract | you want to know what is available, or resolve a kernel by name |
| `lightgpu-build` | `fatbin("cuda/kernels.cu", "kernels.fatbin")` for your `build.rs` | you have your own kernels and want the arch list and numeric flags in one place |

Everything CUDA-related is inert unless the `cuda` feature is on, so
`--no-default-features` gives a pure-Rust CPU build with no CUDA toolchain - the
same convention every consumer engine uses. The pure-Rust layers (`json`, `mmap`,
`safetensors`, `ops`' CPU twins) are not behind it: they have no CUDA dependency
and are always compiled.

## What is in the kernel set

47 kernels - the generic ops engines share, plus the machinery to prove
them. Each consumer compiles only the subset it calls:

* **norms** - `rms_norm`, `layer_norm` (one-pass E[x^2]-mean^2 by default, two-pass available), `channel_layer_norm`
* **NCHW channel ops** - `channel_affine` (the folded-BatchNorm op, with `null`
  scale or shift and `in` allowed to alias `out`), `channel_scale` (its
  `shift = null` specialisation, for when the per-channel vector is an
  *activation* rather than a weight), `channel_mean` (per-channel global average
  pool, static shared scratch, fixed summation order)
* **rope** - `rope_neox` (half-split), `rope_2d` (adjacent pairs)
* **elementwise** - `silu_mul`, `gelu_tanh`, `gelu_erf`, `relu`, `sigmoid`,
  `add`, `add_inplace`, `scale`, `row_affine`, `copy`
* **reductions** - `argmax`, `channel_mean` (per-channel spatial mean, i.e. the
  global average pool - listed under reductions as well as above because that is
  what it is)
* **convolutions** - `conv1x1`, `conv3x3s1p1`, `conv3x3_winograd`
  (F(4x4,3x3), the same op at 4/9ths the multiplies), `conv4x4s4`, `conv_kxk`,
  `linear_1x1`
* **vision I/O and layout** - `upsample2x_nearest` (the integer-halving form),
  `pixel_unshuffle2` (space-to-depth with PyTorch's channel order), `lrelu`
  (slope as an argument), `add_scaled` (`a + s*b`, the fused residual)
* **layouts** - `extract_rows`, `merge_2x2`, `get_row_q8_0_aligned`, `copy_row`,
  `set_i32`
* **attention** - `attn_gqa` (DRAM K/V, warp per query), `attn_flash`
  (shared-memory staged)
* **GEMM** - `f32_gemm`, `f32_gemm_tiled`, `quantize_q8_0`, `q8_0_gemm_dp4a`,
  `q8_0_gemm_aligned`, `q8_0_gemv`
* **Fourier** - `fft2_r2c`, `fft2_c2r` (batched 2-D real transforms, n <= 64,
  unnormalised, half spectrum - the pair a spectral convolution calls; they are
  what lets an engine drop cuFFT)
* **misc** - `noop`, `linear` (token layout)

`cuda/CONVENTIONS.md` documents the signature and layout rules every kernel
obeys, the per-operation merge decisions (what was chosen, what was rejected,
and why), and the numerics policy. `src/ops/mod.rs` carries the same op table
with a one-line contract each, and `NAMES` there is the list `gpuinfo` resolves
against - it must stay in step with `cuda/kernels.cu`, or the completeness check
silently under-reports.

## Build

```
cargo build --release          # CUDA: needs nvcc, embeds the fatbin
cargo build --release --no-default-features   # CPU-only, no CUDA toolchain
```

`NVCC=/path/to/nvcc` picks the compiler; `LA_CUDA_ARCH` overrides the arch list
(default: SASS for `sm_61`, `sm_75`, `sm_80` plus PTX for 8.0, so a newer driver
JITs it). The GPU runtime loads the NVIDIA driver through the standard system
loader path.

## Check it

```
cargo run --release --bin gpuinfo
```

```
driver    : libcuda.so.1
device    : NVIDIA GeForce GTX 1080 (sm_61)
SMs       : 20
smem/block: 49152 bytes
free VRAM : 7.82 GiB
fatbin    : 1123456 bytes
kernels   : 47 resolved in the module
launch    : lg_noop(1,1) ok
cpu twins : ok
```

`gpuinfo` is the smoke test for the whole combination: it loads the embedded
fatbin, resolves **every** name in the op table against the module (so the table
and the `.cu` cannot drift apart silently), launches a kernel on the device, and
runs the CPU kernels' self-test. A missing symbol or a failed launch fails the
command - nothing more: agreement between the two backends is a development-time
check of the arithmetic, and neither replaces a golden fixture from the upstream
reference implementation.

## Using it from an engine

```toml
[dependencies]
lightgpu = { git = "https://github.com/jacobsparts/lightgpu.git" }

[build-dependencies]
lightgpu-build = { git = "https://github.com/jacobsparts/lightgpu.git", package = "lightgpu-build" }
```

For local coordinated development, replace both Git dependencies with paths in a
workspace checkout. Published engine repositories use the Git form so cloning an
engine does not require cloning this repository separately.

```rust
// build.rs, for your own kernel file
fn main() { lightgpu_build::fatbin("cuda/my_kernels.cu", "my_kernels.fatbin"); }
```

```rust
let dev = lightgpu::vm::device()?;          // driver + primary context
let m = lightgpu::vm::Module::load(lightgpu::FATBIN)?;
let mut a = lightgpu::vm::Args::new();
a.ptr(x).ptr(w).ptr(y).i32(ne0).i32(nrows).f32(eps);
a.launch(&m, "lg_rms_norm", lightgpu::vm::Launch::new((nrows as u32, 1, 1), (256, 1, 1)))?;
lightgpu::vm::sync()?;
```

## Credits

The kernel set merges work from five engines: LocateAnything-3B (MoonViT +
Qwen2 - transformer kernels and the q8_0 path), RMBG-2.0 (BiRefNet - vision
kernels), Real-ESRGAN (RRDBNet - the F(4x4,3x3) Winograd convolution and the
vision I/O ops), lama-inpaint-rs (big-lama - the batched 2-D Fourier
transforms, which replaced its own in-tree FFT so that no engine needs cuFFT)
and maxim-rs (MAXIM-2S Enhancement/LOL - the NCHW per-channel affine and global
average pool, plus the plain ReLU/`shift = null` scale contracts their ops
turned out to share).
Model weights are not included: this crate is only the toolkit.

## Embedding only the kernels you call

A binary should carry the kernels it uses, not every kernel the toolkit defines.
`lightgpu-build::fatbin_entries` passes your list to nvcc's `--entries`, so no
code is generated for anything else:

```rust
// build.rs
const KERNELS: &[&str] = &["lg_rms_norm", "lg_attn_gqa", "lg_f32_gemm_tiled"];
fn main() {
    let src = lightgpu_build::toolkit_kernels_cu().expect("lightgpu's kernels.cu");
    lightgpu_build::fatbin_entries(&src.to_string_lossy(), "kernels.fatbin", KERNELS);
}
```

Measured on one consumer: naming 24 of the kernels takes the embedded fatbin
from 1,741,512 B to 415,448 B, and the binary from 3.63 MB to 2.16 MB.

`toolkit_kernels_cu()` finds the source through `DEP_LIGHTGPU_KERNELS_CU`, which
cargo sets for any package that depends on `lightgpu` (the `links = "lightgpu"`
key in its manifest). No relative-path guessing; `LA_GPU_DIR` remains as an
override and the evident relative locations as a fallback.

### Two things that will silently defeat this

* **`-lineinfo` makes `--entries` a no-op.** In nvcc 12.4 a build with line
  tables keeps every kernel: the same 24-kernel list produced 1,151,688 B with
  all 56 reachable instead of 415,448 B with none. It is off by default here;
  set `LA_CUDA_LINEINFO=1` only for profiling, and re-check your binary size.
* **nvcc does not reliably overwrite an existing `-o` fatbin.** `fatbin_entries`
  deletes the output first - without that, a stale wider image from an earlier
  build stays in place and gets embedded.

`gpuinfo` is the exception: it embeds the full set on purpose, because its job
is to prove every kernel the toolkit defines resolves on this machine.

