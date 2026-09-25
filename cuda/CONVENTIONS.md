# Kernel conventions

These rules are why the kernels in `kernels.cu` can be lifted between engines.
They are the contract; a new kernel that breaks one should say why in a comment.

## 1. Signatures

* `extern "C" __global__` and the `lg_` name prefix, so the driver looks the
  kernel up by name with no C++ mangling.
* Raw pointers and scalars only. No structs defined by an engine, no globals, no
  `__constant__` tables that a caller must populate out of band. A caller who
  cannot pass something as a pointer plus an int cannot use the kernel.
* `const T *__restrict__` for inputs, `T *__restrict__` for outputs. No kernel
  both reads and writes the same buffer unless it says so: the in-place ops are
  `lg_silu_mul` and `lg_softmax_rows`, plus `lg_channel_affine`, whose element
  `idx` reads and writes only `idx` and which the folded-BatchNorm call site
  invokes with `out == in`.
* Sizes as `int` when the element count provably fits in 2^31 (any single
  tensor in these models does), `long` when a caller could pass a batch that
  overflows (`lg_copy`, `lg_mul_broadcast`, the conv family, the gather pair).
  Getting this wrong fails silently on large inputs, so it is per-kernel, not
  per-file.

## 2. Layout

Two layouts exist, and every kernel documents which it takes.

**ggml transformer layout** (what the quantized checkpoints use):

* weights are `[ne1][ne0]` with `ne0` the reduction dimension and contiguous;
* activations are `[ne0, nrows]` with `ne0` contiguous - i.e. one row per
  output element, column-major over rows;
* attention tensors are `[hd, nh, ntok]` with token stride `nh * hd`;
* `q8_0` weights are ggml blocks (34 bytes: f16 scale + 32 int8). Two variants
  are supported on purpose: the native 34-byte blocks, read straight from a
  mmapped file, and the 36-byte aligned form produced when repacking for upload.
  The aligned form exists because every 4-byte word of the 34-byte layout
  straddles a block boundary, which costs ~8 instructions against 8 `dp4a`;
  measured 3.1x on the large GEMMs.

**NCHW / token layout** (what the vision models use):

* `NCHW` as `[c][h][w]` planes, contiguous;
* token layout `[n_tokens][C]`, i.e. row-major with channels innermost - the
  form a patch embed or a linear wants.

Kernels that convert between the two say so in their name
(`lg_nchw_to_tokens`, `lg_tokens_to_nchw`) rather than assuming a caller's
convention. A kernel that takes a pitch or a stride parameter must state whether
the pitch is in elements or bytes.

## 3. Merging decisions

Where the two source kernel sets implemented the same operation differently, the
better implementation won. The decisions, and why:

| op | chosen | rejected | why |
| --- | --- | --- | --- |
| layer norm variance | two-pass: mean, then variance of `(x - mean)` | one-pass `E[x^2] - mean^2` | the one-pass form cancels catastrophically when the mean is large relative to the spread; the two-pass form is what both engines' CPU twins do |
| `erf` | Abramowitz & Stegun 7.1.26 | hardware `erff` | the hardware variant differs from a plain Rust CPU twin by more than the pipeline's own error, which blunts the per-op GPU-vs-CPU diff. Same formula on both sides means an erf mismatch can never explain a difference |
| elementwise | out-of-place `in, out` | in-place | costs nothing, removes an aliasing rule the caller had to know, and lets the input survive for a residual. `lg_silu_mul` stays in-place because the fused gated-MLP path reuses the gate buffer on purpose |
| bias add / per-row affine | `lg_row_affine(x, scale, shift, ne0, nrows)` | `add_bias` | `lg_add_bias` is `lg_row_affine` with `scale = null`. The vector is indexed by the **contiguous** dimension, so this covers a bias add and a last-axis affine. |
| NCHW per-channel affine | `lg_channel_affine(in, out, scale, shift, c, hw)` | `affine_channels` | **CORRECTION.** An earlier note claimed `lg_row_affine` subsumes the NCHW channel affine with the caller's layout choice. It does not: this op indexes its vector by **channel** and applies it across each channel's contiguous `hw`, and no choice of `ne0`/`nrows` reproduces that, because NCHW fixes which dimension is contiguous. Two kernels. Either parameter may be null; `in` may alias `out`. |
| conv | specialised `lg_conv3x3s1p1` plus generic `lg_conv_kxk` | routing everything through the generic path | the 3x3 is BiRefNet's hot path and its specialised index arithmetic is measurably better |
| NCHW per-channel scale | `lg_channel_scale(in, s, out, c, hw)`, documented as the `shift = null` specialisation | folding the scale-only case into `lg_channel_affine` | the same operation with one parameter dropped, kept as its own entry point because the scale-only callers already pass the three-pointer `(in, s, out)` order and their per-channel vector is an ACTIVATION, not a weight; the affine's `(in, out, scale, shift, c, hw)` order would be a call-site hazard for no gain. Both allow `in` to alias `out`. |
| NCHW channel mean | one kernel: `lg_channel_mean`, static shared scratch, summation order part of the contract | dynamic shared scratch; a reduction tree sized to `blockDim` | `docs/MAINTAINING.md`: a shared kernel must not acquire a requirement its callers satisfy implicitly, and reduction scratch should be sized for the largest legal `blockDim` (1024). The order, the tree geometry and the aliasing rule are written on the kernel, and the CPU twin takes the block size so the two agree exactly. |
| standalone sigmoid | one toolkit `lg_sigmoid` | a copy per engine | three engines need a plain sigmoid as its own op (MAXIM's CALayer, lama's output layer, rmbg's attention), so it does not belong folded into any one engine's fused kernel. `lg_sigmoid` uses `__expf`: moving a caller onto it from an `expf` copy is an arithmetic change, not a rename. |
| bilinear resize | `lg_resize_bilinear` and `mx_resize_axis` stay separate | one shared resampler | different rate rules, not two spellings of one. `lg_resize_bilinear` samples with the torch/PIL coordinate rule (`src = (dst + 0.5) * scale - 0.5`, clamped, `align_corners` as an argument); `mx_resize_axis` follows jax's antialiased rule (a filter over `scale` taps, renormalised at the border, and different plane sizes per pass). Same output size, different arithmetic. |
| `la_bilinear_zero` | a `__device__` helper local to its caller's `.cu` | promoting the helper alone | a device helper is inlined into its caller's translation unit, while this toolkit is compiled as one source file that no consumer `#include`s; sharing it would need a header of device code, a mechanism this toolkit deliberately does not have. |
| 3x3 conv, second implementation | `lg_conv3x3s1p1` (direct) **and** `lg_conv3x3_winograd` (F(4x4,3x3)) | dropping either | the same op at 4/9ths the multiplies when the transform amortises, and 4x the multiplies when it does not. The Winograd form is the only one worth using from c_in=32 up and the direct form is the only one worth using at c_in=3, which is a whole-network first layer; both are kept, with the crossover documented on the op |
| patch merge | two kernels: `lg_merge_2x2` (MoonViT tap order, `(2a+b)*gw + (2c+e)` source) and `lg_patch_merge` (Swin order, 2x2 taps) | one merged kernel | they are genuinely different operations with different tap orders and different source indexing; merging them would need a mode flag that no caller wants |
| attention | `lg_attn_gqa` (DRAM K/V, warp per query) **and** `lg_attn_flash` (smem-staged) | dropping either | short key sequences favour the former (staging costs more than it saves); long ones favour the latter (the naive version is ~9.7 GB of K/V traffic per ViT layer). Both are kept, with the crossover documented per kernel |
| 1x1 conv | `lg_conv1x1`, one thread per output, weight `[c_out][c_in]` | a tiled form reading a transposed `[c_in][c_out]` blob | the transposed layout is a second device allocation and a second loader path in every consumer; the direct form stays defined regardless, because that is what a before/after speedup is measured against. The signature `int c_in, int c_out, int h, int wd` is the one every consumer is built against |
| 3x3 conv, tiled form | `lg_conv3x3s1p1`, one thread per output | a tile staging a 64-pixel row segment plus its halo | such a tile needs `wd % 64 == 0`. As a toolkit precondition that means both kernels stay linked in every consumer and every call site grows a width test; generalising to a partial trailing segment makes the shared-memory size a runtime quantity, i.e. a dynamic shared argument, which `docs/MAINTAINING.md` forbids |
| blocked-layout gating matmul | the engine that owns the layout | a shared kernel | `docs/MAINTAINING.md`: it is an implicit GEMM over one architecture's blocked `[c][outer][inner]` layout - a fusion, a layout and a tap order |
| index type | `long` where a batch can overflow | all-`int` | silent truncation on large inputs is the worst failure mode; the cost is negligible outside the hot loops |

## 4. Numerics

* Compiled with `--ftz=false --prec-div=true --prec-sqrt=true --fmad=true`: no
  fast math, no flush-to-zero. Every kernel has an arithmetic twin on the CPU and
  the two must agree, so the GPU side may not win precision by dropping IEEE
  behaviour.
* Reductions keep the CPU twin's summation order where that is cheap, so a
  backend mismatch points at a bug rather than at floating-point reordering.
  Where the order does differ, the kernel says so.
* A fully-masked attention row must produce a finite output: additive masks
  should use `-FLT_MAX/4` rather than `-inf`, since `-inf` produces NaN through
  the online-softmax rescale.

## 5. Adding a kernel

1. Write the CUDA kernel here, following the rules above.
2. Add the CPU twin in `src/ops/cpu.rs`, same name, same arithmetic order.
3. Re-run `gpuinfo`: it resolves every name in `src/ops/mod.rs` against the
   loaded module, so a kernel that is added to the `.cu` but not to the table
   (or vice versa) fails the smoke test rather than a caller's first launch.
4. If the kernel is one of a pair where two implementations are both worth
   keeping, keep both and document the crossover point, as the attention
   section does.

## 6. Driver layer

The rules that keep the driver layer boring, and the bugs that motivated them:

* Prefer the `_v2` symbol whenever the driver exports one. The legacy
  unversioned entry points have narrower ABIs - `cuMemGetInfo(unsigned int*,
  unsigned int*)`, `cuCtxCreate` without a flags argument - and calling the
  plain symbol through a `size_t*` signature misreports as
  `CUDA_ERROR_INVALID_CONTEXT`. This exact bug cost an afternoon here.
* Bind device 0's primary context inside initialisation, not in the device-query
  path, so that a module load or an allocation works without a caller having
  called `device()` first.
* `LA_CUDA_LIB=/path/to/libcuda.so.1` overrides the library. This is not a
  convenience: on a machine whose userspace driver does not match the loaded
  kernel module, the default path fails `cuInit` with error 100/804 while a
  matching copy of the library works.
