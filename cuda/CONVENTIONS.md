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
| NCHW per-channel affine | `lg_channel_affine(in, out, scale, shift, c, hw)` | `affine_channels` | **CORRECTION.** An earlier note claimed `lg_row_affine` subsumes the NCHW channel affine with the caller's layout choice. It does not: this op indexes its vector by **channel** and applies it across each channel's contiguous `hw`, and no choice of `ne0`/`nrows` reproduces that, because NCHW fixes which dimension is contiguous. Two kernels. Either parameter may be null; `in` may alias `out`. **PROMOTED** from rmbg-rs's `cuda/swin.cu`, which is where it had been living while this file and `kernels.cu` both described it as toolkit API. |
| conv | specialised `lg_conv3x3s1p1` plus generic `lg_conv_kxk` | routing everything through the generic path | the 3x3 is BiRefNet's hot path and its specialised index arithmetic is measurably better |
| NCHW per-channel scale | two entry points: `lg_channel_affine(in, out, scale, shift, c, hw)` and `lg_channel_scale(in, s, out, c, hw)` | folding the scale-only form into the affine | they ARE the same operation (`lg_channel_scale` is the `shift = null` case), so one kernel with an optional shift would be defensible. Kept separate because the scale-only callers already build the `(in, s, out)` argument order and `s` there is an ACTIVATION buffer (MAXIM's `x * sigmoid(y)`), not a weight - a variant that reordered their arguments would be a call-site hazard for no gain. Documented as a specialisation rather than an independent op. Both allow `in` to alias `out`, which the folded-BatchNorm call site relies on. |
| NCHW channel mean | ONE kernel, static shared scratch, fixed summation order | two kernels (rmbg's strided-loop-plus-tree and MAXIM's one-thread-per-channel serial loop) | they were NOT the same kernel, and this is the trap the merge would have hit. The order is now written down as contract: zero all 1024 scratch slots, each thread accumulates its own strided partial into slot `threadIdx.x`, then a halving tree over the full 1024 slots, then thread 0 divides. The tree is over 1024 slots rather than over `blockDim` so it needs no power-of-two block and no unpaired-trailing-partial case - and that case is exactly where the tempting `else if (t == half) r[half - 1] += r[half]` is a RACE. Static scratch also removes rmbg's dynamic `threads * 4` argument, per `docs/MAINTAINING.md`. |
| standalone sigmoid | the toolkit's `lg_sigmoid` | per-project copies (`mx_sigmoid`, `k_sigmoid`) | three engines need a plain sigmoid as its own op. Before this, `mx_sigmoid`'s comment claimed the toolkit had none because its transformer engines fold it into an attention kernel - false, and exactly the drift a shared table prevents. NOTE the arithmetic: `lg_sigmoid` uses `__expf` and `mx_sigmoid` used `expf`, so pointing MAXIM at the toolkit version is a real (tiny) numerical change, not a rename. |
| bilinear resize | nothing merged - the two calls stay separate | "one resampler for the family" | NOT duplicates, and this is the one pair a naive pass would have merged. `lg_resize_bilinear` (rmbg-rs) samples with the torch/PIL coordinate rule: `src = (dst + 0.5) * scale - 0.5`, clamped, `align_corners` as an argument. `mx_resize_axis` (maxim-rs) is a two-pass separable resize following jax's ANTIALIASED rule - it builds a filter over `scale` taps and RENORMALISES at the border, and its two passes have different plane sizes (the horizontal pass writes `hin * wout`, not `hout * wout`). Same output size, different arithmetic, different kernels. Keeping both is correct; a merged "resize" would silently change whichever consumer lost. |
| `la_bilinear_zero` | stays in rmbg-rs, as a device helper | promoting the helper alone | it is the one genuinely engine-agnostic piece of `cuda/swin.cu` - a bilinear sample with zero padding, i.e. torchvision's `grid_sample` rule - but a device helper is INLINED into its caller's translation unit, and this toolkit is compiled as its own source file that no consumer `#include`s. Sharing it would mean either a header of device code (a new mechanism, and the toolkit's whole contract is "one .cu, looked up by name through the driver") or duplicating the inside of `lg_deform_conv`, which is a swin-specific op with a torchvision-compatible offset/mask layout. The cost of sharing exceeds the cost of the six duplicated lines. |
| 3x3 conv, second implementation | `lg_conv3x3s1p1` (direct) **and** `lg_conv3x3_winograd` (F(4x4,3x3)) | dropping either | the same op at 4/9ths the multiplies when the transform amortises, and 4x the multiplies when it does not. The Winograd form is the only one worth using from c_in=32 up and the direct form is the only one worth using at c_in=3, which is a whole-network first layer; both are kept, with the crossover documented on the op |
| patch merge | two kernels: `lg_merge_2x2` (MoonViT tap order, `(2a+b)*gw + (2c+e)` source) and `lg_patch_merge` (Swin order, 2x2 taps) | one merged kernel | they are genuinely different operations with different tap orders and different source indexing; merging them would need a mode flag that no caller wants |
| attention | `lg_attn_gqa` (DRAM K/V, warp per query) **and** `lg_attn_flash` (smem-staged) | dropping either | short key sequences favour the former (staging costs more than it saves); long ones favour the latter (the naive version is ~9.7 GB of K/V traffic per ViT layer). Both are kept, with the crossover documented per kernel |
| index type | `long` where a batch can overflow | all-`int` | silent truncation on large inputs is the worst failure mode; the cost is negligible outside the hot loops |

## 4. Tiled convolutions: evaluated, NOT promoted

`maxim-rs` carries tiled replacements for two of this file's convolutions, with
measured speedups (`--legacy-ops` runs the originals from the same binary):

| op | here | in maxim-rs | at 512x512 |
| --- | --- | --- | --- |
| 1x1 conv, 444 ops | `lg_conv1x1` | `mx_conv1x1_t` | 5.39 -> 1.56 ms/op |
| 3x3 conv, 66 ops | `lg_conv3x3s1p1` | `mx_conv3x3_t4`/`_t2` | 28.8 -> 3.42 ms/op |
| gating matmul, 112 ops | (none) | `mx_gate_mm_t0`/`_t1` | 18.8 -> 0.89 ms/op |

The third is not a candidate at all and never was: `mx_gate_mm_tiling` is an
implicit GEMM over the gMLP's blocked `[c][outer][inner]` layout, which is a
MAXIM-family contract, so by `docs/MAINTAINING.md` ("does it exist because of one
architecture's fusion, layout, or tap order? keep it in the project") it stays in
`maxim-rs`. It is recorded here so the conclusion is on the record rather than
implied by its absence.

The first two are real speedups of a generic op and this pass DID evaluate
promoting them. They are not promoted, because neither precondition has a clean
fallback, and the rule above ("if the fallback story is not clean, do not
promote") is the one that decides it. A mis-shaped launch here is a silent wrong
answer or an address error, and this file's conv signatures are the ones every
consumer already builds against.

**`mx_conv3x3_t*` needs `wd % 64 == 0`.** It stages a 64-pixel row segment plus a
one-column halo per input channel in shared memory. A kernel needing that cannot
be a drop-in `lg_conv3x3s1p1`: either it carries a host-side guard with a fallback
to the direct kernel - which means every consumer keeps both kernels linked AND
every call site grows a width test, i.e. the preconditions leak into the toolkit's
public contract - or the tile generalises to a partial trailing segment, which
turns "64 pixels" into a runtime width and gives back part of what the tiling
bought (the halo and the smem sizing both become dynamic).

A partial trailing segment is the better of the two, and it is still not clean:
the shared-memory size would become `(span + 2) * (ty + 2)` with `span` a runtime
value, so the launch needs a dynamic shared-memory argument - the exact implicit
requirement `docs/MAINTAINING.md` forbids - or a fixed maximum, which wastes smem
on the narrow cases. Removing the alignment hazard that the fixed 64 buys would
also mean re-deriving the tile-fill guard whose absence faulted at 640x448 in
`maxim-rs` (an integer-derived float4 load at an aligned-in-theory address).

Per consumer, the widths that would actually occur:

* **rmbg-rs** - `conv()` is driven by BiRefNet feature-map sizes: the input is
  resized to 1024x1024, the patch embed is 4x4 stride 4 (256x256), and every
  later width comes from a patch merge or a window reshape, so they are multiples
  of small powers of two and NOT of 64 in general. The crate's own
  `--cuda-selftest` exercises a 7-wide 3x3 conv (16x9x7), which is the shape that
  would have to fall back. So: needs a fallback, and the fallback would be the
  common case for the 7x7 and 1x1 convs it also runs through this path.
* **realesrgan-rs** - not applicable twice over: its production 3x3 conv is
  `lg_conv3x3_winograd`, not `lg_conv3x3s1p1`, and its first layer is c_in=3,
  which this file already documents as the case where the Winograd transform does
  not amortise. Its own direct conv (`lg_conv3x3_res`) has a different signature
  and a different purpose (the fallback for shapes Winograd cannot take).
* **lama-inpaint-rs** - not applicable: it calls neither conv. Its convolutions
  are its own `k_conv7x7`, `k_im2col` + `k_sgemm_slab`, and `k_conv_transpose`.
* **locate-anything-rs** - not applicable: no convolution kernels at all.
* **maxim-rs** - already uses both, and only because its plan pads the input to a
  multiple of 64 *and* carries the transposed weight blob. That is a whole-engine
  choice; it is not something the toolkit can require of a caller.

**`mx_conv1x1_t` needs a transposed `[c_in][c_out]` weight.** That is a new device
allocation and a new layout in every consumer's weight loader, and it is the
reason MAXIM's GPU weights are 74.1 MiB against 54.1 MiB CPU-only. For a consumer
that adopts it, the cost is not only the extra blob but the requirement that the
plan build BOTH layouts under the `cuda` feature, which is what `maxim-rs`'s
`Op::Conv1x1 { wt, .. }` threads through. And `lg_conv1x1` cannot simply be
rewritten in place, for a different reason: the originals must stay defined and
launchable so the speedup stays MEASURABLE (`--legacy-ops`) rather than asserted,
and that requirement comes from the same pass that produced these numbers.

**Verdict:** both stay in `maxim-rs`. `lg_conv1x1` and `lg_conv3x3s1p1` are
unchanged here - they are what the tiled forms are measured against, and they
remain the right kernel for a caller whose widths are not a multiple of 64, which
is most of them.

## 5. Numerics

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

## 6. Adding a kernel

1. Write the CUDA kernel here, following the rules above.
2. Add the CPU twin in `src/ops/cpu.rs`, same name, same arithmetic order.
3. Re-run `gpuinfo`: it resolves every name in `src/ops/mod.rs` against the
   loaded module, so a kernel that is added to the `.cu` but not to the table
   (or vice versa) fails the smoke test rather than a caller's first launch.
4. If the kernel is one of a pair where two implementations are both worth
   keeping, keep both and document the crossover point, as the attention
   section does.

## 7. Driver layer

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
