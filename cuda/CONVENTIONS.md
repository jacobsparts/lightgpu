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
  `lg_silu_mul` and `lg_add_inplace` (out-of-place forms need one more buffer of
  the activation size per layer) plus `lg_channel_affine`, whose element `idx`
  reads and writes only `idx` and which the folded-BatchNorm call site invokes
  with `out == in`.
* Sizes as `int` when the element count provably fits in 2^31 (any single
  tensor in these models does), `long` when a caller could pass a batch that
  overflows (`lg_copy`, the conv family, the gather pair). Getting this wrong
  fails silently on large inputs - reading a length from a `long` slot when the
  kernel wants an `int` is a wrong number, not an error - so it is per-kernel,
  not per-file, and a caller moving onto a toolkit kernel must check the width.
  Moving MAXIM's `mx_mul` (a `long`) onto `lg_mul` (an `int`) is a live example.

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

Kernels that convert between the two say so in their name (`lg_extract_rows`,
`lg_merge_2x2`, `lg_pixel_unshuffle2`) rather than assuming a caller's
convention. A kernel that takes a pitch or a stride parameter must state whether
the pitch is in elements or bytes.

A name appearing in this document is not necessarily a toolkit kernel: the
merge-decision table below cites REJECTED candidates and consumer-side kernels by
name, and six of those (`lg_mul_broadcast`, `lg_softmax_rows`,
`lg_resize_bilinear`, `lg_patch_merge`, `lg_nchw_to_tokens`, `lg_tokens_to_nchw`)
live in `rmbg-rs/cuda/swin.cu` and `lg_add_bias` in no repository at all. The
kernel set is `cuda/kernels.cu` and `src/ops/mod.rs`, and nothing else.

## 3. Merging decisions

Where the two source kernel sets implemented the same operation differently, the
better implementation won. The decisions, and why.

THE TABLE IS A RECORD OF DECISIONS, NOT A CLAIM THAT THEY ARE FINAL. It was written
at merge time, from the two implementations' code and their engines' tests. Two of
its rejections have since been overturned by a MEASUREMENT rather than by an argument,
and rather than edit the history away the corrections are rows of their own, naming
what was measured and on what hardware. Read a "rejected" column as "rejected for
the reason given, at the time" - two of those reasons turned out not to be the ones
that dominate.


| op | chosen | rejected | why |
| --- | --- | --- | --- |
| layer norm variance | two-pass: mean, then variance of `(x - mean)` | one-pass `E[x^2] - mean^2` | the one-pass form cancels catastrophically when the mean is large relative to the spread; the two-pass form is what both engines' CPU twins do |
| `erf` | Abramowitz & Stegun 7.1.26 | hardware `erff` | the hardware variant differs from a plain Rust CPU twin by more than the pipeline's own error, which blunts the per-op GPU-vs-CPU diff. Same formula on both sides means an erf mismatch can never explain a difference |
| elementwise | out-of-place `in, out` | in-place | costs nothing, removes an aliasing rule the caller had to know, and lets the input survive for a residual. `lg_silu_mul` stays in-place because the fused gated-MLP path reuses the gate buffer on purpose |
| elementwise multiply | promoted to the toolkit as `lg_mul(a, b, y, n)` | leaving it where it was, or folding it into a broadcast multiply | generic on the "could another family call this" test, and the GATED architectures do: NAFNet's SimpleGate multiplies the two halves of a channel split and MAXIM's gMLP multiplies a gate by a value. It is not `lg_silu_mul` (fused, in place, consumes its gate) and not `lg_mul_broadcast`, which scales every channel by one shared spatial map and at `hw == 1` is the scalar `a[0]` - the two are not the same op and neither subsumes the other |
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
| 1x1 conv and token linear, REGISTER-BLOCKED (**CORRECTION**, turn 4178) | `lg_conv1x1_rb` and `lg_linear_rb`, one `lg_rb_body<4,4,16,PLANE,BIAS_FIRST>` serving both layouts, 4x4 outputs a thread | the row above, read as a rejection of tiling the 1x1 in its native layout | **This does not contradict the row above, and saying why is the point.** That entry rejects a tiled form whose WEIGHT BLOB IS TRANSPOSED to `[c_in][c_out]` - a second device allocation, a second loader path and a signature no consumer is built against - and that rejection stands. What was rejected by ARGUMENT and later overturned by MEASUREMENT is a tiled form in the NATIVE `[c_out][c_in]` layout: `scunet-rs` wrote one (`sc_conv1x1_t`, 16x16 shared tile, one output a thread) and `examples/k1x1.rs` measured it over the seven stage geometries that engine runs at 3.075 ms a forward against the register-blocked GEMM's 1.345 ms - 1.9-2.5x per geometry. A 4x4 register tile divides the shared-memory loads per FMA by `(TM*TN)/(TM+TN)`, which is 2 per 16 FMAs instead of 1 per 1, and the shape is capped by REGISTERS rather than by the load ratio: `nvcc -Xptxas -v` on sm_61 gives 2x4 79 registers, 4x4 124 (two blocks an SM), 2x8 127, 4x8 164, 8x4 190, 8x8 240, and 256 threads times 240 already exceeds the SM's 65536-register file. `lg_conv1x1_rb` is a bit-exact swap for `lg_conv1x1` - the bias is folded into the accumulator before the first multiply in both, and the accumulation order over `c_in` is unchanged - so both stay, with the plain form as the measurement's other arm |
| layer norm, ONE WARP PER ROW (**CORRECTION**, turn 4178) | `lg_layer_norm_warp`, an addition rather than a replacement | the row above, read as a rejection of a warp-per-row norm | The variance DECISION above (one-pass `E[x^2] - mean^2` as the default, `lg_layer_norm_2pass` on request) is untouched and `lg_layer_norm_warp` uses the same one-pass form and the same reduction tree, so the two agree bit for bit on the part that is a reduction. What the block-per-row kernel got wrong for a TRANSFORMER's rows is the GEOMETRY: two shared-memory trees and two barriers are amortised over `blockDim` elements, and a Swin or BiRefNet normalization row is 32-256 wide, so there is nothing to amortise them against. Measured 2.2x at a 256-wide row and 8.6x at 32 wide (`examples/kpromote.rs`, both kernels in one window). It is an ADDITION, not a replacement: the block form is still the better kernel for the wide rows it was written for, and `lg_layer_norm` is unchanged |
| window index map (gather/scatter), non-merge | `lg_window_gather`/`lg_window_scatter`, signature `(x, tok, nw, n, nww, win, hp, wp, c, shift, w0)` | `swin2sr-rs`'s `ss_window_gather` | It is a SUPERSET, not the common denominator: it fuses the channel LayerNorm and its `eps` into the gather, so a caller that wants only the permutation cannot use it, and its layout is swin2sr's own. Merging it would mean every other engine passing `scale`/`shift`/`eps` it has no use for, or the fused kernel growing a mode branch. `rmbg-rs`'s private pair WAS the common denominator and is gone - its call is this pair at `shift = 0, w0 = 0`, which the same permutation produces because `nw * n == hp * wp` there |
| 3x3 conv, tiled form | `lg_conv3x3s1p1`, one thread per output | a tile staging a 64-pixel row segment plus its halo | such a tile needs `wd % 64 == 0`. As a toolkit precondition that means both kernels stay linked in every consumer and every call site grows a width test; generalising to a partial trailing segment makes the shared-memory size a runtime quantity, i.e. a dynamic shared argument, which `docs/MAINTAINING.md` forbids |
| 3x3 conv and 1x1 conv, TILED WITH STATIC SHARED MEMORY | `lg_conv3x3_tile` and `lg_conv1x1_tile` - the generalization the row above said was impossible | leaving both to the engines that wrote them (`ifan-rs`'s `if_conv_tile_kernel`, measured 9.7x-18.3x here) | **The row above is not contradicted; it is the reason these two exist.** That entry rejects a tile whose width is a PRECONDITION (`wd % 64 == 0` would have to be checked at every call site) or a RUNTIME quantity (dynamic shared memory). `lg_conv3x3_tile` has neither: the tile is fixed at 128 output columns with a partial trailing tile handled by a bounds test on the store, which is statically sized and needs no precondition. Two things it does NOT do, both stated on the op: its accumulation order is (ci, ky, kx) where `lg_conv3x3s1p1` is (ky, kx, ci), so it is a magnitude-bound swap (measured 2.03e-06 worst) and not an equality swap; and its bias is added after the sum rather than folded in first. `lg_conv1x1_tile` is the same treatment for the 1x1 and is the narrower of the two - it wins 4.1x-5.4x where the output is very wide and the plane is not tiny, and LOSES at 64x64 - so it is documented as an addition for one shape, not as a general replacement. The earlier rejected tiled 1x1 remains rejected: this one reads the native `[c_out][c_in]` blob, not a transposed one |
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
