//! The kernel set, by name, with each op's layout contract.
//!
//! The CUDA implementation is `cuda/kernels.cu` (embedded as [`crate::FATBIN`]);
//! the arithmetic twins are in [`cpu`]. An engine looks kernels up by name, so
//! this table is the toolkit's public surface: [`NAMES`] is what
//! `Module::func` will successfully resolve, and [`OPS`] carries the one-line
//! contract each op obeys.

pub mod cpu;

/// One kernel: its fatbin name and the layout contract it implements.
///
/// No `PartialEq`: two `fn` pointers are not reliably comparable.
#[derive(Debug, Clone, Copy)]
pub struct Op {
    pub name: &'static str,
    pub doc: &'static str,
}

/// Every kernel the toolkit ships, in the order `cuda/kernels.cu` declares them.
pub const OPS: &[Op] = &[
    Op { name: "lg_noop", doc: "Trivial entry point: loading the module and calling this proves the whole driver path works (`lightgpu`'s `gpuinfo` does exactly that)." },
    Op { name: "lg_rms_norm", doc: "RMSNorm (Qwen2 LM): y = x * rsqrt(mean(x^2) + eps) * w, per-row." },
    Op { name: "lg_layer_norm", doc: "LayerNorm: y = (x - mean) * rsqrt(var + eps) * w + b, per-row." },
    Op { name: "lg_layer_norm_2pass", doc: "LayerNorm, two-pass form (mean, then the variance of (x - mean)). Stabler when the mean is large relative to the spread; see the note above on why it is not the default for the LocateAnything checkpoints." },
    Op { name: "lg_rope_neox", doc: "NEOX (half-split) convention - Qwen2 LM. head_dim must be even. x: [hd, nh, ntok] (hd contiguous), pair i <-> i + hd/2, pos per token." },
    Op { name: "lg_rope_2d", doc: "2-D RoPE (MoonViT): head_dim split into adjacent pairs; cos/sin tables are precomputed host-side (they depend on the token grid) and passed as [n_pairs, ntok]." },
    Op { name: "lg_silu_mul", doc: "y = silu(gate) * up" },
    Op { name: "lg_gelu_tanh", doc: "y = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))" },
    Op { name: "lg_gelu_erf", doc: "GELU, erf form: y = 0.5 x (1 + erf(x / sqrt(2)))" },
    Op { name: "lg_relu", doc: "y = max(x, 0), elementwise over n elements. The plain ReLU, with no slope argument (contrast `lg_lrelu`)." },
    Op { name: "lg_sigmoid", doc: "y = 1 / (1 + exp(-x)), elementwise over n elements. A standalone op rather than something folded into a fused kernel, because three engines need it alone (MAXIM's CALayer, lama's output layer, rmbg's attention). NCHW-agnostic: any length, and `y` may be the same buffer as `x`." },
    Op { name: "lg_add", doc: "y = a + b" },
    Op { name: "lg_add_inplace", doc: "y += x, in place. The second in-place op in this file, and for the same reason as `lg_silu_mul`: a residual accumulate where an out-of-place form would need one more buffer of the activation size per layer." },
    Op { name: "lg_scale", doc: "y = x * s (scalar)" },
    Op { name: "lg_row_affine", doc: "y[i][j] = x[i][j] * scale[j] (+ shift[j] when non-null). x is [ne0, nrows] with ne0 contiguous, so the per-row vector is indexed by the CONTIGUOUS dimension. MERGE DECISION: this replaces lg_add_bias (scale=1). CORRECTION: it does NOT subsume a per-channel affine over NCHW - that op indexes its vector by CHANNEL, and no choice of ne0/nrows reproduces it because NCHW fixes which dimension is contiguous. Use lg_channel_affine for that." },
    Op { name: "lg_channel_layer_norm", doc: "LayerNorm over the CHANNEL axis of an NCHW tensor: for each of the hw spatial positions, normalize the c values spaced hw apart, with w/b indexed by channel. Distinct from lg_layer_norm, which normalizes the CONTIGUOUS ne0 within each row and therefore reads the wrong axis whenever an NCHW tensor has hw > 1. One block per spatial position; the reduction scratch is static shared memory sized for the largest legal blockDim (1024), so the caller passes NO dynamic shared memory." },
    Op { name: "lg_channel_affine", doc: "NCHW per-CHANNEL affine: out[c][p] = in[c][p] * scale[c] + shift[c], p over each channel's contiguous hw. The folded-BatchNorm op. Either scale or shift may be null (pure scale / pure bias), and `in` may alias `out`. NOT expressible with lg_row_affine, which indexes the CONTIGUOUS dimension - for NCHW that is the width, not the channel." },
    Op { name: "lg_channel_scale", doc: "out[c][p] = in[c][p] * s[c] - the `shift = null` specialisation of lg_channel_affine, kept as its own entry point because its three-pointer (in, s, out) argument order is what the scale-only callers already build. Use it where `s` is an activation rather than a weight." },
    Op { name: "lg_copy", doc: "y = x, n elements. Kept as a kernel because engines use it to make an explicit device-side copy when they cannot alias the buffer." },
    Op { name: "lg_channel_mean", doc: "Per-channel spatial mean over an NCHW plane: out[c] = mean_p in[c][p], one block per channel, grid = (c, 1, 1). SUMMATION ORDER IS THE CONTRACT: strided per-thread partials (i = t, t + blockDim, ...) followed by a halving tree over the full 1024 scratch slots, so it needs no power-of-two block and has no unpaired-partial case. The CPU twin reproduces that order for the same block size, so a backend difference is a bug and not a reordering. Static shared scratch (1024), so the caller passes NO dynamic shared memory." },
    Op { name: "lg_argmax", doc: "Argmax over the ne0 rows of a [ne0, ncols] buffer, one block per column. Also writes the max value." },
    Op { name: "lg_conv1x1", doc: "out[n][oc][y][x] = bias[oc] + sum_ic w[oc][ic] * in[n][ic][y][x]" },
    Op { name: "lg_conv3x3s1p1", doc: "3x3, stride 1, pad 1. Accumulation order matches the CPU twin: ky, kx, ci." },
    Op { name: "lg_conv3x3_winograd", doc: "3x3, stride 1, pad 1 - the SAME op as lg_conv3x3s1p1, with an F(4x4,3x3) Winograd implementation: 4/9ths the multiplies at the cost of a transform. Same layout and weight order ([c_out][c_in][3][3], ci contiguous). Extra args: c_chunk (input-channel chunk), ocb (output channels per CTA, <= 16), act (0 none, 1 relu, 2 lrelu with slope act_p). grid = (ceil(wd/16), ceil(h/16), ceil(c_out/ocb)), block = (256,1,1), shared = (c_chunk*16 + c_chunk*ocb)*36 floats. Choose by c_in: it LOSES at c_in=3, where the transform is not amortised, and wins from c_in=32 up." },
    Op { name: "lg_conv4x4s4", doc: "4x4 patch embed, stride 4, no padding. Order: ci, ky, kx." },
    Op { name: "lg_conv_kxk", doc: "Generic k x k, stride 1, pad k/2. Order: ky, kx, ci." },
    Op { name: "lg_linear_1x1", doc: "1x1 conv with a 1x1 spatial input (the ASPP gap branch): out[oc] = sum_ic w * x" },
    Op { name: "lg_extract_rows", doc: "dst [nrows, ntok] <- src [src_stride, ntok] (first nrows rows). Used to split the ViT qkv buffer (3456 rows) into q/k/v (1152 rows each). One block per TOKEN: consecutive threads then touch consecutive addresses in both source and destination. Iterating rows with grid.x instead would make every thread in a warp touch a different row, i.e. ~13.8 KB apart - fully uncoalesced (measured: ~1 s of the ViT)." },
    Op { name: "lg_merge_2x2", doc: "2x2 patch merge (MoonViT -> connector), matching merge_patches(): out[m*4C + e_idx*C + d] = vf[tok*C + d] e_idx = b*2+e, tok = (2a+b)*gw + (2c+e), m = a*mw + c" },
    Op { name: "lg_get_row_q8_0_aligned", doc: "Decode helper: dequantize ONE row of an aligned 36-byte q8_0 weight table, with the row index passed as a kernel ARGUMENT (no ids upload). grid = (ceil(ne0/256), 1), block = (256,1,1). Writes ne0 f32." },
    Op { name: "lg_copy_row", doc: "Async device-to-device row append for a KV cache. Replaces per-layer synchronous cuMemcpyDtoD calls that flush the pipeline." },
    Op { name: "lg_set_i32", doc: "Write a single i32 from a host value without a pageable H2D copy (used for the decode position scalar)." },
    Op { name: "lg_attn_gqa", doc: "" },
    Op { name: "lg_attn_flash", doc: "Self-attention with K/V staged in shared memory (ViT and full prefill). Identical arithmetic to lg_attn_gqa, but a block owns (head, W query tokens) and reads each K/V element from DRAM once per block instead of once per query token. For the ViT the naive version is ~9.7 GB of traffic per layer (~39 ms); staging into smem cuts it by ~W." },
    Op { name: "lg_linear", doc: "Linear on the row-major token layout: out[i*C_out + o] = bias[o] + sum_c x[i*C_in + c] * w[o*C_in + c], accumulated in c order. Tiled 16x16 with the shared-memory trick that makes the weight tile readable as ws[tx][k] against xs[ty][k]." },
    Op { name: "lg_f32_gemm", doc: "Naive one-thread-per-output. Correct and layout-obvious; the tiled variant below is what large shapes should call." },
    Op { name: "lg_f32_gemm_tiled", doc: "f32 GEMM, v8-style tiling: 64 rows x 32 columns per 256-thread block (32 row lanes x 8 column slots, 2 rows x 4 columns per thread = 8 accumulators). The 32 columns' current k-step (one float4 each) is staged in shared memory so the activation float4 is loaded once per block-k-step instead of once per row lane. No k-split: the tiles are numerous enough. grid = (ceil(ne1/64), ceil(ncols/32)). Requires ne0 % 4 == 0." },
    Op { name: "lg_quantize_q8_0", doc: "Standalone activation quantizer: x [ne0, ncols] -> int8 qs [ne0, ncols] + per-32-block f32 scales sc [ne0/32, ncols]. One warp per (block, column). Running this once per GEMM removes the per-row-block recomputation that made the fused dp4a kernel slow. NOTE: scales are stored [ncols][nb] (column-major over blocks) so the GEMM can walk a column's scales with stride 1: sc[c * nb + b]." },
    Op { name: "lg_q8_0_gemm_dp4a", doc: "dp4a GEMM over the ALIGNED 36-byte layout: 64 rows x 32 cols per 256-thread block, 32 row lanes x 8 column slots, 2 rows x 4 columns per thread. The k-block's 32 int8 weights and the staged activation are both read as aligned 4-byte ints (8 int loads per block instead of 64 byte ops). grid = (ceil(ne1/64), ceil(ncols/32))." },
    Op { name: "lg_q8_0_gemm_aligned", doc: "Scalar q8_0 GEMM over the ALIGNED 36-byte layout: one thread per output element, used for the single-column lm_head projection. grid = (ceil(ne1/64), ceil(ncols/32)), block = (256,1,1). y[col][row]." },
    Op { name: "lg_q8_0_gemv", doc: "" },
    Op { name: "lg_lrelu", doc: "y = x >= 0 ? x : slope * x, whole-plane, slope as an argument. Distinct from lg_relu, whose `int n` token-length signature does not carry a float operand." },
    Op { name: "lg_add_scaled", doc: "y = a + s * b, whole-plane: `out = skip + scale * branch` without materialising s*b in another plane. Length is `long`, matching lg_copy rather than lg_add's `int`." },
    Op { name: "lg_upsample2x_nearest", doc: "Nearest-neighbour 2x upsample, NCHW: out[c][oy][ox] = in[c][oy/2][ox/2]. The integer halving is the CONTRACT - it is what PyTorch's F.interpolate(scale_factor=2, mode='nearest') does; mapping through (o+0.5)/2 gives different pixels. grid = (ceil(2w/32), ceil(2h/8)), block = (32,8,1)." },
    Op { name: "lg_pixel_unshuffle2", doc: "Space-to-depth: [C][H][W] -> [4C][H/2][W/2] with output channel c*4 + dy*2 + dx from input channel c at (2y+dy, 2x+dx) - PyTorch's pixel_unshuffle(x, 2). The PERMUTATION is the contract, not just the shape: a following conv's weights are ordered by it. grid = (ceil((w/2)/32), ceil((h/2)/8)). NOT lg_merge_2x2, which merges the same taps into a token-major layout." },
    Op { name: "lg_fft2_r2c", doc: "Batched 2-D real-to-complex FFT. in is [batch][n][n] row-major, out is [batch][n][n/2+1] INTERLEAVED complex - the half spectrum rfftn defines. n <= 64 (the plane lives in shared memory), nb = log2(n) selects the bit-reversal width. UNNORMALISED, like cuFFT: pass scale = 1.0f and apply 1/sqrt(n*n) yourself, or fold the ortho factor in by passing it as scale. grid = (batch, 1, 1), block = (256, 1, 1), no dynamic shared memory." },
    Op { name: "lg_fft2_c2r", doc: "Batched 2-D complex-to-real inverse FFT: in is the [batch][n][n/2+1] interleaved half spectrum lg_fft2_r2c produces, out is [batch][n][n] real. The full spectrum is rebuilt by conjugate symmetry internally, so the caller stores only the half spectrum. UNNORMALISED: pass scale = 1/sqrt(n*n) for the ortho inverse (and undo any scale the forward folded in). grid = (batch, 1, 1), block = (256, 1, 1)." },
];

/// Kernel names, for a completeness check against a loaded module.
pub static NAMES: &[&str] = &[
    "lg_noop",
    "lg_rms_norm",
    "lg_layer_norm",
    "lg_layer_norm_2pass",
    "lg_rope_neox",
    "lg_rope_2d",
    "lg_silu_mul",
    "lg_gelu_tanh",
    "lg_gelu_erf",
    "lg_relu",
    "lg_sigmoid",
    "lg_add",
    "lg_add_inplace",
    "lg_scale",
    "lg_row_affine",
    "lg_channel_layer_norm",
    "lg_channel_affine",
    "lg_channel_scale",
    "lg_copy",
    "lg_channel_mean",
    "lg_argmax",
    "lg_conv1x1",
    "lg_conv3x3s1p1",
    "lg_conv3x3_winograd",
    "lg_conv4x4s4",
    "lg_conv_kxk",
    "lg_linear_1x1",
    "lg_extract_rows",
    "lg_merge_2x2",
    "lg_get_row_q8_0_aligned",
    "lg_copy_row",
    "lg_set_i32",
    "lg_attn_gqa",
    "lg_attn_flash",
    "lg_linear",
    "lg_f32_gemm",
    "lg_f32_gemm_tiled",
    "lg_quantize_q8_0",
    "lg_q8_0_gemm_dp4a",
    "lg_q8_0_gemm_aligned",
    "lg_q8_0_gemv",
    "lg_lrelu",
    "lg_add_scaled",
    "lg_upsample2x_nearest",
    "lg_pixel_unshuffle2",
    "lg_fft2_r2c",
    "lg_fft2_c2r",
];

/// Is this name part of the shipped set?
pub fn has(name: &str) -> bool {
    NAMES.contains(&name)
}
