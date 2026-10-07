// lightgpu kernel set - the shared toolkit behind the dependency-light inference
// engines (LocateAnything-3B/MoonViT+Qwen2 and RMBG-2.0/BiRefNet today).
//
// This file is the merger of two independently developed kernel sets:
//   * the transformer set from locate-anything-rs (RMSNorm, RoPE, GQA/flash
//     attention, q8_0 GEMM/GEMV, ggml layouts), and
//   * the vision/CNN set from rmbg-rs (conv, deformable conv, resize, window
//     attention, NCHW and token layouts).
// Where the two implemented the same op differently, the better one won; see
// CONVENTIONS.md for the per-op decisions and the resulting rules.
//
// EVERY kernel here follows the rules in CONVENTIONS.md. The two that matter
// most when lifting a kernel between engines:
//
//   1. `extern "C"` names with the `lg_` prefix, no app-specific structs, no
//      globals, raw pointers plus scalars only - so the same file serves any
//      engine and is looked up by name through the driver API.
//   2. One definition per op and an arithmetic twin on the CPU
//      (`src/ops/cpu.rs`) that keeps the same order of operations, so a GPU
//      change that still matches it has not changed the arithmetic. The twin is
//      also the CPU backend, and is tuned on its own terms.
//
// Numerics: no fast math, no flush-to-zero (--ftz=false --prec-div=true
// --prec-sqrt=true --fmad=true). Reductions keep the same order as the CPU
// twin wherever that was cheap, so a mismatch between backends means a real
// bug rather than a different summation order.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define LA_DEVI __device__ __forceinline__

// ===========================================================================
// 1. Smoke test
// ===========================================================================

// Trivial entry point: loading the module and calling this proves the whole
// driver path works (`lightgpu`'s `gpuinfo` does exactly that).
extern "C" __global__ void lg_noop() {}

// ===========================================================================
// 2. Norms (rows are the reduction axis; x is [ne0, nrows] with ne0 contiguous)
// ===========================================================================

// RMSNorm (Qwen2 LM): y = x * rsqrt(mean(x^2) + eps) * w, per-row.
extern "C" __global__ void lg_rms_norm(
    const float *__restrict__ x, const float *__restrict__ w, float *__restrict__ y,
    int ne0, int nrows, float eps)
{
    const int r = blockIdx.x;
    if (r >= nrows) return;
    const float *xr = x + (size_t)r * ne0;
    float *yr = y + (size_t)r * ne0;
    float ss = 0.f;
    for (int i = threadIdx.x; i < ne0; i += blockDim.x) ss += xr[i] * xr[i];
    __shared__ float red[256];
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float scale = rsqrtf(red[0] / (float)ne0 + eps);
    for (int i = threadIdx.x; i < ne0; i += blockDim.x) yr[i] = xr[i] * scale * w[i];
}

// LayerNorm: y = (x - mean) * rsqrt(var + eps) * w + b, per-row.
//
// MERGE DECISION: the two source sets differed here. rmbg-rs computed the mean
// then the variance of (x - mean), which is the numerically stabler form for
// spread-out data. locate-anything-rs used the one-pass E[x^2] - mean^2, which
// is what the LocateAnything checkpoints were verified against - and this model
// is hypersensitive to tiny perturbations (a 2e-5 seed per layer is amplified by
// the int8 activation quantization, see the engine's README), so the two forms
// are NOT interchangeable in practice even though both are "correct". The
// one-pass form is therefore the default here. Use `lg_layer_norm_2pass` when
// a caller wants the stable form on well-conditioned data.
extern "C" __global__ void lg_layer_norm(
    const float *__restrict__ x, const float *__restrict__ w, const float *__restrict__ b,
    float *__restrict__ y, int ne0, int nrows, float eps)
{
    const int r = blockIdx.x;
    if (r >= nrows) return;
    const float *xr = x + (size_t)r * ne0;
    float *yr = y + (size_t)r * ne0;
    float s1 = 0.f, s2 = 0.f;
    for (int i = threadIdx.x; i < ne0; i += blockDim.x) { s1 += xr[i]; s2 += xr[i] * xr[i]; }
    // Fixed arrays sized to the LARGEST legal blockDim (1024), not to 256: a
    // 256-entry array is out of bounds for any caller that sizes blockDim from
    // a channel count above 256, which corrupted shared memory silently and
    // surfaced as NaN deep inside a consumer's model (no small-tensor test here
    // could see it). Fixed rather than dynamic on purpose: a kernel must not
    // acquire a shared-memory requirement that existing callers cannot satisfy,
    // and `lg_layer_norm` is launched by callers that pass no shared memory at
    // all - making it dynamic turned that into an illegal shared store.
    __shared__ float r1[1024], r2[1024];
    r1[threadIdx.x] = s1; r2[threadIdx.x] = s2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) { r1[threadIdx.x] += r1[threadIdx.x + s]; r2[threadIdx.x] += r2[threadIdx.x + s]; }
        __syncthreads();
    }
    const float n = (float)ne0;
    const float mean = r1[0] / n;
    const float var = r2[0] / n - mean * mean;
    const float scale = rsqrtf(fmaxf(var, 0.f) + eps);
    for (int i = threadIdx.x; i < ne0; i += blockDim.x) yr[i] = (xr[i] - mean) * scale * w[i] + b[i];
}

// LayerNorm, two-pass form (mean, then the variance of (x - mean)). Stabler
// when the mean is large relative to the spread; see the note above on why it is
// not the default for the LocateAnything checkpoints.
extern "C" __global__ void lg_layer_norm_2pass(
    const float *__restrict__ x, const float *__restrict__ w, const float *__restrict__ b,
    float *__restrict__ y, int ne0, int nrows, float eps)
{
    const int r = blockIdx.x;
    if (r >= nrows) return;
    __shared__ float rsum[256];
    __shared__ float rvar[256];
    const float *xr = x + (size_t)r * ne0;
    float *yr = y + (size_t)r * ne0;
    float s = 0.f;
    for (int i = threadIdx.x; i < ne0; i += blockDim.x) s += xr[i];
    rsum[threadIdx.x] = s;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if ((int)threadIdx.x < st) rsum[threadIdx.x] += rsum[threadIdx.x + st];
        __syncthreads();
    }
    const float mean = rsum[0] / (float)ne0;
    float v = 0.f;
    for (int i = threadIdx.x; i < ne0; i += blockDim.x) {
        const float d = xr[i] - mean;
        v += d * d;
    }
    rvar[threadIdx.x] = v;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if ((int)threadIdx.x < st) rvar[threadIdx.x] += rvar[threadIdx.x + st];
        __syncthreads();
    }
    const float rstd = rsqrtf(rvar[0] / (float)ne0 + eps);
    for (int i = threadIdx.x; i < ne0; i += blockDim.x) yr[i] = (xr[i] - mean) * rstd * w[i] + b[i];
}

// ===========================================================================
// 3. RoPE
// ===========================================================================

// NEOX (half-split) convention - Qwen2 LM. head_dim must be even.
//   x: [hd, nh, ntok] (hd contiguous), pair i <-> i + hd/2, pos per token.
extern "C" __global__ void lg_rope_neox(
    float *__restrict__ x, const int *__restrict__ pos,
    int hd, int nh, int ntok, float theta)
{
    const int half = hd / 2;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y;
    const int t = blockIdx.z;
    if (i >= half || t >= ntok) return;
    const float freq = powf(theta, -2.f * (float)i / (float)hd);
    const float ang = (float)pos[t] * freq;
    const float c = cosf(ang), s = sinf(ang);
    // [hd, nh, ntok] with hd contiguous and ntok slowest: token stride nh*hd.
    float *base = x + ((size_t)t * nh + h) * hd;
    const float x0 = base[i];
    const float x1 = base[i + half];
    base[i]        = x0 * c - x1 * s;
    base[i + half] = x0 * s + x1 * c;
}

// 2-D RoPE (MoonViT): head_dim split into adjacent pairs; cos/sin tables are
// precomputed host-side (they depend on the token grid) and passed as
// [n_pairs, ntok].
extern "C" __global__ void lg_rope_2d(
    float *__restrict__ x, const float *__restrict__ cosb, const float *__restrict__ sinb,
    int hd, int nh, int ntok)
{
    const int np = hd / 2;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y;
    const int t = blockIdx.z;
    if (k >= np || t >= ntok) return;
    const float c = cosb[(size_t)k * ntok + t];
    const float s = sinb[(size_t)k * ntok + t];
    float *base = x + ((size_t)t * nh + h) * hd;
    const float x0 = base[2 * k];
    const float x1 = base[2 * k + 1];
    base[2 * k]     = x0 * c - x1 * s;
    base[2 * k + 1] = x0 * s + x1 * c;
}

// ===========================================================================
// 4. Elementwise
// ===========================================================================

// MERGE DECISION: locate-anything-rs's elementwise kernels wrote in place
// (gate, up -> gate) while rmbg-rs's took separate in/out. The out-of-place
// form wins: it costs nothing, removes an aliasing rule the caller had to know,
// and lets a caller keep the input for a residual. `lg_silu_mul` is the only
// one kept in-place, because the engine's fused gated-MLP path reuses the gate
// buffer as the output on purpose.

// y = silu(gate) * up
extern "C" __global__ void lg_silu_mul(
    float *__restrict__ gate, const float *__restrict__ up, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) gate[i] = (gate[i] / (1.f + expf(-gate[i]))) * up[i];
}

// y = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
extern "C" __global__ void lg_gelu_tanh(
    const float *__restrict__ x, float *__restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    const float c = 0.7978845608028654f;
    y[i] = 0.5f * v * (1.f + tanhf(c * (v + 0.044715f * v * v * v)));
}

// erf(x) by the Abramowitz & Stegun 7.1.26 approximation (|error| ~1.5e-7).
//
// MERGE DECISION: locate-anything-rs called the hardware `erff`, rmbg-rs used
// this approximation so that its GPU and CPU paths agree exactly. The
// approximation wins for a toolkit: the hardware `erff` differs from a plain
// Rust CPU twin by more than the rest of the pipeline, which would blunt the
// per-op GPU-vs-CPU diff that catches real bugs. Both backends here call the
// same formula, so an erf mismatch can never be the explanation for a
// difference.
LA_DEVI float la_erf(float v) {
    const float sign = v < 0.0f ? -1.0f : 1.0f;
    const float x = fabsf(v);
    const float t = 1.0f / (1.0f + 0.3275911f * x);
    const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f)
                     * t * __expf(-x * x);
    return sign * y;
}

// GELU, erf form: y = 0.5 x (1 + erf(x / sqrt(2)))
extern "C" __global__ void lg_gelu_erf(
    const float *__restrict__ x, float *__restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    y[i] = 0.5f * v * (1.0f + la_erf(v * 0.70710678118654752440f));
}

// y = max(x, 0), elementwise over n elements. Unlike `lg_lrelu` this carries no
// slope argument, so it is the plain ReLU the vision path applies after a conv.
extern "C" __global__ void lg_relu(
    const float *__restrict__ x, float *__restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    y[i] = v < 0.0f ? 0.0f : v;
}

// y = 1 / (1 + __expf(-x)), elementwise over n elements. Standalone rather than
// folded into a fused kernel, because the engines that need it need it alone.
// `x` may alias `y`: the kernel reads element i and writes element i and nothing
// else, so an in-place call is the same pointer twice.
//
// NUMERICS: `__expf` is the fast exponential. The CPU twin uses the same form,
// so the two agree; a caller that needs `expf` bit-exactness wants its own
// kernel rather than this one.
extern "C" __global__ void lg_sigmoid(
    const float *__restrict__ x, float *__restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = 1.0f / (1.0f + __expf(-x[i]));
}

// y = a + b
extern "C" __global__ void lg_add(
    const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

// y = a * b, whole-plane, elementwise. The multiplicative twin of `lg_add` and
// the plainest op there is, but it had no home in the toolkit: the one engine
// that needed it shipped its own copy (MAXIM's `mx_mul`), because nothing in
// the transformer engines needed a multiply. It is NOT
// `lg_silu_mul` (which is a fused gated-MLP step, in place, and consumes its
// gate), and it is NOT `lg_mul_broadcast`, which is a different op: that one
// multiplies every channel by ONE shared spatial map (`x[i] *= a[i % hw]`),
// and at hw == 1 it would multiply by the scalar `a[0]` rather than elementwise.
//
// The callers this is for are the gated architectures: NAFNet's SimpleGate
// splits the channels in half and multiplies the halves, and MAXIM's gMLP
// multiplies a gate by a value. `int n` rather than `long`, like `lg_add` and
// `lg_scale` beside it: a single plane is the largest operand any of these
// models produces, and none of them approaches 2^31 elements.
extern "C" __global__ void lg_mul(
    const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i];
}

// y += x, in place. The second in-place op in this file, and for the same
// reason as `lg_silu_mul`: a residual accumulate where an out-of-place form
// would need one more buffer of the activation size per layer.
extern "C" __global__ void lg_add_inplace(float *__restrict__ a, const float *__restrict__ b, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

// y = x * s (scalar)
extern "C" __global__ void lg_scale(
    const float *__restrict__ x, float *__restrict__ y, float s, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * s;
}

// y[i][j] = x[i][j] * scale[j] (+ shift[j] when non-null)
// x is [ne0, nrows] with ne0 contiguous and the per-row vector of length ne0.
// MERGE DECISION: this replaces la_add_bias (bias add: scale=1) and any
// affine whose vector indexes the CONTIGUOUS dimension.
// CORRECTION: an earlier note here claimed this also subsumes the NCHW
// per-channel affine (rmbg-rs's k_affine_channels). It does NOT: that op
// indexes its vector by CHANNEL and applies it across each channel's
// contiguous hw, and no caller-side choice of ne0/nrows reproduces it,
// because NCHW fixes which dimension is contiguous. See lg_channel_affine.
extern "C" __global__ void lg_row_affine(
    float *__restrict__ x, const float *__restrict__ scale, const float *__restrict__ shift,
    int ne0, int nrows)
{
    const int r = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne0 || r >= nrows) return;
    float *p = x + (size_t)r * ne0 + i;
    // Both parameters are optional so that one kernel covers a pure bias add
    // (null scale), a pure scale (null shift) and an affine. A null scale is
    // NOT an error: it is how `add_bias` is expressed.
    float v = *p * (scale ? scale[i] : 1.0f);
    if (shift) v += shift[i];
    *p = v;
}

// LayerNorm over the CHANNEL axis of an NCHW tensor: for each of the hw
// spatial positions, normalize the c values spaced hw apart, with w/b indexed
// by channel. This is NOT lg_layer_norm: that one normalizes over the
// CONTIGUOUS ne0 within each row (the last-axis case, xr = x + r*ne0), and for
// an NCHW tensor whose contiguous dimension is hw the two differ as soon as
// hw > 1 - lg_layer_norm would then sum across channels of the wrong axis.
//
// ONE THREAD PER SPATIAL POSITION, each walking the c values serially, and no
// shared memory. Nothing here needs a block: the reduction is over c, which the
// same thread can do better than a block can.
//
// THIS REPLACED a one-BLOCK-per-spatial-position form: grid (hw, 1, 1), the c
// values reduced by a halving tree over 1024-slot shared arrays r1[]/r2[]. It
// was launched with blockDim 256 and MAXIM's c is 32, so 224 of the 256 threads
// accumulated nothing and then idled through an 8-step tree to collapse 32 live
// values - and the gather `xp[i * hw]` is strided by the whole plane, so a warp
// asked for 32 separate cache lines instead of one. Measured on the GTX 1080 at
// c=32, hw=286720 (MAXIM's largest shape):
//     block-per-position   4.17-4.29 ms/launch    25.7 GB/s
//     this form            0.470 ms/launch      234 GB/s        (8.9x)
// and 4.5x / 3.0x / 8.3x at the shapes below it (c=64 hw=71680, c=128 hw=17920,
// c=32 hw=16384). Consecutive threads now read consecutive `p` for the same
// channel, so each (channel, warp) load is one coalesced transaction.
//
// THE SUMMATION ORDER IS THEREFORE SERIAL AND ASCENDING OVER c, where the old
// form's was a tree. That is a last-bit difference (measured: 2030138 of
// 9175040 elements at c=32 hw=286720, worst 4.77e-07 - one ulp) and no comment
// here ever promised the tree, so this is not a documented contract being
// broken. It is still a behaviour change, and it is deliberate: an engine whose
// CPU twin reproduces the old tree order now differs from this kernel in the
// last bit, and its twin should be treated as the thing to update, since a
// serial sum is both faster here and the ordinary order for a scalar reference.
// (Contrast lg_channel_mean, whose summation order IS contractual and is written
// out below for that reason.)
extern "C" __global__ void lg_channel_layer_norm(
    const float *__restrict__ x, const float *__restrict__ w, const float *__restrict__ b,
    float *__restrict__ y, int c, int hw, float eps)
{
    const long p = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= hw) return;
    const float *xp = x + p;
    float *yp = y + p;
    float s1 = 0.f, s2 = 0.f;
    for (int i = 0; i < c; ++i) {
        const float v = xp[(size_t)i * hw];
        s1 += v;
        s2 += v * v;
    }
    const float n = (float)c;
    const float mean = s1 / n;
    // One-pass E[x^2] - mean^2, as this kernel has always used: the two-pass
    // form is numerically better but would be a second, larger change.
    const float var = s2 / n - mean * mean;
    const float scale = rsqrtf(fmaxf(var, 0.f) + eps);
    for (int i = 0; i < c; ++i) {
        const size_t o = (size_t)i * hw;
        yp[o] = (xp[o] - mean) * scale * w[i] + b[i];
    }
}

// NCHW per-CHANNEL affine: out[c][p] = in[c][p] * scale[c] + shift[c], p running
// over the contiguous hw of each channel. The op an NCHW engine needs after
// folding a BatchNorm.
//
// MERGE DECISION: it is NOT expressible with `lg_row_affine`, which indexes its
// vector by the CONTIGUOUS dimension - for NCHW that is the width, not the
// channel. See the correction on `lg_row_affine`.
//
// Either parameter may be null for a pure scale (shift = null) or a pure bias
// (scale = null, then scale[c] = 1). `in` may alias `out`: element idx reads and
// writes only idx, and the channel index is derived from idx itself, so a call
// with out == in is safe and is the form an in-place folded BatchNorm wants.
extern "C" __global__ void lg_channel_affine(
    const float *__restrict__ in, float *__restrict__ out,
    const float *__restrict__ scale, const float *__restrict__ shift,
    int c, int hw)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)c * hw;
    if (idx >= total) return;
    const int ch = (int)(idx / hw);
    float v = in[idx] * (scale ? scale[ch] : 1.0f);
    if (shift) v += shift[ch];
    out[idx] = v;
}

// out[c][p] = in[c][p] * s[c] - the `shift = null` case of lg_channel_affine,
// kept as its own entry point rather than folded into it.
//
// MERGE DECISION: they are the same operation, so one kernel with an optional
// shift would do; two are kept because the scale-only callers (MAXIM's CALayer
// `x * sigmoid(y)`, whose s is an ACTIVATION buffer that must not be confused
// with a weight) already build the three-pointer list (in, s, out), and a
// variant that reads the affine's `(in, out, scale, shift, ...)` argument order
// would be a call-site hazard for no gain. lg_channel_scale is the documented
// specialisation; lg_channel_affine is the general form.
extern "C" __global__ void lg_channel_scale(
    const float *__restrict__ in, const float *__restrict__ s, float *__restrict__ out,
    int c, int hw)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)c * hw;
    if (idx >= total) return;
    const int ch = (int)(idx / hw);
    out[idx] = in[idx] * s[ch];
}

// y = x, n elements. Kept as a kernel because engines use it to make an
// explicit device-side copy when they cannot alias the buffer.
extern "C" __global__ void lg_copy(
    const float *__restrict__ x, float *__restrict__ y, long n)
{
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}

// ===========================================================================
// 5. Reductions
// ===========================================================================

// Mean over the spatial plane, per channel: out[c] = mean_p in[c][p]. NCHW,
// one block per channel (grid = (c, 1, 1), block <= 1024, no dynamic shared
// memory), so c can exceed any launchable block size.
//
// SUMMATION ORDER IS PART OF THE CONTRACT - it is a parity decision, because a
// reordering here moves a model's last bits with nothing else to show for it:
//
//   1. every one of the 1024 scratch slots is zeroed, then
//   2. thread `t` accumulates its own strided partial in slot `t`:
//      r[t] = sum of p[t], p[t + blockDim], p[t + 2*blockDim], ...
//   3. a halving tree over the FULL 1024 slots (r[t] += r[t + s] for s from 512
//      down to 1) reduces them,
//   4. thread 0 writes r[0] / hw.
//
// The tree runs over 1024 slots regardless of blockDim, so it needs no
// power-of-two block and no special case for an odd count: the zeroed tail is
// the padding, and adding zeros cannot change the value. Sizing the tree to the
// actual block instead would have to handle an unpaired trailing partial, and
// the obvious way to write that (`else if (t == half) r[half - 1] += r[half]`)
// is a RACE: the thread that owns slot `half` writes r[half - 1] while its
// owner, still alive in that step, reads r[half].
//
// The CPU twin (`cpu::channel_mean`) takes the block size as a parameter and
// reproduces steps 2 and 3 exactly, so the two backends agree bit for bit and a
// difference is a bug rather than a reordering. The scratch is STATIC, sized for
// the largest legal blockDim: docs/MAINTAINING.md's rule that a shared kernel
// must not acquire a requirement its callers satisfy implicitly.
extern "C" __global__ void lg_channel_mean(
    const float *__restrict__ x, float *__restrict__ out, int c, int hw)
{
    __shared__ float r[1024];
    const int ch = blockIdx.x;
    if (ch >= c) return;  // uniform per block: blockIdx.x only, so no early exit
                          // can split a __syncthreads below
    const int t = threadIdx.x;
    const float *p = x + (size_t)ch * hw;
    for (int i = t; i < 1024; i += blockDim.x) r[i] = 0.0f;
    __syncthreads();
    for (int i = t; i < hw; i += blockDim.x) r[t] += p[i];
    __syncthreads();
    for (int s = 512; s > 0; s >>= 1) {
        if (t < s) r[t] += r[t + s];
        __syncthreads();
    }
    if (t == 0) out[ch] = r[0] / (float)hw;
}

// Argmax over the ne0 rows of a [ne0, ncols] buffer, one block per column.
// Also writes the max value.
extern "C" __global__ void lg_argmax(
    const float *__restrict__ x, int *__restrict__ idx_out, float *__restrict__ val_out,
    int ne0, int ncols)
{
    __shared__ float sval[256];
    __shared__ int sidx[256];
    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    float best = -INFINITY;
    int bi = 0;
    for (int i = tid; i < ne0; i += blockDim.x) {
        const float v = x[(size_t)c * ne0 + i];
        if (v > best) { best = v; bi = i; }
    }
    sval[tid] = best; sidx[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sval[tid + s] > sval[tid]) { sval[tid] = sval[tid + s]; sidx[tid] = sidx[tid + s]; }
        }
        __syncthreads();
    }
    if (tid == 0) { idx_out[c] = sidx[0]; val_out[c] = sval[0]; }
}

// ===========================================================================
// 6. Convolutions (NCHW)
// ===========================================================================
//
// Indices that can exceed 2^31 are `long` (the element count of a conv over a
// large batch); channel counts are `int`. `lg_conv3x3s1p1` is kept separate from
// the generic `lg_conv_kxk` because its specialized index arithmetic is
// measurably better on the hot path.

// out[n][oc][y][x] = bias[oc] + sum_ic w[oc][ic] * in[n][ic][y][x]
extern "C" __global__ void lg_conv1x1(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)c_out * h * wd;
    if (idx >= total) return;
    const int x = (int)(idx % wd);
    const long t = idx / wd;
    const int y = (int)(t % h);
    const int oc = (int)(t / h);
    const size_t plane = (size_t)h * wd;
    float acc = bias ? bias[oc] : 0.0f;
    const float *wp = w + (size_t)oc * c_in;
    const float *ip = in + (size_t)y * wd + x;
    for (int ic = 0; ic < c_in; ++ic) acc += wp[ic] * ip[(size_t)ic * plane];
    out[idx] = acc;
}

// 3x3, stride 1, pad 1. Accumulation order matches the CPU twin: ky, kx, ci.
extern "C" __global__ void lg_conv3x3s1p1(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)c_out * h * wd;
    if (idx >= total) return;
    const int x = (int)(idx % wd);
    const long t = idx / wd;
    const int y = (int)(t % h);
    const int oc = (int)(t / h);
    const size_t plane = (size_t)h * wd;
    float acc = bias ? bias[oc] : 0.0f;
    for (int ky = 0; ky < 3; ++ky) {
        const int iy = y + ky - 1;
        if (iy < 0 || iy >= h) continue;
        for (int kx = 0; kx < 3; ++kx) {
            const int ix = x + kx - 1;
            if (ix < 0 || ix >= wd) continue;
            const size_t off = (size_t)iy * wd + ix;
            const float *wp = w + ((size_t)oc * c_in) * 9 + ky * 3 + kx;
            for (int ci = 0; ci < c_in; ++ci) {
                const float wv = wp[(size_t)ci * 9];
                if (wv == 0.0f) continue;
                acc += wv * in[(size_t)ci * plane + off];
            }
        }
    }
    out[idx] = acc;
}

// 4x4 patch embed, stride 4, no padding. Order: ci, ky, kx.
extern "C" __global__ void lg_conv4x4s4(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd)
{
    const int oh = h / 4, ow = wd / 4;
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)c_out * oh * ow;
    if (idx >= total) return;
    const int ox = (int)(idx % ow);
    const long t = idx / ow;
    const int oy = (int)(t % oh);
    const int oc = (int)(t / oh);
    const size_t iplane = (size_t)h * wd;
    float acc = bias ? bias[oc] : 0.0f;
    for (int ci = 0; ci < c_in; ++ci) {
        const float *ip = in + (size_t)ci * iplane;
        const float *wp = w + ((size_t)oc * c_in + ci) * 16;
        for (int ky = 0; ky < 4; ++ky) {
            const int iy = oy * 4 + ky;
            for (int kx = 0; kx < 4; ++kx) {
                acc += ip[(size_t)iy * wd + ox * 4 + kx] * wp[ky * 4 + kx];
            }
        }
    }
    out[idx] = acc;
}

// Generic k x k, stride 1, pad k/2. Order: ky, kx, ci.
extern "C" __global__ void lg_conv_kxk(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, int k)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)c_out * h * wd;
    if (idx >= total) return;
    const int x = (int)(idx % wd);
    const long t = idx / wd;
    const int y = (int)(t % h);
    const int oc = (int)(t / h);
    const int pad = k / 2;
    const int k2 = k * k;
    const size_t plane = (size_t)h * wd;
    float acc = bias ? bias[oc] : 0.0f;
    for (int ky = 0; ky < k; ++ky) {
        const int iy = y + ky - pad;
        if (iy < 0 || iy >= h) continue;
        for (int kx = 0; kx < k; ++kx) {
            const int ix = x + kx - pad;
            if (ix < 0 || ix >= wd) continue;
            const size_t off = (size_t)iy * wd + ix;
            const float *wp = w + ((size_t)oc * c_in) * k2 + ky * k + kx;
            for (int ci = 0; ci < c_in; ++ci) {
                const float wv = wp[(size_t)ci * k2];
                if (wv == 0.0f) continue;
                acc += wv * in[(size_t)ci * plane + off];
            }
        }
    }
    out[idx] = acc;
}

// 1x1 conv with a 1x1 spatial input (the ASPP gap branch): out[oc] = sum_ic w * x
extern "C" __global__ void lg_linear_1x1(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out)
{
    const int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= c_out) return;
    const float *wr = w + (size_t)oc * c_in;
    float acc = bias ? bias[oc] : 0.0f;
    for (int i = 0; i < c_in; ++i) acc += wr[i] * in[i];
    out[oc] = acc;
}


// ---------------------------------------------------------------------------
// Winograd F(4x4,3x3): 3x3 stride-1 pad-1 convolution with a bias and an
// optional activation, at 4/9ths the multiplies of the direct form. The same
// operator as lg_conv3x3s1p1, offered as an alternative implementation rather
// than a new op. It wins only when c_in is large enough to amortise the
// transform, so a 3-channel input should use the direct op and a deep one this.
//
// TWO THINGS ABOUT THIS KERNEL ARE CONTRACTS, NOT STYLE:
//
// 1. THE INVERSE TRANSFORM RUNS ONCE PER THREAD, outside the c_in loop, because
//    A^T m A is linear in m: sum_ci A^T (u_ci * v_ci) A == A^T (sum_ci u_ci *
//    v_ci) A. Running it per ci costs ~96 flops against the 36 multiplies it
//    feeds, which stops the transform being cheaper than what it accelerates.
// 2. THE STAGED STRIDE IS 36. Padding it breaks the 16-byte alignment that lets
//    the inner loop issue 128-bit shared loads, which costs more than the bank
//    conflict it removes.
//
// The caller's shared size is (c_chunk*ntiles + c_chunk*ocb)*36 floats: two
// transformed arrays per channel of the chunk, the input [ci][tile][36] and the
// weights [ci][oc][36]. Sizing it for the input alone puts the weight transform
// back inside the ci loop.
// ---------------------------------------------------------------------------

#define WG4_T     4     // output tile edge (F(4,3): 4 outputs from a 6x6 patch)
#define WG4_ALPHA 6     // transformed tile edge
#define WG4_TILES 4     // tiles per side per CTA -> 4x4 tiles = 16x16 outputs
#define WG4_NT   (WG4_TILES * WG4_TILES)
#define WG4_STRIDE 36   // (not 37: the alignment note above)

// B^T: 6 -> 6, the input transform. Rows then columns.
LA_DEVI void la_wg4_bt(const float *d, float *v) {
    v[0] = 4.0f*d[0] - 5.0f*d[2] + d[4];
    v[1] = -4.0f*d[1] - 4.0f*d[2] + d[3] + d[4];
    v[2] = 4.0f*d[1] - 4.0f*d[2] - d[3] + d[4];
    v[3] = -2.0f*d[1] - d[2] + 2.0f*d[3] + d[4];
    v[4] = 2.0f*d[1] - d[2] - 2.0f*d[3] + d[4];
    v[5] = 4.0f*d[1] - 5.0f*d[3] + d[5];
}

// G: 3 -> 6, the weight transform.
LA_DEVI void la_wg4_g(const float *g, float *u) {
    u[0] = 0.25f*g[0];
    u[1] = -0.1666666667f*g[0] - 0.1666666667f*g[1] - 0.1666666667f*g[2];
    u[2] = -0.1666666667f*g[0] + 0.1666666667f*g[1] - 0.1666666667f*g[2];
    u[3] = 0.0416666667f*g[0] + 0.0833333333f*g[1] + 0.1666666667f*g[2];
    u[4] = 0.0416666667f*g[0] - 0.0833333333f*g[1] + 0.1666666667f*g[2];
    u[5] = g[2];
}

// A^T: 6 -> 4, the output transform.
LA_DEVI void la_wg4_at(const float *m, float *y) {
    y[0] = m[0] + m[1] + m[2] + m[3] + m[4];
    y[1] = m[1] - m[2] + 2.0f*m[3] - 2.0f*m[4];
    y[2] = m[1] + m[2] + 4.0f*m[3] + 4.0f*m[4];
    y[3] = m[1] - m[2] + 8.0f*m[3] - 8.0f*m[4] + m[5];
}

// B^T d B for one 6x6 patch, into 36 values. Two passes of six temporaries.
LA_DEVI void la_wg4_input(const float *patch, int stride, float *v) {
    float col[WG4_ALPHA], out[WG4_ALPHA];
#pragma unroll
    for (int r = 0; r < WG4_ALPHA; ++r) {
#pragma unroll
        for (int c = 0; c < WG4_ALPHA; ++c) col[c] = patch[r * stride + c];
        la_wg4_bt(col, out);
#pragma unroll
        for (int c = 0; c < WG4_ALPHA; ++c) v[r * WG4_ALPHA + c] = out[c];
    }
#pragma unroll
    for (int c = 0; c < WG4_ALPHA; ++c) {
#pragma unroll
        for (int r = 0; r < WG4_ALPHA; ++r) col[r] = v[r * WG4_ALPHA + c];
        la_wg4_bt(col, out);
#pragma unroll
        for (int r = 0; r < WG4_ALPHA; ++r) v[r * WG4_ALPHA + c] = out[r];
    }
}

// G g G^T for one 3x3 weight, into 36 values.
LA_DEVI void la_wg4_weight(const float *wp, float *u) {
    float col[3], out[WG4_ALPHA];
    float mid[WG4_ALPHA * 3];
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        la_wg4_g(wp + r * 3, out);
#pragma unroll
        for (int c = 0; c < WG4_ALPHA; ++c) mid[r * WG4_ALPHA + c] = out[c];
    }
#pragma unroll
    for (int c = 0; c < WG4_ALPHA; ++c) {
#pragma unroll
        for (int r = 0; r < 3; ++r) col[r] = mid[r * WG4_ALPHA + c];
        la_wg4_g(col, out);
#pragma unroll
        for (int r = 0; r < WG4_ALPHA; ++r) u[r * WG4_ALPHA + c] = out[r];
    }
}

// A^T m A, accumulating the 4x4 result.
LA_DEVI void la_wg4_output(const float *m, float *acc) {
    float col[WG4_ALPHA], out[WG4_T];
    float mid[WG4_T * WG4_ALPHA];
#pragma unroll
    for (int r = 0; r < WG4_ALPHA; ++r) {
#pragma unroll
        for (int c = 0; c < WG4_ALPHA; ++c) col[c] = m[r * WG4_ALPHA + c];
        la_wg4_at(col, out);
#pragma unroll
        for (int c = 0; c < WG4_T; ++c) mid[r * WG4_T + c] = out[c];
    }
#pragma unroll
    for (int c = 0; c < WG4_T; ++c) {
#pragma unroll
        for (int r = 0; r < WG4_ALPHA; ++r) col[r] = mid[r * WG4_T + c];
        la_wg4_at(col, out);
#pragma unroll
        for (int r = 0; r < WG4_T; ++r) acc[r * WG4_T + c] += out[r];
    }
}

// F(4x4,3x3) convolution: NCHW in/out, weights [c_out][c_in][3][3] with c_in
// contiguous (stride 9) - the same layout lg_conv3x3s1p1 takes, so a caller can
// swap the two. Grid: (ceil(wd/16), ceil(h/16), ceil(c_out/ocb)), block
// (256,1,1); one thread owns one (tile, output channel) with 36 accumulators.
//
// act: 0 = none, 1 = relu, 2 = leaky relu with slope act_p. Fusing these three
// IS the contract; a tanh- or erf-based activation is a different op with its
// own cost, not a mode flag on this one.
//
// ocb (output channels per CTA) and c_chunk (input channels staged per pass) are
// RUNTIME arguments, so one instantiation serves every channel count. THE CALLER
// MUST AGREE WITH THIS KERNEL ABOUT BOTH: grid.z is ceil(c_out/ocb), ocb must not
// exceed 256/WG4_NT = 16 (the output channels the CTA's threads can own one
// apiece), and the shared size is derived from ocb and c_chunk. Disagreement here
// is silent, not an error.
LA_DEVI void la_wg4_conv(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, int c_chunk, int ocb,
    int act, float act_p)
{
    constexpr int T = WG4_T;
    constexpr int A = WG4_ALPHA;
    constexpr int NT = WG4_NT;

    extern __shared__ float smem[];
    float *tbuf = smem;                                // [ci][tile][36]
    float *ubuf = smem + (size_t)c_chunk * NT * WG4_STRIDE;   // [ci][oc][36]

    const int tid = threadIdx.x;
    const int ox0 = blockIdx.x * (WG4_TILES * T);
    const int oy0 = blockIdx.y * (WG4_TILES * T);
    const int oc0 = blockIdx.z * ocb;

    const int tile = tid % NT;
    const int oc = oc0 + (tid / NT);
    // NOT an early return: there are __syncthreads() calls below and a thread
    // that returns while its CTA-mates wait hangs the kernel.
    const bool active = (oc < c_out);

    const int tr = tile / WG4_TILES, tc = tile % WG4_TILES;
    const int oy = oy0 + tr * T;
    const int ox = ox0 + tc * T;

    const size_t plane = (size_t)h * wd;
    float acc[T * T];
#pragma unroll
    for (int i = 0; i < T * T; ++i) acc[i] = 0.0f;
    float macc[A * A];
#pragma unroll
    for (int i = 0; i < A * A; ++i) macc[i] = 0.0f;

    for (int c0 = 0; c0 < c_in; c0 += c_chunk) {
        const int cc = min(c_chunk, c_in - c0);

        // ---- stage the transformed input: 16 tiles / 256 threads ----
        for (int ci = tid / NT; ci < cc; ci += 256 / NT) {
            const int ci_tile = tid % NT;
            const int r = ci_tile / WG4_TILES, c = ci_tile % WG4_TILES;
            const int ny = oy0 + r * T - 1;
            const int nx = ox0 + c * T - 1;
            float *dst = tbuf + (ci * NT + ci_tile) * WG4_STRIDE;
            float patch[A * A];
            #pragma unroll
            for (int i = 0; i < A; ++i) {
                const int iy = ny + i;
                #pragma unroll
                for (int j = 0; j < A; ++j) {
                    const int ix = nx + j;
                    float val = 0.0f;
                    if (iy >= 0 && iy < h && ix >= 0 && ix < wd)
                        val = in[(size_t)(c0 + ci) * plane + (size_t)iy * wd + ix];
                    patch[i * A + j] = val;
                }
            }
            la_wg4_input(patch, A, dst);
        }
        __syncthreads();

        // ---- stage the transformed weights: cc*ocb pairs, cooperatively ----
        {
            const int npair = cc * ocb;
            for (int p = tid; p < npair; p += 256) {
                const int pci = p / ocb;
                const int poc = p % ocb;
                const int woc = oc0 + poc;
                if (woc >= c_out) continue;   // slot unused; never read
                float *dst = ubuf + (pci * ocb + poc) * WG4_STRIDE;
                const float *wp = w + ((size_t)woc * c_in + (size_t)(c0 + pci)) * 9;
                float u[A * A];
                la_wg4_weight(wp, u);
                #pragma unroll
                for (int i = 0; i < A * A; ++i) dst[i] = u[i];
            }
        }
        __syncthreads();

        // ---- this thread's (tile, oc) reduction ----
        const float *v = tbuf + tile * WG4_STRIDE;
        const int oc_local = tid / NT;
        for (int ci = 0; active && ci < cc; ++ci) {
            const float *u = ubuf + (ci * ocb + oc_local) * WG4_STRIDE;
            const float *vc = v + ci * NT * WG4_STRIDE;
#pragma unroll
            for (int i = 0; i < A * A; ++i) macc[i] += u[i] * vc[i];
        }
        __syncthreads();
    }

    // ONE inverse transform per thread, over the whole reduction (note 1).
    if (active) la_wg4_output(macc, acc);

    if (!active) return;
    const float b = bias ? bias[oc] : 0.0f;
    float *op = out + (size_t)oc * plane;
#pragma unroll
    for (int i = 0; i < T; ++i) {
        const int y = oy + i;
#pragma unroll
        for (int j = 0; j < T; ++j) {
            const int x = ox + j;
            if (x >= wd || y >= h) continue;
            float a = acc[i * T + j] + b;
            if (act == 1) a = a > 0.0f ? a : 0.0f;
            else if (act == 2) a = a >= 0.0f ? a : act_p * a;
            op[(size_t)y * wd + x] = a;
        }
    }
}

// The launch bound is load-bearing: 128 registers x 256 threads is half the
// SM's register file, so two CTAs are resident. Asking ptxas for more resident
// warps takes that working set away by force and spills it to local memory.
extern "C" __global__ void __launch_bounds__(256, 2) lg_conv3x3_winograd(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, int c_chunk, int ocb,
    int act, float act_p)
{
    la_wg4_conv(in, w, bias, out, c_in, c_out, h, wd, c_chunk, ocb, act, act_p);
}



// ===========================================================================
// 7. Layout transposes and gathers
// ===========================================================================

// dst [nrows, ntok] <- src [src_stride, ntok] (first nrows rows).
// Used to split the ViT qkv buffer (3456 rows) into q/k/v (1152 rows each).
// One block per TOKEN: consecutive threads then touch consecutive addresses in
// both source and destination. Iterating rows with grid.x instead would make
// every thread in a warp touch a different row, i.e. ~13.8 KB apart - fully
// uncoalesced (measured: ~1 s of the ViT).
extern "C" __global__ void lg_extract_rows(
    float *__restrict__ dst, const float *__restrict__ src,
    int src_stride, int nrows, int ntok)
{
    const int t = blockIdx.x;
    const float *srow = src + (size_t)t * src_stride;
    float *drow = dst + (size_t)t * nrows;
    for (int i = blockIdx.y * blockDim.x + threadIdx.x; i < nrows; i += gridDim.y * blockDim.x) {
        drow[i] = srow[i];
    }
}

// 2x2 patch merge (MoonViT -> connector), matching merge_patches():
//   out[m*4C + e_idx*C + d] = vf[tok*C + d]
//   e_idx = b*2+e, tok = (2a+b)*gw + (2c+e), m = a*mw + c
extern "C" __global__ void lg_merge_2x2(
    float *__restrict__ out, const float *__restrict__ vf,
    int gh, int gw, int c)
{
    const int m = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 4 * c) return;
    const int mw = gw / 2;
    const int a = m / mw, cc = m % mw;
    const int e_idx = i / c;
    const int d = i % c;
    const int b = e_idx / 2, e = e_idx % 2;
    const int tok = (2 * a + b) * gw + (2 * cc + e);
    out[(size_t)m * 4 * c + i] = vf[(size_t)tok * c + d];
}

// Decode helper: dequantize ONE row of an aligned 36-byte q8_0 weight table,
// with the row index passed as a kernel ARGUMENT (no ids upload).
// grid = (ceil(ne0/256), 1), block = (256,1,1). Writes ne0 f32.
extern "C" __global__ void lg_get_row_q8_0_aligned(
    const uint8_t *__restrict__ w, int row, float *__restrict__ dst, int ne0)
{
    const int nb = ne0 / 32;
    const uint8_t *wr = w + (size_t)row * nb * 36;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ne0; i += gridDim.x * blockDim.x) {
        const int b = i >> 5;
        const int lane = i & 31;
        const uint8_t *blk = wr + (size_t)b * 36;
        const float d = __half2float(*reinterpret_cast<const __half *>(blk));
        dst[i] = d * (float)reinterpret_cast<const int8_t *>(blk + 4)[lane];
    }
}

// Async device-to-device row append for a KV cache. Replaces per-layer
// synchronous cuMemcpyDtoD calls that flush the pipeline.
extern "C" __global__ void lg_copy_row(const float4 *__restrict__ src, float4 *__restrict__ dst, int n4)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

// Write a single i32 from a host value without a pageable H2D copy (used for
// the decode position scalar).
extern "C" __global__ void lg_set_i32(int *__restrict__ dst, int value)
{
    dst[0] = value;
}

// ===========================================================================
// 8. Attention
// ===========================================================================

// GQA attention, flash-style: one warp per (query token, query head).
//   q: [hd, n_qh, ntq]   k,v: [hd, n_kvh, ntk]   out: [hd, n_qh, ntq]
// Lane l holds the d-slice {l, l+32, ...}; keys are iterated by the warp and
// the running (max, denom) is kept per lane in the online-softmax form.
// Mask, when non-null, is [ntq, ntk] additive (use -FLT_MAX/4 rather than -inf
// so a fully-masked row still produces a finite output).
//
// Kept alongside lg_attn_flash: this one reads K/V from DRAM per query token
// and is the right choice for short key sequences, where the shared-memory
// staging of the flash variant costs more than it saves.
#define LA_HD_MAX 128
#define LA_DPL (LA_HD_MAX / 32)

extern "C" __global__ void lg_attn_gqa(
    const float *__restrict__ q, const float *__restrict__ k, const float *__restrict__ v,
    const float *__restrict__ mask, float *__restrict__ out,
    int hd, int n_qh, int n_kvh, int ntq, int ntk, float scale)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warps = blockDim.x / 32;
    const int group = n_qh / n_kvh;
    const int nwarp_total = n_qh * ntq;
    const int job = blockIdx.x * warps + warp;
    if (job >= nwarp_total) return;
    const int tq = job / n_qh;
    const int h = job % n_qh;
    const int hkv = h / group;
    // hd is 128 for the LM but 72 for the ViT: ceil(hd/32) registers per lane,
    // with bounds guards so the ragged tail is covered too.
    const int dpl = (hd + 31) / 32;

    // ggml layout: [hd, nh, ntok] with ntok slowest -> stride nh*hd per token.
    const float *qp = q + ((size_t)tq * n_qh + h) * hd;
    float qreg[LA_DPL];
    for (int i = 0; i < dpl; ++i) {
        const int d = lane + 32 * i;
        qreg[i] = (d < hd) ? qp[d] : 0.f;
    }

    float m = -INFINITY;
    float l = 0.f;
    float acc[LA_DPL];
    for (int i = 0; i < dpl; ++i) acc[i] = 0.f;

    for (int tk = 0; tk < ntk; ++tk) {
        const float *kp = k + ((size_t)tk * n_kvh + hkv) * hd;
        float dot = 0.f;
        for (int i = 0; i < dpl; ++i) {
            const int d = lane + 32 * i;
            if (d < hd) dot += qreg[i] * kp[d];
        }
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
        float s = dot * scale;
        if (mask) s += mask[(size_t)tq * ntk + tk];
        const float mnew = fmaxf(m, s);
        const float corr = __expf(m - mnew);
        const float p = __expf(s - mnew);
        l = l * corr + p;
        const float *vp = v + ((size_t)tk * n_kvh + hkv) * hd;
        for (int i = 0; i < dpl; ++i) {
            const int d = lane + 32 * i;
            if (d < hd) acc[i] = acc[i] * corr + p * vp[d];
        }
        m = mnew;
    }

    float *op = out + ((size_t)tq * n_qh + h) * hd;
    const float inv = (l > 0.f) ? 1.f / l : 0.f;
    for (int i = 0; i < dpl; ++i) {
        const int d = lane + 32 * i;
        if (d < hd) op[d] = acc[i] * inv;
    }
}

// Split-K "flash-decode" attention for the decode step (ntq == 1).
//
// lg_attn_gqa is warp-per-(head,query-token): for decode that is only n_qh
// warps (16 for the LM = 2 blocks on 20 SMs), each walking all ntk keys in
// SERIAL. At ntk=1391 that is 85 ms/token across 36 layers - the entire decode
// gap - for ~0.2 GFLOP of work. The fix is to split the key range across P
// independent chunks, compute an online-softmax partial (m, l, acc) per chunk
// in parallel, then combine: m = max(m_p), l = sum_p l_p*exp(m_p-m),
// acc = sum_p acc_p*exp(m_p-m). The combine is associative, so this is the same
// arithmetic up to f32 reassociation. grid = (P, n_qh), block = 32 (one warp).
//
// Precondition: ntq == 1 (one query token per decode step); mask (may be null)
// is mask[tq*ntk + tk] = mask[tk] for tq=0.
extern "C" __global__ void lg_attn_gqa_sk_p1(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, const float *__restrict__ mask,
    float *__restrict__ pm, float *__restrict__ pl, float *__restrict__ pa,
    int hd, int n_qh, int n_kvh, int ntq, int ntk, int P, float scale)
{
    const int lane = threadIdx.x;
    const int h = blockIdx.y;
    const int p = blockIdx.x;
    const int group = n_qh / n_kvh;
    const int hkv = h / group;
    const int dpl = (hd + 31) / 32;
    const int chunk = (ntk + P - 1) / P;
    const int lo = p * chunk;
    const int hi = lo + chunk < ntk ? lo + chunk : ntk;

    // q index tq = 0 (decode).
    const float *qp = q + (size_t)h * hd;
    float qreg[LA_DPL];
    for (int i = 0; i < dpl; ++i) {
        const int d = lane + 32 * i;
        qreg[i] = (d < hd) ? qp[d] : 0.f;
    }

    float m = -INFINITY, l = 0.f, acc[LA_DPL];
    for (int i = 0; i < dpl; ++i) acc[i] = 0.f;

    for (int tk = lo; tk < hi; ++tk) {
        const float *kp = k + ((size_t)tk * n_kvh + hkv) * hd;
        float dot = 0.f;
        for (int i = 0; i < dpl; ++i) {
            const int d = lane + 32 * i;
            if (d < hd) dot += qreg[i] * kp[d];
        }
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
        float s = dot * scale;
        if (mask) s += mask[tk];
        const float mnew = fmaxf(m, s);
        const float corr = __expf(m - mnew);
        const float pv = __expf(s - mnew);
        l = l * corr + pv;
        const float *vp = v + ((size_t)tk * n_kvh + hkv) * hd;
        for (int i = 0; i < dpl; ++i) {
            const int d = lane + 32 * i;
            if (d < hd) acc[i] = acc[i] * corr + pv * vp[d];
        }
        m = mnew;
    }

    const size_t idx = (size_t)h * P + p;
    if (lane == 0) {
        pm[idx] = m;
        pl[idx] = l;
    }
    for (int i = 0; i < dpl; ++i) {
        const int d = lane + 32 * i;
        if (d < hd) pa[idx * hd + d] = acc[i];
    }
}

// Combine step: one warp per head over its P partials. Associative reduction
// of the online-softmax state (m = max, then rescale every partial to the
// global max and sum). grid = (n_qh), block = 32.
extern "C" __global__ void lg_attn_gqa_sk_p2(
    const float *__restrict__ pm, const float *__restrict__ pl,
    const float *__restrict__ pa, float *__restrict__ out,
    int hd, int n_qh, int P)
{
    const int lane = threadIdx.x;
    const int h = blockIdx.x;
    const int dpl = (hd + 31) / 32;

    float m = -INFINITY;
    for (int p = 0; p < P; ++p) {
        const size_t idx = (size_t)h * P + p;
        if (pl[idx] > 0.f) m = fmaxf(m, pm[idx]);
    }
    float l = 0.f, acc[LA_DPL];
    for (int i = 0; i < dpl; ++i) acc[i] = 0.f;
    for (int p = 0; p < P; ++p) {
        const size_t idx = (size_t)h * P + p;
        const float lp = pl[idx];
        if (lp <= 0.f) continue;
        const float e = __expf(pm[idx] - m);
        l += lp * e;
        for (int i = 0; i < dpl; ++i) {
            const int d = lane + 32 * i;
            if (d < hd) acc[i] += pa[idx * hd + d] * e;
        }
    }
    const float inv = (l > 0.f) ? 1.f / l : 0.f;
    float *op = out + (size_t)h * hd;
    for (int i = 0; i < dpl; ++i) {
        const int d = lane + 32 * i;
        if (d < hd) op[d] = acc[i] * inv;
    }
}

// Self-attention with K/V staged in shared memory (ViT and full prefill).
// Identical arithmetic to lg_attn_gqa, but a block owns (head, W query tokens)
// and reads each K/V element from DRAM once per block instead of once per query
// token. For the ViT the naive version is ~9.7 GB of traffic per layer (~39 ms);
// staging into smem cuts it by ~W.
//
//   q,k,v,out: [hd, nh, ntok] (token stride nh*hd)
//   grid = (nh, ceil(ntq / W)), block = (W*32), dynamic smem = 2*kc*hd*4 bytes
//   requires n_qh == n_kvh; the mask (may be null) is mask[tq*ntk + tk]
extern "C" __global__ void lg_attn_flash(
    const float *__restrict__ q, const float *__restrict__ k, const float *__restrict__ v,
    const float *__restrict__ mask, float *__restrict__ out,
    int hd, int nh, int ntq, int ntk, float scale, int kc)
{
    extern __shared__ float smem[];
    float *ks = smem;
    float *vs = smem + (size_t)kc * hd;

    const int h = blockIdx.x;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int W = blockDim.x / 32;
    const int tq = blockIdx.y * W + warp;
    const int dpl = (hd + 31) / 32;

    float qreg[LA_DPL];
    float acc[LA_DPL];
    for (int i = 0; i < dpl; ++i) {
        const int d = lane + 32 * i;
        qreg[i] = (tq < ntq && d < hd) ? q[((size_t)tq * nh + h) * hd + d] : 0.f;
        acc[i] = 0.f;
    }
    float m = -INFINITY;
    float l = 0.f;

    for (int base = 0; base < ntk; base += kc) {
        const int n = min(kc, ntk - base);
        for (int i = threadIdx.x; i < n * hd; i += blockDim.x) {
            const int tk = base + i / hd;
            const int d = i % hd;
            ks[i] = k[((size_t)tk * nh + h) * hd + d];
            vs[i] = v[((size_t)tk * nh + h) * hd + d];
        }
        __syncthreads();
        if (tq < ntq) {
            for (int j = 0; j < n; ++j) {
                const float *kp = ks + (size_t)j * hd;
                float dot = 0.f;
                for (int i = 0; i < dpl; ++i) {
                    const int d = lane + 32 * i;
                    if (d < hd) dot += qreg[i] * kp[d];
                }
                for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
                float s = dot * scale;
                if (mask) s += mask[(size_t)tq * ntk + (base + j)];
                const float mnew = fmaxf(m, s);
                const float corr = __expf(m - mnew);
                const float p = __expf(s - mnew);
                l = l * corr + p;
                const float *vp = vs + (size_t)j * hd;
                for (int i = 0; i < dpl; ++i) {
                    const int d = lane + 32 * i;
                    if (d < hd) acc[i] = acc[i] * corr + p * vp[d];
                }
                m = mnew;
            }
        }
        __syncthreads();
    }

    if (tq < ntq) {
        float *op = out + ((size_t)tq * nh + h) * hd;
        const float inv = (l > 0.f) ? 1.f / l : 0.f;
        for (int i = 0; i < dpl; ++i) {
            const int d = lane + 32 * i;
            if (d < hd) op[d] = acc[i] * inv;
        }
    }
}

// ---------------------------------------------------------------------------
// Batched-GEMM prefill attention: S = scale*QK^T + mask, row softmax, O = PV.
//
// The THIRD attention form, and the only one of the three that needs a score
// matrix. `lg_attn_gqa` and `lg_attn_flash` both walk keys inside the softmax
// loop (online rescale), which is what makes them right for DECODE - one query
// token, or a growing cache - and wasteful for PREFILL, where the whole key set
// is known up front and the same K/V is re-read once per query. Prefill is a
// dense GEMM followed by a row softmax followed by a GEMM, and each stage is a
// cache-friendly sweep rather than a per-key walk.
//
// Measured on the promoting engine (GTX 1080, sm_61, 1391-token prompt, 16
// heads, hd 128, 36 layers): 5153 ms against `lg_attn_gqa`'s 8456 ms, -39% of
// the LM prefill, with per-layer 14.1 ms scores / 1.2 ms softmax / 12.0 ms out.
// The first version of this was a straight elementwise sweep - one thread per
// (query, key) pair walking the dot product out of global memory - and it was
// CORRECT BUT NO FASTER: 73 ms for the scores stage alone, because a k row was
// re-read once per query row. Staging Q and K tiles in shared memory and giving
// each thread a 4x4 register block is the whole difference.
//
//   q: [hd, n_qh, ntq]   k, v: [hd, n_kvh, ntk]   out: [hd, n_qh, ntq]
//   s: [n_qh, ntq, ntk]  - the head is SLOWEST, so one head's slice is
//                          contiguous and the softmax stage walks a row direct
//   mask (may be null): [ntq, ntk] additive; use -FLT_MAX/4 rather than -inf
//   ALL THREE require hd <= LA_HD_MAX (128). That is a documented precondition,
//   not a silent assumption: the shared tiles are sized by it and stage 3 uses
//   one thread per output channel at a 128-thread block. Reuse `LA_HD_MAX`
//   rather than a bare 128 so raising it raises all three together.
//
// GQA: `n_kvh` may be less than `n_qh`; each kv head is used for
// n_qh/n_kvh consecutive query heads, as in `lg_attn_gqa`.
// ---------------------------------------------------------------------------

#define LA_PF_QT 32    // query rows per block (scores)
#define LA_PF_KT 32    // keys per block (scores)
#define LA_PF_T  256   // staging threads: only the first 64 own a register block
#define LA_PF_SM_T 256 // softmax reduction width
#define LA_PF_OQT 16   // query rows per block (out)
#define LA_PF_OKT 16   // keys staged per iteration (out)

// Stage 1: s[h][i][j] = scale * dot(q_i, k_j) + mask[i][j].
// A block owns a (query tile x key tile) of one head, stages both in shared
// memory once, and then each thread multiplies a 4x4 register block - four query
// rows against four keys - so each staged value feeds 4 dot products.
// grid = (ceil(ntk/LA_PF_KT), ceil(ntq/LA_PF_QT), n_qh), block = (LA_PF_T).
extern "C" __global__ void lg_attn_prefill_scores(
    const float *__restrict__ q,
    const float *__restrict__ k,
    const float *__restrict__ mask,
    float *__restrict__ s,
    int hd, int n_qh, int n_kvh, int ntq, int ntk, float scale)
{
    // blockIdx.x = key tile, blockIdx.y = query tile, blockIdx.z = head
    const int h = blockIdx.z;
    const int q0 = blockIdx.y * LA_PF_QT;
    const int k0 = blockIdx.x * LA_PF_KT;
    if (q0 >= ntq || k0 >= ntk) return;

    const int gq = h / (n_qh / n_kvh);   // kv head repeated n_qh/n_kvh times
    const int tid = threadIdx.x;
    // 8 x 8 grid of 4x4 register blocks inside the 32x32 tile.
    const int rq = (tid / 8) * 4;        // first query row this thread owns
    const int rk = (tid % 8) * 4;        // first key this thread owns

    __shared__ float qs[LA_PF_QT * LA_HD_MAX];
    __shared__ float ks[LA_PF_KT * LA_HD_MAX];

    // Stage the tiles: thread t loads row (t/hd), element (t%hd).
    const int nq = min(LA_PF_QT, ntq - q0);
    const int nk = min(LA_PF_KT, ntk - k0);
    for (int t = tid; t < LA_PF_QT * hd; t += LA_PF_T) {
        const int r = t / hd, c = t % hd;
        qs[(size_t)r * hd + c] = (r < nq)
            ? q[(size_t)(q0 + r) * n_qh * hd + (size_t)h * hd + c]
            : 0.0f;
    }
    for (int t = tid; t < LA_PF_KT * hd; t += LA_PF_T) {
        const int r = t / hd, c = t % hd;
        ks[(size_t)r * hd + c] = (r < nk)
            ? k[(size_t)(k0 + r) * n_kvh * hd + (size_t)gq * hd + c]
            : 0.0f;
    }
    __syncthreads();

    // Only the first 64 threads own a 4x4 tile coordinate (8x8 blocks cover the
    // 32x32 tile); the rest helped stage shared memory and must not compute, or
    // they would index past the tile.
    if (tid >= 64) return;

    float acc[4][4];
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) acc[a][b] = 0.0f;

    for (int d = 0; d < hd; ++d) {
        // Four query rows and four keys, one d at a time: each shared-memory
        // read is shared by the whole 4x4 block's accumulators.
        const float qv[4] = {
            qs[(size_t)(rq + 0) * hd + d],
            qs[(size_t)(rq + 1) * hd + d],
            qs[(size_t)(rq + 2) * hd + d],
            qs[(size_t)(rq + 3) * hd + d],
        };
        const float kv[4] = {
            ks[(size_t)(rk + 0) * hd + d],
            ks[(size_t)(rk + 1) * hd + d],
            ks[(size_t)(rk + 2) * hd + d],
            ks[(size_t)(rk + 3) * hd + d],
        };
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b) acc[a][b] = fmaf(qv[a], kv[b], acc[a][b]);
    }

    for (int a = 0; a < 4; ++a) {
        const int i = q0 + rq + a;
        if (i >= ntq) break;
        for (int b = 0; b < 4; ++b) {
            const int j = k0 + rk + b;
            if (j >= ntk) break;
            float v = acc[a][b] * scale;
            if (mask) v += mask[(size_t)i * ntk + j];
            s[((size_t)h * ntq + i) * ntk + j] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 2: in-place row softmax over the key axis of s[h][i][:].
// One block per (head, query), so the max-then-sum two-pass is a block-wide
// reduction over a contiguous row. 1.2 ms per layer against 73 ms for the
// scores stage on the promoting engine, so it is deliberately left simple.
// grid = (1, ntq, n_qh), block = (LA_PF_SM_T).
extern "C" __global__ void lg_attn_prefill_softmax(
    float *__restrict__ s, int ntq, int ntk)
{
    const int h = blockIdx.z;
    const int i = blockIdx.y;
    float *row = s + ((size_t)h * ntq + i) * ntk;
    const int tid = threadIdx.x;

    __shared__ float red[LA_PF_SM_T];
    float m = -INFINITY;   // as lg_attn_gqa: this file does not include math_constants.h
    for (int j = tid; j < ntk; j += LA_PF_SM_T) m = fmaxf(m, row[j]);
    red[tid] = m;
    __syncthreads();
    for (int off = LA_PF_SM_T / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();

    // exp relative to the row max. The mask is additive (-FLT_MAX/4 on masked
    // pairs), so a masked entry subtracts to exp(-FLT_MAX/4 - m) = 0 and drops
    // out of both the numerator and the denominator - no special case needed.
    float l = 0.0f;
    for (int j = tid; j < ntk; j += LA_PF_SM_T) {
        const float e = __expf(row[j] - m);
        row[j] = e;
        l += e;
    }
    red[tid] = l;
    __syncthreads();
    for (int off = LA_PF_SM_T / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    l = red[0];
    // A fully-masked row would give l == 0. A causal mask always leaves the
    // diagonal unmasked so this is unreachable there, but 1/0 would put NaN in
    // the residual and NaN is far harder to trace than a zero row.
    const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    for (int j = tid; j < ntk; j += LA_PF_SM_T) row[j] *= inv;
}

extern "C" __global__ void lg_attn_prefill_softmax_cached(float* s, int ntq, int ntk) {
    if(ntk>8192){
    const int h = blockIdx.z;
    const int i = blockIdx.y;
    float *row = s + ((size_t)h * ntq + i) * ntk;
    const int tid = threadIdx.x;

    __shared__ float red[LA_PF_SM_T];
    float m = -INFINITY;   // as lg_attn_gqa: this file does not include math_constants.h
    for (int j = tid; j < ntk; j += LA_PF_SM_T) m = fmaxf(m, row[j]);
    red[tid] = m;
    __syncthreads();
    for (int off = LA_PF_SM_T / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();

    // exp relative to the row max. The mask is additive (-FLT_MAX/4 on masked
    // pairs), so a masked entry subtracts to exp(-FLT_MAX/4 - m) = 0 and drops
    // out of both the numerator and the denominator - no special case needed.
    float l = 0.0f;
    for (int j = tid; j < ntk; j += LA_PF_SM_T) {
        const float e = __expf(row[j] - m);
        row[j] = e;
        l += e;
    }
    red[tid] = l;
    __syncthreads();
    for (int off = LA_PF_SM_T / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    l = red[0];
    // A fully-masked row would give l == 0. A causal mask always leaves the
    // diagonal unmasked so this is unreachable there, but 1/0 would put NaN in
    // the residual and NaN is far harder to trace than a zero row.
    const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    for (int j = tid; j < ntk; j += LA_PF_SM_T) row[j] *= inv;

        return;
    }
    float* row=s+((size_t)blockIdx.z*ntq+blockIdx.y)*ntk;
    int tid=threadIdx.x;
    float vals[32];
    float m=-INFINITY;
    #pragma unroll
    for(int z=0;z<32;++z){int j=tid+z*256; vals[z]=j<ntk?row[j]:-INFINITY; m=fmaxf(m,vals[z]);}
    // Keep the original reduction tree, so changing memory traffic does not
    // change the denominator's floating-point association.
    __shared__ float red[256];
    red[tid]=m; __syncthreads();
    for(int off=128;off>0;off>>=1){if(tid<off) red[tid]=fmaxf(red[tid],red[tid+off]); __syncthreads();}
    m=red[0]; __syncthreads();
    float l=0;
    #pragma unroll
    for(int z=0;z<32;++z){if(tid+z*256<ntk){vals[z]=__expf(vals[z]-m); l+=vals[z];}}
    red[tid]=l; __syncthreads();
    for(int off=128;off>0;off>>=1){if(tid<off) red[tid]+=red[tid+off]; __syncthreads();}
    float inv=red[0]>0?1.f/red[0]:0.f;
    #pragma unroll
    for(int z=0;z<32;++z) if(tid+z*256<ntk) row[tid+z*256]=vals[z]*inv;
}

// Stage 3: out[h][i][d] = sum_j p[h][i][j] * v[j][gq][d], as a GEMM.
// Transposed-w form: w[k][r] = v[(gq*hd)+k*ldw+r] (r = channel), x[c][k] =
// p[h*ntq*ntk + c*ntk + k] (c = query, k = key), so out[c][r] = sum_k x[c][k] *
// w[k][r]. A 256-thread block computes a 64x32 output tile (BM x BN) over an
// 8-deep K tile, both operands staged in shared once and a TM x TN register
// block per thread. This is 12.9x the previous one-thread-per-channel sweep
// (measured at the true query-tiled shape: 12.9 ms vs 165 ms per layer) and is
// bit-exact with it. The head is a grid axis; gq selects the kv head.
// grid = (ceil(hd/64), ceil(ntq/32), n_qh), block = 256.
extern "C" __global__ void lg_attn_prefill_out(
    const float *__restrict__ p,
    const float *__restrict__ v,
    float *__restrict__ out,
    int hd, int n_qh, int n_kvh, int ntq, int ntk)
{
    const int BM = 64, BN = 32, BK = 8, TM = 4, TN = 2;
    __shared__ float As[BK][BM + 4];
    __shared__ float Bs[BK][BN + 4];

    const int h = blockIdx.z;
    const int gq = h / (n_qh / n_kvh);
    const int ldw = n_kvh * hd;   // v token stride
    const int sy  = n_qh * hd;    // out token stride

    const int tid = threadIdx.x;
    const int ntc = BN / TN;
    const int tr = tid / ntc;
    const int tc = tid % ntc;
    const int r0 = blockIdx.x * BM;   // first output channel (hd axis)
    const int c0 = blockIdx.y * BN;   // first query row (ntq axis)

    float acc[TM][TN];
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    const float *__restrict__ wbase = v + (size_t)gq * hd;
    const float *__restrict__ xbase = p + (size_t)h * ntq * ntk;
    const int ntiles = (ntk + BK - 1) / BK;

    for (int t = 0; t < ntiles; ++t) {
        for (int i = tid; i < BK * BM; i += 256) {
            const int r = i / BK, k = i % BK;
            const int rr = r0 + r, kk = t * BK + k;
            As[k][r] = (rr < hd && kk < ntk) ? wbase[(size_t)kk * ldw + rr] : 0.0f;
        }
        for (int i = tid; i < BK * BN; i += 256) {
            const int c = i / BK, k = i % BK;
            const int cc = c0 + c, kk = t * BK + k;
            Bs[k][c] = (cc < ntq && kk < ntk) ? xbase[(size_t)cc * ntk + kk] : 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
            for (int i = 0; i < TM; ++i) a[i] = As[k][tr * TM + i];
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tc * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        const int r = r0 + tr * TM + i;
        if (r < hd)
            for (int j = 0; j < TN; ++j) {
                const int c = c0 + tc * TN + j;
                if (c < ntq) out[(size_t)c * sy + (size_t)h * hd + r] = acc[i][j];
            }
    }
}

// ===========================================================================
// 9. Linear layer in the token layout (C, 1, h*w)
//    (y, x, c) at c*h*w + y*w + x)
// ===========================================================================


// ===== NEW scores GEMM: S[h][i][j] = scale*dot(q_i,k_j) + mask[i][j]
// GEMM m=ntq(rows i) x n=ntk(cols j) x k=hd. BM=64 BN=64 BK=36 TM=4 TN=4, 256 thr.
// All threads compute 4x4 outputs. Q/K staged to smem in float4 (d contiguous).
extern "C" __global__ void lg_attn_prefill_scores2(const float*__restrict__ q, const float*__restrict__ k,
    const float*__restrict__ mask, float*__restrict__ s,
    int hd, int n_qh, int n_kvh, int ntq, int ntk, float scale)
{
    const int BM=64, BN=64, BK=36, TM=4, TN=4;
    __shared__ float As[BK][BM+2];
    __shared__ float Bs[BK][BN+2];
    const int h = blockIdx.z;
    const int tid = threadIdx.x;
    const int tr = tid / (BN/TN);
    const int tc = tid % (BN/TN);
    const int i0 = blockIdx.y * BM;
    const int j0 = blockIdx.x * BN;
    const int gq = h / (n_qh / n_kvh);   // kv head repeated n_qh/n_kvh times
    const int ldk = n_kvh * hd;
    const float* qs = q + (size_t)i0*n_qh*hd + (size_t)h*hd;
    const float* ks = k + (size_t)j0*ldk   + (size_t)gq*hd;
    float acc[TM][TN];
    for(int a=0;a<TM;++a) for(int b=0;b<TN;++b) acc[a][b]=0.f;
    const int ntq_rem = ntq - i0, ntk_rem = ntk - j0;
    for(int db=0; db<hd; db+=BK){
        for(int x=tid; x<BM*(BK/4); x+=256){
            const int r=x/(BK/4), d=4*(x%(BK/4));
            float4 a=make_float4(0,0,0,0), b=a;
            if(i0+r<ntq && db+d+3<hd && hd%4==0) a=*(const float4*)(qs+(size_t)r*n_qh*hd+db+d);
            else for(int z=0;z<4;++z) if(i0+r<ntq && db+d+z<hd) ((float*)&a)[z]=qs[(size_t)r*n_qh*hd+db+d+z];
            if(j0+r<ntk && db+d+3<hd && hd%4==0) b=*(const float4*)(ks+(size_t)r*ldk+db+d);
            else for(int z=0;z<4;++z) if(j0+r<ntk && db+d+z<hd) ((float*)&b)[z]=ks[(size_t)r*ldk+db+d+z];
            As[d][r]=a.x; As[d+1][r]=a.y; As[d+2][r]=a.z; As[d+3][r]=a.w;
            Bs[d][r]=b.x; Bs[d+1][r]=b.y; Bs[d+2][r]=b.z; Bs[d+3][r]=b.w;
        }
        __syncthreads();
        for(int d=0; d<BK; ++d){
            float a[TM], b[TN];
            for(int x=0;x<TM;++x) a[x]=As[d][tr*TM+x];
            for(int y=0;y<TN;++y) b[y]=Bs[d][tc*TN+y];
            for(int x=0;x<TM;++x) for(int y=0;y<TN;++y) acc[x][y]=fmaf(a[x],b[y],acc[x][y]);
        }
        __syncthreads();
    }
    for(int x=0;x<TM;++x){
        const int i=i0+tr*TM+x;
        if(i<ntq && x<ntq_rem){
            for(int y=0;y<TN;++y){
                const int j=j0+tc*TN+y;
                if(j<ntk && y<ntk_rem){
                    float v = acc[x][y]*scale;
                    if(mask) v += mask[(size_t)i*ntk + j];
                    s[((size_t)h*ntq + i)*ntk + j] = v;
                }
            }
        }
    }
}

// Row statistics for softmax/PV fusion; ntk <= 8192, block = 256.
extern "C" __global__ void lg_attn_prefill_stats(const float* s, float* stats, int ntq, int ntk) {
    const float* row=s+((size_t)blockIdx.z*ntq+blockIdx.y)*ntk;
    int tid=threadIdx.x;
    float vals[32];
    float m=-INFINITY;
    #pragma unroll
    for(int z=0;z<32;++z){int j=tid+z*256; vals[z]=j<ntk?row[j]:-INFINITY; m=fmaxf(m,vals[z]);}
    // Keep the original reduction tree, so changing memory traffic does not
    // change the denominator's floating-point association.
    __shared__ float red[256];
    red[tid]=m; __syncthreads();
    for(int off=128;off>0;off>>=1){if(tid<off) red[tid]=fmaxf(red[tid],red[tid+off]); __syncthreads();}
    m=red[0]; __syncthreads();
    float l=0;
    #pragma unroll
    for(int z=0;z<32;++z){if(tid+z*256<ntk){vals[z]=__expf(vals[z]-m); l+=vals[z];}}
    red[tid]=l; __syncthreads();
    for(int off=128;off>0;off>>=1){if(tid<off) red[tid]+=red[tid+off]; __syncthreads();}
    float inv=red[0]>0?1.f/red[0]:0.f;
    if(tid==0){stats[((size_t)blockIdx.z*ntq+blockIdx.y)*2]=m;stats[((size_t)blockIdx.z*ntq+blockIdx.y)*2+1]=inv;}
}

extern "C" __global__ void lg_attn_prefill_out_softmax(const float*__restrict__ p, const float*__restrict__ stats, const float*__restrict__ v, float*__restrict__ out,
    int hd, int n_qh, int n_kvh, int ntq, int ntk)
{
    const int BM=64, BN=64, BK=32, TM=4, TN=4;
    __shared__ float As[BK][BM+2];
    __shared__ float Bs[BK][BN+2];
    const int h = blockIdx.z;
    const int gq = h / (n_qh / n_kvh);
    const int ldw = n_kvh * hd;
    const int sy  = n_qh * hd;
    const int tid = threadIdx.x;
    const int tr = tid / (BN/TN);
    const int tc = tid % (BN/TN);
    const int r0 = blockIdx.x * BM;
    const int i0 = blockIdx.y * BN;
    float acc[TM][TN];
    for(int a=0;a<TM;++a) for(int b=0;b<TN;++b) acc[a][b]=0.f;
    const int ntiles = (ntk+BK-1)/BK;
    for(int t=0;t<ntiles;++t){
        for(int x=tid; x<BM*BK/4; x+=256){
            const int r=4*(x%(BM/4)), jj=x/(BM/4), rr=r0+r, kidx=t*BK+jj;
            float4 a=make_float4(0,0,0,0);
            if(rr+3<hd && kidx<ntk && hd%4==0) a=*(const float4*)(v+(size_t)kidx*ldw+gq*hd+rr);
            else for(int z=0;z<4;++z) if(rr+z<hd && kidx<ntk) ((float*)&a)[z]=v[(size_t)kidx*ldw+gq*hd+rr+z];
            As[jj][r]=a.x; As[jj][r+1]=a.y; As[jj][r+2]=a.z; As[jj][r+3]=a.w;
        }
        for(int y=tid; y<BN*BK/4; y+=256){
            const int c=y/(BK/4), jj=4*(y%(BK/4)), ii=i0+c, kidx=t*BK+jj;
            float4 b=make_float4(0,0,0,0);
            if(ii<ntq && kidx+3<ntk && ntk%4==0) b=*(const float4*)(p+((size_t)h*ntq+ii)*ntk+kidx);
            else for(int z=0;z<4;++z) if(ii<ntq && kidx+z<ntk) ((float*)&b)[z]=p[((size_t)h*ntq+ii)*ntk+kidx+z];
            float m=0,inv=0;
            if(ii<ntq){m=stats[((size_t)h*ntq+ii)*2];inv=stats[((size_t)h*ntq+ii)*2+1];}
            Bs[jj][c]=(kidx<ntk)?__expf(b.x-m)*inv:0;
            Bs[jj+1][c]=(kidx+1<ntk)?__expf(b.y-m)*inv:0;
            Bs[jj+2][c]=(kidx+2<ntk)?__expf(b.z-m)*inv:0;
            Bs[jj+3][c]=(kidx+3<ntk)?__expf(b.w-m)*inv:0;
        }
        __syncthreads();
        for(int d=0; d<BK; ++d){
            float a[TM], b[TN];
            for(int x=0;x<TM;++x) a[x]=As[d][tr*TM+x];
            for(int y=0;y<TN;++y) b[y]=Bs[d][tc*TN+y];
            for(int x=0;x<TM;++x) for(int y=0;y<TN;++y) acc[x][y]=fmaf(a[x],b[y],acc[x][y]);
        }
        __syncthreads();
    }
    for(int x=0;x<TM;++x){
        const int r=r0+tr*TM+x;
        if(r<hd){
            for(int y=0;y<TN;++y){
                const int i=i0+tc*TN+y;
                if(i<ntq) out[(size_t)i*sy + (size_t)h*hd + r] = acc[x][y];
            }
        }
    }
}


// ===== NEW out GEMM (transposed-w): out[i][r] = sum_j p[i][j]*v[j][r]
// rows r=hd (blockIdx.x), cols i=ntq (blockIdx.y), reduce j=ntk. BM=64 BN=64 BK=32 TM=4 TN=4.
extern "C" __global__ void lg_attn_prefill_out2(const float*__restrict__ p, const float*__restrict__ v, float*__restrict__ out,
    int hd, int n_qh, int n_kvh, int ntq, int ntk)
{
    const int BM=64, BN=64, BK=32, TM=4, TN=4;
    __shared__ float As[BK][BM+2];
    __shared__ float Bs[BK][BN+2];
    const int h = blockIdx.z;
    const int gq = h / (n_qh / n_kvh);
    const int ldw = n_kvh * hd;
    const int sy  = n_qh * hd;
    const int tid = threadIdx.x;
    const int tr = tid / (BN/TN);
    const int tc = tid % (BN/TN);
    const int r0 = blockIdx.x * BM;
    const int i0 = blockIdx.y * BN;
    float acc[TM][TN];
    for(int a=0;a<TM;++a) for(int b=0;b<TN;++b) acc[a][b]=0.f;
    const int ntiles = (ntk+BK-1)/BK;
    for(int t=0;t<ntiles;++t){
        for(int x=tid; x<BM*BK/4; x+=256){
            const int r=4*(x%(BM/4)), jj=x/(BM/4), rr=r0+r, kidx=t*BK+jj;
            float4 a=make_float4(0,0,0,0);
            if(rr+3<hd && kidx<ntk && hd%4==0) a=*(const float4*)(v+(size_t)kidx*ldw+gq*hd+rr);
            else for(int z=0;z<4;++z) if(rr+z<hd && kidx<ntk) ((float*)&a)[z]=v[(size_t)kidx*ldw+gq*hd+rr+z];
            As[jj][r]=a.x; As[jj][r+1]=a.y; As[jj][r+2]=a.z; As[jj][r+3]=a.w;
        }
        for(int y=tid; y<BN*BK/4; y+=256){
            const int c=y/(BK/4), jj=4*(y%(BK/4)), ii=i0+c, kidx=t*BK+jj;
            float4 b=make_float4(0,0,0,0);
            if(ii<ntq && kidx+3<ntk && ntk%4==0) b=*(const float4*)(p+((size_t)h*ntq+ii)*ntk+kidx);
            else for(int z=0;z<4;++z) if(ii<ntq && kidx+z<ntk) ((float*)&b)[z]=p[((size_t)h*ntq+ii)*ntk+kidx+z];
            Bs[jj][c]=b.x; Bs[jj+1][c]=b.y; Bs[jj+2][c]=b.z; Bs[jj+3][c]=b.w;
        }
        __syncthreads();
        for(int d=0; d<BK; ++d){
            float a[TM], b[TN];
            for(int x=0;x<TM;++x) a[x]=As[d][tr*TM+x];
            for(int y=0;y<TN;++y) b[y]=Bs[d][tc*TN+y];
            for(int x=0;x<TM;++x) for(int y=0;y<TN;++y) acc[x][y]=fmaf(a[x],b[y],acc[x][y]);
        }
        __syncthreads();
    }
    for(int x=0;x<TM;++x){
        const int r=r0+tr*TM+x;
        if(r<hd){
            for(int y=0;y<TN;++y){
                const int i=i0+tc*TN+y;
                if(i<ntq) out[(size_t)i*sy + (size_t)h*hd + r] = acc[x][y];
            }
        }
    }
}

// ===========================================================================
// 10. Plain f32 GEMM (ggml layout: W is [ne1][ne0], x is [ne0, ncols],
//     y is [ncols][ne1])
// ===========================================================================

// Naive one-thread-per-output. Correct and layout-obvious; the tiled variant
// below is what large shapes should call.
extern "C" __global__ void lg_f32_gemm(
    const float *__restrict__ w, const float *__restrict__ x, float *__restrict__ y,
    int ne0, int ne1, int ncols)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y;
    if (j >= ne1) return;
    const float *wrow = w + (size_t)j * ne0;
    const float *xcol = x + (size_t)c * ne0;
    float acc = 0.f;
    for (int k = 0; k < ne0; ++k) acc += wrow[k] * xcol[k];
    y[(size_t)c * ne1 + j] = acc;
}

// f32 GEMM: 64 rows x 32 columns per 256-thread block. Both the w tile
// (BM x BK) and the x tile (BN x BK) are staged in shared memory once per
// k-step, and each thread owns a 4x2 register block (TM=4, TN=2) so every
// staged value feeds several accumulators. The staging loops walk the tile
// as (r = i/BK, k = i%BK) so consecutive threads read consecutive k at a
// fixed row - a coalesced load. The older body left w in registers and let
// the 8 column-slot threads re-read the same float4, costing 8x the w
// traffic; staging both tiles is what removes that. No k-split: the tiles
// are numerous enough.
// grid = (ceil(ne1/64), ceil(ncols/32)), block = 256. Requires ne1,ncols > 0.
extern "C" __global__ void lg_f32_gemm_tiled(
    const float *__restrict__ w, const float *__restrict__ x, float *__restrict__ y,
    int ne0, int ne1, int ncols)
{
    const int BM = 64, BN = 32, BK = 8, TM = 4, TN = 2;
    __shared__ float As[BK][BM + 4];
    __shared__ float Bs[BK][BN + 4];
    const int tid = threadIdx.x;
    const int ntc = BN / TN;                 // thread columns per block
    const int tr = tid / ntc;
    const int tc = tid % ntc;
    const int r0 = blockIdx.x * BM;
    const int c0 = blockIdx.y * BN;

    float acc[TM][TN];
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    const int ntiles = (ne0 + BK - 1) / BK;
    for (int t = 0; t < ntiles; ++t) {
        // Stage w[r0..r0+BM, t*BK..] and x[c0..c0+BN, t*BK..].
        for (int i = tid; i < BK * BM; i += 256) {
            const int r = i / BK, k = i % BK;
            const int rr = r0 + r, kk = t * BK + k;
            As[k][r] = (rr < ne1 && kk < ne0) ? w[(size_t)rr * ne0 + kk] : 0.f;
        }
        for (int i = tid; i < BK * BN; i += 256) {
            const int c = i / BK, k = i % BK;
            const int cc = c0 + c, kk = t * BK + k;
            Bs[k][c] = (cc < ncols && kk < ne0) ? x[(size_t)cc * ne0 + kk] : 0.f;
        }
        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
            for (int i = 0; i < TM; ++i) a[i] = As[k][tr * TM + i];
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tc * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        const int r = r0 + tr * TM + i;
        if (r < ne1)
            for (int j = 0; j < TN; ++j) {
                const int c = c0 + tc * TN + j;
                if (c < ncols) y[(size_t)c * ne1 + r] = acc[i][j];
            }
    }
}


// f32 GEMM v2: cuBLAS-style register-tiled GEMM. All 256 threads compute
// 4x4 (TM=4 TN=4) output tiles from a BK=32 smem tile, so ne0/BK __syncthreads
// replaces the tiled variant's ne0/8, and float4 vectorized staging coalesces
// both inputs. Measured 2.0-2.5x over lg_f32_gemm_tiled at the ViT fc1 shape
// (4304x1152x5408), bit-exact vs cuBLAS at divisible shapes.
// grid = (ceil(ne1/64), ceil(ncols/64)), block = 256.
// Requires ne0 % 4 == 0 and 16-byte-aligned w/x rows (same contract as the
// other float4 kernels); ragged ne0/ncols edges are handled with scalar loads.
extern "C" __global__ void lg_f32_gemm_v2(
    const float *__restrict__ w, const float *__restrict__ x, float *__restrict__ y,
    int ne0, int ne1, int ncols)
{
    const int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
    __shared__ float As[BK][BM + 2];
    __shared__ float Bs[BK][BN + 2];
    const int tid = threadIdx.x;
    const int ntc = BN / TN;                 // 16 thread-columns
    const int tr = tid / ntc;
    const int tc = tid % ntc;
    const int r0 = blockIdx.x * BM;
    const int c0 = blockIdx.y * BN;

    float acc[TM][TN];
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    const int ntiles = (ne0 + BK - 1) / BK;
    const int NA = BK * BM / 4, NB = BK * BN / 4;
    for (int t = 0; t < ntiles; ++t) {
        for (int i = tid; i < NA; i += 256) {
            const int r = i / (BK / 4), k4 = i % (BK / 4);
            const int rr = r0 + r, kk = t * BK + k4 * 4;
            if (rr < ne1 && kk + 3 < ne0) {
                float4 v = *reinterpret_cast<const float4 *>(&w[(size_t)rr * ne0 + kk]);
                As[k4 * 4 + 0][r] = v.x; As[k4 * 4 + 1][r] = v.y;
                As[k4 * 4 + 2][r] = v.z; As[k4 * 4 + 3][r] = v.w;
            } else {
                for (int s = 0; s < 4; ++s)
                    As[k4 * 4 + s][r] = (rr < ne1 && kk + s < ne0) ? w[(size_t)rr * ne0 + kk + s] : 0.f;
            }
        }
        for (int i = tid; i < NB; i += 256) {
            const int c = i / (BK / 4), k4 = i % (BK / 4);
            const int cc = c0 + c, kk = t * BK + k4 * 4;
            if (cc < ncols && kk + 3 < ne0) {
                float4 v = *reinterpret_cast<const float4 *>(&x[(size_t)cc * ne0 + kk]);
                Bs[k4 * 4 + 0][c] = v.x; Bs[k4 * 4 + 1][c] = v.y;
                Bs[k4 * 4 + 2][c] = v.z; Bs[k4 * 4 + 3][c] = v.w;
            } else {
                for (int s = 0; s < 4; ++s)
                    Bs[k4 * 4 + s][c] = (cc < ncols && kk + s < ne0) ? x[(size_t)cc * ne0 + kk + s] : 0.f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[k][tr * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tc * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        const int r = r0 + tr * TM + i;
        if (r < ne1)
            for (int j = 0; j < TN; ++j) {
                const int c = c0 + tc * TN + j;
                if (c < ncols) y[(size_t)c * ne1 + r] = acc[i][j];
            }
    }
}

// ===========================================================================
// 11. q8_0 GEMM/GEMV (ggml-compatible block, 34 bytes: half d + int8 qs[32])
// ===========================================================================
//
// Two layouts are supported on purpose:
//
//   * the NATIVE 34-byte layout, read directly from a mmapped model file (no
//     repack, so a CPU-hosted model and the quantized file agree byte for byte);
//   * the ALIGNED 36-byte layout, produced by repacking during upload. Every
//     4-byte word of the 34-byte layout straddles a block boundary, so the
//     unaligned loads cost ~8 instructions against 8 dp4a; with the 4-byte
//     padding the weights are read as aligned ints instead. Measured 3.1x on
//     the large GEMMs, which is why both exist.

// Standalone activation quantizer: x [ne0, ncols] -> int8 qs [ne0, ncols] +
// per-32-block f32 scales sc [ne0/32, ncols]. One warp per (block, column).
// Running this once per GEMM removes the per-row-block recomputation that made
// the fused dp4a kernel slow.
// NOTE: scales are stored [ncols][nb] (column-major over blocks) so the GEMM
// can walk a column's scales with stride 1: sc[c * nb + b].
extern "C" __global__ void lg_quantize_q8_0(
    const float *__restrict__ x, int8_t *__restrict__ qs, float *__restrict__ sc,
    int ne0, int ncols)
{
    const int b = blockIdx.x;
    const int c = blockIdx.y;
    const int lane = threadIdx.x;
    const float *xp = x + (size_t)c * ne0 + b * 32;
    float v = fabsf(xp[lane]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    const float amax = v;
    __shared__ float sh[32];
    if (lane == 0) sh[0] = amax / 127.f;
    __syncthreads();
    const float d = sh[0];
    const float id = d > 0.f ? 1.f / d : 0.f;
    if (lane == 0) sc[(size_t)c * (ne0 / 32) + b] = d;
    qs[(size_t)c * ne0 + b * 32 + lane] = (int8_t)__float2int_rn(xp[lane] * id);
}

// dp4a GEMM over the ALIGNED 36-byte layout: 64 rows x 32 cols per 256-thread
// block, 32 row lanes x 8 column slots, 2 rows x 4 columns per thread. The
// k-block's 32 int8 weights and the staged activation are both read as aligned
// 4-byte ints (8 int loads per block instead of 64 byte ops).
// grid = (ceil(ne1/64), ceil(ncols/32)).
extern "C" __global__ void lg_q8_0_gemm_v2(const uint8_t*w,const int8_t*qs,const float*sc,float*y,int ne0,int ne1,int ncols){
 const int BM=32,BN=64,BK=32,TM=4,TN=2; __shared__ int8_t A[BM][BK],B[BN][BK]; __shared__ float sa[BM],sb[BN];
 int tid=threadIdx.x, tr=tid/(BN/TN), tc=tid%(BN/TN), r0=blockIdx.x*BM,c0=blockIdx.y*BN,nb=ne0/32;
 float acc[TM][TN]; 
#pragma unroll
 for(int z=0;z<TM*TN;z++)((float*)acc)[z]=0;
 for(int blk=0;blk<nb;blk++){
  for(int z=tid;z<BM*BK;z+=256){int r=z/BK,d=z%BK,rr=r0+r;int8_t v=0;float q=0;if(rr<ne1){const uint8_t*p=w+((size_t)rr*nb+blk)*36;v=(int8_t)p[4+d];if(!d)q=__half2float(*(const __half*)p);}A[r][d]=v;if(!d)sa[r]=q;}
  for(int z=tid;z<BN*BK;z+=256){int c=z/BK,d=z%BK,cc=c0+c;int8_t v=0;float q=0;if(cc<ncols){v=qs[(size_t)cc*ne0+blk*32+d];if(!d)q=sc[(size_t)cc*nb+blk];}B[c][d]=v;if(!d)sb[c]=q;}
  __syncthreads();
  if(tid<256){int ia[TM][TN]; 
#pragma unroll
   for(int z=0;z<TM*TN;z++)((int*)ia)[z]=0;
   for(int w4=0;w4<8;w4++){int au[TM],bu[TN];for(int i=0;i<TM;i++)au[i]=*(const int*)&A[tr*TM+i][w4*4];for(int j=0;j<TN;j++)bu[j]=*(const int*)&B[tc*TN+j][w4*4];for(int i=0;i<TM;i++)for(int j=0;j<TN;j++)ia[i][j]=__dp4a(au[i],bu[j],ia[i][j]);}
   for(int i=0;i<TM;i++){float x=sa[tr*TM+i];for(int j=0;j<TN;j++)acc[i][j]+=x*sb[tc*TN+j]*(float)ia[i][j];}
  }
  __syncthreads();
 }
 if(tid<256)for(int i=0;i<TM;i++){int r=r0+tr*TM+i;if(r<ne1)for(int j=0;j<TN;j++){int c=c0+tc*TN+j;if(c<ncols)y[(size_t)c*ne1+r]=acc[i][j];}}
}


extern "C" __global__ void lg_q8_0_gemm_dp4a(
    const uint8_t *__restrict__ w, const int8_t *__restrict__ qs,
    const float *__restrict__ sc, float *__restrict__ y,
    int ne0, int ne1, int ncols)
{
    __shared__ int xs[32 * 8];
    const int tid = threadIdx.x;
    const int jl  = tid & 31;
    const int cs  = tid >> 5;
    const int j0  = blockIdx.x * 64;
    const int c0  = blockIdx.y * 32;
    const int nb  = ne0 / 32;

    const int r0 = j0 + jl;
    const int r1 = j0 + jl + 32;
    const int cA = c0 + cs * 4;

    const bool ur0 = r0 < ne1;
    const bool ur1 = r1 < ne1;
    const bool uc0 = cA + 0 < ncols, uc1 = cA + 1 < ncols;
    const bool uc2 = cA + 2 < ncols, uc3 = cA + 3 < ncols;

    const uint8_t *w0 = w + (size_t)r0 * nb * 36 + 4;
    const uint8_t *w1 = w + (size_t)r1 * nb * 36 + 4;
    const float *s0 = sc + (size_t)(cA + 0) * nb;
    const float *s1 = sc + (size_t)(cA + 1) * nb;
    const float *s2 = sc + (size_t)(cA + 2) * nb;
    const float *s3 = sc + (size_t)(cA + 3) * nb;

    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a03 = 0.f;
    float a10 = 0.f, a11 = 0.f, a12 = 0.f, a13 = 0.f;

    for (int b = 0; b < nb; ++b) {
        {
            const int c = tid >> 3;
            const int wd = tid & 7;
            const int col = c0 + c;
            int v4 = 0;
            if (col < ncols) {
                v4 = *reinterpret_cast<const int *>(qs + (size_t)col * ne0 + b * 32 + wd * 4);
            }
            xs[c * 8 + wd] = v4;
        }
        __syncthreads();

        const int *xc0 = &xs[(cs * 4 + 0) * 8];
        const int *xc1 = &xs[(cs * 4 + 1) * 8];
        const int *xc2 = &xs[(cs * 4 + 2) * 8];
        const int *xc3 = &xs[(cs * 4 + 3) * 8];
        const uint8_t *p0 = w0 + (size_t)b * 36;
        const uint8_t *p1 = w1 + (size_t)b * 36;
        const float d0 = __half2float(*reinterpret_cast<const __half *>(p0 - 4));
        const float d1 = __half2float(*reinterpret_cast<const __half *>(p1 - 4));

        int i00 = 0, i01 = 0, i02 = 0, i03 = 0, i10 = 0, i11 = 0, i12 = 0, i13 = 0;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int x0 = xc0[t];
            const int x1 = xc1[t];
            const int x2 = xc2[t];
            const int x3 = xc3[t];
            const int u0 = *reinterpret_cast<const int *>(p0 + t * 4);
            const int u1 = *reinterpret_cast<const int *>(p1 + t * 4);
            i00 = __dp4a(u0, x0, i00); i01 = __dp4a(u0, x1, i01);
            i02 = __dp4a(u0, x2, i02); i03 = __dp4a(u0, x3, i03);
            i10 = __dp4a(u1, x0, i10); i11 = __dp4a(u1, x1, i11);
            i12 = __dp4a(u1, x2, i12); i13 = __dp4a(u1, x3, i13);
        }
        if (ur0) {
            if (uc0) a00 += d0 * s0[b] * (float)i00;
            if (uc1) a01 += d0 * s1[b] * (float)i01;
            if (uc2) a02 += d0 * s2[b] * (float)i02;
            if (uc3) a03 += d0 * s3[b] * (float)i03;
        }
        if (ur1) {
            if (uc0) a10 += d1 * s0[b] * (float)i10;
            if (uc1) a11 += d1 * s1[b] * (float)i11;
            if (uc2) a12 += d1 * s2[b] * (float)i12;
            if (uc3) a13 += d1 * s3[b] * (float)i13;
        }
        __syncthreads();
    }
    if (ur0) {
        if (uc0) y[(size_t)(cA + 0) * ne1 + r0] = a00;
        if (uc1) y[(size_t)(cA + 1) * ne1 + r0] = a01;
        if (uc2) y[(size_t)(cA + 2) * ne1 + r0] = a02;
        if (uc3) y[(size_t)(cA + 3) * ne1 + r0] = a03;
    }
    if (ur1) {
        if (uc0) y[(size_t)(cA + 0) * ne1 + r1] = a10;
        if (uc1) y[(size_t)(cA + 1) * ne1 + r1] = a11;
        if (uc2) y[(size_t)(cA + 2) * ne1 + r1] = a12;
        if (uc3) y[(size_t)(cA + 3) * ne1 + r1] = a13;
    }
}

// Scalar q8_0 GEMM over the ALIGNED 36-byte layout: one thread per output
// element, used for the single-column lm_head projection.

// Tiled q8_0 GEMM with 4x the arithmetic intensity of lg_q8_0_gemm_dp4a:
// BM=128 x BN=64 output tile, BK=32 reduction tile (= exactly one q8_0 block),
// TM=8 x TN=4 = 32 int outputs per thread. The per-block int dp4a accumulator
// is folded against the fp16 scale once per BK tile (matching the reference
// accumulation order, so results stay bit-exact). locate-anything exclusive
// (grid differs from the shared dispatch contract). y[col][row].
// grid = (ceil(ne1/128), ceil(ncols/64)), block = (256,1,1).
extern "C" __global__ void lg_q8_0_gemm_tiled2(
    const uint8_t *__restrict__ w, const int8_t *__restrict__ qs,
    const float *__restrict__ sc, float *__restrict__ y,
    int ne0, int ne1, int ncols)
{
    const int BM = 128, BN = 64, BK = 32, TM = 8, TN = 4;
    __shared__ int8_t As[BM][BK];
    __shared__ int8_t Bs[BN][BK];
    __shared__ float sA[BM];
    __shared__ float sB[BN];

    const int tid = threadIdx.x;
    const int tr = tid / (BN / TN); // 0..15
    const int tc = tid % (BN / TN); // 0..15
    const int r0 = blockIdx.x * BM;
    const int c0 = blockIdx.y * BN;
    const int nb = ne0 / 32;

    float acc[TM][TN];
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (int blk = 0; blk < nb; ++blk) {
        for (int i = tid; i < BM * 32; i += 256) {
            const int r = i / 32, d = i % 32, rr = r0 + r;
            int8_t v = 0;
            float scv = 0.0f;
            if (rr < ne1) {
                const uint8_t *pb = w + ((size_t)rr * nb + blk) * 36;
                v = (int8_t)pb[4 + d];
                if (d == 0) scv = __half2float(*(const __half *)pb);
            }
            As[r][d] = v;
            if (d == 0) sA[r] = scv;
        }
        for (int i = tid; i < BN * 32; i += 256) {
            const int c = i / 32, d = i % 32, cc = c0 + c;
            int8_t v = 0;
            float scv = 0.0f;
            if (cc < ncols) {
                v = qs[(size_t)cc * ne0 + blk * 32 + d];
                if (d == 0) scv = sc[(size_t)cc * nb + blk];
            }
            Bs[c][d] = v;
            if (d == 0) sB[c] = scv;
        }
        __syncthreads();

        int iacc[TM][TN];
        for (int i = 0; i < TM; ++i)
            for (int j = 0; j < TN; ++j) iacc[i][j] = 0;
        for (int w4 = 0; w4 < 8; ++w4) {
            int au[TM], bu[TN];
            for (int i = 0; i < TM; ++i) au[i] = *(const int *)&As[tr * TM + i][w4 * 4];
            for (int j = 0; j < TN; ++j) bu[j] = *(const int *)&Bs[tc * TN + j][w4 * 4];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) iacc[i][j] = __dp4a(au[i], bu[j], iacc[i][j]);
        }
        for (int i = 0; i < TM; ++i) {
            const float sa = sA[tr * TM + i];
            for (int j = 0; j < TN; ++j)
                acc[i][j] += sa * sB[tc * TN + j] * (float)iacc[i][j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        const int r = r0 + tr * TM + i;
        if (r < ne1)
            for (int j = 0; j < TN; ++j) {
                const int c = c0 + tc * TN + j;
                if (c < ncols) y[(size_t)c * ne1 + r] = acc[i][j];
            }
    }
}
// grid = (ceil(ne1/64), ceil(ncols/32)), block = (256,1,1). y[col][row].
extern "C" __global__ void lg_q8_0_gemm_aligned(
    const uint8_t *__restrict__ w, const float *__restrict__ x, float *__restrict__ y,
    int ne0, int ne1, int ncols)
{
    const int row = blockIdx.x * 64 + (threadIdx.x & 63);
    const int col = blockIdx.y * 32 + (threadIdx.x >> 6);
    if (row >= ne1 || col >= ncols) return;
    const int nb = ne0 / 32;
    const uint8_t *wr = w + (size_t)row * nb * 36;
    const float *xr = x + (size_t)col * ne0;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = wr + (size_t)b * 36;
        const float d = __half2float(*reinterpret_cast<const __half *>(blk));
        const int8_t *q = reinterpret_cast<const int8_t *>(blk + 4);
        float dot = 0.f;
#pragma unroll
        for (int t = 0; t < 32; ++t) dot += (float)q[t] * xr[b * 32 + t];
        acc += d * dot;
    }
    y[(size_t)col * ne1 + row] = acc;
}

// Decode-oriented q8_0 GEMV (few columns): ONE WARP PER OUTPUT ROW, lane l
// walking the row's k-blocks in strides of 32 blocks. For a fixed k-block index
// the 32 lanes read 32 CONSECUTIVE 36-byte blocks, i.e. perfectly coalesced
// traffic; the k-blocks are then reduced across the warp with shuffles. The
// whole activation row is staged in DYNAMIC shared memory (ne0 bytes) at the
// top. An optional output bias serves the decode GEMMs with a bias epilogue.
// grid = (ceil(ne1 / 8), ncols), block = (32 * 8).
#define LA_GEMV_WARPS 8

extern "C" __global__ void lg_q8_0_gemv(
    const uint8_t *__restrict__ w, const int8_t *__restrict__ qs,
    const float *__restrict__ sc, float *__restrict__ y,
    int ne0, int ne1, int ncols, const float *__restrict__ bias)
{
    extern __shared__ signed char xs[];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int c = blockIdx.y;
    const int nb = ne0 / 32;
    const bool col_ok = (c < ncols);
    const int8_t *x = qs + (size_t)c * ne0;
    const float *s = sc + (size_t)c * nb;

    for (int i = tid; i < ne0; i += blockDim.x) xs[i] = col_ok ? x[i] : (signed char)0;
    __syncthreads();

    const int row = blockIdx.x * LA_GEMV_WARPS + warp;
    if (row >= ne1 || !col_ok) return;

    const uint8_t *wrow = w + (size_t)row * nb * 36;
    float acc = 0.f;
    for (int b = lane; b < nb; b += 32) {
        const uint8_t *blk = wrow + (size_t)b * 36;
        const float d = __half2float(*reinterpret_cast<const __half *>(blk));
        const int8_t *q = reinterpret_cast<const int8_t *>(blk + 4);
        int dot = 0;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int uw = *reinterpret_cast<const int *>(q + t * 4);
            const int ux = *reinterpret_cast<const int *>(xs + b * 32 + t * 4);
            dot = __dp4a(uw, ux, dot);
        }
        acc += d * s[b] * (float)dot;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[(size_t)c * ne1 + row] = bias ? acc + bias[row] : acc;
}

// ---------------------------------------------------------------------------
// Vision I/O and layout ops.
// ---------------------------------------------------------------------------

// y = x >= 0 ? x : slope * x, whole-plane. The slope is an argument rather than
// part of the name, so one kernel serves every model's constant.
extern "C" __global__ void lg_lrelu(
    const float *__restrict__ x, float *__restrict__ y, float slope, long n)
{
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    y[i] = v >= 0.0f ? v : slope * v;
}

// y = a + s * b, whole-plane. A fused kernel rather than `lg_add` + `lg_scale`
// because materialising s*b costs another plane per layer, and because lg_add's
// signature (an `int` length) serves its own hot path.
extern "C" __global__ void lg_add_scaled(
    const float *__restrict__ a, const float *__restrict__ b,
    float *__restrict__ out, long n, float scale)
{
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + scale * b[i];
}

// Nearest-neighbour 2x upsample, NCHW, out[c][oy][ox] = in[c][oy/2][ox/2].
//
// The integer halving is the contract: it is what PyTorch's
// F.interpolate(scale_factor=2, mode='nearest') does, and several frameworks
// instead map through (o+0.5)/2. This kernel reproduces the integer form, so a
// caller matching a PyTorch reference gets the reference's pixels.
// grid = (ceil(2w/32), ceil(2h/8)), block = (32, 8, 1).
extern "C" __global__ void lg_upsample2x_nearest(
    const float *__restrict__ in, float *__restrict__ out,
    int c, int h, int wd)
{
    const int oh = h * 2, ow = wd * 2;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ow || y >= oh) return;
    const size_t iplane = (size_t)h * wd;
    const size_t oplane = (size_t)oh * ow;
    for (int ci = 0; ci < c; ++ci) {
        out[(size_t)ci * oplane + (size_t)y * ow + x] =
            in[(size_t)ci * iplane + (size_t)(y >> 1) * wd + (x >> 1)];
    }
}

// Space-to-depth: `pixel_unshuffle(x, 2)` from [C][H][W] to [4C][H/2][W/2],
// with output channel c*4 + dy*2 + dx taken from input channel c at
// (2y+dy, 2x+dx). The permutation - not just the shape - is the contract: a
// checkpoint's following conv is ordered by it, so a different channel order
// produces a different (and wrong) model. Distinct from lg_merge_2x2, which is
// the same tap geometry on a token-major layout.
// grid = (ceil((w/2)/32), ceil((h/2)/8)), block = (32, 8, 1).
extern "C" __global__ void lg_pixel_unshuffle2(
    const float *__restrict__ in, float *__restrict__ out,
    int c, int h, int wd)
{
    const int oh = h / 2, ow = wd / 2;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ow || y >= oh) return;
    const size_t iplane = (size_t)h * wd;
    const size_t oplane = (size_t)oh * ow;
    for (int ci = 0; ci < c; ++ci) {
        const float *ip = in + (size_t)ci * iplane;
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                out[(size_t)(ci * 4 + dy * 2 + dx) * oplane + (size_t)y * ow + x] =
                    ip[(size_t)(2 * y + dy) * wd + (2 * x + dx)];
            }
        }
    }
}

// Depth-to-space: `pixel_shuffle(x, r)` from [r*r*C][H][W] to [C][r*H][r*W],
// with output channel c at (r*y+dy, r*x+dx) taken from input channel
// c*r*r + dy*r + dx at (y, x). THE PERMUTATION IS THE CONTRACT, not the shape
// (as for the unshuffle above): a checkpoint's following conv is ordered by it,
// so a different channel order produces a different and wrong model.
//
// Three engines in this family wrote this op out privately - nafnet-rs and
// swin2sr-rs at r = 2, hat-rs parametric because HAT ships x3 as well as x4 -
// and swin2sr-rs's GPU REFUSED a scale-3 head for want of a general kernel while
// its CPU had one.
//
// `r` is a RUNTIME argument: hat-rs launches r = 2 (the two x4 octaves) and
// r = 3 (its x3 head) from one place. That is why the r == 2 and r == 3 cases
// are written out separately, and the reason is measured rather than stylistic:
// at the same grid, the generic `oy / r` form is 6-8% SLOWER than nafnet's
// fixed-2 kernel at r == 2 (0.0399 ms against 0.0369 ms on an sm_61 card), so
// the shift path is what keeps this at least as fast as the copy it replaces;
// and r == 3 needs a literal divisor so the compiler emits a magic multiply
// instead of a runtime division. Any other r takes the generic form, which is
// exact but not tuned. Measured against every copy it replaces, at each
// engine's own geometry, interleaved in one process: r == 2 TIES nafnet's
// kernel exactly (0.0369 ms at [64][64][64]) and is 1.9-2.1x hat-rs's flat-grid
// form; r == 3 is 1.88x hat-rs's (0.0799 against 0.1505 ms).
//
// grid = (ceil(r*w/32), ceil(r*h/8), c), block = (32, 8, 1). One thread per
// output element with the channel in blockIdx.z, which is where the speed comes
// from: the two divisions by the runtime output width and height disappear, and
// a warp's four-element group stays inside one input row.
extern "C" __global__ void lg_pixel_shuffle(
    const float *__restrict__ src, float *__restrict__ dst,
    int c, int h, int w, int r)
{
    const int oh = h * r, ow = w * r;
    const int ch = blockIdx.z;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= c || oy >= oh || ox >= ow) return;
    int y, x, dy, dx;
    if (r == 2) {
        y = oy >> 1; x = ox >> 1; dy = oy & 1; dx = ox & 1;
    } else if (r == 3) {
        y = oy / 3; x = ox / 3; dy = oy - y * 3; dx = ox - x * 3;
    } else {
        y = oy / r; x = ox / r; dy = oy - y * r; dx = ox - x * r;
    }
    const int sch = ch * r * r + dy * r + dx;
    dst[((size_t)ch * oh + oy) * ow + ox] = src[((size_t)sch * h + y) * w + x];
}

// Device-side image marshalling is deliberately NOT here: at 4K the host-side
// interleaved-RGB/planar-NCHW conversion is negligible against the forward pass,
// so such kernels would add an unexercised code path for no gain.

// ---------------------------------------------------------------------------
// Batched 2-D real Fourier transforms (R2C / C2R), for any model that runs a
// spectral convolution or any other FFT stage. No cuFFT: the point of this
// toolkit is a driver-only install, so the transform is part of the kernel set.
//
// Geometry: square planes of side n (the shared-memory footprint caps it at 64),
// batch = number of planes, and the half spectrum PyTorch's
// `rfftn(norm='ortho')` defines: [batch][n][n/2+1] interleaved complex, scaled by
// 1/sqrt(n*n). `lg_fft2_c2r` consumes exactly that and returns the real plane.
//
// Both transforms are UNNORMALISED, exactly like the cuFFT pair they replace, so
// a caller that wants the ortho convention applies `scale` itself: pass 1.0 for a
// bare forward transform, or 1/sqrt(n*n) to fold the ortho factor in on the way
// out (which is what a caller wanting a stacked real/imag layout does in its own
// packing pass). `lg_fft2_c2r` takes 1/sqrt(n*n) to produce the ortho inverse.
//
// One block per plane, 8 warps / 256 threads. The plane lives in shared memory as
// float2 with one complex pad per row, so the column pass rides the same stride.
// Each warp owns one whole row (or column) at a time; the transform is radix-2 DIT
// with the bit reversal folded into the load, twiddles read from a shared
// per-stage table, and __syncwarp() - not a block barrier - between butterfly
// stages. Only three __syncthreads per plane (load, between the two passes,
// before the store).
//
// Warp-per-transform rather than the whole block cooperating on one row: the
// block-cooperative form serialises 2*n*log2(n) dependent wide barriers per plane
// and measured 1.147 ms per 64x64x128 call against 0.083 ms for this form.
//
// grid = (batch, 1, 1), block = (256, 1, 1), no dynamic shared memory.
// ---------------------------------------------------------------------------

#define LG_FFT_MAX_N 64
#define LG_FFT_ROW_STRIDE (LG_FFT_MAX_N + 1)

__device__ __forceinline__ float2 lg_fft_cmul(float2 a, float2 b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ int lg_fft_bitrev(int i, int nb)
{
    int j = 0;
    for (int b = 0; b < nb; ++b) if (i & (1 << b)) j |= (1 << (nb - 1 - b));
    return j;
}

// One twiddle table per plane size: stage len contributes entries [len/2 - 1,
// len - 1], so the table holds n - 1 complex values and stage len starts at
// len/2 - 1.
__device__ void lg_fft_twiddles(float2 *__restrict__ tw, int n, int sign)
{
    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        for (int k = threadIdx.x; k < half; k += blockDim.x) {
            float s, c;
            __sincosf((float)sign * 2.0f * (float)M_PI * (float)k / (float)len, &s, &c);
            tw[(half - 1) + k] = make_float2(c, s);
        }
    }
}

// Transform n points of one row/column in place, natural order in and out.
// Butterfly m of a stage is (i, i + half) with i = (m / half) * len + (m % half);
// m enumerates exactly n/2 butterflies per stage.
__device__ void lg_fft_warp(float2 *__restrict__ row, const float2 *__restrict__ tw, int n, int nb)
{
    const int lane = threadIdx.x & 31;
    const int per = n >> 5;
    const int n_half = n >> 1;
    // Fixed unrolled trip count over the 6 stages LG_FFT_MAX_N allows, ended by
    // `len > n`. The unroll is worth ~20% at n=64: with a dynamic loop bound the
    // compiler stops strength-reducing the shared-memory indexing in the butterfly.
    #pragma unroll
    for (int st = 0; st < 6; ++st) {
        const int len = 2 << st;
        if (len > n) break;
        const int half = len >> 1;
        for (int e = 0; e < per; ++e) {
            const int m = e * 32 + lane;
            if (m >= n_half) continue;
            const int i = (m / half) * len + (m % half);
            const int j = i + half;
            const float2 w = tw[(half - 1) + (m % half)];
            const float2 u = row[i];
            const float2 v = lg_fft_cmul(row[j], w);
            row[i] = make_float2(u.x + v.x, u.y + v.y);
            row[j] = make_float2(u.x - v.x, u.y - v.y);
        }
        __syncwarp();
    }
    (void)nb;
}

// Batched R2C. `in` is [batch][n][n] row-major; `out` is [batch][n][n/2+1]
// interleaved complex, unnormalised (pass scale = 1/sqrt(n*n) for the ortho form).
extern "C" __global__ void lg_fft2_r2c(
    const float *__restrict__ in, float *__restrict__ out,
    int batch, int n, int nb, float scale)
{
    __shared__ float2 s[LG_FFT_MAX_N * LG_FFT_ROW_STRIDE];
    __shared__ float2 scr[8][LG_FFT_MAX_N];
    __shared__ float2 tw[LG_FFT_MAX_N];
    const int plane = blockIdx.x;
    if (plane >= batch) return;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int per = n >> 5;
    const int hw = n / 2 + 1;

    lg_fft_twiddles(tw, n, -1);
    __syncthreads();

    // Load with the horizontal bit reversal already applied, so the row pass
    // starts in bit-reversed order and writes natural order back.
    const float *src = in + (size_t)plane * n * n;
    for (int i = threadIdx.x; i < n * n; i += blockDim.x) {
        const int y = i / n;
        const int x = i - y * n;
        s[y * LG_FFT_ROW_STRIDE + lg_fft_bitrev(x, nb)] = make_float2(src[i], 0.0f);
    }
    __syncthreads();

    for (int r = warp; r < n; r += 8)
        lg_fft_warp(s + r * LG_FFT_ROW_STRIDE, tw, n, nb);
    __syncthreads();

    // Column pass per spectral column; the gather into `scr` applies the vertical
    // bit reversal, and the store is contiguous along y because the spectrum
    // column index in the transform output is the same as the transform column.
    for (int c = warp; c < hw; c += 8) {
        for (int e = 0; e < per; ++e) {
            const int i = e * 32 + lane;
            scr[warp][lg_fft_bitrev(i, nb)] = s[i * LG_FFT_ROW_STRIDE + c];
        }
        __syncwarp();
        lg_fft_warp(scr[warp], tw, n, nb);
        for (int e = 0; e < per; ++e) {
            const int y = e * 32 + lane;
            const float2 v = scr[warp][y];
            ((float2 *)out)[((size_t)plane * n + y) * hw + c] =
                make_float2(v.x * scale, v.y * scale);
        }
    }
}

// Batched C2R. `in` is [batch][n][n/2+1] interleaved complex, `out` is
// [batch][n][n]. The full spectrum is rebuilt by conjugate symmetry,
// X[u][x] = conj(X[(n-u) % n][n-x]) for x > n/2, then the column pass runs before
// the row pass (the row pass then starts from natural order) with +1 twiddles,
// and the real part is scaled.
extern "C" __global__ void lg_fft2_c2r(
    const float *__restrict__ in, float *__restrict__ out,
    int batch, int n, int nb, float scale)
{
    __shared__ float2 s[LG_FFT_MAX_N * LG_FFT_ROW_STRIDE];
    __shared__ float2 scr[8][LG_FFT_MAX_N];
    __shared__ float2 tw[LG_FFT_MAX_N];
    const int plane = blockIdx.x;
    if (plane >= batch) return;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int per = n >> 5;
    const int hw = n / 2 + 1;

    lg_fft_twiddles(tw, n, +1);
    __syncthreads();

    const float2 *src = ((const float2 *)in) + (size_t)plane * n * hw;
    for (int i = threadIdx.x; i < n * n; i += blockDim.x) {
        const int u = i / n;
        const int x = i - u * n;
        float2 v;
        if (x <= n / 2) {
            v = src[(size_t)u * hw + x];
        } else {
            const int uu = (n - u) % n;
            const int xx = n - x;
            const float2 t = src[(size_t)uu * hw + xx];
            v = make_float2(t.x, -t.y);
        }
        s[u * LG_FFT_ROW_STRIDE + lg_fft_bitrev(x, nb)] = v;
    }
    __syncthreads();

    for (int x = warp; x < n; x += 8) {
        for (int e = 0; e < per; ++e) {
            const int u = e * 32 + lane;
            scr[warp][lg_fft_bitrev(u, nb)] = s[u * LG_FFT_ROW_STRIDE + x];
        }
        __syncwarp();
        lg_fft_warp(scr[warp], tw, n, nb);
        for (int e = 0; e < per; ++e) {
            const int u = e * 32 + lane;
            s[u * LG_FFT_ROW_STRIDE + x] = scr[warp][u];
        }
        __syncwarp();
    }
    __syncthreads();

    for (int r = warp; r < n; r += 8)
        lg_fft_warp(s + r * LG_FFT_ROW_STRIDE, tw, n, nb);
    __syncthreads();

    float *dst = out + (size_t)plane * n * n;
    for (int i = threadIdx.x; i < n * n; i += blockDim.x) {
        const int u = i / n;
        const int x = i - u * n;
        dst[i] = s[u * LG_FFT_ROW_STRIDE + x].x * scale;
    }
}

// ===========================================================================
// 12. Promoted from the vision engines
// ===========================================================================
//
// Four families that were private to one engine each. Every number quoted here
// comes from that engine's own per-kernel instrument (buffers resident, one
// clock window, the ratio interleaved):
//
//   * `lg_linear_rb` / `lg_conv1x1_rb` - the register-blocked GEMM in the two
//     layouts that were already in this file: 2.0x `lg_linear` and 2.3x the
//     tiled form, 10x `lg_conv1x1`, on the geometries scunet-rs runs.
//   * `lg_layer_norm_warp` - 2.2x `lg_layer_norm` at a 256-wide row and 8.6x at
//     32 wide. The rows a transformer normalizes are narrow, which is where
//     holding a row in registers wins; `lg_layer_norm` remains the better
//     choice for the wide rows it was written for, so this is an ADDITION.
//   * `lg_conv2x2s2` / `lg_conv_t2x2` - no spelling existed here, and nafnet,
//     maxim and ifan each carry a private stride-2 stage.
//   * `lg_window_gather` / `lg_window_scatter` - the Swin window assembly that
//     rmbg, swin2sr and scunet each wrote out again, with the same index map.
//
// The twins in `src/ops/cpu.rs` implement the accumulation order each kernel
// states. Where an op is a replacement for one already here, the order was kept
// identical as well, so a swap can be checked by equality rather than by
// tolerance.

// ---------------------------------------------------------------------------
// Register-blocked GEMM, in the token layout (`lg_linear`) and the plane layout
// (a 1x1 convolution over NCHW).
//
// ONE FMA PER TWO SHARED-MEMORY LOADS is what the 16x16-tile kernels above do:
// one output element per thread, so the inner loop is `acc += xs[ty][k] *
// ws[tx][k]` and shared bandwidth binds at about 850 GFLOP/s on an sm_61 card.
// Hoisting a TM x TN output tile into registers divides the loads per FMA by
// (TM*TN)/(TM+TN): 4x4 gives 2 loads per 16 FMAs instead of 1 per 1.
//
// A tile shape above 4x4 measured WORSE, not better: 256 threads times the
// 4x4 shape's 124 registers already allows two blocks an SM, while 8x8's 240
// registers allow one. The registers, not the load ratio, are the constraint.
// ---------------------------------------------------------------------------

#define LG_RB_PAD 1   // odd, so a warp's shared-memory strides land on distinct banks

// BM x BN outputs per block, TM x TN of them per thread.
template <int TM, int TN, int BK, bool PLANE, bool BIAS_FIRST>
__device__ void lg_rb_body(
    const float *__restrict__ x, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int rows, int c_in, int c_out)
{
    constexpr int BM = 16 * TM;
    constexpr int BN = 16 * TN;
    __shared__ float xs[BM][BK + LG_RB_PAD];
    __shared__ float ws[BN][BK + LG_RB_PAD];

    const int tid = threadIdx.y * 16 + threadIdx.x;
    // Rows on the x axis for the plane layout and on y for the token layout: a
    // warp has to write consecutive `out[o*rows + r]` (plane) or consecutive
    // `out[r*c_out + o]` (token), and `gridDim.y` stops at 65535 while
    // `gridDim.x` holds 2^31-1 - a 1024x1024 plane is already 65536 pixel tiles.
    const int r0 = (PLANE ? blockIdx.x : blockIdx.y) * BM;
    const int o0 = (PLANE ? blockIdx.y : blockIdx.x) * BN;
    const int tr = (PLANE ? threadIdx.x : threadIdx.y) * TM;
    const int to = (PLANE ? threadIdx.y : threadIdx.x) * TN;

    // `lg_conv1x1` folds the bias into the accumulator before the first
    // multiply and `lg_linear` adds it after the last one. Both orders are
    // reproduced here, one per instantiation, so replacing either kernel is a
    // bit-exact change rather than a rounding-level one.
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = (BIAS_FIRST && bias != nullptr && o0 + to + j < c_out)
                            ? bias[o0 + to + j] : 0.0f;

    // A full tile is the common case at every released checkpoint's shape, and
    // it is the unrolled one; the guards serve a caller that is not that shape.
    const bool full = (r0 + BM <= rows) && (o0 + BN <= c_out) && (c_in % BK == 0);
    for (int k0 = 0; k0 < c_in; k0 += BK) {
#pragma unroll
        for (int t = 0; t < BM * BK / 256; ++t) {
            const int idx = tid + t * 256;
            // CONSECUTIVE THREADS TAKE THE CONTIGUOUS AXIS, which is `k` in the
            // token layout and `row` in the plane layout: the decomposition, not
            // just the address, changes with the layout.
            const int row = PLANE ? (idx % BM) : (idx / BK);
            const int k = PLANE ? (idx / BM) : (idx % BK);
            float v = 0.0f;
            if (full || (r0 + row < rows && k0 + k < c_in)) {
                v = PLANE ? x[(size_t)(k0 + k) * rows + r0 + row]
                          : x[(size_t)(r0 + row) * c_in + k0 + k];
            }
            xs[row][k] = v;
        }
#pragma unroll
        for (int t = 0; t < BN * BK / 256; ++t) {
            const int idx = tid + t * 256;
            const int n = idx / BK;
            const int k = idx % BK;
            // The weight tile is the same read in both layouts: `w[o][k]`
            // row-major with consecutive threads on consecutive `k`.
            ws[n][k] = (full || (o0 + n < c_out && k0 + k < c_in))
                           ? w[(size_t)(o0 + n) * c_in + k0 + k] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float b[TN];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = ws[to + j][k];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const float a = xs[tr + i][k];
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a * b[j];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int r = r0 + tr + i;
            const int o = o0 + to + j;
            if (r < rows && o < c_out) {
                const float b = (BIAS_FIRST || bias == nullptr) ? 0.0f : bias[o];
                if (PLANE) out[(size_t)o * rows + r] = acc[i][j] + b;
                else out[(size_t)r * c_out + o] = acc[i][j] + b;
            }
        }
    }
}

// Token layout, the same contract as `lg_linear`: out[i*C_out + o] =
// bias[o] + sum_c x[i*C_in + c] * w[o*C_in + c], c ascending, bias added last.
// grid = (ceil(c_out/64), ceil(rows/64), 1), block = (16,16,1) = 256 threads.
extern "C" __global__ void lg_linear_rb(
    const float *__restrict__ x, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int rows, int c_in, int c_out)
{
    lg_rb_body<4, 4, 16, false, false>(x, w, bias, out, rows, c_in, c_out);
}

// Plane layout, the same contract as `lg_conv1x1`: for every pixel p,
// out[o*plane + p] = bias[o] + sum_c w[o*c_in + c] * in[c*plane + p], c
// ascending, bias folded into the accumulator first. `plane` is h*w.
// grid = (ceil(plane/64), ceil(c_out/64), 1), block = (16,16,1).
extern "C" __global__ void lg_conv1x1_rb(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd)
{
    lg_rb_body<4, 4, 16, true, true>(in, w, bias, out, h * wd, c_in, c_out);
}

// ---------------------------------------------------------------------------
// LayerNorm, ONE WARP PER ROW.
//
// The row of a transformer's normalization is 32-256 wide, and at that width a
// whole row fits in one lane's register array (128 floats = 32 per lane at the
// 32 lane stride). Holding it there turns `lg_layer_norm`'s three global reads
// and one write into one read and one write, with both reductions in registers.
//
// The reduction tree is `lg_layer_norm`'s own (a shfl_down from 16 down to 1,
// then the lane-0 broadcast), so the two kernels agree bit for bit on the part
// that is a reduction. mean and var are the one-pass E[x^2] - mean^2 form both
// kernels use, and the CPU twin reproduces the tree rather than a serial sum.
// ---------------------------------------------------------------------------

#define LG_LN_WARP_MAX 128   // floats a lane array can hold at a 32-lane stride

__device__ float lg_warp_sum(float v)
{
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return __shfl_sync(0xffffffffu, v, 0);
}

// grid = (ceil(nrows/8), 1, 1), block = (256,1,1) - eight rows a block.
extern "C" __global__ void lg_layer_norm_warp(
    const float *__restrict__ x, const float *__restrict__ w, const float *__restrict__ b,
    float *__restrict__ y, int ne0, int nrows, float eps)
{
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= nrows) return;
    const float *xr = x + (size_t)row * ne0;
    float *yr = y + (size_t)row * ne0;

    // The register path, and the one every released vision checkpoint takes.
    if (ne0 <= LG_LN_WARP_MAX) {
        float v[LG_LN_WARP_MAX / 32];
        float s = 0.0f, q = 0.0f;
        for (int i = lane, k = 0; i < ne0; i += 32, ++k) {
            v[k] = xr[i];
            s += v[k];
            q += v[k] * v[k];
        }
        s = lg_warp_sum(s);
        q = lg_warp_sum(q);
        const float n = (float)ne0;
        const float mean = s / n;
        const float var = q / n - mean * mean;
        const float rstd = rsqrtf(fmaxf(var, 0.0f) + eps);
        for (int i = lane, k = 0; i < ne0; i += 32, ++k)
            yr[i] = (v[k] - mean) * rstd * w[i] + b[i];
        return;
    }

    // A wider row re-reads x for the output pass: the same reductions, one more
    // pass over memory. This path is why the kernel is a strict generalisation of
    // the register one rather than a special case.
    float s = 0.0f, q = 0.0f;
    for (int i = lane; i < ne0; i += 32) {
        const float t = xr[i];
        s += t;
        q += t * t;
    }
    s = lg_warp_sum(s);
    q = lg_warp_sum(q);
    const float n = (float)ne0;
    const float mean = s / n;
    const float rstd = rsqrtf(fmaxf(q / n - mean * mean, 0.0f) + eps);
    for (int i = lane; i < ne0; i += 32)
        yr[i] = (xr[i] - mean) * rstd * w[i] + b[i];
}

// ---------------------------------------------------------------------------
// The tiled convolution: 3x3 stride-1 pad-1 and 1x1, staged in shared memory.
//
// THESE ARE ALTERNATIVE IMPLEMENTATIONS of `lg_conv3x3s1p1` and `lg_conv1x1`,
// not new operators. Same NCHW layout, same weight layout ([c_out][c_in][2+][2+]
// with ci contiguous across the taps), same nullable bias. What changes is WHERE
// THE REUSE COMES FROM. Section 6's kernels give one thread one output element
// held in one register, so the inner loop is a serial FMA chain behind a fresh
// load of both operands every iteration - fine when the input is the bottleneck
// and the bottleneck is DRAM, but it leaves the FMA pipes idle: measured at 89
// to 224 GFLOP/s on an sm_61 card, against the 1600-2067 this reaches. Here a
// block stages a tile of the input in shared memory and each thread holds
// OC_TILE accumulators over TPX pixels, so one input value serves every output
// channel in the tile and the shared traffic is TPX + OC_TILE loads for
// OC_TILE * TPX multiply-adds.
//
// THE ACCUMULATION ORDER IS THEREFORE DIFFERENT, AND THAT IS NOT A BUG TO FIX:
// (ci, ky, kx) instead of section 6's (ky, kx, ci). Staging channels in tiles
// makes the channel tile the OUTER loop; keeping (ky, kx, ci) would need every
// one of c_in channels resident at once, which is 166 KB of shared memory for 32
// channels of a 130x10 tile against a 48 KB limit. Two consequences follow that
// a caller has to accept rather than work around:
//
//   * THE RESULT IS NOT BIT-IDENTICAL to `lg_conv3x3s1p1`. Measured on 32->32 at
//     128x128 with the activation off: 489413 of 524288 elements differ, worst
//     2.03e-06. This is a magnitude-bound swap, NOT the equality swap that
//     `lg_conv1x1_rb`/`lg_linear_rb` offer for their operators - a caller that
//     needs bit-exactness must keep the section 6 kernel for that op;
//   * the halo is ZERO-FILLED rather than skipped, which is the same sum with
//     zeros added: harmless in itself, and the same class of last-bit movement.
//
// WHAT THEY ARE WORTH, measured against the kernels they duplicate on an idle
// GTX 1080, minimum of five reps, three tensors resident per launch so the
// memory-bound forms are not penalised:
//
//     3x3  32->32     512x512    2.99 ms   1618 GFLOP/s    28.85 ms    167   9.7x
//     3x3  128->128   512x512   37.4  ms   2067           683.2  ms    113  18.3x
//     3x3  3->32      512x512    0.50 ms    903             5.10 ms     89  10.2x
//     3x3  64->64     720x720   20.7  ms   1845           336.3  ms    114  16.2x
//     3x3  32->32     1080p     23.6  ms   1620           170.9  ms    224   7.2x
//     1x1  128->128   512x512   13.4  ms   5767            71.6  ms   1080   5.3x
//     1x1  128->15232 64x128    48.4  ms   5936           197.6  ms   1455   4.1x
//     1x1  128->15232 64x64     46.5  ms   3091            33.3  ms   4310   0.7x
//
// ONE GEOMETRY SERVES EVERY CHANNEL COUNT for the 3x3 - c_in 3, 32, 64 and 128
// all win, at 512x512 and at 1080p - which is why it is offered as a general op
// rather than as a tuned special case.
//
// THE 1x1 IS MORE NARROWLY USEFUL and is here for the case it was written for: a
// very wide output (15232 channels) over a small plane, where the cost is the
// INPUT RE-READ (c_out/OC passes over the same input) rather than the
// arithmetic. It LOSES at 64x64 because there is not enough spatial extent to
// amortise that, so a caller with a small plane and many input channels should
// use `lg_conv1x1` or `lg_conv1x1_rb` instead. Note the three 1x1 forms differ
// in exactly one respect each: `lg_conv1x1` and `lg_conv1x1_rb` agree bit for
// bit (bias first, c ascending), and this one neither agrees with them nor is
// ordered like them, which is why it is a separate name rather than a
// replacement.
//
// SHARED MEMORY IS STATIC AND SIZED PER KERNEL. nvcc sizes DYNAMIC shared memory
// for the whole kernel, so one kernel cannot let an instantiation pick its own
// tile shape; each kernel below therefore carries its own arrays and derives the
// staged extent from its own constants. The 3x3's cost is 42752 bytes -
// sh[8][10][130] for an eight-channel tile of a 130x10 halo'd region plus
// sw[8][4][9] for its weights - which is inside sm_75's 48 KB static limit with
// room for three blocks an SM, so no opt-in carveout is needed on any target
// this toolkit builds (the 64 KiB opt-in is only required from sm_80 up) and
// neither launch calls cudaFuncSetAttribute.
// ---------------------------------------------------------------------------

// A 32x8 block with four columns per thread: a 128x8 output tile, and 32
// accumulators per thread. The columns a thread owns are TBX apart rather than
// adjacent, so that consecutive threads read consecutive shared addresses; four
// adjacent columns per thread would make every shared load a four-way bank
// conflict. The staged region is one column and one row wider than the output
// tile at each edge, which IS the 3x3's pad of 1.
constexpr int LG_C3_TBX = 32, LG_C3_TBY = 8, LG_C3_TPX = 4;
constexpr int LG_C3_TW = LG_C3_TBX * LG_C3_TPX;   // 128 output columns per tile
constexpr int LG_C3_TH = LG_C3_TBY;               // 8 output rows per tile
constexpr int LG_C3_CI = 4;                       // input channels per staged tile
constexpr int LG_C3_OC = 8;                       // accumulators per thread
constexpr int LG_C3_SW = LG_C3_TW + 2;            // staged width, with the halo
constexpr int LG_C3_SH = LG_C3_TH + 2;            // staged height, with the halo

// out[oc][y][x] = act(bias[oc] + sum_{ci,ky,kx} w[oc][ci][ky][kx]
//                                      * in[ci][y+ky-1][x+kx-1])
// Weight layout [c_out][c_in][3][3], ci contiguous. ORDER: ci, ky, kx - see the
// block comment above for why that is not the section 6 order.
// act: 0 none, 1 relu, 2 leaky relu with `act_p` as the slope (the convention
// `lg_conv3x3_winograd` uses).
// grid = (ceil(wd/128), ceil(h/8), ceil(c_out/8)), block = (32,8,1).
extern "C" __global__ void __launch_bounds__(LG_C3_TBX * LG_C3_TBY) lg_conv3x3_tile(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, int act, float act_p)
{
    __shared__ float sh[LG_C3_CI][LG_C3_SH][LG_C3_SW];
    __shared__ float sw[LG_C3_OC][LG_C3_CI][9];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * LG_C3_TBX + tx;
    const int ox0 = blockIdx.x * LG_C3_TW;
    const int oy0 = blockIdx.y * LG_C3_TH;
    const int oc0 = blockIdx.z * LG_C3_OC;
    // The input column and row the staged tile starts at: the output tile's
    // origin shifted back by the pad.
    const int gx0 = ox0 - 1, gy0 = oy0 - 1;

    float acc[LG_C3_OC][LG_C3_TPX];
#pragma unroll
    for (int o = 0; o < LG_C3_OC; ++o)
#pragma unroll
        for (int p = 0; p < LG_C3_TPX; ++p) acc[o][p] = 0.0f;

    for (int ci0 = 0; ci0 < c_in; ci0 += LG_C3_CI) {
        // Stage the input tile. Out-of-range positions are ZERO, which is what
        // the padded convolution's boundary contributes, so the compute loop
        // below needs no boundary test at all. A channel tile past c_in is also
        // zero, and contributes nothing for the same reason.
#pragma unroll
        for (int k = 0; k < LG_C3_CI; ++k) {
            const int ci = ci0 + k;
            for (int i = tid; i < LG_C3_SW * LG_C3_SH; i += LG_C3_TBX * LG_C3_TBY) {
                const int sy = i / LG_C3_SW;
                const int sx = i - sy * LG_C3_SW;
                const int gy = gy0 + sy;
                const int gx = gx0 + sx;
                float v = 0.0f;
                if (ci < c_in && gy >= 0 && gy < h && gx >= 0 && gx < wd)
                    v = in[((size_t)ci * h + gy) * wd + gx];
                sh[k][sy][sx] = v;
            }
        }
        // Stage this tile's weights in the (oc, ci, tap) order the compute loop
        // reads them in, for the same reason.
        for (int i = tid; i < LG_C3_OC * LG_C3_CI * 9; i += LG_C3_TBX * LG_C3_TBY) {
            const int o = i / (LG_C3_CI * 9);
            const int r = i - o * (LG_C3_CI * 9);
            const int k = r / 9;
            const int t = r - k * 9;
            const int oc = oc0 + o, ci = ci0 + k;
            float v = 0.0f;
            if (oc < c_out && ci < c_in) v = w[((size_t)oc * c_in + ci) * 9 + t];
            sw[o][k][t] = v;
        }
        __syncthreads();

        // Per (ci, ky, kx) the TPX input values are loaded once and reused for
        // every output channel in the tile, and each weight is loaded once and
        // reused for every pixel.
#pragma unroll
        for (int k = 0; k < LG_C3_CI; ++k)
#pragma unroll
            for (int ky = 0; ky < 3; ++ky)
#pragma unroll
                for (int kx = 0; kx < 3; ++kx) {
                    float v[LG_C3_TPX];
#pragma unroll
                    for (int p = 0; p < LG_C3_TPX; ++p)
                        v[p] = sh[k][ty + ky][tx + p * LG_C3_TBX + kx];
#pragma unroll
                    for (int o = 0; o < LG_C3_OC; ++o) {
                        const float wv = sw[o][k][ky * 3 + kx];
#pragma unroll
                        for (int p = 0; p < LG_C3_TPX; ++p) acc[o][p] += wv * v[p];
                    }
                }
        // The next channel tile overwrites `sh`, so every reader must be done.
        __syncthreads();
    }

    // The bias is added once, after the whole sum. The output is the same size
    // as the input (stride 1, pad 1), so the tile indices are already output
    // coordinates.
    const int my = oy0 + ty;
#pragma unroll
    for (int o = 0; o < LG_C3_OC; ++o) {
        const int oc = oc0 + o;
        if (oc < c_out && my < h) {
            const float b = bias ? bias[oc] : 0.0f;
#pragma unroll
            for (int p = 0; p < LG_C3_TPX; ++p) {
                const int gx = ox0 + tx + p * LG_C3_TBX;
                if (gx < wd) {
                    float v = acc[o][p] + b;
                    if (act == 1) v = v > 0.0f ? v : 0.0f;
                    else if (act == 2) v = v >= 0.0f ? v : act_p * v;
                    out[((size_t)oc * h + my) * wd + gx] = v;
                }
            }
        }
    }
}

// The same block structure for the 1x1, with a WIDE channel tile instead of a
// halo. A 1x1 has no spatial reuse to stage - one input value serves one output
// pixel - so the shared tile only exists to be re-read once per output-channel
// tile, and what a wide OC buys is fewer passes over the input: at 15232 output
// channels an 8-channel tile re-reads the input 1904 times and a 32-channel tile
// 476 times. The channel tile is 4 rather than 8 because at KW = KH = 1 the
// staged tile is exactly the output tile, so a wider one costs nothing but
// registers in the staging loop.
constexpr int LG_C1_TBX = 32, LG_C1_TBY = 4, LG_C1_TPX = 4;
constexpr int LG_C1_TW = LG_C1_TBX * LG_C1_TPX;   // 128 output columns per tile
constexpr int LG_C1_TH = LG_C1_TBY;               // 4 output rows per tile
constexpr int LG_C1_CI = 4;
constexpr int LG_C1_OC = 32;

// out[oc][y][x] = act(bias[oc] + sum_ci w[oc][ci] * in[ci][y][x])
// Weight layout [c_out][c_in]. ORDER: ci.
// grid = (ceil(wd/128), ceil(h/4), ceil(c_out/32)), block = (32,4,1).
extern "C" __global__ void __launch_bounds__(LG_C1_TBX * LG_C1_TBY) lg_conv1x1_tile(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, int act, float act_p)
{
    __shared__ float sh[LG_C1_CI][LG_C1_TH][LG_C1_TW];
    __shared__ float sw[LG_C1_OC][LG_C1_CI];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * LG_C1_TBX + tx;
    const int ox0 = blockIdx.x * LG_C1_TW;
    const int oy0 = blockIdx.y * LG_C1_TH;
    const int oc0 = blockIdx.z * LG_C1_OC;

    float acc[LG_C1_OC][LG_C1_TPX];
#pragma unroll
    for (int o = 0; o < LG_C1_OC; ++o)
#pragma unroll
        for (int p = 0; p < LG_C1_TPX; ++p) acc[o][p] = 0.0f;

    for (int ci0 = 0; ci0 < c_in; ci0 += LG_C1_CI) {
#pragma unroll
        for (int k = 0; k < LG_C1_CI; ++k) {
            const int ci = ci0 + k;
            for (int i = tid; i < LG_C1_TW * LG_C1_TH; i += LG_C1_TBX * LG_C1_TBY) {
                const int sy = i / LG_C1_TW;
                const int sx = i - sy * LG_C1_TW;
                const int gy = oy0 + sy;
                const int gx = ox0 + sx;
                float v = 0.0f;
                if (ci < c_in && gy < h && gx < wd)
                    v = in[((size_t)ci * h + gy) * wd + gx];
                sh[k][sy][sx] = v;
            }
        }
        for (int i = tid; i < LG_C1_OC * LG_C1_CI; i += LG_C1_TBX * LG_C1_TBY) {
            const int o = i / LG_C1_CI;
            const int k = i - o * LG_C1_CI;
            const int oc = oc0 + o, ci = ci0 + k;
            float v = 0.0f;
            if (oc < c_out && ci < c_in) v = w[(size_t)oc * c_in + ci];
            sw[o][k] = v;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < LG_C1_CI; ++k) {
            float v[LG_C1_TPX];
#pragma unroll
            for (int p = 0; p < LG_C1_TPX; ++p) v[p] = sh[k][ty][tx + p * LG_C1_TBX];
#pragma unroll
            for (int o = 0; o < LG_C1_OC; ++o) {
                const float wv = sw[o][k];
#pragma unroll
                for (int p = 0; p < LG_C1_TPX; ++p) acc[o][p] += wv * v[p];
            }
        }
        __syncthreads();
    }

    const int my = oy0 + ty;
#pragma unroll
    for (int o = 0; o < LG_C1_OC; ++o) {
        const int oc = oc0 + o;
        if (oc < c_out && my < h) {
            const float b = bias ? bias[oc] : 0.0f;
#pragma unroll
            for (int p = 0; p < LG_C1_TPX; ++p) {
                const int gx = ox0 + tx + p * LG_C1_TBX;
                if (gx < wd) {
                    float v = acc[o][p] + b;
                    if (act == 1) v = v > 0.0f ? v : 0.0f;
                    else if (act == 2) v = v >= 0.0f ? v : act_p * v;
                    out[((size_t)oc * h + my) * wd + gx] = v;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The stride-2 2x2 pair.
//
// Both give one thread one output pixel of EIGHT output channels. At stride 2 a
// 2x2 kernel means an input element belongs to exactly ONE output's tap set, so
// there is no reuse over space to exploit - the reuse has to come from the
// channels, and that is what the eight accumulators are. One output channel per
// thread instead costs one weight and one activation per FMA.
//
// The bias is nullable and, when present, is the accumulator's initial value,
// matching `lg_conv3x3s1p1` and `lg_conv_kxk`. Passing null reproduces the
// bias-free kernels these were promoted from, bit for bit.
// ---------------------------------------------------------------------------

#define LG_OC 8   // output channels per thread

// out[oc][y][x] = bias[oc] + sum_{ky,kx,ci} w[oc][ci][ky][kx] * in[ci][2y+ky][2x+kx]
// Weight layout is the forward convolution's: [c_out][c_in][2][2].
// Order: ky, kx, ci. grid = (ceil((h/2)*(w/2)), ceil(c_out/8), 1), block = (256,1,1).
extern "C" __global__ void lg_conv2x2s2(
    const float *__restrict__ in, const float *__restrict__ wt,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int w)
{
    const int ow = w / 2, oh = h / 2;
    const long p = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= (long)oh * ow) return;
    const int x = (int)(p % ow);
    const int y = (int)(p / ow);
    const int oc0 = blockIdx.y * LG_OC;
    const size_t plane = (size_t)h * w;
    float acc[LG_OC];
#pragma unroll
    for (int k = 0; k < LG_OC; ++k)
        acc[k] = (bias != nullptr && oc0 + k < c_out) ? bias[oc0 + k] : 0.0f;

    if (oc0 + LG_OC <= c_out) {
        for (int ky = 0; ky < 2; ++ky) {
            for (int kx = 0; kx < 2; ++kx) {
                const size_t off = (size_t)(2 * y + ky) * w + (2 * x + kx);
                for (int ci = 0; ci < c_in; ++ci) {
                    const float xv = in[(size_t)ci * plane + off];
                    const float *wp = wt + ((size_t)oc0 * c_in + ci) * 4 + ky * 2 + kx;
#pragma unroll
                    for (int k = 0; k < LG_OC; ++k) acc[k] += wp[(size_t)k * c_in * 4] * xv;
                }
            }
        }
    } else {
        for (int ky = 0; ky < 2; ++ky) {
            for (int kx = 0; kx < 2; ++kx) {
                const size_t off = (size_t)(2 * y + ky) * w + (2 * x + kx);
                for (int ci = 0; ci < c_in; ++ci) {
                    const float xv = in[(size_t)ci * plane + off];
                    const float *wp = wt + ((size_t)oc0 * c_in + ci) * 4 + ky * 2 + kx;
#pragma unroll
                    for (int k = 0; k < LG_OC; ++k) {
                        if (oc0 + k < c_out) acc[k] += wp[(size_t)k * c_in * 4] * xv;
                    }
                }
            }
        }
    }
#pragma unroll
    for (int k = 0; k < LG_OC; ++k) {
        if (oc0 + k < c_out) out[((size_t)(oc0 + k) * oh + y) * ow + x] = acc[k];
    }
}

// out[oc][2y+ky][2x+kx] = bias[oc] + sum_ci w[ci][oc][ky][kx] * in[ci][y][x], NO
// tap flip. Weight layout is the transposed convolution's: [c_in][c_out][2][2] -
// the same four numbers as the forward form, arranged the other way round, so
// reading the wrong one is a wrong answer at every geometry rather than a
// rounding difference. Order: ci. grid = (ceil(4*h*w), ceil(c_out/8), 1),
// block = (256,1,1).
extern "C" __global__ void lg_conv_t2x2(
    const float *__restrict__ in, const float *__restrict__ wt,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int w)
{
    const int ow = 2 * w, oh = 2 * h;
    const long p = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= (long)oh * ow) return;
    const int ox = (int)(p % ow);
    const int oy = (int)(p / ow);
    const int iy = oy >> 1, ky = oy & 1;
    const int ix = ox >> 1, kx = ox & 1;
    const int oc0 = blockIdx.y * LG_OC;
    const size_t plane = (size_t)h * w;
    const size_t tap = (size_t)ky * 2 + kx;
    float acc[LG_OC];
#pragma unroll
    for (int k = 0; k < LG_OC; ++k)
        acc[k] = (bias != nullptr && oc0 + k < c_out) ? bias[oc0 + k] : 0.0f;

    if (oc0 + LG_OC <= c_out) {
        for (int ci = 0; ci < c_in; ++ci) {
            const float xv = in[(size_t)ci * plane + (size_t)iy * w + ix];
            const float *wp = wt + ((size_t)ci * c_out + oc0) * 4 + tap;
#pragma unroll
            for (int k = 0; k < LG_OC; ++k) acc[k] += wp[(size_t)k * 4] * xv;
        }
    } else {
        for (int ci = 0; ci < c_in; ++ci) {
            const float xv = in[(size_t)ci * plane + (size_t)iy * w + ix];
            const float *wp = wt + ((size_t)ci * c_out + oc0) * 4 + tap;
#pragma unroll
            for (int k = 0; k < LG_OC; ++k) {
                if (oc0 + k < c_out) acc[k] += wp[(size_t)k * 4] * xv;
            }
        }
    }
#pragma unroll
    for (int k = 0; k < LG_OC; ++k) {
        if (oc0 + k < c_out) out[((size_t)(oc0 + k) * oh + oy) * ow + ox] = acc[k];
    }
}

// ---------------------------------------------------------------------------
// Swin window assembly.
//
// `lg_window_gather` turns an NCHW plane into `[nw][n][c]` tokens, `n` = win*win,
// and `lg_window_scatter` inverts it at the same shift. The shift is a CYCLIC
// offset, folded into the plane index as a modulo wrap on both axes: the
// reference rolls the plane by -shift before windowing and by +shift after, so
// the sign is part of the contract rather than a convention.
//
// `w0` is the index of the chunk's first window. A caller that windows a whole
// plane at once passes 0; a caller that processes a bounded number of tokens at
// a time passes the chunk's base, because the index map has to keep counting
// from it. The token buffer is per chunk, `nw` windows wide.
// ---------------------------------------------------------------------------

__device__ void lg_window_index(int wl, int t, int nww, int win, int hp, int wp, int shift,
                                int *py, int *px)
{
    const int i = t / win;
    const int j = t % win;
    const int wh = wl / nww;
    const int ww = wl % nww;
    *py = (wh * win + i + shift) % hp;
    *px = (ww * win + j + shift) % wp;
}

// grid = (ceil(nw*n*c / 256), 1, 1), block = (256,1,1). `x` is NCHW, so its
// channel stride is hp*wp; `tok` is [nw][n][c] with c contiguous.
extern "C" __global__ void lg_window_gather(
    const float *__restrict__ x, float *__restrict__ tok,
    int nw, int n, int nww, int win, int hp, int wp, int c, int shift, int w0)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)nw * n * c) return;
    const int ch = (int)(idx % c);
    const long t2 = idx / c;
    const int t = (int)(t2 % n);
    const int wl = (int)(t2 / n);
    int y, xx;
    lg_window_index(w0 + wl, t, nww, win, hp, wp, shift, &y, &xx);
    tok[(size_t)(wl * n + t) * c + ch] = x[(size_t)ch * ((size_t)hp * wp) + (size_t)y * wp + xx];
}

// The same index map in the other direction: `lg_window_scatter(gather(p)) == p`
// for every element, at any shift.
extern "C" __global__ void lg_window_scatter(
    const float *__restrict__ tok, float *__restrict__ x,
    int nw, int n, int nww, int win, int hp, int wp, int c, int shift, int w0)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)nw * n * c) return;
    const int ch = (int)(idx % c);
    const long t2 = idx / c;
    const int t = (int)(t2 % n);
    const int wl = (int)(t2 / n);
    int y, xx;
    lg_window_index(w0 + wl, t, nww, win, hp, wp, shift, &y, &xx);
    x[(size_t)ch * ((size_t)hp * wp) + (size_t)y * wp + xx] = tok[(size_t)(wl * n + t) * c + ch];
}

// ---------------------------------------------------------------------------
// The PARAMETERISED TILE BODY: lg_conv3x3_q2 / _q2ng2 / _catq0 / _catq0ng2
// ---------------------------------------------------------------------------
//
// 3x3, stride 1, pad 1 - the SAME op family as `lg_conv3x3_tile`, promoted from
// hcflow-rs where it was written, measured and left. WHAT IT ADDS OVER
// `lg_conv3x3_tile` IS THREE THINGS, and the first is why it is a second entry
// point rather than a replacement:
//
//   1. THE TILE SHAPE IS A TEMPLATE PARAMETER. `lg_conv3x3_tile` is fixed at a
//      128-column tile (TBX 32 x TPX 4), 8 rows, CI 4 staged input channels and
//      OC 8 output channels per thread; this body takes all of them (TBX, TBY,
//      TPX, OCM, CIM) so a caller can retile for its own shapes without a second
//      copy of the algorithm. `lg_tile_body<32,8,4>` reproduced `lg_conv3x3_tile`'s
//      geometry exactly on the promoting engine.
//   2. NG OUTPUT-CHANNEL GROUPS PER BLOCK (default 1). Each group is a separate
//      CTA elsewhere in the family, so it costs its own copy of the staged input
//      tile; putting NG groups in one block as a threadIdx.z axis makes them
//      share one. It does not hide the staging latency, it amortizes it: only the
//      accumulator init, the weight fetch and the epilogue become NG-way and
//      `acc[OC][TPX]` per thread is unchanged. Measured 1.17-1.58x over NG=1 on
//      the promoting engine's shapes, which is why the NG=2 entries below exist.
//      THE BLOCK SHAPE DIFFERS (block = (TBX, TBY, NG)), so this is a separate
//      name and not a flag on `lg_conv3x3_tile`.
//   3. UP TO FIVE CONCATENATED SOURCE PLANES. `i1..i5` are extra input planes and
//      `g1..g5` the channel index at which each BEGINS, so the conv reads a
//      virtual plane whose channels are the concatenation. A dense block grows its
//      input by one plane per layer, and materialising that concatenation would be
//      a copy of the whole activation per layer; these folds it into the read.
//      With `ng <= 1` the body is a single plane and this is inert.
//
// THE ACCUMULATION ORDER IS (ci, ky, kx) WITH THE BIAS ADDED AFTER THE SUM,
// exactly as `lg_conv3x3_tile`, and NOT `lg_conv3x3s1p1`'s (ky, kx, ci) with the
// bias first. So the NG==1 entries are BIT-IDENTICAL to `lg_conv3x3_tile` at the
// same geometry (the promoting engine measured equality, not a tolerance), and
// their CPU twin below demands equality too.
//
// THE EXTRA POINTERS ARE AN ABI HAZARD. These entries take FOURTEEN arguments
// where `lg_conv3x3_tile` takes ten, and a launch that passes the wrong count is a
// wrong number rather than an error: a caller moving onto them must check the
// width (see cuda/CONVENTIONS.md on `int` versus `long`).
//
template <int TBX, int TBY, int TPX, int OCM = 8, int CIM = 4, int ZFIRST = 1,
          bool PREZERO = false, int ZSWAP = 0, int NG = 1>
__device__ __forceinline__ void lg_tile_body(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, int act, float act_p,
    const float *__restrict__ i1 = nullptr, const float *__restrict__ i2 = nullptr,
    const float *__restrict__ i3 = nullptr, const float *__restrict__ i4 = nullptr,
    const float *__restrict__ i5 = nullptr,
    int g1 = 0, int g2 = 0, int g3 = 0, int g4 = 0, int g5 = 0, int ng = 1)
{
    constexpr int TW = TBX * TPX;   // output columns per tile
    constexpr int TH = TBY;         // output rows per tile
    constexpr int CI = CIM;         // input channels per staged tile
    constexpr int OC = OCM;         // output channels per tile, accumulators per thread
    // Staged width, with the halo, rounded so that the ROW STRIDE does not put
    // two rows of a warp into the same banks. A warp covers two rows of the
    // staged tile (TBX = 16 threads to a row), and the row stride is SW floats,
    // so with SW = TW + 2 = 66 the second row sits 2 banks from the first and
    // the two 16-wide reads overlap in 14 banks - a two-way conflict on every
    // input load in the inner loop. A stride of 16 (mod 32) puts them in
    // complementary halves and removes it entirely. Costs a few hundred bytes
    // of shared memory per channel tile.
    constexpr int SW = ((TW + 2) % 32 >= 16) ? (TW + 2)
                                             : (TW + 2 + 16 - ((TW + 2) % 32));
    constexpr int SH = TH + 2;      // staged height, with the halo

    __shared__ float sh[CI][SH][SW];
    // [tap][ci][oc]: the eight `oc` weights of one (tap, ci) are contiguous, so
    // the inner loop reads them with two 16-byte loads instead of eight scalars.
    // The NG channel groups of this block sit SIDE BY SIDE along the last index,
    // group `z` owning [z*OC, z*OC + OC); with NG == 1 that is the original
    // layout exactly, so the NG == 1 instantiation is unchanged.
    __shared__ float sw[9][CI][NG * OC];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TBX + tx;
    // `tyz`/`tidz` fold the NG axis into the STAGING thread space: with NG = 2 the
    // block's 256 threads split the SAME rows between them (stride TBY * NG), so
    // the staging loop costs the block exactly what it cost at NG = 1 while
    // feeding NG times the FMAs. Without this the z-slabs would each stage the
    // whole tile - NG times the global traffic, which is the opposite of the
    // point. At NG == 1 both reduce to `ty` and `tid`.
    const int tyz = ty + (NG > 1 ? threadIdx.z : 0) * TBY;
    const int tidz = tyz * TBX + tx;
    // WHICH AXIS IS WHICH. CUDA launches blocks with `blockIdx.x` varying
    // FASTEST, and the blocks that share an input tile are exactly the ones
    // that differ only in their output-channel group. With the channels on
    // `z` (the default) those blocks are launched a whole plane apart and each
    // one re-reads the input from DRAM; with the channels on `x` they run
    // back to back and the second one's read hits L2. `ZFIRST` selects the
    // former (the layout this kernel shipped with), `CHANX` the latter.
    // THE THREE AXIS MAPS, all reachable through the same template.
    //   ZFIRST = 1                tile column on x, channel group on z (shipped)
    //   ZFIRST = 0, ZSWAP = 0     channel group on x, tile column on z
    //   ZFIRST = 0, ZSWAP = 1     BOTH on x: the tile column occupies the HIGH
    //                             bits of blockIdx.x and the channel group the
    //                             low ones, so `c_out/OC` consecutive x values
    //                             are the channel groups of ONE tile column.
    // The third is the interesting one at small plane sizes, where there are not
    // enough tiles to fill 20 SMs: it lets a kernel whose tile count is short
    // still spread its channel groups across the machine instead of running
    // `c_out/OC` of them per SM back to back. The launch must supply
    // `(GC * GW, GH, 1)` - see the launch geometry on each named entry below.
    const int GC_ = (c_out + OC - 1) / OC;
    const int bx = blockIdx.x;
    const int ox0 = (ZFIRST ? bx : (ZSWAP ? bx / GC_ : blockIdx.z)) * TW;
    const int oy0 = blockIdx.y * TH;
    // NG CHANNEL GROUPS PER BLOCK. Each group is a separate CTA elsewhere in
    // the family, so it costs its own copy of the STAGED INPUT TILE; putting
    // several groups in one block makes them share one, and that is the whole
    // point. The groups are spread over threadIdx.z (the block is
    // (TBX, TBY, NG)), and since NG does not enter the input tile at all, the
    // staging loop, its bounds predicates and the re-read are amortized over NG
    // times the arithmetic.
    //
    // WHY THIS AND NOT MORE OC. Raising `OC` cuts the re-read the same way but
    // costs acc[OC][TPX] and wv[OC] IN EVERY THREAD - `r16x`/`r32a-c` measure
    // it, and they are 9.0-11.5 ms against the 64-column tile's 6.16 at 64->64@512x512. NG
    // leaves regs/thread alone: only the acc initialisation, the `wv` fetch and
    // the epilogue become NG-way, and every thread's accumulator set stays
    // `acc[OC][TPX]`.
    //
    // WHY NOT dbuf/pipe/p1. Those try to HIDE the exposed load latency and all
    // lost: a second shared buffer is a tie end to end, register prefetch is
    // -30% (the 64-column tile is at its register cap), deferred stores -22% with a stack frame.
    // NG does not hide the latency, it AMORTIZES it - the same measurement that
    // condemns prefetching (72% of the kernel is the global load a shared store
    // waits on) is what endorses this one.
    //
    // MEASURED COST MODEL IT ATTACKS. One call stages GC * c_in * (TH+2)(TW+2) *
    // 4 bytes of input plus 9 * c_in * c_out * 4 * tiles of weights, the second
    // term independent of GC. At 64->64@512x512 that is 692.1 + 75.5 = 767.6 MB
    // for 19.33 GFLOP = 24.0 FLOP per staged byte, and 24.0 x the ~137 GB/s the
    // kernel actually achieves inside itself is 3288 GFLOP/s against a measured
    // 3162 - the model reproduces the ceiling, so the ceiling is the staged
    // traffic. NG=2 gives 43.7, NG=4 74.2 FLOP/byte; reaching the 8938 GFLOP/s
    // FMA ceiling at NG=1 would need 372 GB/s, above the card's DRAM peak.
    const int z = NG > 1 ? threadIdx.z : 0;
    const int oc0 = ((ZFIRST ? blockIdx.z : (ZSWAP ? bx % GC_ : bx)) * NG + z) * OC;
    const int gx0 = ox0 - 1, gy0 = oy0 - 1;

    float acc[OC][TPX];
#pragma unroll
    for (int o = 0; o < OC; ++o)
#pragma unroll
        for (int p = 0; p < TPX; ++p) acc[o][p] = 0.0f;

    for (int ci0 = 0; ci0 < c_in; ci0 += CI) {
        if (PREZERO) {
            // ONE write of the whole tile, then the staging loop only writes the
            // elements it actually has a value for. Every element of `sh` is
            // written exactly once per channel tile either way, so this is the
            // same instruction count for the tile as a whole with the bounds
            // predicates moved out of the per-element path - which is the
            // per-channel-tile overhead the staging probe showed dominates the
            // kernel (measured on the promoting engine: 8-18% of the time is arithmetic).
            float *shf = &sh[0][0][0];
            for (int i = tidz; i < CI * SH * SW; i += TBX * TBY * NG) shf[i] = 0.0f;
        }
        // Staged WITHOUT a flat index and therefore without an integer
        // division: `i / SW` is a multiply-shift sequence per staged element,
        // and there are CI of them per channel tile. The nested form walks the
        // rows and columns directly instead.
#pragma unroll
        for (int k = 0; k < CI; ++k) {
            const int ci = ci0 + k;
            const bool ci_ok = ci < c_in;
            for (int sy = tyz; sy < SH; sy += TBY * NG) {
                const int gy = gy0 + sy;
                const bool y_ok = ci_ok && gy >= 0 && gy < h;
                // WHICH PLANE this channel lives in, and at what offset within
                // it. With `ng == 1` the input is a single plane and this
                // reduces to the original expression. With more, the input is a
                // CONCATENATION of `ng` planes along the channel axis - the
                // dense block's growing input `cat([x, h1, h2, ...])` - and the
                // concatenated channel index is mapped to a (plane, channel
                // within plane) pair through the group starts. The comparison
                // chain runs once per staged element, its operand is the channel
                // loop index so it is uniform across a warp, and the kernel
                // reads the same bytes the concatenated copy would have held -
                // it just skips the copy (15 of them per dense block, 11% of the
                // LR64 run and 12% of the LR256 one).
                const int ci_u = ci_ok ? ci : 0;
                int base = 0;
                const float *plane = in;
                if (ng > 1 && ci_u >= g1) { base = g1; plane = i1; }
                if (ng > 2 && ci_u >= g2) { base = g2; plane = i2; }
                if (ng > 3 && ci_u >= g3) { base = g3; plane = i3; }
                if (ng > 4 && ci_u >= g4) { base = g4; plane = i4; }
                if (ng > 5 && ci_u >= g5) { base = g5; plane = i5; }
                const float *src = plane + ((size_t)(ci_u - base) * h + (gy < 0 ? 0 : gy)) * wd;
                if (PREZERO) {
                    // The whole tile was zeroed before this loop, so an element
                    // that is out of the plane is left holding that zero instead
                    // of being written with one - and the ROW predicate, which is
                    // uniform across the warp (every thread of a row shares `gy`),
                    // becomes a branch around the column loop rather than a test
                    // inside it. What remains per element is the column check
                    // alone.
                    //
                    // The row branch is also what keeps the halo at zero: rows 0
                    // and SH-1 are the halo rows and their `gy` is outside the
                    // plane for a full tile, so they take the branch and keep the
                    // zeroed value. An element at an IN-PLANE halo row (a tile
                    // that starts at row 0 has its row 0 halo row also in plane)
                    // is read back through the column check instead.
                    if (!ci_ok || gy < 0 || gy >= h) {
                        // nothing to load; the zeros stay
                    } else {
                        for (int sx = tx; sx < SW; sx += TBX) {
                            const int gx = gx0 + sx;
                            if (gx >= 0 && gx < wd) sh[k][sy][sx] = src[gx];
                        }
                    }
                } else {
                    for (int sx = tx; sx < SW; sx += TBX) {
                        const int gx = gx0 + sx;
                        float v = 0.0f;
                        if (y_ok && gx >= 0 && gx < wd) v = src[gx];
                        sh[k][sy][sx] = v;
                    }
                }
            }
        }
        // The staged weight tile covers the NG groups' weights for THIS channel
        // tile, so its total size is 9 * CI * OC * NG - but each `z` slab owns a
        // DIFFERENT range of it (`sw[..][..][z*OC + o]`), so the split has to be
        // PER SLAB and not across the whole block: this loop's range is one
        // group's 9 * CI * OC entries and its stride is one block's worth of
        // threads. Splitting it as `tidz .. 9*CI*OC step TBX*TBY*NG` (the shape
        // the input staging correctly uses, where the staged data IS shared)
        // leaves the upper z slabs' weights unwritten - which is a WRONG PLANE,
        // not a slow one, and is exactly how this was first written and caught by
        // `--ng-test`'s NG=2 correctness row at 1.04e1 relative. The block's
        // weight-staging work therefore still grows with NG (it must - the tile
        // is NG times bigger), while the input staging does not (it must not -
        // the tile is shared).
        for (int wi = tid; wi < 9 * CI * OC; wi += TBX * TBY) {
            const int tap = wi / (CI * OC);
            const int k = (wi / OC) % CI;
            const int o = wi % OC;
            const int ci = ci0 + k;
            const int oc = oc0 + o;
            float v = 0.0f;
            if (ci < c_in && oc < c_out) v = w[((size_t)oc * c_in + ci) * 9 + tap];
            sw[tap][k][z * OC + o] = v;
        }
        __syncthreads();

        // Per (ci, tap) the TPX input values are loaded once and reused for
        // every output channel in the tile; the OC weights are loaded as OC/4
        // float4s and reused for every pixel. Raising OC raises the reuse of
        // both, at the cost of accumulators: OC * TPX floats in registers.
        //
        // With NG groups in the block, the input tile is SHARED and the weight
        // slice is not: `z` picks its group's OC weights out of the row, which is
        // still one contiguous, 16-byte-aligned run so the float4 loads below are
        // unchanged.
#pragma unroll
        for (int k = 0; k < CI; ++k)
#pragma unroll
            for (int ky = 0; ky < 3; ++ky)
#pragma unroll
                for (int kx = 0; kx < 3; ++kx) {
                    const int tap = ky * 3 + kx;
                    float v[TPX];
#pragma unroll
                    for (int p = 0; p < TPX; ++p)
                        v[p] = sh[k][ty + ky][tx + p * TBX + kx];
                    const float4 *wp =
                        reinterpret_cast<const float4 *>(&sw[tap][k][z * OC]);
                    float wv[OC];
#pragma unroll
                    for (int o4 = 0; o4 < OC / 4; ++o4) {
                        const float4 wq = wp[o4];
                        wv[o4 * 4 + 0] = wq.x;
                        wv[o4 * 4 + 1] = wq.y;
                        wv[o4 * 4 + 2] = wq.z;
                        wv[o4 * 4 + 3] = wq.w;
                    }
#pragma unroll
                    for (int o = 0; o < OC; ++o)
#pragma unroll
                        for (int p = 0; p < TPX; ++p) acc[o][p] += wv[o] * v[p];
                }
        __syncthreads();
    }

    const int my = oy0 + ty;
#pragma unroll
    for (int o = 0; o < OC; ++o) {
        const int oc = oc0 + o;
        if (oc < c_out && my < h) {
            const float b = bias ? bias[oc] : 0.0f;
#pragma unroll
            for (int p = 0; p < TPX; ++p) {
                const int gx = ox0 + tx + p * TBX;
                if (gx < wd) {
                    float v = acc[o][p] + b;
                    if (act == 1) v = v > 0.0f ? v : 0.0f;
                    else if (act == 2) v = v >= 0.0f ? v : act_p * v;
                    out[((size_t)oc * h + my) * wd + gx] = v;
                }
            }
        }
    }
}

// The four named instantiations the promoting engine's graph dispatches, so a
// consumer can call them without being able to instantiate the template itself
// (a separate translation unit per fatbin, and `--entries` prunes an unnamed
// global). `q2`'s tile is 64 output columns x 8 rows with 2 staged input
// channels, 8 blocks per SM; `catq0` is the same tile with the concat inputs.
//
// LAUNCH GEOMETRY, which is part of the contract because each template reads
// blockIdx differently:
//   q2 / catq0        grid = (ceil(wd/64), ceil(h/8), ceil(c_out/8)),
//                     block = (16, 8, 1), channels on blockIdx.z
//   q2ng2 / catq0ng2  grid = (ceil(ceil(c_out/8)/NG), ceil(h/8), ceil(wd/64)),
//                     block = (16, 8, NG) - channels on blockIdx.x, NG on z.
//                     NG must divide ceil(c_out/8), or part of the last group
//                     goes unwritten.
extern "C" __global__ void __launch_bounds__(16 * 8, 8)
lg_conv3x3_q2(const float *__restrict__ in, const float *__restrict__ w,
              const float *__restrict__ bias, float *__restrict__ out,
              int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<16, 8, 4, 8, 2>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

extern "C" __global__ void __launch_bounds__(16 * 8 * 2, 4)
lg_conv3x3_q2ng2(const float *__restrict__ in, const float *__restrict__ w,
                 const float *__restrict__ bias, float *__restrict__ out,
                 int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<16, 8, 4, 8, 2, 0, false, 0, 2>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

extern "C" __global__ void __launch_bounds__(16 * 8, 8)
lg_conv3x3_catq0(const float *__restrict__ in, const float *__restrict__ w,
                 const float *__restrict__ bias, float *__restrict__ out,
                 int c_in, int c_out, int h, int wd, int act, float act_p,
                 const float *__restrict__ i1, const float *__restrict__ i2,
                 const float *__restrict__ i3, const float *__restrict__ i4,
                 const float *__restrict__ i5,
                 int g1, int g2, int g3, int g4, int g5, int ng)
{
    lg_tile_body<16, 8, 4, 8, 2, 0>(in, w, bias, out, c_in, c_out, h, wd, act, act_p,
                                    i1, i2, i3, i4, i5, g1, g2, g3, g4, g5, ng);
}

extern "C" __global__ void __launch_bounds__(16 * 8 * 2, 4)
lg_conv3x3_catq0ng2(const float *__restrict__ in, const float *__restrict__ w,
                    const float *__restrict__ bias, float *__restrict__ out,
                    int c_in, int c_out, int h, int wd, int act, float act_p,
                    const float *__restrict__ i1, const float *__restrict__ i2,
                    const float *__restrict__ i3, const float *__restrict__ i4,
                    const float *__restrict__ i5,
                    int g1, int g2, int g3, int g4, int g5, int ng)
{
    lg_tile_body<16, 8, 4, 8, 2, 0, false, 0, 2>(in, w, bias, out, c_in, c_out, h, wd, act, act_p,
                                                 i1, i2, i3, i4, i5, g1, g2, g3, g4, g5, ng);
}
