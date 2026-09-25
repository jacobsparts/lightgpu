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

// ===========================================================================
// 9. Linear layer in the token layout (C, 1, h*w)
//    (y, x, c) at c*h*w + y*w + x)
// ===========================================================================


// Linear on the row-major token layout: out[i*C_out + o] =
// bias[o] + sum_c x[i*C_in + c] * w[o*C_in + c], accumulated in c order.
// Tiled 16x16 with the shared-memory trick that makes the weight tile
// readable as ws[tx][k] against xs[ty][k].
extern "C" __global__ void lg_linear(
    const float *__restrict__ x, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int rows, int c_in, int c_out)
{
    __shared__ float xs[16][17];
    __shared__ float ws[16][17];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int r0 = blockIdx.y * 16;
    const int o0 = blockIdx.x * 16;
    const int r = r0 + ty;
    const int o = o0 + tx;
    float acc = 0.0f;
    for (int c0 = 0; c0 < c_in; c0 += 16) {
        const int cx = c0 + tx;
        xs[ty][tx] = (r < rows && cx < c_in) ? x[(size_t)r * c_in + cx] : 0.0f;
        const int wr = o0 + ty;
        const int wc = c0 + tx;
        ws[ty][tx] = (wr < c_out && wc < c_in) ? w[(size_t)wr * c_in + wc] : 0.0f;
        __syncthreads();
        for (int k = 0; k < 16; ++k) acc += xs[ty][k] * ws[tx][k];
        __syncthreads();
    }
    if (r < rows && o < c_out) out[(size_t)r * c_out + o] = acc + (bias ? bias[o] : 0.0f);
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

// f32 GEMM, v8-style tiling: 64 rows x 32 columns per 256-thread block (32 row
// lanes x 8 column slots, 2 rows x 4 columns per thread = 8 accumulators). The
// 32 columns' current k-step (one float4 each) is staged in shared memory so
// the activation float4 is loaded once per block-k-step instead of once per
// row lane. No k-split: the tiles are numerous enough.
// grid = (ceil(ne1/64), ceil(ncols/32)). Requires ne0 % 4 == 0.
extern "C" __global__ void lg_f32_gemm_tiled(
    const float *__restrict__ w, const float *__restrict__ x, float *__restrict__ y,
    int ne0, int ne1, int ncols)
{
    __shared__ float xs[32 * 4];
    const int tid = threadIdx.x;
    const int jl  = tid & 31;
    const int cs  = tid >> 5;
    const int j0  = blockIdx.x * 64;
    const int c0  = blockIdx.y * 32;

    const int r0 = j0 + jl;
    const int r1 = r0 + 32;
    const int cA = c0 + cs * 4;
    const bool ur0 = r0 < ne1;
    const bool ur1 = r1 < ne1;
    const bool uc0 = cA + 0 < ncols, uc1 = cA + 1 < ncols;
    const bool uc2 = cA + 2 < ncols, uc3 = cA + 3 < ncols;

    const float *w0 = w + (size_t)r0 * ne0;
    const float *w1 = w + (size_t)r1 * ne0;

    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a03 = 0.f;
    float a10 = 0.f, a11 = 0.f, a12 = 0.f, a13 = 0.f;

    for (int k = 0; k + 4 <= ne0; k += 4) {
        if (tid < 32) {
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (c0 + tid < ncols) {
                v = *reinterpret_cast<const float4 *>(x + (size_t)(c0 + tid) * ne0 + k);
            }
            xs[tid * 4 + 0] = v.x; xs[tid * 4 + 1] = v.y;
            xs[tid * 4 + 2] = v.z; xs[tid * 4 + 3] = v.w;
        }
        __syncthreads();
        const float4 xv0 = *reinterpret_cast<const float4 *>(&xs[(cs * 4 + 0) * 4]);
        const float4 xv1 = *reinterpret_cast<const float4 *>(&xs[(cs * 4 + 1) * 4]);
        const float4 xv2 = *reinterpret_cast<const float4 *>(&xs[(cs * 4 + 2) * 4]);
        const float4 xv3 = *reinterpret_cast<const float4 *>(&xs[(cs * 4 + 3) * 4]);
        if (ur0) {
            const float4 wv = *reinterpret_cast<const float4 *>(w0 + k);
            if (uc0) { a00 = fmaf(wv.x, xv0.x, a00); a00 = fmaf(wv.y, xv0.y, a00); a00 = fmaf(wv.z, xv0.z, a00); a00 = fmaf(wv.w, xv0.w, a00); }
            if (uc1) { a01 = fmaf(wv.x, xv1.x, a01); a01 = fmaf(wv.y, xv1.y, a01); a01 = fmaf(wv.z, xv1.z, a01); a01 = fmaf(wv.w, xv1.w, a01); }
            if (uc2) { a02 = fmaf(wv.x, xv2.x, a02); a02 = fmaf(wv.y, xv2.y, a02); a02 = fmaf(wv.z, xv2.z, a02); a02 = fmaf(wv.w, xv2.w, a02); }
            if (uc3) { a03 = fmaf(wv.x, xv3.x, a03); a03 = fmaf(wv.y, xv3.y, a03); a03 = fmaf(wv.z, xv3.z, a03); a03 = fmaf(wv.w, xv3.w, a03); }
        }
        if (ur1) {
            const float4 wv = *reinterpret_cast<const float4 *>(w1 + k);
            if (uc0) { a10 = fmaf(wv.x, xv0.x, a10); a10 = fmaf(wv.y, xv0.y, a10); a10 = fmaf(wv.z, xv0.z, a10); a10 = fmaf(wv.w, xv0.w, a10); }
            if (uc1) { a11 = fmaf(wv.x, xv1.x, a11); a11 = fmaf(wv.y, xv1.y, a11); a11 = fmaf(wv.z, xv1.z, a11); a11 = fmaf(wv.w, xv1.w, a11); }
            if (uc2) { a12 = fmaf(wv.x, xv2.x, a12); a12 = fmaf(wv.y, xv2.y, a12); a12 = fmaf(wv.z, xv2.z, a12); a12 = fmaf(wv.w, xv2.w, a12); }
            if (uc3) { a13 = fmaf(wv.x, xv3.x, a13); a13 = fmaf(wv.y, xv3.y, a13); a13 = fmaf(wv.z, xv3.z, a13); a13 = fmaf(wv.w, xv3.w, a13); }
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
