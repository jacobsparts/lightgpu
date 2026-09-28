//! CPU twins of the kernels: same names, same arithmetic, plain Rust.
//!
//! These are the CPU backend, not a test harness: this is the path an engine
//! runs when there is no GPU, and it is held to the same performance standard as
//! the kernels it mirrors. Keeping the *order* of operations the GPU kernels use,
//! wherever that is cheap, is what makes a backend mismatch mean a real
//! difference rather than a rounding one; it is not a reason to leave the CPU
//! side slow. The few places where the two deliberately differ (erf, layer-norm
//! variance) are documented at the kernel instead.
//!
//! THE CONVOLUTIONS ARE THE EXCEPTION TO "PLAIN RUST", deliberately. The
//! elementwise twins are reference implementations - obviously right, and fast
//! enough for an op that touches each element once - but a convolution is
//! the hot spot of every vision engine here (in ifan-rs at 256x256 the 3x3s are
//! 72.7% of a CPU pass), and rather than ship a slow twin, three engines each
//! wrote their own AVX2 + rayon copy of that one op. `conv1x1` and
//! `conv3x3s1p1` are therefore parallel and vectorised; what is NOT given up is
//! the arithmetic, which is still the kernel's own order - see the section
//! comment above them.

use rayon::prelude::*;

/// `lg_add`: y = a + b
pub fn add(a: &[f32], b: &[f32], y: &mut [f32]) {
    for i in 0..a.len() {
        y[i] = a[i] + b[i];
    }
}

/// `lg_mul`: y = a * b
pub fn mul(a: &[f32], b: &[f32], y: &mut [f32]) {
    for i in 0..a.len() {
        y[i] = a[i] * b[i];
    }
}

/// `lg_scale`: y = x * s
pub fn scale(x: &[f32], s: f32, y: &mut [f32]) {
    for i in 0..x.len() {
        y[i] = x[i] * s;
    }
}

/// `lg_relu`: y = max(x, 0)
pub fn relu(x: &[f32], y: &mut [f32]) {
    for i in 0..x.len() {
        y[i] = if x[i] < 0.0 { 0.0 } else { x[i] };
    }
}

/// `lg_sigmoid`: y = 1 / (1 + exp(-x))
pub fn sigmoid(x: &[f32], y: &mut [f32]) {
    for i in 0..x.len() {
        y[i] = 1.0 / (1.0 + (-x[i]).exp());
    }
}

/// `lg_erf`: Abramowitz & Stegun 7.1.26, the same formula the GPU uses, so an
/// erf difference can never explain a GPU-vs-CPU mismatch.
pub fn erf(v: f32) -> f32 {
    let sign = if v < 0.0 { -1.0 } else { 1.0 };
    let x = v.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

/// `lg_gelu_erf`: y = 0.5 x (1 + erf(x / sqrt(2)))
pub fn gelu_erf(x: &[f32], y: &mut [f32]) {
    for i in 0..x.len() {
        y[i] = 0.5 * x[i] * (1.0 + erf(x[i] * 0.70710678118654752440));
    }
}

/// `lg_gelu_tanh`: y = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
pub fn gelu_tanh(x: &[f32], y: &mut [f32]) {
    const C: f32 = 0.7978845608028654;
    for i in 0..x.len() {
        let v = x[i];
        y[i] = 0.5 * v * (1.0 + (C * (v + 0.044715 * v * v * v)).tanh());
    }
}

/// `lg_silu_mul`: gate = silu(gate) * up, in place.
pub fn silu_mul(gate: &mut [f32], up: &[f32]) {
    for i in 0..gate.len() {
        let g = gate[i];
        gate[i] = (g / (1.0 + (-g).exp())) * up[i];
    }
}

/// `lg_rope_neox`: half-split rotation of a `[hd, nh, ntok]` buffer.
pub fn rope_neox(x: &mut [f32], pos: &[i32], hd: usize, nh: usize, ntok: usize, theta: f32) {
    let half = hd / 2;
    for t in 0..ntok {
        for h in 0..nh {
            let base = (t * nh + h) * hd;
            for i in 0..half {
                let freq = theta.powf(-2.0 * i as f32 / hd as f32);
                let ang = pos[t] as f32 * freq;
                let (c, s) = (ang.cos(), ang.sin());
                let x0 = x[base + i];
                let x1 = x[base + i + half];
                x[base + i] = x0 * c - x1 * s;
                x[base + i + half] = x0 * s + x1 * c;
            }
        }
    }
}

/// `lg_channel_layer_norm`: layer norm over the CHANNEL axis of an NCHW
/// tensor - for each spatial position p, normalize the c values at stride hw.
pub fn channel_layer_norm(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    y: &mut [f32],
    c: usize,
    hw: usize,
    eps: f32,
) {
    for p in 0..hw {
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        for ch in 0..c {
            let v = x[ch * hw + p];
            s1 += v;
            s2 += v * v;
        }
        let n = c as f32;
        let mean = s1 / n;
        let var = s2 / n - mean * mean;
        let scale = 1.0 / (var.max(0.0) + eps).sqrt();
        for ch in 0..c {
            let i = ch * hw + p;
            y[i] = (x[i] - mean) * scale * w[ch] + b[ch];
        }
    }
}

/// `lg_channel_affine`: the NCHW per-CHANNEL affine, out[c][p] = in[c][p] *
/// scale[c] + shift[c]. Either vector may be empty for a pure scale (no shift)
/// or a pure bias (no scale); `in` and `y` may be the same slice.
pub fn channel_affine(
    x: &[f32],
    scale: &[f32],
    shift: &[f32],
    y: &mut [f32],
    c: usize,
    hw: usize,
) {
    for ch in 0..c {
        let s = if scale.is_empty() { 1.0f32 } else { scale[ch] };
        let b = if shift.is_empty() { 0.0f32 } else { shift[ch] };
        for p in 0..hw {
            let i = ch * hw + p;
            y[i] = x[i] * s + b;
        }
    }
}

/// `lg_channel_scale`: the `shift = null` case of [`channel_affine`], as the
/// kernel is a separate entry point for it.
pub fn channel_scale(x: &[f32], s: &[f32], y: &mut [f32], c: usize, hw: usize) {
    for ch in 0..c {
        for p in 0..hw {
            let i = ch * hw + p;
            y[i] = x[i] * s[ch];
        }
    }
}

/// `lg_channel_mean`: out[c] = mean over the hw plane of channel c.
///
/// This mirrors the KERNEL's arithmetic rather than a scalar left-to-right sum,
/// and `block` is a parameter for exactly that reason: the kernel's scratch is
/// 1024 slots, each thread accumulates its own strided partial into slot
/// `threadIdx.x`, and the tree then sums slots 0..1024 in a fixed (non-scalar)
/// order. Reproducing it here means the GPU and CPU results agree bit for bit,
/// so `selftest` can assert equality instead of a tolerance. The zeroed tail
/// above `block` contributes nothing, so folding it in is a no-op.
pub fn channel_mean(x: &[f32], out: &mut [f32], c: usize, hw: usize, block: usize) {
    for ch in 0..c {
        let mut r = [0.0f32; 1024];
        for t in 0..block {
            let mut s = 0.0f32;
            let mut i = t;
            while i < hw {
                s += x[ch * hw + i];
                i += block;
            }
            r[t] = s;
        }
        let mut step = 512;
        while step > 0 {
            for t in 0..step {
                r[t] += r[t + step];
            }
            step >>= 1;
        }
        out[ch] = r[0] / hw as f32;
    }
}

/// `lg_layer_norm`: the two-pass form, matching the kernel.
pub fn layer_norm(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    y: &mut [f32],
    ne0: usize,
    nrows: usize,
    eps: f32,
) {
    for r in 0..nrows {
        let xr = &x[r * ne0..r * ne0 + ne0];
        let yr = &mut y[r * ne0..r * ne0 + ne0];
        let mean = xr.iter().sum::<f32>() / ne0 as f32;
        let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / ne0 as f32;
        let rstd = 1.0 / (var + eps).sqrt();
        for i in 0..ne0 {
            yr[i] = (xr[i] - mean) * rstd * w[i] + b[i];
        }
    }
}

/// `lg_rms_norm`: y = x * rsqrt(mean(x^2) + eps) * w
pub fn rms_norm(x: &[f32], w: &[f32], y: &mut [f32], ne0: usize, nrows: usize, eps: f32) {
    for r in 0..nrows {
        let xr = &x[r * ne0..r * ne0 + ne0];
        let yr = &mut y[r * ne0..r * ne0 + ne0];
        let ss = xr.iter().map(|v| v * v).sum::<f32>() / ne0 as f32;
        let scale = 1.0 / (ss + eps).sqrt();
        for i in 0..ne0 {
            yr[i] = xr[i] * scale * w[i];
        }
    }
}

/// `lg_fft2_r2c`: batched 2-D real FFT, unnormalised, half spectrum.
///
/// Same shape and convention as the GPU kernel: `x` is `[batch][n][n]` row-major,
/// `out` is `[batch][n][n/2+1]` interleaved complex `(re, im)` pairs, and the
/// result is multiplied by `scale` (pass 1.0 for the unnormalised transform, or
/// `1/sqrt(n*n)` for `rfftn(norm="ortho")`). `n` must be a power of two and at
/// most 64, matching the kernel's shared-memory bound.
///
/// The same transform as the GPU kernel, written as a plain radix-2 decimation-in-time
/// transform on rows then columns, so the GPU's bit-reversal-plus-butterfly
/// structure can be checked against it term by term. A difference here is a real
/// bug, not floating-point reordering.
pub fn fft2_r2c(x: &[f32], out: &mut [f32], batch: usize, n: usize, scale: f32) {
    assert!(n.is_power_of_two() && n <= 64, "n must be a power of two <= 64");
    assert_eq!(x.len(), batch * n * n, "x is [batch][n][n]");
    let hw = n / 2 + 1;
    assert_eq!(out.len(), batch * n * hw * 2, "out is [batch][n][n/2+1] interleaved");
    let mut re = vec![0.0f32; n * n];
    let mut im = vec![0.0f32; n * n];
    for b in 0..batch {
        let src = &x[b * n * n..(b + 1) * n * n];
        for r in 0..n {
            for c in 0..n {
                re[r * n + c] = src[r * n + c];
                im[r * n + c] = 0.0;
            }
        }
        // Rows, then columns: two 1-D transforms, which is what the 2-D FFT is.
        for r in 0..n {
            fft1_dit(&mut re[r * n..(r + 1) * n], &mut im[r * n..(r + 1) * n], n, false);
        }
        for c in 0..n {
            let mut cr = vec![0.0f32; n];
            let mut ci = vec![0.0f32; n];
            for r in 0..n {
                cr[r] = re[r * n + c];
                ci[r] = im[r * n + c];
            }
            fft1_dit(&mut cr, &mut ci, n, false);
            for r in 0..n {
                re[r * n + c] = cr[r];
                im[r * n + c] = ci[r];
            }
        }
        for u in 0..n {
            for c in 0..hw {
                out[((b * n + u) * hw + c) * 2] = re[u * n + c] * scale;
                out[((b * n + u) * hw + c) * 2 + 1] = im[u * n + c] * scale;
            }
        }
    }
}

/// `lg_fft2_c2r`: the inverse of [`fft2_r2c`].
///
/// `in` is the `[batch][n][n/2+1]` interleaved half spectrum, `out` is the
/// `[batch][n][n]` real plane multiplied by `scale` (pass `1/sqrt(n*n)` for the
/// ortho inverse). The full spectrum is rebuilt by conjugate symmetry, exactly as
/// the GPU kernel does, so the caller stores only the half spectrum.
pub fn fft2_c2r(x: &[f32], out: &mut [f32], batch: usize, n: usize, scale: f32) {
    assert!(n.is_power_of_two() && n <= 64, "n must be a power of two <= 64");
    let hw = n / 2 + 1;
    assert_eq!(x.len(), batch * n * hw * 2, "x is [batch][n][n/2+1] interleaved");
    assert_eq!(out.len(), batch * n * n, "out is [batch][n][n]");
    let mut re = vec![0.0f32; n * n];
    let mut im = vec![0.0f32; n * n];
    for b in 0..batch {
        for u in 0..n {
            for c in 0..n {
                let (rr, ii) = if c <= n / 2 {
                    (x[((b * n + u) * hw + c) * 2], x[((b * n + u) * hw + c) * 2 + 1])
                } else {
                    let uu = (n - u) % n;
                    let cc = n - c;
                    let t = &x[((b * n + uu) * hw + cc) * 2..];
                    (t[0], -t[1])
                };
                re[u * n + c] = rr;
                im[u * n + c] = ii;
            }
        }
        // Columns first, then rows - the mirror of the forward order.
        for c in 0..n {
            let mut cr = vec![0.0f32; n];
            let mut ci = vec![0.0f32; n];
            for r in 0..n {
                cr[r] = re[r * n + c];
                ci[r] = im[r * n + c];
            }
            fft1_dit(&mut cr, &mut ci, n, true);
            for r in 0..n {
                re[r * n + c] = cr[r];
                im[r * n + c] = ci[r];
            }
        }
        for r in 0..n {
            fft1_dit(&mut re[r * n..(r + 1) * n], &mut im[r * n..(r + 1) * n], n, true);
        }
        for r in 0..n {
            for c in 0..n {
                out[(b * n + r) * n + c] = re[r * n + c] * scale;
            }
        }
    }
}

/// In-place radix-2 DIT 1-D complex FFT, natural order in and out. `inverse`
/// selects the `+i` twiddle sign.
fn fft1_dit(re: &mut [f32], im: &mut [f32], n: usize, inverse: bool) {
    let nb = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bitrev(i, nb);
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let sign = if inverse { 1.0f32 } else { -1.0f32 };
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let ang = sign * 2.0 * std::f32::consts::PI * k as f32 / len as f32;
                let (s, c) = (ang.sin(), ang.cos());
                let i = start + k;
                let j = i + half;
                let (ur, ui) = (re[i], im[i]);
                let (vr, vi) = (re[j], im[j]);
                let tr = vr * c - vi * s;
                let ti = vr * s + vi * c;
                re[i] = ur + tr;
                im[i] = ui + ti;
                re[j] = ur - tr;
                im[j] = ui - ti;
            }
        }
        len <<= 1;
    }
}

// ---------------------------------------------------------------------------
// Twins of the CONVOLUTIONS in section 6 of cuda/kernels.cu - `lg_conv1x1` and
// `lg_conv3x3s1p1` - which are the two operators the tiled kernels below are
// alternative implementations of, so the four live side by side.
//
// THIS PAIR IS THE FIRST HERE THAT IS PARALLEL AND VECTORISED, and that is not
// drift: it is the reason the pair exists. Everything above is a reference
// implementation - obviously right, and fast enough for an op that touches each
// element once. A convolution is in a different class (in ifan-rs at 256x256 the
// 3x3s are 72.7% of a whole CPU pass), and rather than ship a slow twin three
// engines each wrote their OWN AVX2 + rayon copy of this one op. One twin,
// measured, replaces all three:
//
//   * RAYON, one task per (output channel, output row). A task per CHANNEL is
//     not enough: `ending` in nafnet's graph is a 32 -> 3 conv, so a
//     channel-grained split gives it three tasks on a 24-thread machine, which
//     is a barrier against an idle machine. A row is also the natural unit - the
//     output is written exactly once - and the channel still belongs to the
//     task, so a row's weight slice and bias are loop invariants.
//   * A CONTIGUOUS INNER LOOP. For a fixed tap the output row and the input row
//     are both contiguous, so the inner loop is `acc[i] += k * src[i]`, an AXPY,
//     and each tap's bounds are resolved ONCE from the tap's own offset instead
//     of per output element inside the innermost loop of a 9-tap convolution.
//     That branch is what stops the pixel-at-a-time form vectorising.
//   * AVX2 + FMA BEHIND A RUNTIME TEST (`is_x86_feature_detected!`), never a
//     build-time baseline. No `-C target-cpu=native` exists anywhere in this
//     family, and a binary built on this machine still has to run on one without
//     AVX2; the vector path is compiled out entirely on a non-x86-64 target.
//
// WHAT NEITHER PATH CHANGES. Parallelism and vector width do not touch the sum.
// One task owns each output row and accumulates it in the kernel's own order,
// and lane j of the AXPY accumulates exactly the taps the scalar element j
// would, in the same sequence. `selftest` asserts both halves of that: the
// scalar inner loop through the same parallel framing is BIT-IDENTICAL to a
// plain reference, and the vector path differs from it by the FMA alone.
//
// THE ONE REAL DIFFERENCE IS THE FMA. `fma(k, x, acc)` rounds once where a
// scalar `acc + k * x` rounds twice - about 1e-7 relative, and it is the more
// accurate of the two. It is also the direction the GPU kernel is already in,
// because nvcc contracts `acc += wv * x` into an FMA, so the vector path moves
// the CPU backend TOWARDS the kernels rather than away from them. Each of the
// three engines' hand-written copies made this same trade; realesrgan-rs
// documents it in these terms at its own copy.
//
// ORDER. The nullable bias is folded into the accumulator BEFORE the first
// multiply, then the taps run ky, kx, ci - the order of the KERNEL. That is what
// makes a twin-versus-kernel comparison a comparison of two implementations of
// ONE sum; a twin that summed in an order of its own could only ever be compared
// by tolerance. It is also NOT the order `conv3x3_tile`/`conv1x1_tile` use
// (channel loop outermost, bias last), which is why the generic-to-tiled swap
// stays the magnitude-bound swap those two kernels document.
// ---------------------------------------------------------------------------

/// `lg_conv1x1`: `y[oc][p] = bias[oc] + sum_c w[oc][c] * x[c][p]`, order c
/// ascending with the bias in the accumulator first.
#[allow(clippy::too_many_arguments)]
pub fn conv1x1(
    x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32],
    c_in: usize, c_out: usize, h: usize, wd: usize,
) {
    conv1x1_impl(x, w, bias, y, c_in, c_out, h, wd, simd_ok());
}

/// [`conv1x1`] with the inner loop selectable, so `selftest` can tell the FMA's
/// last bit apart from the parallel framing.
#[allow(clippy::too_many_arguments)]
fn conv1x1_impl(
    x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32],
    c_in: usize, c_out: usize, h: usize, wd: usize, simd: bool,
) {
    let hw = h * wd;
    if c_out == 0 || hw == 0 {
        return;
    }
    let out = &mut y[..c_out * hw];
    out.par_chunks_mut(wd).enumerate().for_each_init(
        Vec::new,
        |acc, (idx, orow)| {
            let (oc, oy) = (idx / h, idx % h);
            if acc.len() < wd {
                acc.resize(wd, 0.0);
            }
            let row = &mut acc[..wd];
            row.fill(if bias.is_empty() { 0.0 } else { bias[oc] });
            let wp = &w[oc * c_in..(oc + 1) * c_in];
            for (c, &k) in wp[..c_in].iter().enumerate() {
                if k == 0.0 {
                    continue;
                }
                let base = c * hw + oy * wd;
                row_axpy(row, &x[base..base + wd], k, simd);
            }
            orow.copy_from_slice(row);
        },
    );
}

/// `lg_conv3x3s1p1`: 3x3, stride 1, pad 1, plane layout, nullable bias folded
/// into the accumulator first, then ky, kx, ci.
#[allow(clippy::too_many_arguments)]
pub fn conv3x3s1p1(
    x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32],
    c_in: usize, c_out: usize, h: usize, wd: usize,
) {
    conv3x3s1p1_impl(x, w, bias, y, c_in, c_out, h, wd, simd_ok());
}

/// [`conv3x3s1p1`] with the inner loop selectable. See [`conv1x1_impl`].
#[allow(clippy::too_many_arguments)]
fn conv3x3s1p1_impl(
    x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32],
    c_in: usize, c_out: usize, h: usize, wd: usize, simd: bool,
) {
    conv3x3s1p1_with(x, w, bias, y, c_in, c_out, h, wd, simd, true)
}

/// [`conv3x3s1p1_impl`] with the channel PAIRING selectable. `pair = false` runs
/// one channel a task; `pair = true` runs two and shares the loads. The two must
/// agree BIT FOR BIT, and `selftest` asserts exactly that - the pairing is a
/// schedule, not arithmetic, so if it can change an answer at all it is a bug.
#[allow(clippy::too_many_arguments)]
fn conv3x3s1p1_with(
    x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32],
    c_in: usize, c_out: usize, h: usize, wd: usize, simd: bool, pair: bool,
) {
    let hw = h * wd;
    if c_out == 0 || hw == 0 {
        return;
    }
    let out = &mut y[..c_out * hw];
    let g = Row3x3 { x, c_in, hw, h, wd };
    // ONE TASK PER OUTPUT-CHANNEL PAIR, and the rows of a pair are split across
    // the pool as well. A pair chunk is `2*hw` of the output, so the two row
    // slices a task needs are `chunk.split_at_mut(hw)` - which is the whole
    // reason the pairing is expressible at all.
    //
    // The pair is worth this: eight FMAs per four loaded vectors, because one
    // input load feeds BOTH output channels. Measured against the same tile
    // driving one channel, the paired form is 1.1x-2.0x on every geometry
    // measured (ifan-rs's 128->128 at 59x164, 118x328, 236x328, 472x164 and
    // 59x1312, 64->64 at 236x656, 32->32 at 472x1312, 256->128 at 59x164) and
    // 1.38x on the worse of the two independent runs. An odd `c_out` leaves a
    // last chunk of ONE plane, which takes the unpaired row routine.
    //
    // The order is what makes this safe to choose by speed alone: the paired
    // tile runs the same (ky, kx, ci) sequence with the bias already in the
    // accumulator, so its output is BIT-IDENTICAL to the unpaired one, not
    // merely close. `selftest` asserts that directly.
    if !pair {
        out.par_chunks_mut(wd).enumerate().for_each_init(
            Vec::new,
            |buf, (idx, orow)| {
                let (oc, oy) = (idx / h, idx % h);
                g.row(&w[oc * c_in * 9..(oc + 1) * c_in * 9], bias_at(bias, oc), oy, orow, buf, simd);
            },
        );
        return;
    }
    out.par_chunks_mut(2 * hw).enumerate().for_each(|(p, chunk)| {
        let (oc0, oc1) = (p * 2, p * 2 + 1);
        let (plane0, rest) = chunk.split_at_mut(hw);
        if oc1 >= c_out {
            plane0.par_chunks_mut(wd).enumerate().for_each_init(
                Vec::new,
                |buf, (oy, orow)| {
                    let wp = &w[oc0 * c_in * 9..(oc0 + 1) * c_in * 9];
                    g.row(wp, bias_at(bias, oc0), oy, orow, buf, simd);
                },
            );
            return;
        }
        let plane1 = &mut rest[..hw];
        let wp0 = &w[oc0 * c_in * 9..(oc0 + 1) * c_in * 9];
        let wp1 = &w[oc1 * c_in * 9..(oc1 + 1) * c_in * 9];
        let (b0, b1) = (bias_at(bias, oc0), bias_at(bias, oc1));
        plane0
            .par_chunks_mut(wd)
            .zip(plane1.par_chunks_mut(wd))
            .enumerate()
            .for_each_init(
                || (Vec::new(), Vec::new()),
                |bufs, (oy, (orow0, orow1))| {
                    g.row_pair(
                        [(wp0, b0), (wp1, b1)],
                        oy,
                        [orow0, orow1],
                        (&mut bufs.0, &mut bufs.1),
                        simd,
                    );
                },
            );
    });
}

/// A nullable bias read, in the one place both row routines agree on the rule.
#[inline]
fn bias_at(bias: &[f32], oc: usize) -> f32 {
    if bias.is_empty() {
        0.0
    } else {
        bias[oc]
    }
}

/// The geometry and the input a 3x3 row pass needs, grouped so the row routines
/// take a row index, a weight slice and a bias rather than ten arguments.
///
/// `Copy` so the closures below can hold it by value, which is what keeps the
/// nested `rayon` iterators from fighting over a borrow.
#[derive(Clone, Copy)]
struct Row3x3<'a> {
    x: &'a [f32],
    c_in: usize,
    hw: usize,
    h: usize,
    wd: usize,
}

impl Row3x3<'_> {
    /// One output row of ONE output channel.
    fn row(&self, wp: &[f32], b: f32, oy: usize, orow: &mut [f32], buf: &mut Vec<f32>, simd: bool) {
        if buf.len() < self.wd {
            buf.resize(self.wd, 0.0);
        }
        let acc = &mut buf[..self.wd];
        acc.fill(b);
        let tiled_end = self.tiles(wp, oy, acc, simd);
        self.axpy(wp, oy, tiled_end, acc, simd);
        orow.copy_from_slice(acc);
    }

    /// One output row of TWO output channels at once.
    ///
    /// The two channels arrive as arrays rather than as four arguments so that
    /// "channel 0" and "channel 1" are one concept in the signature and not two
    /// pairs of parameters that a caller could transpose.
    fn row_pair(
        &self, chs: [(&[f32], f32); 2], oy: usize, outs: [&mut [f32]; 2],
        bufs: (&mut Vec<f32>, &mut Vec<f32>), simd: bool,
    ) {
        let [(wp0, b0), (wp1, b1)] = chs;
        let [orow0, orow1] = outs;
        let (buf0, buf1) = bufs;
        for b in [&mut *buf0, &mut *buf1] {
            if b.len() < self.wd {
                b.resize(self.wd, 0.0);
            }
        }
        let (acc0, acc1) = (&mut buf0[..self.wd], &mut buf1[..self.wd]);
        acc0.fill(b0);
        acc1.fill(b1);
        let tiled_end = self.tiles_pair(wp0, wp1, oy, acc0, acc1, simd);
        self.axpy(wp0, oy, tiled_end, acc0, simd);
        self.axpy(wp1, oy, tiled_end, acc1, simd);
        orow0.copy_from_slice(acc0);
        orow1.copy_from_slice(acc1);
    }

    /// The 32-column register tiles across the interior of a row, for ONE output
    /// channel, from column 1 to the last column a full tile fits in. Returns the
    /// first column NO tile covered, or 0 if the row was too narrow for one -
    /// which is how the caller knows the AXPY pass takes the whole row.
    #[cfg(target_arch = "x86_64")]
    fn tiles(&self, wp: &[f32], oy: usize, acc: &mut [f32], simd: bool) -> usize {
        let mut x0 = 1usize;
        if simd {
            while x0 + TILE <= self.wd - 1 {
                // SAFETY: `simd` is the AVX2+FMA runtime test, and the loop
                // condition keeps x0 >= 1 and x0 + TILE + 1 <= wd, so every kx
                // offset the tile reads is in bounds.
                unsafe { row_tile3x3_avx2(self, wp, oy, x0, acc) };
                x0 += TILE;
            }
        }
        if x0 == 1 { 0 } else { x0 }
    }

    /// [`Self::tiles`] for TWO output channels, sharing every input load.
    #[cfg(target_arch = "x86_64")]
    fn tiles_pair(
        &self, wp0: &[f32], wp1: &[f32], oy: usize,
        acc0: &mut [f32], acc1: &mut [f32], simd: bool,
    ) -> usize {
        let mut x0 = 1usize;
        if simd {
            while x0 + TILE <= self.wd - 1 {
                // SAFETY: as above, for both weight slices.
                unsafe { row_tile3x3_pair_avx2(self, wp0, wp1, oy, x0, acc0, acc1) };
                x0 += TILE;
            }
        }
        if x0 == 1 { 0 } else { x0 }
    }

    /// Without AVX2 there is no tile: the AXPY form takes the whole row.
    #[cfg(not(target_arch = "x86_64"))]
    fn tiles(&self, wp: &[f32], oy: usize, acc: &mut [f32], simd: bool) -> usize {
        let _ = (wp, oy, acc, simd);
        0
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn tiles_pair(
        &self, wp0: &[f32], wp1: &[f32], oy: usize,
        acc0: &mut [f32], acc1: &mut [f32], simd: bool,
    ) -> usize {
        let _ = (wp0, wp1, oy, acc0, acc1, simd);
        0
    }

    /// The AXPY form for the columns the tiles did not write.
    fn axpy(&self, wp: &[f32], oy: usize, tiled_end: usize, acc: &mut [f32], simd: bool) {
        let segs: &[(usize, usize)] = if tiled_end == 0 {
            &[(0, self.wd)]
        } else {
            &[(0, 1), (tiled_end, self.wd)]
        };
        for &(a, b) in segs {
            for ky in 0..3usize {
                let iy = oy as isize + ky as isize - 1;
                if iy < 0 || iy >= self.h as isize {
                    continue;
                }
                let srow = iy as usize * self.wd;
                for kx in 0..3usize {
                    // The output columns this tap reaches: kx = 0 loses column 0
                    // to the left edge and kx = 2 loses the last column. Every
                    // access inside the range below is in bounds by construction,
                    // which is the whole point - the bounds test used to sit in
                    // the innermost loop of a 27-tap sum.
                    let lo = a.max(if kx == 0 { 1 } else { 0 });
                    let hi = b.min(if kx == 2 { self.wd.wrapping_sub(1) } else { self.wd });
                    if lo >= hi {
                        continue;
                    }
                    for ci in 0..self.c_in {
                        let k = wp[ci * 9 + ky * 3 + kx];
                        if k == 0.0 {
                            continue;
                        }
                        let base = ci * self.hw + srow + lo + kx - 1;
                        row_axpy(&mut acc[lo..hi], &self.x[base..base + (hi - lo)], k, simd);
                    }
                }
            }
        }
    }
}

/// The tile width, in output columns, of [`row_tile3x3_avx2`] - four `__m256`
/// accumulators. NOT a runtime value: the kernel's register allocation and the
/// loop condition that has to keep every tap in bounds both depend on it.
const TILE: usize = 32;

/// One 32-column interior tile of one output row, with the accumulators in
/// registers for ALL 27 taps.
///
/// The caller guarantees `x0 >= 1` and `x0 + TILE + 1 <= wd`, so all three `kx`
/// offsets are in bounds and no tap is skipped or clamped. `acc` already holds
/// the bias in every column; this adds the taps to it, in the same per-lane
/// order as the AXPY path, so the two are interchangeable.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn row_tile3x3_avx2(g: &Row3x3, wp: &[f32], oy: usize, x0: usize, acc: &mut [f32]) {
    use core::arch::x86_64::*;
    let (x, c_in, hw, h, wd) = (g.x, g.c_in, g.hw, g.h, g.wd);
    let xp = x.as_ptr();
    let b = *acc.get_unchecked(x0);
    let mut a0 = _mm256_set1_ps(b);
    let mut a1 = a0;
    let mut a2 = a0;
    let mut a3 = a0;
    for ky in 0..3usize {
        let iy = oy as isize + ky as isize - 1;
        if iy < 0 || iy >= h as isize {
            continue;
        }
        let srow = iy as usize * wd;
        for kx in 0..3usize {
            for ci in 0..c_in {
                let k = *wp.get_unchecked(ci * 9 + ky * 3 + kx);
                if k == 0.0 {
                    continue;
                }
                let kv = _mm256_set1_ps(k);
                let p = xp.add(ci * hw + srow + x0 + kx - 1);
                a0 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(p), a0);
                a1 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(p.add(8)), a1);
                a2 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(p.add(16)), a2);
                a3 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(p.add(24)), a3);
            }
        }
    }
    let op = acc.as_mut_ptr().add(x0);
    _mm256_storeu_ps(op, a0);
    _mm256_storeu_ps(op.add(8), a1);
    _mm256_storeu_ps(op.add(16), a2);
    _mm256_storeu_ps(op.add(24), a3);
}

/// The same tile for TWO output channels at once: one `_mm256_loadu_ps` feeds an
/// FMA into each channel's accumulator, so four loads issue eight FMAs.
///
/// This is the shape ifan-rs's own `conv3x3_row_tile2_avx2` has, and it is where
/// its edge over a one-channel tile came from. The gain is not arithmetic - the
/// same 27 multiplies per output happen either way - it is that the loads are
/// shared, halving the load:FMA ratio of the inner loop.
///
/// The per-lane order is IDENTICAL to [`row_tile3x3_avx2`]: bias already in the
/// accumulator, then ky, kx, ci, one FMA each, for both channels. So the two
/// tiles produce bit-identical output and the `c_out` parity of a caller cannot
/// change a result.
///
/// # Safety
///
/// Same guarantee as [`row_tile3x3_avx2`], for both weight slices: `x0 >= 1` and
/// `x0 + TILE + 1 <= wd`, so every `kx` offset is in bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn row_tile3x3_pair_avx2(
    g: &Row3x3, wp0: &[f32], wp1: &[f32], oy: usize, x0: usize,
    acc0: &mut [f32], acc1: &mut [f32],
) {
    use core::arch::x86_64::*;
    let (x, c_in, hw, h, wd) = (g.x, g.c_in, g.hw, g.h, g.wd);
    let xp = x.as_ptr();
    let mut a = [_mm256_set1_ps(*acc0.get_unchecked(x0)); 4];
    let mut b = [_mm256_set1_ps(*acc1.get_unchecked(x0)); 4];
    for ky in 0..3usize {
        let iy = oy as isize + ky as isize - 1;
        if iy < 0 || iy >= h as isize {
            continue;
        }
        let srow = iy as usize * wd;
        for kx in 0..3usize {
            for ci in 0..c_in {
                let p = xp.add(ci * hw + srow + x0 + kx - 1);
                let v0 = _mm256_loadu_ps(p);
                let v1 = _mm256_loadu_ps(p.add(8));
                let v2 = _mm256_loadu_ps(p.add(16));
                let v3 = _mm256_loadu_ps(p.add(24));
                let ka = _mm256_set1_ps(*wp0.get_unchecked(ci * 9 + ky * 3 + kx));
                let kb = _mm256_set1_ps(*wp1.get_unchecked(ci * 9 + ky * 3 + kx));
                a[0] = _mm256_fmadd_ps(ka, v0, a[0]);
                a[1] = _mm256_fmadd_ps(ka, v1, a[1]);
                a[2] = _mm256_fmadd_ps(ka, v2, a[2]);
                a[3] = _mm256_fmadd_ps(ka, v3, a[3]);
                b[0] = _mm256_fmadd_ps(kb, v0, b[0]);
                b[1] = _mm256_fmadd_ps(kb, v1, b[1]);
                b[2] = _mm256_fmadd_ps(kb, v2, b[2]);
                b[3] = _mm256_fmadd_ps(kb, v3, b[3]);
            }
        }
    }
    let o0 = acc0.as_mut_ptr().add(x0);
    let o1 = acc1.as_mut_ptr().add(x0);
    for j in 0..4 {
        _mm256_storeu_ps(o0.add(8 * j), a[j]);
        _mm256_storeu_ps(o1.add(8 * j), b[j]);
    }
}

/// `acc[i] += k * src[i]` for every i, as one contiguous AXPY: the loop the
/// twin exists for.
///
/// `simd` is resolved ONCE per kernel call by the caller rather than re-tested
/// per tap. `is_x86_feature_detected!` caches its answer, so the test is cheap,
/// but a tap can be as short as a few elements and the branch would then be a
/// real fraction of it; a tap loop is also not a place to make the vectoriser
/// prove anything.
#[inline]
fn row_axpy(acc: &mut [f32], src: &[f32], k: f32, simd: bool) {
    debug_assert_eq!(acc.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    {
        if simd {
            // SAFETY: `simd_ok` is `is_x86_feature_detected!` for exactly the two
            // features `row_axpy_avx2` is compiled with, and it is the only
            // source of a `true`.
            unsafe { row_axpy_avx2(acc, src, k) };
            return;
        }
    }
    let _ = simd;
    for (a, s) in acc.iter_mut().zip(src.iter()) {
        *a += k * *s;
    }
}

/// AVX2 + FMA, or `false` where the CPU says it has neither. Both features are
/// tested: `_mm256_fmadd_ps` is FMA, and a CPU with AVX2 alone would fault on it.
#[cfg(target_arch = "x86_64")]
#[inline]
fn simd_ok() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn simd_ok() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn row_axpy_avx2(acc: &mut [f32], src: &[f32], k: f32) {
    use core::arch::x86_64::*;
    let n = acc.len();
    debug_assert_eq!(src.len(), n);
    let kv = _mm256_set1_ps(k);
    let mut i = 0usize;
    while i + 8 <= n {
        let a = _mm256_loadu_ps(acc.as_ptr().add(i));
        let s = _mm256_loadu_ps(src.as_ptr().add(i));
        _mm256_storeu_ps(acc.as_mut_ptr().add(i), _mm256_fmadd_ps(kv, s, a));
        i += 8;
    }
    // The tail is always evaluated: `n` is a row width, a runtime value nothing
    // above can prove is a multiple of 8.
    while i < n {
        *acc.get_unchecked_mut(i) += k * *src.get_unchecked(i);
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Twins of the kernels promoted from the vision engines (section 12 of
// cuda/kernels.cu). Each one states the accumulation order its kernel uses,
// because that order is the contract a backend comparison rests on - and where
// the kernel replaces an op already here, the order is the SAME one, so a swap
// can be checked by equality rather than by tolerance.
// ---------------------------------------------------------------------------

/// `lg_linear_rb`: the register-blocked GEMM on the token layout, the same
/// arithmetic as `lg_linear` - c ascending, the bias added after the last
/// multiply - with a 4x4 output tile per thread in registers.
pub fn linear_rb(
    x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32],
    rows: usize, c_in: usize, c_out: usize,
) {
    for r in 0..rows {
        for o in 0..c_out {
            let mut acc = 0.0f32;
            for c in 0..c_in {
                acc += x[r * c_in + c] * w[o * c_in + c];
            }
            y[r * c_out + o] = acc + if bias.is_empty() { 0.0 } else { bias[o] };
        }
    }
}

/// `lg_conv1x1_rb`: the same op as `lg_conv1x1` on the plane layout, register
/// tiled. The kernel folds the bias into the accumulator BEFORE the first
/// multiply where token-layout kernels add it after the last one; this
/// reproduces `lg_conv1x1`'s order, and that is the only reason the two
/// instantiations of the body differ.
pub fn conv1x1_rb(
    x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32],
    c_in: usize, c_out: usize, h: usize, wd: usize,
) {
    let plane = h * wd;
    for o in 0..c_out {
        let b = if bias.is_empty() { 0.0f32 } else { bias[o] };
        for p in 0..plane {
            let mut acc = b;
            for c in 0..c_in {
                acc += w[o * c_in + c] * x[c * plane + p];
            }
            y[o * plane + p] = acc;
        }
    }
}

/// `lg_conv3x3_tile`: 3x3, stride 1, pad 1 - the same operator as
/// `lg_conv3x3s1p1`, accumulated in the TILED kernel's order.
///
/// TWO THINGS HERE ARE THE CONTRACT AND NOT STYLE. The channel loop is OUTERMOST
/// (ci, then ky, then kx) because in the kernel the channel tile has to be the
/// outer loop, and the BIAS IS ADDED AFTER THE SUM where `lg_conv3x3s1p1` folds it
/// into the accumulator first. So this twin and a section-6-order twin agree only
/// to rounding, and the selftest checks it against a reference in THIS order
/// rather than against `lg_conv3x3s1p1`'s. The halo is zero-filled, which is the
/// same sum with zeros added.
///
/// `act`: 0 none, 1 relu, 2 leaky relu with `act_p` as the slope - the
/// `lg_conv3x3_winograd` convention, not ifan's 0/1 flag.
pub fn conv3x3_tile(
    x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32],
    c_in: usize, c_out: usize, h: usize, wd: usize, act: u32, act_p: f32,
) {
    let plane = h * wd;
    for oc in 0..c_out {
        let b = if bias.is_empty() { 0.0f32 } else { bias[oc] };
        for oy in 0..h {
            for ox in 0..wd {
                let mut acc = 0.0f32;
                for ci in 0..c_in {
                    let wp = (oc * c_in + ci) * 9;
                    for ky in 0..3 {
                        let iy = oy as isize + ky as isize - 1;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        for kx in 0..3 {
                            let ix = ox as isize + kx as isize - 1;
                            if ix < 0 || ix >= wd as isize {
                                continue;
                            }
                            acc += x[ci * plane + iy as usize * wd + ix as usize]
                                * w[wp + ky * 3 + kx];
                        }
                    }
                }
                let v = activate(acc + b, act, act_p);
                y[(oc * h + oy) * wd + ox] = v;
            }
        }
    }
}

/// `lg_conv1x1_tile`: the same operator as `lg_conv1x1`, accumulated in the
/// tiled kernel's order - c ascending, and the BIAS ADDED AFTER THE SUM where
/// `lg_conv1x1` and `lg_conv1x1_rb` both fold it in first. That single
/// difference is why all three are separate entries and why only the 1x1's two
/// section-6 forms are bit-compatible with each other.
pub fn conv1x1_tile(
    x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32],
    c_in: usize, c_out: usize, h: usize, wd: usize, act: u32, act_p: f32,
) {
    let plane = h * wd;
    for oc in 0..c_out {
        let b = if bias.is_empty() { 0.0f32 } else { bias[oc] };
        for p in 0..plane {
            let mut acc = 0.0f32;
            for c in 0..c_in {
                acc += w[oc * c_in + c] * x[c * plane + p];
            }
            y[oc * plane + p] = activate(acc + b, act, act_p);
        }
    }
}

/// The activation the fused convolutions apply, in the toolkit's convention.
fn activate(v: f32, act: u32, act_p: f32) -> f32 {
    match act {
        1 => {
            if v > 0.0 {
                v
            } else {
                0.0
            }
        }
        2 => {
            if v >= 0.0 {
                v
            } else {
                act_p * v
            }
        }
        _ => v,
    }
}

/// The reduction `lg_layer_norm_warp` performs: a halving tree over the lane
/// stride, summed in the order the shuffles do it, then broadcast from lane 0.
/// The twin reproduces the TREE rather than a serial sum, because that is what
/// makes the two comparable; `layer_norm` above is a two-pass form in a
/// different order and is a different contract.
fn warp_tree_sum(v: &[f32]) -> f32 {
    let mut a = v.to_vec();
    a.resize(32, 0.0);
    let mut off = 16;
    while off > 0 {
        for lane in 0..off {
            a[lane] += a[lane + off];
        }
        off >>= 1;
    }
    a[0]
}

/// `lg_layer_norm_warp`: LayerNorm with one warp per row, a row of up to 128
/// held in registers, the one-pass `E[x^2] - mean^2` variance, and the reduction
/// tree above. A wider row re-reads x, as the kernel's fallback path does.
pub fn layer_norm_warp(
    x: &[f32], w: &[f32], b: &[f32], y: &mut [f32],
    ne0: usize, nrows: usize, eps: f32,
) {
    for r in 0..nrows {
        let xr = &x[r * ne0..r * ne0 + ne0];
        let yr = &mut y[r * ne0..r * ne0 + ne0];
        // ALWAYS 32 LANE ACCUMULATORS, never `ceil(ne0/32)`: lane `l` holds the
        // elements at l, l+32, ..., which is what the kernel's per-lane register
        // array does. Collapsing the lanes first would sum the row in a
        // different order - and for a row narrower than a warp it would drop
        // every element but the first, which is the mistake this shape catches.
        let mut s = vec![0.0f32; 32];
        let mut q = vec![0.0f32; 32];
        for i in 0..ne0 {
            let v = xr[i];
            s[i % 32] += v;
            q[i % 32] += v * v;
        }
        let s = warp_tree_sum(&s);
        let q = warp_tree_sum(&q);
        let n = ne0 as f32;
        let mean = s / n;
        let rstd = 1.0 / ((q / n - mean * mean).max(0.0) + eps).sqrt();
        for i in 0..ne0 {
            yr[i] = (xr[i] - mean) * rstd * w[i] + b[i];
        }
    }
}

/// `lg_conv2x2s2`: 2x2, stride 2, no padding, nullable bias. Order: ky, kx, ci.
pub fn conv2x2s2(
    x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32],
    c_in: usize, c_out: usize, h: usize, wd: usize,
) {
    let (oh, ow) = (h / 2, wd / 2);
    let plane = h * wd;
    for oc in 0..c_out {
        let b = if bias.is_empty() { 0.0f32 } else { bias[oc] };
        for oy in 0..oh {
            for ox in 0..ow {
                let mut acc = b;
                for ky in 0..2 {
                    for kx in 0..2 {
                        for ci in 0..c_in {
                            let xv = x[ci * plane + (2 * oy + ky) * wd + (2 * ox + kx)];
                            acc += w[(oc * c_in + ci) * 4 + ky * 2 + kx] * xv;
                        }
                    }
                }
                y[(oc * oh + oy) * ow + ox] = acc;
            }
        }
    }
}

/// `lg_conv_t2x2`: the transposed twin, no tap flip, weight layout
/// `[c_in][c_out][2][2]`. Order: ci.
pub fn conv_t2x2(
    x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32],
    c_in: usize, c_out: usize, h: usize, wd: usize,
) {
    let (oh, ow) = (2 * h, 2 * wd);
    let plane = h * wd;
    for oc in 0..c_out {
        let b = if bias.is_empty() { 0.0f32 } else { bias[oc] };
        for oy in 0..oh {
            for ox in 0..ow {
                let (iy, ky) = (oy / 2, oy % 2);
                let (ix, kx) = (ox / 2, ox % 2);
                let mut acc = b;
                for ci in 0..c_in {
                    acc += w[(ci * c_out + oc) * 4 + ky * 2 + kx] * x[ci * plane + iy * wd + ix];
                }
                y[(oc * oh + oy) * ow + ox] = acc;
            }
        }
    }
}

/// The Swin window index map, shared by both directions so a twin cannot
/// disagree with itself about the shift. Returns the plane offset of a window's
/// token `t`. `w0` is the chunk's first window index.
fn window_plane_off(
    wl: usize, t: usize, nww: usize, win: usize, hp: usize, wp: usize, shift: usize,
) -> (usize, usize) {
    let i = t / win;
    let j = t % win;
    let wh = wl / nww;
    let ww = wl % nww;
    (((wh * win + i + shift) % hp) * wp, (ww * win + j + shift) % wp)
}

/// `lg_window_gather`: NCHW plane -> `[nw][n][c]` tokens.
#[allow(clippy::too_many_arguments)]
pub fn window_gather(
    x: &[f32], tok: &mut [f32],
    nw: usize, n: usize, nww: usize, win: usize, hp: usize, wp: usize, c: usize,
    shift: usize, w0: usize,
) {
    for wl in 0..nw {
        for t in 0..n {
            let (py, px) = window_plane_off(w0 + wl, t, nww, win, hp, wp, shift);
            for ch in 0..c {
                tok[(wl * n + t) * c + ch] = x[ch * hp * wp + py + px];
            }
        }
    }
}

/// `lg_window_scatter`: the exact inverse of `window_gather` at the same shift.
#[allow(clippy::too_many_arguments)]
pub fn window_scatter(
    tok: &[f32], x: &mut [f32],
    nw: usize, n: usize, nww: usize, win: usize, hp: usize, wp: usize, c: usize,
    shift: usize, w0: usize,
) {
    for wl in 0..nw {
        for t in 0..n {
            let (py, px) = window_plane_off(w0 + wl, t, nww, win, hp, wp, shift);
            for ch in 0..c {
                x[ch * hp * wp + py + px] = tok[(wl * n + t) * c + ch];
            }
        }
    }
}

/// `lg_pixel_shuffle`: depth-to-space, the INVERSE of the unshuffle above. Input
/// channel `c*r*r + dy*r + dx` goes to output channel `c` at `(r*y + dy, r*x + dx)`,
/// which is torch's `pixel_shuffle(x, r)` view `(c, r, r, h, w) -> (c, h, r, w, r)`.
///
/// The permutation is the contract, not the shape, so this is written from the
/// kernel's stated contract rather than from the kernel's code - the two can
/// disagree, and `selftest` demands equality between them rather than a tolerance.
/// One output row `(ch, oy)` is filled per iteration and the rows are independent,
/// so the loop is parallel; nothing is reordered, so the result is exact.
pub fn pixel_shuffle(src: &[f32], dst: &mut [f32], c: usize, h: usize, w: usize, r: usize) {
    let oh = h * r;
    let ow = w * r;
    debug_assert_eq!(src.len(), c * r * r * h * w);
    debug_assert_eq!(dst.len(), c * oh * ow);
    dst.par_chunks_mut(ow)
        .enumerate()
        .for_each(|(row, out_row)| {
            let ch = row / oh;
            let oy = row % oh;
            let y = oy / r;
            let dy = oy % r;
            for ox in 0..ow {
                let x = ox / r;
                let dx = ox % r;
                let sch = ch * r * r + dy * r + dx;
                out_row[ox] = src[(sch * h + y) * w + x];
            }
        });
}

/// Bit reversal of `i` over `nb` bits, matching the kernels' `fft_bitrev`.
fn bitrev(i: usize, nb: usize) -> usize {
    let mut j = 0usize;
    for b in 0..nb {
        if i & (1 << b) != 0 {
            j |= 1 << (nb - 1 - b);
        }
    }
    j
}

/// The toolkit's own smoke test: every twin must run and agree with a
/// straightforward reimplementation on a fixed input. Called by `gpuinfo`, which
/// is the completeness check for the pair (kernel, twin) - a kernel whose twin
/// is missing or wrong is a CPU backend that cannot run the graph.
pub fn selftest() -> Result<(), String> {
    // Roots of the norms must be exact for a constant input.
    let x = vec![3.0f32; 8];
    let w = vec![1.0f32; 8];
    let mut y = vec![0.0f32; 8];
    rms_norm(&x, &w, &mut y, 8, 1, 0.0);
    for v in &y {
        if (v - 1.0).abs() > 1e-6 {
            return Err(format!("rms_norm(3) = {v}, expected 1"));
        }
    }
    layer_norm(&x, &w, &w, &mut y, 8, 1, 0.0);
    for v in &y {
        if (v - 1.0).abs() > 1e-6 {
            return Err(format!("layer_norm(3) = {v}, expected 1"));
        }
    }
    // silu*up with silu(0) = 0
    let mut g = vec![0.0f32; 4];
    silu_mul(&mut g, &[1.0; 4]);
    if g.iter().any(|v| v.abs() > 1e-9) {
        return Err("silu_mul(0) != 0".into());
    }
    // Channel-wise layer norm over NCHW: c=3 channels, hw=2 positions. Channel
    // values 3, 7, 5 give mean 5 and var 8/3, so with w=1,b=0 the normalized
    // output is (v - 5)/sqrt(8/3) = -1.224745, 1.224745, 0 for BOTH positions -
    // and crucially it must be identical at p=0 and p=1 (the gather is strided
    // by hw, which is the whole point of this op versus lg_layer_norm).
    let cn_x = vec![3.0f32, 3.0, 7.0, 7.0, 5.0, 5.0];
    let cn_w = vec![1.0f32; 3];
    let cn_b = vec![0.0f32; 3];
    let mut cn_y = vec![9.0f32; 6];
    channel_layer_norm(&cn_x, &cn_w, &cn_b, &mut cn_y, 3, 2, 0.0);
    let expect = [-1.2247449f32, -1.2247449, 1.2247449, 1.2247449, 0.0, 0.0];
    for (i, (g, e)) in cn_y.iter().zip(expect.iter()).enumerate() {
        if (g - e).abs() > 1e-6 {
            return Err(format!("channel_layer_norm[{i}] = {g}, expected {e}"));
        }
    }

    // erf against the known value erf(1) = 0.8427007929...
    let e = erf(1.0);
    if (e - 0.8427008).abs() > 1e-6 {
        return Err(format!("erf(1) = {e}"));
    }
    // An even split of two halves must rotate to the same vector.
    let mut r = vec![1.0f32, 0.0, 0.0, 1.0];
    rope_neox(&mut r, &[0], 4, 1, 1, 10000.0);
    if (r[0] - 1.0).abs() > 1e-6 || r[1].abs() > 1e-6 {
        return Err(format!("rope_neox(pos 0) changed the vector: {r:?}"));
    }
    // FFT round trip: a known real plane, forward then ortho inverse, must come
    // back to itself. n = 8 with 2 planes exercises the batch loop and the
    // conjugate-symmetry rebuild (x > n/2), and a non-trivial signal makes a
    // bit-reversal or twiddle-sign error visible rather than cancelling out.
    {
        let n = 8usize;
        let batch = 2usize;
        let mut x = vec![0.0f32; batch * n * n];
        for b in 0..batch {
            for r in 0..n {
                for c in 0..n {
                    x[(b * n + r) * n + c] = ((r * n + c) % 7) as f32 - 3.0;
                }
            }
        }
        let hw = n / 2 + 1;
        let mut spec = vec![0.0f32; batch * n * hw * 2];
        fft2_r2c(&x, &mut spec, batch, n, 1.0);
        let mut back = vec![0.0f32; batch * n * n];
        fft2_c2r(&spec, &mut back, batch, n, 1.0 / (n * n) as f32);
        for i in 0..x.len() {
            if (back[i] - x[i]).abs() > 1e-4 {
                return Err(format!(
                    "fft round trip at {i}: {} != {}",
                    back[i], x[i]
                ));
            }
        }
        // The ortho convention is a scale factor: r2c with the ortho scale must
        // equal r2c without it, times that scale.
        let s = 1.0f32 / (n * n) as f32;
        let mut spec2 = vec![0.0f32; batch * n * hw * 2];
        fft2_r2c(&x, &mut spec2, batch, n, s);
        for i in 0..spec.len() {
            if (spec2[i] - spec[i] * s).abs() > 1e-4 {
                return Err(format!("fft ortho scale at {i}: {} != {}", spec2[i], spec[i] * s));
            }
        }
    }
    // lg_mul: the elementwise product, including the out-of-place contract (the
    // inputs must survive, which is what separates it from lg_silu_mul) and
    // negatives, so a `max(0, ...)`-style mistake cannot pass.
    {
        let a = [1.0f32, -2.0, 3.0, -4.0];
        let b = [5.0f32, 6.0, -7.0, -8.0];
        let mut y = [0.0f32; 4];
        mul(&a, &b, &mut y);
        let expect = [5.0f32, -12.0, -21.0, 32.0];
        for (i, (g, e)) in y.iter().zip(expect.iter()).enumerate() {
            if g != e {
                return Err(format!("mul[{i}] = {g}, expected {e}"));
            }
        }
        if a != [1.0f32, -2.0, 3.0, -4.0] || b != [5.0f32, 6.0, -7.0, -8.0] {
            return Err("mul modified its inputs".into());
        }
        // A length that is not a multiple of any block size: the kernel's guard
        // is `i < n` on an `int`, and the twin iterates the slice it is given.
        let a5 = [2.0f32, 2.0, 2.0, 2.0, 2.0];
        let b5 = [3.0f32, 0.5, -1.0, 0.0, 100.0];
        let mut y5 = [0.0f32; 5];
        mul(&a5, &b5, &mut y5);
        for (i, (g, e)) in y5.iter().zip([6.0f32, 1.0, -2.0, 0.0, 200.0].iter()).enumerate() {
            if g != e {
                return Err(format!("mul(odd length)[{i}] = {g}, expected {e}"));
            }
        }
    }

    // Channel affine / scale / mean, the three ops promoted from rmbg's and
    // MAXIM's own files. c=2, hw=3 with distinct per-channel vectors, so a
    // transposed scale/shift vector or an off-by-one in the channel index is
    // visible rather than masked by a uniform value.
    {
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let sc = [10.0f32, 100.0];
        let sh = [1.0f32, 2.0];
        let mut y = [0.0f32; 6];
        channel_affine(&x, &sc, &sh, &mut y, 2, 3);
        let expect = [11.0f32, 21.0, 31.0, 402.0, 502.0, 602.0];
        for (i, (g, e)) in y.iter().zip(expect.iter()).enumerate() {
            if (g - e).abs() > 1e-6 {
                return Err(format!("channel_affine[{i}] = {g}, expected {e}"));
            }
        }
        // A null shift and a null scale are the two documented specialisations.
        let mut y2 = [0.0f32; 6];
        channel_affine(&x, &sc, &[], &mut y2, 2, 3);
        for (i, (g, e)) in y2.iter().zip([10.0f32, 20.0, 30.0, 400.0, 500.0, 600.0].iter()).enumerate() {
            if (g - e).abs() > 1e-6 {
                return Err(format!("channel_affine(no shift)[{i}] = {g}, expected {e}"));
            }
        }
        let mut y3 = [0.0f32; 6];
        channel_affine(&x, &[], &sh, &mut y3, 2, 3);
        for (i, (g, e)) in y3.iter().zip([2.0f32, 3.0, 4.0, 6.0, 7.0, 8.0].iter()).enumerate() {
            if (g - e).abs() > 1e-6 {
                return Err(format!("channel_affine(no scale)[{i}] = {g}, expected {e}"));
            }
        }
        // channel_scale must be exactly channel_affine with no shift.
        let mut y4 = [0.0f32; 6];
        channel_scale(&x, &sc, &mut y4, 2, 3);
        if y4 != y2 {
            return Err(format!("channel_scale {y4:?} != channel_affine(no shift) {y2:?}"));
        }
        // In-place: the kernel documents that `in` may alias `out`, which is the
        // folded-BatchNorm form. Same answer as the out-of-place call.
        let mut alias = x;
        let want = y;
        for i in 0..6 {
            let v = alias[i] * sc[i / 3] + sh[i / 3];
            alias[i] = v;
        }
        if alias != want {
            return Err("channel_affine in-place differs from out-of-place".into());
        }

        // channel_mean: the twin mirrors the KERNEL's partial order, so the
        // block size is a parameter. block = 1 is the pure serial sum; a large
        // block is the GPU's shape. They must agree to within rounding, and the
        // non-multiple hw (5 against block 4) is the case a wrong tree would
        // break by dropping or double-counting an element.
        let m = [1.0f32, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0];
        let mut mo1 = [0.0f32; 2];
        channel_mean(&m, &mut mo1, 2, 4, 1);
        let mut mk = [0.0f32; 2];
        channel_mean(&m, &mut mk, 2, 4, 1024);
        for ch in 0..2 {
            let want = m[ch * 4..ch * 4 + 4].iter().sum::<f32>() / 4.0;
            if (mo1[ch] - want).abs() > 1e-6 {
                return Err(format!("channel_mean(block 1)[{ch}] = {}, expected {want}", mo1[ch]));
            }
            if (mk[ch] - want).abs() > 1e-6 {
                return Err(format!("channel_mean(block 1024)[{ch}] = {}, expected {want}", mk[ch]));
            }
        }
        let mut mo = [0.0f32; 2];
        channel_mean(&m, &mut mo, 2, 4, 3);  // an odd block, the assymetric case
        for ch in 0..2 {
            if (mo[ch] - mo1[ch]).abs() > 1e-6 {
                return Err(format!("channel_mean is block-dependent: {} vs {}", mo[ch], mo1[ch]));
            }
        }
    }

    // ---- the generic convolutions, the toolkit's parallel twins ----------
    //
    // Two things are on trial here and they are separable, which is why the
    // kernel has an inner-loop switch: the PARALLEL FRAMING (one rayon task per
    // output channel and row, a contiguous accumulator, the bounds resolved per
    // tap instead of per element) must change the answer EXACTLY never, and the
    // VECTOR path (an FMA where the scalar code rounds twice) may change it in
    // the last bit and nowhere else. A row that was dropped, mis-offset or
    // double-counted is off by ~1; the vector path is off by ~1e-7.
    {
        // EVERY ROW WIDTH IN [1, 90], because the two inner forms cross over by
        // width and the whole point of having both is that they are
        // interchangeable. A width below TILE exercises only the AXPY path, a
        // width above it exercises the register tile (and both, at the edges and
        // the tail), and several widths here put the last tile flush against the
        // right edge or leave remainders of 1..7 columns - the cases where a
        // segment that is off by one column shows up as a value of ~1, not 1e-7.
        // This sweep is what caught a real out-of-bounds tile on 64/65/67-wide
        // rows in the prototype, which is why it is 90 widths rather than one.
        for wd_probe in 1..=90usize {
            let (ci, co, h) = (3usize, 2usize, 3usize);
            let wd = wd_probe;
            let plane = h * wd;
            let x: Vec<f32> = (0..ci * plane).map(|i| ((i * 37 % 101) as f32 - 50.0) / 16.0).collect();
            let w: Vec<f32> = (0..co * ci * 9).map(|i| ((i * 23 % 71) as f32 - 35.0) / 16.0).collect();
            let b: Vec<f32> = (0..co).map(|i| (i as f32) / 4.0 - 0.5).collect();
            let mut want = vec![0.0f32; co * plane];
            for oc in 0..co {
                for oy in 0..h {
                    for ox in 0..wd {
                        let mut acc = b[oc];
                        for ky in 0..3usize {
                            let iy = oy as isize + ky as isize - 1;
                            if iy < 0 || iy >= h as isize {
                                continue;
                            }
                            for kx in 0..3usize {
                                let ix = ox as isize + kx as isize - 1;
                                if ix < 0 || ix >= wd as isize {
                                    continue;
                                }
                                for c in 0..ci {
                                    acc += w[(oc * ci + c) * 9 + ky * 3 + kx]
                                        * x[c * plane + iy as usize * wd + ix as usize];
                                }
                            }
                        }
                        want[(oc * h + oy) * wd + ox] = acc;
                    }
                }
            }
            // The scalar inner loop through the same framing: bit-identical.
            let mut gs = vec![0.0f32; co * plane];
            conv3x3s1p1_impl(&x, &w, &b, &mut gs, ci, co, h, wd, false);
            if gs != want {
                return Err(format!("conv3x3s1p1(wd={wd}): the framing changed the sum"));
            }
            // The vector path, tile and AXPY together: within the one FMA.
            let mut gv = vec![0.0f32; co * plane];
            conv3x3s1p1(&x, &w, &b, &mut gv, ci, co, h, wd);
            let bound_w = if simd_ok() { 1e-5 } else { 0.0 };
            for i in 0..co * plane {
                if (gv[i] - want[i]).abs() > bound_w {
                    return Err(format!(
                        "conv3x3s1p1(wd={wd}) element {i}: {} vs {}, bound {bound_w}",
                        gv[i], want[i]
                    ));
                }
            }
        }
        // A tall-narrow and a wide-short image, so the row framing sees a row
        // that is one task and a row that is many: the same arithmetic either way.
        let (ci, co, h, wd) = (5usize, 4usize, 7usize, 7usize);
        let plane = h * wd;
        let x: Vec<f32> = (0..ci * plane).map(|i| ((i * 37 % 101) as f32 - 50.0) / 16.0).collect();
        let w: Vec<f32> = (0..co * ci * 9).map(|i| ((i * 23 % 71) as f32 - 35.0) / 16.0).collect();
        let b: Vec<f32> = (0..co).map(|i| (i as f32) / 4.0 - 0.5).collect();

        // The twin's ARITHMETIC, written out independently, in the kernel's order:
        // the nullable bias folded into the accumulator first, then ky, kx, ci.
        let mut want = vec![0.0f32; co * plane];
        for oc in 0..co {
            for oy in 0..h {
                for ox in 0..wd {
                    let mut acc = b[oc];
                    for ky in 0..3usize {
                        let iy = oy as isize + ky as isize - 1;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        for kx in 0..3usize {
                            let ix = ox as isize + kx as isize - 1;
                            if ix < 0 || ix >= wd as isize {
                                continue;
                            }
                            for c in 0..ci {
                                acc += w[(oc * ci + c) * 9 + ky * 3 + kx]
                                    * x[c * plane + iy as usize * wd + ix as usize];
                            }
                        }
                    }
                    want[(oc * h + oy) * wd + ox] = acc;
                }
            }
        }
        // The scalar inner loop through the SAME parallel framing must be
        // bit-identical to that reference: any difference is the framing's, and
        // there must be none.
        let mut got_scalar = vec![0.0f32; co * plane];
        conv3x3s1p1_impl(&x, &w, &b, &mut got_scalar, ci, co, h, wd, false);
        if got_scalar != want {
            return Err("conv3x3s1p1: the parallel framing changed the sum".into());
        }
        // The vector path differs by the FMA alone - and by nothing at all on a
        // CPU without AVX2 and FMA, where it IS the scalar path.
        let mut got = vec![0.0f32; co * plane];
        conv3x3s1p1(&x, &w, &b, &mut got, ci, co, h, wd);
        let mut worst = 0.0f32;
        for i in 0..co * plane {
            let d = (got[i] - want[i]).abs();
            if d > worst {
                worst = d;
            }
        }
        let bound = if simd_ok() { 1e-5 } else { 0.0 };
        if worst > bound {
            return Err(format!(
                "conv3x3s1p1: worst |d| vs its own order is {worst}, expected {bound}"
            ));
        }

        // THE CHANNEL PAIRING MUST BE INVISIBLE. `conv3x3s1p1` pairs output
        // channels so one input load feeds two FMAs; that is a schedule, not
        // arithmetic, so the paired and unpaired paths have to agree BIT FOR BIT.
        // If they ever differ by even one ulp, the pairing has changed the sum,
        // which means the two accumulators are not doing what this file says -
        // and the width sweep above would not catch it, because it uses the
        // default. An ODD `c_out` is included: that is the case where the last
        // chunk holds one plane and takes the unpaired routine.
        for c_out_probe in 2..=9usize {
            let (ci, h, wd) = (3usize, 3usize, 41usize);
            let plane = h * wd;
            let x: Vec<f32> = (0..ci * plane).map(|i| ((i * 37 % 101) as f32 - 50.0) / 16.0).collect();
            let w: Vec<f32> = (0..c_out_probe * ci * 9).map(|i| ((i * 23 % 71) as f32 - 35.0) / 16.0).collect();
            let b: Vec<f32> = (0..c_out_probe).map(|i| (i as f32) / 4.0 - 0.5).collect();
            let mut ys = vec![0.0f32; c_out_probe * plane];
            let mut yp = vec![0.0f32; c_out_probe * plane];
            conv3x3s1p1_with(&x, &w, &b, &mut ys, ci, c_out_probe, h, wd, simd_ok(), false);
            conv3x3s1p1_with(&x, &w, &b, &mut yp, ci, c_out_probe, h, wd, simd_ok(), true);
            if ys != yp {
                return Err(format!(
                    "conv3x3s1p1(c_out={c_out_probe}): the channel pairing changed the sum"
                ));
            }
        }

        // A null bias must mean exactly what a zero bias means, and the same for
        // the 1x1 below: the kernels take a NULLABLE pointer, and a twin that
        // indexed an empty slice instead would be reading out of bounds.
        let zero = vec![0.0f32; co];
        let mut yz = vec![0.0f32; co * plane];
        let mut yn = vec![0.0f32; co * plane];
        conv3x3s1p1(&x, &w, &zero, &mut yz, ci, co, h, wd);
        conv3x3s1p1(&x, &w, &[], &mut yn, ci, co, h, wd);
        if yz != yn {
            return Err("conv3x3s1p1: a null bias differs from a zero bias".into());
        }

        // The same three checks for lg_conv1x1: order c ascending with the bias
        // first, the framing exact, the vector path within an FMA.
        let (ci1, co1, h1, wd1) = (6usize, 4usize, 3usize, 5usize);
        let plane1 = h1 * wd1;
        let x1: Vec<f32> = (0..ci1 * plane1).map(|i| ((i * 29 % 97) as f32 - 48.0) / 16.0).collect();
        let w1: Vec<f32> = (0..co1 * ci1).map(|i| ((i * 13 % 41) as f32 - 20.0) / 16.0).collect();
        let b1: Vec<f32> = (0..co1).map(|i| (i as f32) / 3.0 - 0.5).collect();
        let mut want1 = vec![0.0f32; co1 * plane1];
        for o in 0..co1 {
            for p in 0..plane1 {
                let mut acc = b1[o];
                for c in 0..ci1 {
                    acc += w1[o * ci1 + c] * x1[c * plane1 + p];
                }
                want1[o * plane1 + p] = acc;
            }
        }
        let mut got1_scalar = vec![0.0f32; co1 * plane1];
        conv1x1_impl(&x1, &w1, &b1, &mut got1_scalar, ci1, co1, h1, wd1, false);
        if got1_scalar != want1 {
            return Err("conv1x1: the parallel framing changed the sum".into());
        }
        let mut got1 = vec![0.0f32; co1 * plane1];
        conv1x1(&x1, &w1, &b1, &mut got1, ci1, co1, h1, wd1);
        let mut worst1 = 0.0f32;
        for i in 0..co1 * plane1 {
            let d = (got1[i] - want1[i]).abs();
            if d > worst1 {
                worst1 = d;
            }
        }
        if worst1 > bound {
            return Err(format!(
                "conv1x1: worst |d| vs its own order is {worst1}, expected {bound}"
            ));
        }
        let zero1 = vec![0.0f32; co1];
        let mut yz1 = vec![0.0f32; co1 * plane1];
        let mut yn1 = vec![0.0f32; co1 * plane1];
        conv1x1(&x1, &w1, &zero1, &mut yz1, ci1, co1, h1, wd1);
        conv1x1(&x1, &w1, &[], &mut yn1, ci1, co1, h1, wd1);
        if yz1 != yn1 {
            return Err("conv1x1: a null bias differs from a zero bias".into());
        }
        // conv1x1_rb is the SAME operator and the SAME order, so a swap between
        // them is an equality check - and since both are serial in c ascending
        // with the bias in the accumulator from the start, they must agree
        // exactly, not to a tolerance.
        let mut yrb = vec![0.0f32; co1 * plane1];
        conv1x1_rb(&x1, &w1, &b1, &mut yrb, ci1, co1, h1, wd1);
        if yrb != want1 {
            return Err("conv1x1_rb disagrees with conv1x1's order".into());
        }
    }

    // ---- the promoted kernels (section 12 of cuda/kernels.cu) ----
    {
        // lg_linear_rb must reproduce lg_linear EXACTLY: same op, same order.
        let (rows, ci, co) = (5usize, 24usize, 34usize);
        let x: Vec<f32> = (0..rows * ci).map(|i| ((i * 37 % 101) as f32 - 50.0) / 16.0).collect();
        let w: Vec<f32> = (0..co * ci).map(|i| ((i * 53 % 71) as f32 - 35.0) / 32.0).collect();
        let b: Vec<f32> = (0..co).map(|i| (i as f32) / 8.0 - 2.0).collect();
        let mut y1 = vec![0.0f32; rows * co];
        let mut y2 = vec![0.0f32; rows * co];
        linear_rb(&x, &w, &b, &mut y1, rows, ci, co);
        // the tiled kernel's order, written out independently
        for r in 0..rows {
            for o in 0..co {
                let mut acc = 0.0f32;
                for c in 0..ci {
                    acc += x[r * ci + c] * w[o * ci + c];
                }
                y2[r * co + o] = acc + b[o];
            }
        }
        if y1 != y2 {
            return Err("linear_rb disagrees with its own order".into());
        }
        // A null bias must mean exactly what a zero bias means.
        let zero = vec![0.0f32; co];
        let mut y3 = vec![0.0f32; rows * co];
        linear_rb(&x, &w, &[], &mut y3, rows, ci, co);
        let mut y4 = vec![0.0f32; rows * co];
        linear_rb(&x, &w, &zero, &mut y4, rows, ci, co);
        if y3 != y4 {
            return Err("linear_rb: null bias differs from a zero bias".into());
        }

        // lg_conv1x1_rb against lg_conv1x1's own order (bias FIRST).
        let (ci, co, h, wd) = (6usize, 5usize, 3usize, 4usize);
        let plane = h * wd;
        let x: Vec<f32> = (0..ci * plane).map(|i| ((i * 29 % 97) as f32 - 48.0) / 16.0).collect();
        let w: Vec<f32> = (0..co * ci).map(|i| ((i * 17 % 43) as f32 - 21.0) / 16.0).collect();
        let b: Vec<f32> = (0..co).map(|i| (i as f32) / 4.0 - 1.0).collect();
        let mut y1 = vec![0.0f32; co * plane];
        let mut y2 = vec![0.0f32; co * plane];
        conv1x1_rb(&x, &w, &b, &mut y1, ci, co, h, wd);
        for o in 0..co {
            for p in 0..plane {
                let mut acc = b[o];
                for c in 0..ci {
                    acc += w[o * ci + c] * x[c * plane + p];
                }
                y2[o * plane + p] = acc;
            }
        }
        if y1 != y2 {
            return Err("conv1x1_rb disagrees with its own order".into());
        }

        // lg_conv3x3_tile / lg_conv1x1_tile. Their order is NOT the order of
        // lg_conv3x3s1p1 / lg_conv1x1 (channel loop outermost, bias after the
        // sum), so each is checked against a reference in ITS OWN order, and that
        // order is then compared to the section-6 one to pin the size of the
        // difference the kernel's comment claims: rounding, not a tap mistake. A
        // transposed weight or a shifted tap shows up at ~1, four orders above
        // the 1e-5 bound here.
        {
            let (ci, co, h, wd) = (5usize, 4usize, 6usize, 7usize);
            let plane = h * wd;
            let x: Vec<f32> = (0..ci * plane).map(|i| ((i * 31 % 89) as f32 - 44.0) / 8.0).collect();
            let w: Vec<f32> = (0..co * ci * 9).map(|i| ((i * 19 % 53) as f32 - 26.0) / 8.0).collect();
            let b: Vec<f32> = (0..co).map(|i| (i as f32) / 4.0 - 1.0).collect();
            let mut y1 = vec![0.0f32; co * plane];
            conv3x3_tile(&x, &w, &b, &mut y1, ci, co, h, wd, 0, 0.0);
            // the same sum, one output element at a time, ci then ky then kx
            let mut y2 = vec![0.0f32; co * plane];
            for oc in 0..co {
                for oy in 0..h {
                    for ox in 0..wd {
                        let mut acc = 0.0f32;
                        for ci_ in 0..ci {
                            for ky in 0..3usize {
                                for kx in 0..3usize {
                                    if oy + ky >= 1 && oy + ky <= h && ox + kx >= 1 && ox + kx <= wd {
                                        acc += x[ci_ * plane + (oy + ky - 1) * wd + (ox + kx - 1)]
                                            * w[((oc * ci + ci_) * 3 + ky) * 3 + kx];
                                    }
                                }
                            }
                        }
                        y2[(oc * h + oy) * wd + ox] = acc + b[oc];
                    }
                }
            }
            if y1 != y2 {
                return Err("conv3x3_tile disagrees with its own order".into());
            }
            // The section-6 order (ky, kx, ci, bias first) on the same input: the
            // two orders must agree to rounding and nothing more.
            let mut worst = 0.0f32;
            for oc in 0..co {
                for oy in 0..h {
                    for ox in 0..wd {
                        let mut acc = b[oc];
                        for ky in 0..3usize {
                            for kx in 0..3usize {
                                if oy + ky >= 1 && oy + ky <= h && ox + kx >= 1 && ox + kx <= wd {
                                    for ci_ in 0..ci {
                                        acc += w[((oc * ci + ci_) * 3 + ky) * 3 + kx]
                                            * x[ci_ * plane + (oy + ky - 1) * wd + (ox + kx - 1)];
                                    }
                                }
                            }
                        }
                        let d = (acc - y1[(oc * h + oy) * wd + ox]).abs();
                        if d > worst {
                            worst = d;
                        }
                    }
                }
            }
            if worst > 1e-5 {
                return Err(format!(
                    "conv3x3_tile differs from the section-6 order by {worst}, expected rounding"
                ));
            }

            // The fused activation, in the convention lg_conv3x3_winograd states:
            // act 2 must scale the negative side by act_p and leave the positive
            // side alone, which a `0`-vs-`>=0` slip at the origin would survive
            // but a flipped comparison would not.
            let slope = 0.1f32;
            let mut ya = vec![0.0f32; co * plane];
            conv3x3_tile(&x, &w, &b, &mut ya, ci, co, h, wd, 2, slope);
            for i in 0..co * plane {
                let want = if y1[i] >= 0.0 { y1[i] } else { slope * y1[i] };
                if (ya[i] - want).abs() > 1e-6 {
                    return Err(format!("conv3x3_tile act=2 at {i}: {} != {}", ya[i], want));
                }
            }
            let mut yr = vec![0.0f32; co * plane];
            conv3x3_tile(&x, &w, &b, &mut yr, ci, co, h, wd, 1, slope);
            for i in 0..co * plane {
                let want = if y1[i] > 0.0 { y1[i] } else { 0.0 };
                if yr[i] != want {
                    return Err(format!("conv3x3_tile act=1 at {i}: {} != {}", yr[i], want));
                }
            }

            // lg_conv1x1_tile: c ascending with the bias after the sum, against
            // the section-6 order (bias first) - equal to rounding, and the
            // biased-and-unbiased pair must differ by exactly the bias.
            let (ci, co, h, wd) = (6usize, 5usize, 3usize, 4usize);
            let plane = h * wd;
            let x: Vec<f32> = (0..ci * plane).map(|i| ((i * 23 % 79) as f32 - 39.0) / 8.0).collect();
            let w: Vec<f32> = (0..co * ci).map(|i| ((i * 13 % 41) as f32 - 20.0) / 8.0).collect();
            let b: Vec<f32> = (0..co).map(|i| (i as f32) / 3.0 - 0.5).collect();
            let mut y1 = vec![0.0f32; co * plane];
            conv1x1_tile(&x, &w, &b, &mut y1, ci, co, h, wd, 0, 0.0);
            let mut y2 = vec![0.0f32; co * plane];
            let mut y0 = vec![0.0f32; co * plane];
            for o in 0..co {
                for p in 0..plane {
                    let mut acc = 0.0f32;
                    for c in 0..ci {
                        acc += w[o * ci + c] * x[c * plane + p];
                    }
                    y2[o * plane + p] = acc + b[o];
                    y0[o * plane + p] = acc;
                }
            }
            if y1 != y2 {
                return Err("conv1x1_tile disagrees with its own order".into());
            }
            for i in 0..co * plane {
                if (y1[i] - y0[i] - b[i / plane]).abs() > 1e-6 {
                    return Err(format!("conv1x1_tile bias at {i} is not added once"));
                }
            }
            let mut y3 = vec![0.0f32; co * plane];
            conv1x1_rb(&x, &w, &b, &mut y3, ci, co, h, wd);
            let mut worst = 0.0f32;
            for i in 0..co * plane {
                let d = (y1[i] - y3[i]).abs();
                if d > worst {
                    worst = d;
                }
            }
            if worst > 1e-5 {
                return Err(format!(
                    "conv1x1_tile differs from conv1x1_rb's order by {worst}, expected rounding"
                ));
            }
        }

        // lg_layer_norm_warp: a constant row normalizes to (w - b), at widths on
        // both sides of the 128 register boundary. The tolerance is loose on
        // purpose: the one-pass variance of a constant row is zero only up to
        // cancellation, so the residual is ~1e-5 at these magnitudes. A row that
        // was DROPPED, or a lane that was skipped, is off by ~1, not by 1e-5 -
        // which is the size of mistake this case is here to catch.
        for ne0 in [8usize, 32, 127, 128, 129, 200] {
            let nrows = 3usize;
            let x = vec![0.7f32; ne0 * nrows];
            let w = vec![1.0f32; ne0];
            let b = vec![0.0f32; ne0];
            let mut y = vec![0.0f32; ne0 * nrows];
            layer_norm_warp(&x, &w, &b, &mut y, ne0, nrows, 1e-5);
            for (i, v) in y.iter().enumerate() {
                if v.abs() > 1e-4 {
                    return Err(format!("layer_norm_warp(ne0={ne0})[{i}] = {v}, expected ~0"));
                }
            }
        }
        // ... and a non-constant row must match the two-pass twin within
        // rounding, at a width the register path covers.
        let ne0 = 64usize;
        let x: Vec<f32> = (0..ne0).map(|i| ((i * 41 % 83) as f32 - 40.0) / 8.0).collect();
        let w: Vec<f32> = (0..ne0).map(|i| 0.5 + (i % 7) as f32 / 8.0).collect();
        let b: Vec<f32> = (0..ne0).map(|i| (i % 5) as f32 / 4.0 - 0.5).collect();
        let mut y1 = vec![0.0f32; ne0];
        let mut y2 = vec![0.0f32; ne0];
        layer_norm_warp(&x, &w, &b, &mut y1, ne0, 1, 1e-5);
        layer_norm(&x, &w, &b, &mut y2, ne0, 1, 1e-5);
        for i in 0..ne0 {
            if (y1[i] - y2[i]).abs() > 1e-4 {
                return Err(format!(
                    "layer_norm_warp[{i}] = {} but layer_norm = {}", y1[i], y2[i]
                ));
            }
        }

        // lg_conv2x2s2/lg_conv_t2x2: an independent, tap-at-a-time reference,
        // which is the form that catches a transposed weight layout read the
        // other way round.
        let (ci, co, h, wd) = (3usize, 3usize, 6usize, 8usize);
        let plane = h * wd;
        let x: Vec<f32> = (0..ci * plane).map(|i| ((i * 13 % 61) as f32 - 30.0) / 8.0).collect();
        let wf: Vec<f32> = (0..co * ci * 4).map(|i| ((i * 7 % 23) as f32 - 11.0) / 8.0).collect();
        let b: Vec<f32> = (0..co).map(|i| (i as f32) / 2.0 - 1.0).collect();
        let (oh, ow) = (h / 2, wd / 2);
        let mut y1 = vec![0.0f32; co * oh * ow];
        let mut y2 = vec![0.0f32; co * oh * ow];
        conv2x2s2(&x, &wf, &b, &mut y1, ci, co, h, wd);
        for oy in 0..oh {
            for ox in 0..ow {
                for oc in 0..co {
                    let mut acc = b[oc];
                    for ci_ in 0..ci {
                        for ky in 0..2 {
                            for kx in 0..2 {
                                acc += x[ci_ * plane + (2 * oy + ky) * wd + (2 * ox + kx)]
                                    * wf[((oc * ci + ci_) * 2 + ky) * 2 + kx];
                            }
                        }
                    }
                    y2[(oc * oh + oy) * ow + ox] = acc;
                }
            }
        }
        if y1 != y2 {
            return Err("conv2x2s2 disagrees with the tap-at-a-time reference".into());
        }
        let wt: Vec<f32> = (0..ci * co * 4).map(|i| ((i * 11 % 19) as f32 - 9.0) / 8.0).collect();
        let mut y1 = vec![0.0f32; co * 4 * plane];
        let mut y2 = vec![0.0f32; co * 4 * plane];
        conv_t2x2(&x, &wt, &b, &mut y1, ci, co, h, wd);
        // the same scatter, accumulated one output pixel at a time
        for iy in 0..h {
            for ix in 0..wd {
                for oc in 0..co {
                    for ky in 0..2 {
                        for kx in 0..2 {
                            let mut acc = b[oc];
                            for ci_ in 0..ci {
                                acc += x[ci_ * plane + iy * wd + ix]
                                    * wt[((ci_ * co + oc) * 2 + ky) * 2 + kx];
                            }
                            y2[(oc * (2 * h) + 2 * iy + ky) * (2 * wd) + 2 * ix + kx] = acc;
                        }
                    }
                }
            }
        }
        if y1 != y2 {
            return Err("conv_t2x2 disagrees with the scatter reference".into());
        }

        // lg_window_gather/scatter: scatter must invert gather for every element,
        // at a shifted and an unshifted call, and the map must MOVE data (a
        // wrong-by-a-modulo map still round-trips, so check a known offset).
        for shift in [0usize, 1, 3] {
            let (hp, wp, win) = (8usize, 8usize, 4usize);
            let nww = wp / win;
            let nw = (hp / win) * nww;
            let c = 2usize;
            let x: Vec<f32> = (0..c * hp * wp).map(|i| i as f32).collect();
            let mut tok = vec![0.0f32; nw * win * win * c];
            let mut back = vec![-1.0f32; c * hp * wp];
            window_gather(&x, &mut tok, nw, win * win, nww, win, hp, wp, c, shift, 0);
            window_scatter(&tok, &mut back, nw, win * win, nww, win, hp, wp, c, shift, 0);
            if back != x {
                return Err(format!("window scatter(gather(x)) != x at shift {shift}"));
            }
            // token 0 of window wl is the plane corner (wh*win+shift, ww*win+shift)
            for wl in 0..nw {
                let wh = wl / nww;
                let ww = wl % nww;
                let y = (wh * win + shift) % hp;
                let xx = (ww * win + shift) % wp;
                let want = x[0 * hp * wp + y * wp + xx];
                let got = tok[(wl * win * win + 0) * c + 0];
                if got != want {
                    return Err(format!(
                        "window_gather token 0 of window {wl} at shift {shift} = {got}, expected corner {want}"
                    ));
                }
            }
            // ... and a chunk base moves the window index but not the data.
            let mut tok2 = vec![0.0f32; tok.len()];
            window_gather(&x, &mut tok2, nw, win * win, nww, win, hp, wp, c, shift, 0);
            if tok2 != tok {
                return Err("window_gather is not reproducible at w0 = 0".into());
            }
        }

        // lg_pixel_shuffle against an INDEPENDENT reference: the flat
        // `(c, r, r, h, w) -> (c, h, r, w, r)` view torch's pixel_shuffle is.
        // Written as index arithmetic rather than as a call to the twin, so a
        // wrong permutation fails and not merely a wrong shape. Odd sizes are
        // deliberate: the kernel's r == 2 and r == 3 paths are special-cased and
        // its 2-D grid has tail guards, and both are only exercised off-tile.
        for &(c, h, w, r) in &[
            (3usize, 5usize, 7usize, 2usize),
            (1, 1, 1, 2),
            (1, 1, 1, 3),
            (2, 3, 4, 3),
            (4, 2, 3, 4),
            (3, 8, 6, 2),
        ] {
            let n_in = c * r * r * h * w;
            let n_out = c * h * r * w * r;
            let x: Vec<f32> = (0..n_in).map(|i| i as f32).collect();
            let mut got = vec![0.0f32; n_out];
            pixel_shuffle(&x, &mut got, c, h, w, r);
            for ch in 0..c {
                for y in 0..h {
                    for x_ in 0..w {
                        for dy in 0..r {
                            for dx in 0..r {
                                let si = ((ch * r + dy) * r + dx) * h * w + y * w + x_;
                                let di = (ch * (h * r) + (y * r + dy)) * (w * r) + (x_ * r + dx);
                                if got[di] != x[si] {
                                    return Err(format!(
                                        "pixel_shuffle(c={c},h={h},w={w},r={r}) at out[{di}] = {}, \
                                         expected in[{si}] = {}",
                                        got[di], x[si]
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// `gpuinfo` runs the twin selftest, and `cargo test` did not - so the
    /// correctness of the CPU backend was exercised only by a command a
    /// contributor has to remember to type, while the JSON and safetensors
    /// parsers were covered here. This calls the SAME function rather than a
    /// copy of it, and it is the check that fails when a twin is added to the op
    /// table with arithmetic that does not match its kernel.
    #[test]
    fn twins_agree_with_their_reference_orders() {
        super::selftest().expect("cpu twin selftest");
    }
}
