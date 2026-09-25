//! CPU twins of the kernels: same names, same arithmetic, plain Rust.
//!
//! These are the CPU backend, not a test harness: this is the path an engine
//! runs when there is no GPU, and it is held to the same performance standard as
//! the kernels it mirrors. Keeping the *order* of operations the GPU kernels use,
//! wherever that is cheap, is what makes a backend mismatch mean a real
//! difference rather than a rounding one; it is not a reason to leave the CPU
//! side slow. The few places where the two deliberately differ (erf, layer-norm
//! variance) are documented at the kernel instead.

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

/// The toolkit's own smoke test: every twin must run and agree with a
/// straightforward reimplementation on a fixed input. Called by `gpuinfo`.
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
    Ok(())
}
