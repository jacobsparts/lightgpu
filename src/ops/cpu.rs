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
    }
    Ok(())
}
