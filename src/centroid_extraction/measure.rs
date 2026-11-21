//! Gaussian-windowed centroid: the position, flux and shape of one star,
//! measured on the raw pixels.
//!
//! The estimate is the fixed point of
//!
//! ```text
//! x ← x + g · Σ wᵢ dᵢ (xᵢ − x) / Σ wᵢ dᵢ ,   wᵢ = exp(−|rᵢ − r|² / 2σw²)
//! ```
//!
//! over the background-subtracted pixels `dᵢ` — the position where the
//! image's correlation with a Gaussian of width `σw` is stationary, i.e.
//! the maximum of the matched filter evaluated in the continuum rather than
//! on the pixel grid. For a Gaussian PSF of that width under uniform
//! (background-limited) noise this is the maximum-likelihood position, and
//! unlike a center of mass over a thresholded footprint it has no footprint:
//! the window is the same for a faint star and a bright one, and a constant
//! background error cancels by symmetry.

/// Variance of a unit pixel's box profile along one axis.
const PIXEL_VARIANCE: f64 = 1.0 / 12.0;

/// Largest window side in pixels (`psf_sigma ≤ 10` → radius ≤ 31).
const WIN_MAX: usize = 64;

/// Gaussian measurement window: its radius and the integer-offset weight
/// table `exp(−j²/2σ²)`, `j = −r..=r`.
#[derive(Clone)]
pub(super) struct Window {
    /// σw².
    pub sigma2: f64,
    /// Radius in pixels; the window is `(2r + 1)²`.
    pub r: isize,
    table: [f32; WIN_MAX],
}

impl Window {
    /// A window of width `sigma` pixels (floored at 0.5 — below that the
    /// weights fall off faster than the pixel grid can sample), truncated
    /// at 3σ.
    pub(super) fn new(sigma: f32) -> Self {
        let sigma = (sigma as f64).max(0.5);
        let r = ((3.0 * sigma).ceil() as isize).clamp(2, WIN_MAX as isize / 2 - 1);
        let mut table = [0.0_f32; WIN_MAX];
        for j in -r..=r {
            table[(j + r) as usize] = (-((j * j) as f64) / (2.0 * sigma * sigma)).exp() as f32;
        }
        Self {
            sigma2: sigma * sigma,
            r,
            table,
        }
    }

    /// Weights `exp(−(j − f)²/2σ²)` for the integer offsets `j = −r..=r`
    /// from the window's center pixel and a sub-pixel shift `f`, into
    /// `out[..2r + 1]`. `(j − f)² = j² − 2jf + f²`, so each weight is the
    /// table entry times `qʲ·exp(−f²/2σ²)` with `q = exp(f/σ²)`: two `exp`
    /// calls and a running product instead of one `exp` per tap.
    #[inline]
    fn weights(&self, f: f64, out: &mut [f32; WIN_MAX]) {
        let r = self.r as usize;
        let c = (-f * f / (2.0 * self.sigma2)).exp();
        let q = (f / self.sigma2).exp();
        let qi = 1.0 / q;
        let (mut up, mut down) = (c, c);
        out[r] = c as f32 * self.table[r];
        for j in 1..=r {
            up *= q;
            down *= qi;
            out[r + j] = up as f32 * self.table[r + j];
            out[r - j] = down as f32 * self.table[r - j];
        }
    }
}

/// One windowed measurement.
#[derive(Clone, Copy, Debug)]
pub(super) struct Measurement {
    /// Position in raw pixel coordinates (pixel centers at integers).
    pub x: f64,
    pub y: f64,
    /// PSF-weighted flux: the total flux of a Gaussian of the window's
    /// width fitted to the pixels.
    pub flux: f64,
    /// Variance of `flux`, and of the position along x and y, **per unit
    /// of raw-pixel noise variance** (multiply by σ² of the pixel noise).
    pub flux_var: f64,
    pub var_x: f64,
    pub var_y: f64,
    /// Windowed central second moments `[xx, yy, xy]`, px².
    pub moments: [f64; 3],
    /// Gaussian amplitude and constant pedestal of a two-parameter fit over
    /// the window (the zero-sum matched filter), and the amplitude's
    /// variance per unit pixel-noise variance.
    pub amp: f64,
    pub pedestal: f64,
    pub amp_var: f64,
    /// Whether the iteration settled (step below 2·10⁻³ px).
    pub converged: bool,
}

impl Measurement {
    /// Mean windowed variance per axis — `σw²/2` for a PSF matching the
    /// window, near zero for a single hot pixel, approaching `σw²` for an
    /// object much broader than the window.
    #[inline]
    pub(super) fn mean_var(&self) -> f64 {
        0.5 * (self.moments[0] + self.moments[1])
    }
}

/// Iterative Gaussian-windowed centroid of `raw − bg` (a `w × h` row-major
/// image), started at `(x0, y0)`.
///
/// Returns `None` when the window does not fit inside the frame (a
/// truncated window biases the estimate toward the interior), the weighted
/// flux is not positive, or the estimate drifts more than `max_drift`
/// pixels from its start along either axis (the start was not a peak of its
/// own). Non-finite pixels count as background.
pub(super) fn measure(
    raw: &[f32],
    (w, h): (usize, usize),
    bg: f32,
    (x0, y0): (f64, f64),
    win: &Window,
    max_drift: f64,
) -> Option<Measurement> {
    const MAX_ITER: usize = 10;
    let (mut x, mut y) = (x0, y0);
    let r = win.r;
    let n1 = (2 * r + 1) as usize;
    let mut wx = [0.0_f32; WIN_MAX];
    let mut wy = [0.0_f32; WIN_MAX];
    for iter in 0..MAX_ITER {
        let (xc, yc) = (x.round(), y.round());
        let (c0, r0) = (xc as isize - r, yc as isize - r);
        if c0 < 0 || r0 < 0 || c0 + n1 as isize > w as isize || r0 + n1 as isize > h as isize {
            return None;
        }
        let (c0, r0) = (c0 as usize, r0 as usize);
        let (fx, fy) = (x - xc, y - yc);
        win.weights(fx, &mut wx);
        win.weights(fy, &mut wy);

        // Accumulate down the columns first: every row updates four
        // per-column sums element-wise (no reduction inside the loop, so it
        // vectorizes), and the x weights are applied once at the end.
        // `i`/`j` are offsets from the window's first row/column, so the
        // moments are about that corner.
        let mut c0s = [0.0_f32; WIN_MAX]; // Σ wy·d
        let mut c1s = [0.0_f32; WIN_MAX]; // Σ wy·i·d
        let mut c2s = [0.0_f32; WIN_MAX]; // Σ wy·i²·d
        let mut cds = [0.0_f32; WIN_MAX]; // Σ d
        for i in 0..n1 {
            let row = &raw[(r0 + i) * w + c0..(r0 + i) * w + c0 + n1];
            let (wy0, yi) = (wy[i], i as f32);
            let (wy1, wy2) = (wy0 * yi, wy0 * yi * yi);
            for (j, &p) in row.iter().enumerate() {
                let d = if p.is_finite() { p - bg } else { 0.0 };
                cds[j] += d;
                c0s[j] += wy0 * d;
                c1s[j] += wy1 * d;
                c2s[j] += wy2 * d;
            }
        }
        let (mut s0, mut sx, mut sy, mut sxx, mut syy, mut sxy, mut sd) = (
            0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64,
        );
        for j in 0..n1 {
            let (wj, xj) = (wx[j] as f64, j as f64);
            let (a0, a1, a2) = (wj * c0s[j] as f64, wj * c1s[j] as f64, wj * c2s[j] as f64);
            sd += cds[j] as f64;
            s0 += a0;
            sx += a0 * xj;
            sxx += a0 * xj * xj;
            sy += a1;
            sxy += a1 * xj;
            syy += a2;
        }
        if s0 <= 0.0 || !s0.is_finite() {
            return None;
        }
        // Windowed mean relative to the current estimate, which sits at
        // offset (r + f) from the window corner.
        let (ox, oy) = (r as f64 + fx, r as f64 + fy);
        let (ax, ay) = (sx / s0, sy / s0);
        let (mx, my) = (ax - ox, ay - oy);
        let moments = [sxx / s0 - ax * ax, syy / s0 - ay * ay, sxy / s0 - ax * ay];
        // For a Gaussian object of variance σo² under a window σw², the
        // windowed mean offset is (true offset)·σw²/(σw² + σo²) and the
        // windowed variance is m = σo²σw²/(σo² + σw²), so the exact step is
        // the mean offset times 1/(1 − m/σw²): 2 for a matched PSF, 1 for a
        // hot pixel. Clamped to [1, 3].
        let m = (0.5 * (moments[0] + moments[1])).clamp(0.0, 2.0 / 3.0 * win.sigma2);
        let gain = 1.0 / (1.0 - m / win.sigma2);
        let (stx, sty) = (gain * mx, gain * my);
        let converged = stx * stx + sty * sty < 4e-6;
        if converged || iter + 1 == MAX_ITER {
            // Sums of the (separable) weights and of the noise terms.
            let sums = |v: &[f32; WIN_MAX], o: f64| {
                let (mut s1, mut s2, mut s2d) = (0.0_f64, 0.0_f64, 0.0_f64);
                for (j, &a) in v[..n1].iter().enumerate() {
                    let a = a as f64;
                    let d = j as f64 - o;
                    s1 += a;
                    s2 += a * a;
                    s2d += a * a * d * d;
                }
                (s1, s2, s2d)
            };
            let (swx, swx2, swx2d) = sums(&wx, ox);
            let (swy, swy2, swy2d) = sums(&wy, oy);
            let (sw, sw2) = (swx * swy, swx2 * swy2);
            let n = (n1 * n1) as f64;
            let wbar = sw / n;
            let denom = (sw2 - n * wbar * wbar).max(1e-12);
            let amp = (s0 - wbar * sd) / denom;
            let g2 = gain * gain / (s0 * s0);
            return Some(Measurement {
                x: x + stx,
                y: y + sty,
                flux: s0 * sw / sw2,
                flux_var: sw * sw / sw2,
                var_x: g2 * swx2d * swy2,
                var_y: g2 * swy2d * swx2,
                moments,
                amp,
                pedestal: (sd - amp * sw) / n,
                amp_var: 1.0 / denom,
                converged,
            });
        }
        x += stx;
        y += sty;
        if (x - x0).abs() > max_drift || (y - y0).abs() > max_drift {
            return None;
        }
    }
    None
}

/// Undo the window's rounding of a measured shape.
///
/// A Gaussian object with covariance `C` seen through a Gaussian window of
/// variance `σw²` measures `(C⁻¹ + I/σw²)⁻¹`, so each eigenvalue `λ` of the
/// windowed moments maps back to `λ / (1 − λ/σw²)`. Returns the unwindowed
/// moments `[xx, yy, xy]` and the elongation `√(λ_max/λ_min)`.
///
/// The result is clamped on both sides. Nothing imaged through square
/// pixels is narrower than a pixel's own variance, 1/12 px², so each
/// eigenvalue is floored there: a hot pixel comes out round rather than
/// with an elongation made of noise, while a one-pixel-wide streak is still
/// long. And `1 − λ/σw² ≥ 0.05`, so an over-broad measurement gives a large
/// finite shape rather than a negative one.
pub(super) fn unwindowed_shape(moments: [f64; 3], sigma2: f64) -> ([f64; 3], f64) {
    let [vxx, vyy, vxy] = moments;
    let tr = vxx + vyy;
    let det = vxx * vyy - vxy * vxy;
    let disc = (tr * tr - 4.0 * det).max(0.0).sqrt();
    let unwin = |l: f64| {
        let l = l.max(0.0);
        (l / (1.0 - l / sigma2).max(0.05)).max(PIXEL_VARIANCE)
    };
    let (l1, l2) = (unwin(0.5 * (tr + disc)), unwin(0.5 * (tr - disc)));
    // Principal axis of the windowed moments (unchanged by the mapping).
    let theta = 0.5 * (2.0 * vxy).atan2(vxx - vyy);
    let (sn, cs) = theta.sin_cos();
    let shape = [
        l1 * cs * cs + l2 * sn * sn,
        l1 * sn * sn + l2 * cs * cs,
        (l1 - l2) * sn * cs,
    ];
    (shape, (l1 / l2).sqrt())
}

/// Elongation `√(λ_max/λ_min)` of plain second moments `[xx, yy, xy]`;
/// `λ_min` is floored so a degenerate (one-pixel-wide) streak comes out
/// very elongated rather than dividing by zero.
pub(super) fn elongation(moments: [f64; 3]) -> f64 {
    let [cxx, cyy, cxy] = moments;
    let tr = cxx + cyy;
    let det = cxx * cyy - cxy * cxy;
    let disc = (tr * tr - 4.0 * det).max(0.0).sqrt();
    ((tr + disc) / (tr - disc).max(1e-12)).sqrt()
}
