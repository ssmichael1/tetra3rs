//! Distortion model fitting from solve results.
//!
//! Given one or more plate-solve results (each with matched catalog star IDs and
//! centroid indices), this module fits distortion models by comparing observed
//! centroid positions to their ideal (pinhole-projected) positions computed from
//! the catalog.
//!
//! The fitting is iterative with sigma-clipping to reject mismatched stars.

use std::collections::HashMap;

use numeris::optim::{least_squares_lm_dyn, LmSettings};
use numeris::{DynMatrix, DynVector, Matrix3};
use tracing::debug;

use crate::centroid::Centroid;
use crate::solver::solve::StarVectors;
use crate::solver::SolverDatabase;
use crate::stats::median_mad_sigma;

use super::calibrate::DistortionModelType;
use super::polynomial::{eval_poly, num_coeffs, term_pairs, PolynomialDistortion};
use super::radial::{brown_conrady_forward, RadialDistortion};
use super::Distortion;

/// Minimum matched points for a Brown-Conrady radial fit. The LM solves for 8
/// parameters (cx, cy, γ, k1–k3, p1, p2), so fewer points is under-determined —
/// [`run_intrinsics_lm`] bails below this and callers must not treat the
/// resulting warm-start as a real fit.
pub(super) const MIN_RADIAL_POINTS: usize = 8;

/// Configuration for distortion fitting.
#[derive(Debug, Clone)]
pub struct DistortionFitConfig {
    /// Sigma threshold for iterative outlier rejection: a point is an inlier
    /// when its residual is within `median + sigma_clip · σ`, where both the
    /// median and the MAD-derived σ are taken over **all** points' residuals
    /// under the current model (robust to < 50% contamination). Estimating σ
    /// over the shrinking inlier set instead ratchets the threshold down on
    /// heteroscedastic data (faint stars scatter more; each refit tightens
    /// on the bright ones) until the mask collapses. Default 3.0.
    pub sigma_clip: f64,
    /// Maximum iterations for iterative fitting. Default 20.
    pub max_iterations: u32,
}

impl Default for DistortionFitConfig {
    fn default() -> Self {
        Self {
            sigma_clip: 3.0,
            max_iterations: 20,
        }
    }
}

// ── Data structures for internal use ────────────────────────────────────────

/// A single matched observation: observed centroid pixel position + ideal (projected) position.
pub(super) struct MatchedPoint {
    /// Observed centroid x (distorted), pixels from image center.
    pub x_obs: f64,
    /// Observed centroid y (distorted), pixels from image center.
    pub y_obs: f64,
    /// Ideal projected x (undistorted pinhole model), pixels from image center.
    pub x_ideal: f64,
    /// Ideal projected y (undistorted pinhole model), pixels from image center.
    pub y_ideal: f64,
}

/// Outcome of [`fit_pooled`]: a sigma-clipped distortion fit over a pooled
/// set of matched points.
///
/// Note there is no jointly-fit CRPIX here: polynomial fits absorb the optical
/// center into their order-0 terms (moved into `CameraModel::crpix` later by
/// `extract_crpix`), and radial fits carry the optical-axis position inside the
/// model ([`RadialDistortion::center`]). Both leave the projection origin
/// (`crpix`) at the image center for the solver's geometry.
pub(super) struct PooledFit {
    /// The fitted distortion model.
    pub model: Distortion,
    /// Correction factor for the anchor focal length (the focal length
    /// implied by the solve FOV used to project the ideal points):
    /// `f_true = focal_scale · f_anchor`. Always `1.0` for polynomial fits;
    /// for radial fits this carries the jointly-fit linear scale term (the
    /// Brown-Conrady model itself has no linear degree of freedom).
    pub focal_scale: f64,
    /// Inlier mask over the input points.
    pub mask: Vec<bool>,
    /// Number of sigma-clip iterations performed.
    pub iterations: u32,
    /// RMS residual in pixels BEFORE distortion correction (inliers only).
    pub rmse_before_px: f64,
    /// RMS residual in pixels AFTER distortion correction (inliers only).
    pub rmse_after_px: f64,
}

impl PooledFit {
    pub fn n_inliers(&self) -> usize {
        self.mask.iter().filter(|&&m| m).count()
    }
}

/// Minimum matched points for `model` to be determined at all.
pub(super) fn min_points(model: DistortionModelType) -> usize {
    match model {
        DistortionModelType::Polynomial { order } => num_coeffs(order),
        DistortionModelType::Radial => MIN_RADIAL_POINTS,
    }
}

/// Fit `model` to pooled matched points with iterative sigma-clipping,
/// reporting before/after RMSE on the same final inlier set. `scale` is the
/// polynomial normalization (typically `image_width / 2`); ignored for radial.
///
/// Returns `None` when there are fewer points than [`min_points`] — below
/// that the fit is under-determined and the LM/LS would hand back its
/// (identity) warm start masquerading as a real fit. Shared by the
/// single-image and multi-image calibration paths.
pub(super) fn fit_pooled(
    points: &[MatchedPoint],
    model: DistortionModelType,
    scale: f64,
    config: &DistortionFitConfig,
) -> Option<PooledFit> {
    if points.len() < min_points(model) {
        return None;
    }
    let fit = match model {
        DistortionModelType::Polynomial { order } => {
            let fit = fit_polynomial_sigma_clip(points, order, scale, config);
            let dist = Distortion::Polynomial(PolynomialDistortion::new(
                order,
                scale,
                fit.a_coeffs,
                fit.b_coeffs,
            ));
            let rmse_after_px = compute_corrected_rmse(points, &fit.mask, &dist);
            PooledFit {
                model: dist,
                focal_scale: 1.0,
                mask: fit.mask,
                iterations: fit.iterations,
                rmse_before_px: 0.0,
                rmse_after_px,
            }
        }
        DistortionModelType::Radial => {
            let fit = fit_radial_centered_sigma_clip(points, config);
            // Residuals under the raw fit (including γ) — identical to the
            // rescaled model evaluated in the corrected frame.
            let residuals = intrinsics_residuals(
                points,
                &[
                    fit.cx, fit.cy, fit.gamma, fit.k1, fit.k2, fit.k3, fit.p1, fit.p2,
                ],
            );
            let rmse_after_px = masked_rms(&residuals, &fit.mask);
            debug!(
                "Brown-Conrady fit: cx={:.2}, cy={:.2}, gamma={:.6}, k1={:.3e}, k2={:.3e}, k3={:.3e}, p1={:.3e}, p2={:.3e}",
                fit.cx, fit.cy, fit.gamma, fit.k1, fit.k2, fit.k3, fit.p1, fit.p2,
            );
            PooledFit {
                model: Distortion::Radial(fit.rescaled_model()),
                focal_scale: fit.gamma,
                mask: fit.mask,
                iterations: fit.iterations,
                rmse_before_px: 0.0,
                rmse_after_px,
            }
        }
    };
    // Raw (uncorrected) obs-vs-ideal RMS on the SAME inlier set, so the
    // before/after pair is a fair comparison.
    let rmse_before_px = compute_corrected_rmse(points, &fit.mask, &Distortion::None);
    debug!(
        "{:?} fit: inliers={}/{}, RMSE {:.3} → {:.3} px",
        model,
        fit.n_inliers(),
        points.len(),
        rmse_before_px,
        fit.rmse_after_px
    );
    Some(PooledFit {
        rmse_before_px,
        ..fit
    })
}

// ── Radial distortion fitting ───────────────────────────────────────────────

/// Result of a sigma-clipped camera-intrinsics fit: optical center,
/// focal-scale factor, and Brown-Conrady distortion (radial + tangential).
pub(super) struct CenteredRadialFitResult {
    /// Optical-center offset in pixels, in the geometric (no-crpix) frame.
    pub cx: f64,
    pub cy: f64,
    /// Focal-length correction factor: `f_true = gamma · f_anchor`, where
    /// `f_anchor` is the focal length used to project the ideal points.
    /// `f_anchor` derives from the solve FOV — a whole-field average biased
    /// by the very distortion being fit (≈1% on TESS) — and Brown-Conrady
    /// has no linear term, so without this degree of freedom the anchor
    /// bias becomes a linear residual (tens of px at the field corner) that
    /// the cubic+ terms can only mimic.
    pub gamma: f64,
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    /// Tangential / decentering coefficients.
    pub p1: f64,
    pub p2: f64,
    pub mask: Vec<bool>,
    pub iterations: u32,
}

impl CenteredRadialFitResult {
    /// The fitted distortion re-expressed in the corrected ideal frame
    /// (focal length × `gamma`): `kᵢ′ = kᵢ/γ^2i`, `p′ = p/γ`, carrying the
    /// fitted optical-axis position as the model's own `center`. Up to a
    /// small constant offset `(γ−1)·c` (absorbed by attitude at solve
    /// time), predicted observed pixels are unchanged.
    pub(super) fn rescaled_model(&self) -> RadialDistortion {
        let g = self.gamma;
        let g2 = g * g;
        RadialDistortion::with_center(
            self.cx,
            self.cy,
            self.k1 / g2,
            self.k2 / (g2 * g2),
            self.k3 / (g2 * g2 * g2),
            self.p1 / g,
            self.p2 / g,
        )
    }
}

/// Joint nonlinear LS fit of the standard camera-intrinsics model — optical
/// center `(cx, cy)`, focal-scale factor `γ`, and Brown-Conrady distortion
/// `(k1, k2, k3, p1, p2)` — via Levenberg-Marquardt with MAD-based
/// sigma-clipping. This is the parameter set OpenCV's `calibrateCamera`
/// fits (free principal point, free focal length, 5-coefficient
/// Brown-Conrady), restricted to square pixels and no skew. Reusable across
/// single-image and multi-image calibration paths.
///
/// The forward model, centered on the fitted optical axis:
///
/// ```text
///     x_n = x_ideal − cx,   y_n = y_ideal − cy
///     r²  = x_n² + y_n²
///     rad = 1 + k1·r² + k2·r⁴ + k3·r⁶
///     x_obs = cx + γ·(x_n·rad + 2·p1·x_n·y_n + p2·(r² + 2·x_n²))
///     y_obs = cy + γ·(y_n·rad + p1·(r² + 2·y_n²) + 2·p2·x_n·y_n)
/// ```
///
/// `(cx, cy)` is deliberately NOT pulled toward the image center: on
/// multi-detector mosaics (TESS, Kepler, Rubin, …) the camera's optical
/// axis lies far off any single detector's center — near a corner for a
/// TESS CCD — and that is where the radial distortion is physically
/// centered. Only a tiny tie-breaking prior on `(cx, cy, p1, p2)`
/// (negligible against any real signal) picks the centered representative
/// when the data cannot distinguish — e.g. a distortion-free pinhole,
/// where the optical center is unidentifiable (a center shift `δc·k1` is
/// first-order indistinguishable from a tangential `p2` term).
///
/// LM is delegated to [`numeris::optim::least_squares_lm_dyn`]. The outer
/// loop performs MAD-based sigma-clipping: after each LM convergence,
/// re-mask inliers based on the residual distribution and re-call LM.
///
/// Warm-starts `(k1, k2, k3)` from a non-centered linear fit
/// ([`fit_radial_ls`]); `γ` starts at 1 and `(cx, cy, p1, p2)` at 0.
pub(super) fn fit_radial_centered_sigma_clip(
    points: &[MatchedPoint],
    config: &DistortionFitConfig,
) -> CenteredRadialFitResult {
    // Normalize coordinates so the field radius is ~1. Without this the
    // Jacobian columns span ~20 orders of magnitude (the γ column is O(r),
    // the k3 column O(r⁷)) and LM hard-fails or converges to wrong local
    // minima at real-camera magnitudes. Dimensionless parameters: ĉ = c/L,
    // k̂ᵢ = kᵢ·L^2i, p̂ = p·L; γ is already dimensionless.
    let norm = points
        .iter()
        .map(|p| p.x_ideal.hypot(p.y_ideal))
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let npoints: Vec<MatchedPoint> = points
        .iter()
        .map(|p| MatchedPoint {
            x_obs: p.x_obs / norm,
            y_obs: p.y_obs / norm,
            x_ideal: p.x_ideal / norm,
            y_ideal: p.y_ideal / norm,
        })
        .collect();

    // Warm-start `k`s from a quick non-centered linear fit.
    let initial_mask = vec![true; npoints.len()];
    let (k1_init, k2_init, k3_init) = fit_radial_ls(&npoints, &initial_mask);
    let mut x = DynVector::<f64>::from_vec(vec![
        0.0,     // cx
        0.0,     // cy
        1.0,     // gamma
        k1_init, // k1
        k2_init, // k2
        k3_init, // k3
        0.0,     // p1
        0.0,     // p2
    ]);
    let mut mask = initial_mask;
    let mut total_lm_iters = 0u32;
    let mut scratch: Vec<f64> = Vec::with_capacity(npoints.len());

    // Outer sigma-clip iterations
    for _outer in 0..config.max_iterations {
        if mask.iter().filter(|&&m| m).count() < MIN_RADIAL_POINTS {
            break;
        }
        let prev_x = x.clone();

        match run_intrinsics_lm(&npoints, &mask, &x) {
            Ok((new_x, iters)) => {
                x = new_x;
                total_lm_iters += iters;
            }
            Err(()) => break,
        }

        // Sigma-clip on residuals at current params. Median and MAD-σ are
        // taken over ALL points (not the shrinking inlier set), so the
        // threshold tracks the model rather than ratcheting down with each
        // refit; the residual magnitudes are non-negative (Rayleigh-like),
        // hence the median offset. See `DistortionFitConfig::sigma_clip`.
        let residuals = intrinsics_residuals(&npoints, x.as_slice());
        scratch.clear();
        scratch.extend_from_slice(&residuals);
        let (median, sigma) = median_mad_sigma(&mut scratch);
        if sigma < 1e-12 / norm {
            break;
        }
        let threshold = median + config.sigma_clip * sigma;
        let new_mask: Vec<bool> = residuals.iter().map(|&r| r <= threshold).collect();
        if new_mask.iter().filter(|&&m| m).count() < MIN_RADIAL_POINTS {
            // Keep the previous (usable) mask rather than commit a degenerate one.
            break;
        }
        let mask_changed = mask.iter().zip(&new_mask).any(|(&a, &b)| a != b);
        mask = new_mask;
        let params_changed = (0..8).any(|i| (x[i] - prev_x[i]).abs() > 1e-12);
        if !mask_changed && !params_changed {
            break;
        }
    }

    // Denormalize back to pixel units.
    let n2 = norm * norm;
    CenteredRadialFitResult {
        cx: x[0] * norm,
        cy: x[1] * norm,
        gamma: x[2],
        k1: x[3] / n2,
        k2: x[4] / (n2 * n2),
        k3: x[5] / (n2 * n2 * n2),
        p1: x[6] / norm,
        p2: x[7] / norm,
        mask,
        iterations: total_lm_iters,
    }
}

/// Predicted observed position under the intrinsics model, for one point.
/// `params` = `[cx, cy, gamma, k1, k2, k3, p1, p2]` (all in the same units
/// as the point coordinates). The Brown-Conrady evaluation itself lives in
/// [`brown_conrady_forward`] — this only adds the optical-center shift and
/// the jointly-fit focal-scale factor γ.
fn intrinsics_predict(p: &MatchedPoint, params: &[f64]) -> (f64, f64) {
    let (cx, cy, gamma) = (params[0], params[1], params[2]);
    let (k1, k2, k3, p1, p2) = (params[3], params[4], params[5], params[6], params[7]);
    let e = brown_conrady_forward(k1, k2, k3, p1, p2, p.x_ideal - cx, p.y_ideal - cy);
    (cx + gamma * e.fx, cy + gamma * e.fy)
}

/// Per-point Euclidean residual under the intrinsics model.
/// `params` as in [`intrinsics_predict`].
pub(super) fn intrinsics_residuals(points: &[MatchedPoint], params: &[f64]) -> Vec<f64> {
    points
        .iter()
        .map(|p| {
            let (px, py) = intrinsics_predict(p, params);
            (p.x_obs - px).hypot(p.y_obs - py)
        })
        .collect()
}

/// Single Levenberg-Marquardt run on the intrinsics model, dispatched to
/// [`numeris::optim::least_squares_lm_dyn`].
///
/// The residual vector has length `2·N_inliers + 4`: two rows per inlier
/// point (x and y) plus four tie-breaking rows `√μ·(cx, cy, p1, p2)`. The
/// tie-break weight is far below any real data term; it only selects the
/// centered, radial-only representative when the data leaves the
/// `(cx, cy) ↔ (p1, p2)` ridge exactly degenerate (see
/// [`fit_radial_centered_sigma_clip`]). Returns `Err(())` if there aren't
/// enough inliers or if LM fails.
fn run_intrinsics_lm(
    points: &[MatchedPoint],
    mask: &[bool],
    x0: &DynVector<f64>,
) -> Result<(DynVector<f64>, u32), ()> {
    let inlier_indices: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i) } else { None })
        .collect();
    if inlier_indices.len() < MIN_RADIAL_POINTS {
        return Err(());
    }
    // Tie-break weight √μ (normalized coordinates): cost μ·ĉ² ≤ 1e-8 even
    // at ĉ ~ 1 — at least three orders below the smallest realistic data
    // term, so it never fights real signal.
    const SQRT_MU_TIE: f64 = 1e-4;
    let m = 2 * inlier_indices.len() + 4;

    // Closures borrow the inlier index list and points. The residual
    // function returns a column vector of length m. The Jacobian returns
    // an m×8 matrix.
    let residual = |x: &DynVector<f64>| -> DynVector<f64> {
        let mut r = DynVector::<f64>::zeros(m);
        for (row_pair, &i) in inlier_indices.iter().enumerate() {
            let p = &points[i];
            let (px, py) = intrinsics_predict(p, x.as_slice());
            r[2 * row_pair] = p.x_obs - px;
            r[2 * row_pair + 1] = p.y_obs - py;
        }
        // Tie-break rows: cost contribution (√μ·cx)² = μ·cx², etc.
        r[m - 4] = SQRT_MU_TIE * x[0];
        r[m - 3] = SQRT_MU_TIE * x[1];
        r[m - 2] = SQRT_MU_TIE * x[6];
        r[m - 1] = SQRT_MU_TIE * x[7];
        r
    };

    let jacobian = |x: &DynVector<f64>| -> DynMatrix<f64> {
        // Numeris LM uses the convention r(x) is the residual; gradient = Jᵀr.
        // R = obs − predicted, so rows are ∂R/∂params = −∂predicted/∂params.
        let cx = x[0];
        let cy = x[1];
        let gamma = x[2];
        let k1 = x[3];
        let k2 = x[4];
        let k3 = x[5];
        let p1 = x[6];
        let p2 = x[7];
        let mut j = DynMatrix::<f64>::zeros(m, 8);
        for (row_pair, &i) in inlier_indices.iter().enumerate() {
            let p = &points[i];
            let xn = p.x_ideal - cx;
            let yn = p.y_ideal - cy;
            // Single-source model evaluation: distorted position, forward-map
            // Jacobian, and radius powers all from brown_conrady_forward.
            let e = brown_conrady_forward(k1, k2, k3, p1, p2, xn, yn);
            let (r2, r4, r6) = (e.r2, e.r4, e.r6);
            let (dx, dy) = (e.fx, e.fy);
            let ddx_dxn = e.j11;
            let ddx_dyn = e.j12;
            let ddy_dyn = e.j22;
            let ddy_dxn = ddx_dyn; // symmetric mixed term
            let row_x = 2 * row_pair;
            let row_y = row_x + 1;
            // x equation: R = x_obs − cx − γ·dx, with ∂xn/∂cx = −1.
            j[(row_x, 0)] = -1.0 + gamma * ddx_dxn;
            j[(row_x, 1)] = gamma * ddx_dyn;
            j[(row_x, 2)] = -dx;
            j[(row_x, 3)] = -gamma * xn * r2;
            j[(row_x, 4)] = -gamma * xn * r4;
            j[(row_x, 5)] = -gamma * xn * r6;
            j[(row_x, 6)] = -gamma * 2.0 * xn * yn;
            j[(row_x, 7)] = -gamma * (r2 + 2.0 * xn * xn);
            // y equation
            j[(row_y, 0)] = gamma * ddy_dxn;
            j[(row_y, 1)] = -1.0 + gamma * ddy_dyn;
            j[(row_y, 2)] = -dy;
            j[(row_y, 3)] = -gamma * yn * r2;
            j[(row_y, 4)] = -gamma * yn * r4;
            j[(row_y, 5)] = -gamma * yn * r6;
            j[(row_y, 6)] = -gamma * (r2 + 2.0 * yn * yn);
            j[(row_y, 7)] = -gamma * 2.0 * xn * yn;
        }
        // Tie-break rows: ∂(√μ·cx)/∂cx = √μ, etc.
        j[(m - 4, 0)] = SQRT_MU_TIE;
        j[(m - 3, 1)] = SQRT_MU_TIE;
        j[(m - 2, 6)] = SQRT_MU_TIE;
        j[(m - 1, 7)] = SQRT_MU_TIE;
        j
    };

    // Coordinates are normalized (field radius ~1, residuals ~1e-4), so the
    // relative tolerances are scale-free. The joint (cx, cy, γ, p1, p2) fit
    // has a long shallow valley; give LM room to walk it — numeris returns
    // a hard error (discarding the iterate) when max_iter is hit.
    let settings = LmSettings::<f64> {
        max_iter: 500,
        f_tol: 1e-11,
        x_tol: 1e-11,
        ..LmSettings::default()
    };
    let result = least_squares_lm_dyn(residual, jacobian, x0, &settings).map_err(|_| ())?;
    Ok((result.x, result.iterations as u32))
}

/// Root-mean-square of the masked-in per-point residuals.
pub(super) fn masked_rms(residuals: &[f64], mask: &[bool]) -> f64 {
    let mut sum_sq = 0.0_f64;
    let mut n = 0usize;
    for (&r, &m) in residuals.iter().zip(mask) {
        if m {
            sum_sq += r * r;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        (sum_sq / n as f64).sqrt()
    }
}

/// Solve `(k1, k2, k3)` from matched points using least squares.
///
/// Model: `x_obs - x_ideal = x_ideal · (k1·r² + k2·r⁴ + k3·r⁶)`
///        `y_obs - y_ideal = y_ideal · (k1·r² + k2·r⁴ + k3·r⁶)`
///
/// Stacks both x and y equations into one system with 3 unknowns.
fn fit_radial_ls(points: &[MatchedPoint], mask: &[bool]) -> (f64, f64, f64) {
    let inlier_count: usize = mask.iter().filter(|&&m| m).count();

    if inlier_count < 3 {
        return (0.0, 0.0, 0.0);
    }

    let nrows = inlier_count * 2;
    let mut a_mat = DynMatrix::<f64>::zeros(nrows, 3);
    let mut b_vec = DynVector::<f64>::zeros(nrows);

    let mut row = 0;
    for (i, p) in points.iter().enumerate() {
        if !mask[i] {
            continue;
        }
        let r2 = p.x_ideal * p.x_ideal + p.y_ideal * p.y_ideal;
        let r4 = r2 * r2;
        let r6 = r2 * r4;

        a_mat[(row, 0)] = p.x_ideal * r2;
        a_mat[(row, 1)] = p.x_ideal * r4;
        a_mat[(row, 2)] = p.x_ideal * r6;
        b_vec[row] = p.x_obs - p.x_ideal;
        row += 1;

        a_mat[(row, 0)] = p.y_ideal * r2;
        a_mat[(row, 1)] = p.y_ideal * r4;
        a_mat[(row, 2)] = p.y_ideal * r6;
        b_vec[row] = p.y_obs - p.y_ideal;
        row += 1;
    }

    let coeffs = a_mat
        .solve_qr(&b_vec)
        .unwrap_or_else(|_| DynVector::zeros(3));

    (coeffs[0], coeffs[1], coeffs[2])
}

// ── Polynomial (SIP-like) distortion fitting ────────────────────────────────

/// Result of a sigma-clipped forward polynomial fit. The inverse (`ap`/`bp`)
/// coefficients are no longer fit or stored — inversion is done numerically
/// (Newton) and `PolynomialDistortion::new` zero-fills the legacy fields.
pub(super) struct PolyFitResult {
    pub a_coeffs: Vec<f64>,
    pub b_coeffs: Vec<f64>,
    pub mask: Vec<bool>,
    pub iterations: u32,
}

/// Per-point radial residual (in pixels) of the forward SIP polynomial model:
/// the Euclidean distance between the observed position and
/// `distort(ideal)` under the given coefficients.
fn poly_point_residuals(
    points: &[MatchedPoint],
    order: u32,
    scale: f64,
    a_coeffs: &[f64],
    b_coeffs: &[f64],
) -> Vec<f64> {
    points
        .iter()
        .map(|p| {
            let u = p.x_ideal / scale;
            let v = p.y_ideal / scale;
            let rx = p.x_obs - p.x_ideal - eval_poly(a_coeffs, order, u, v) * scale;
            let ry = p.y_obs - p.y_ideal - eval_poly(b_coeffs, order, u, v) * scale;
            (rx * rx + ry * ry).sqrt()
        })
        .collect()
}

/// Fit a polynomial distortion model with iterative sigma-clipping.
///
/// The core polynomial loop behind [`fit_pooled`]:
/// 1. Initial forward polynomial LS fit.
/// 2. Iterative sigma-clipping at `median + k·σ`, with both statistics
///    estimated (MAD) over all points so the mask converges instead of
///    shrinking onto the brightest stars.
///
/// `points` are matched observations, `order` is the polynomial order,
/// `scale` is the normalization factor (typically image_width / 2).
pub(super) fn fit_polynomial_sigma_clip(
    points: &[MatchedPoint],
    order: u32,
    scale: f64,
    config: &DistortionFitConfig,
) -> PolyFitResult {
    let n = points.len();
    let ncoeffs = num_coeffs(order);
    let pairs = term_pairs(order);

    let mut mask = vec![true; n];
    let mut iterations = 0u32;
    let mut a_coeffs = vec![0.0; ncoeffs];
    let mut b_coeffs = vec![0.0; ncoeffs];
    let mut scratch: Vec<f64> = Vec::with_capacity(n);

    // Initial fit
    fit_poly_ls(points, &mask, &pairs, scale, &mut a_coeffs, &mut b_coeffs);

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Compute residuals using current model
        let residuals = poly_point_residuals(points, order, scale, &a_coeffs, &b_coeffs);

        // MAD-based robust clipping. Median and MAD-σ are taken over ALL
        // points (not the shrinking inlier set), so the threshold tracks the
        // model rather than ratcheting down with each refit; the residual
        // magnitudes are non-negative (Rayleigh-like), hence the median
        // offset. See `DistortionFitConfig::sigma_clip`.
        scratch.clear();
        scratch.extend_from_slice(&residuals);
        let (median, sigma) = median_mad_sigma(&mut scratch);

        if sigma < 1e-12 {
            break;
        }

        let threshold = median + config.sigma_clip * sigma;
        let new_mask: Vec<bool> = residuals.iter().map(|&r| r <= threshold).collect();

        let n_inliers = new_mask.iter().filter(|&&m| m).count();
        if n_inliers < ncoeffs {
            // Keep the previous (usable) mask rather than commit a degenerate one.
            debug!(
                "Too few inliers ({}) for polynomial fit after sigma-clip",
                n_inliers
            );
            break;
        }

        let changed = mask.iter().zip(&new_mask).any(|(&a, &b)| a != b);
        mask = new_mask;

        if !changed {
            break;
        }

        fit_poly_ls(points, &mask, &pairs, scale, &mut a_coeffs, &mut b_coeffs);
    }

    // The inverse polynomial (distorted → ideal) is no longer fit:
    // PolynomialDistortion::undistort uses Newton iteration on the forward
    // polynomial, which is exact (limited only by forward expressiveness).
    PolyFitResult {
        a_coeffs,
        b_coeffs,
        mask,
        iterations,
    }
}

// ── Internal helpers ────────────────────────────────────────────────────────

/// Build a HashMap from catalog_id → index into star_vectors.
pub(super) fn build_id_lookup(database: &SolverDatabase) -> HashMap<i64, usize> {
    database
        .star_catalog_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect()
}

/// Iterate a solution's matched `(centroid_idx, catalog_star_idx)` pairs.
///
/// Zips the two parallel match vectors (so a length mismatch can't panic),
/// skips centroid indices out of range for the supplied centroid count, and
/// resolves catalog IDs through `id_to_idx` (unknown IDs are skipped). Shared
/// by single-image point gathering and the multi-image calibration's per-image
/// initial-match construction.
pub(super) fn matched_pairs<'a>(
    sol: &'a crate::solver::Solution,
    n_centroids: usize,
    id_to_idx: &'a HashMap<i64, usize>,
) -> impl Iterator<Item = (usize, usize)> + 'a {
    sol.matched_catalog_ids
        .iter()
        .zip(sol.matched_centroid_indices.iter())
        .filter_map(move |(&cat_id, &cent_idx)| {
            if cent_idx >= n_centroids {
                return None;
            }
            id_to_idx.get(&cat_id).map(|&star_idx| (cent_idx, star_idx))
        })
}

/// Append one [`MatchedPoint`] per `(centroid_idx, catalog_star_idx)` pair:
/// the observed centroid paired with the catalog star projected through `rot`
/// at `pixel_scale`. Pairs whose star is behind the camera or whose centroid
/// is non-finite are skipped. Shared by the single-image gather (solve's own
/// rotation) and the multi-image Phase 2 (refined per-image rotations).
pub(super) fn project_matches(
    rot: Matrix3<f32>,
    pairs: impl IntoIterator<Item = (usize, usize)>,
    centroids: &[Centroid],
    star_vectors: StarVectors<'_>,
    parity_sign: f64,
    pixel_scale: f64,
    out: &mut Vec<MatchedPoint>,
) {
    for (cent_idx, star_idx) in pairs {
        let c = &centroids[cent_idx];
        if let Some(mp) = project_to_matched_point(
            rot,
            &star_vectors.get(star_idx),
            parity_sign,
            pixel_scale,
            c.x as f64,
            c.y as f64,
        ) {
            out.push(mp);
        }
    }
}

/// Project a catalog star (ICRS unit vector `sv`) through `rot` into ideal
/// pinhole pixel coordinates and pair it with an observed centroid `(x_obs,
/// y_obs)`.
///
/// Returns `None` when the star projects behind the camera (`cam_z ≤ 0`) or
/// the observed position is non-finite. `parity_sign` is `-1.0` when the image x-axis is flipped; `pixel_scale` is
/// radians per pixel (`1/focal_length_px`).
fn project_to_matched_point(
    rot: Matrix3<f32>,
    sv: &[f32; 3],
    parity_sign: f64,
    pixel_scale: f64,
    x_obs: f64,
    y_obs: f64,
) -> Option<MatchedPoint> {
    if !(x_obs.is_finite() && y_obs.is_finite()) {
        return None;
    }
    let icrs_v = numeris::Vector3::from_array([sv[0], sv[1], sv[2]]);
    let cam_v = rot * icrs_v;
    if cam_v[2] <= 0.0 {
        return None;
    }
    let x_ideal = parity_sign * (cam_v[0] as f64) / (cam_v[2] as f64) / pixel_scale;
    let y_ideal = (cam_v[1] as f64) / (cam_v[2] as f64) / pixel_scale;
    Some(MatchedPoint {
        x_obs,
        y_obs,
        x_ideal,
        y_ideal,
    })
}

/// Compute RMS pixel residual after applying distortion correction to inliers.
pub(super) fn compute_corrected_rmse(
    points: &[MatchedPoint],
    mask: &[bool],
    distortion: &Distortion,
) -> f64 {
    let mut sum_sq = 0.0;
    let mut count = 0;

    for (i, p) in points.iter().enumerate() {
        if !mask[i] {
            continue;
        }
        // Undistort the observed position, then compare to ideal
        let (xu, yu) = distortion.undistort(p.x_obs, p.y_obs);
        let dx = xu - p.x_ideal;
        let dy = yu - p.y_ideal;
        sum_sq += dx * dx + dy * dy;
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }
    (sum_sq / count as f64).sqrt()
}

/// Fit the forward polynomial (ideal → distorted) by least-squares.
///
/// Model: x_obs = x_ideal + Σ A_pq · u^p · v^q   (u = x_ideal/scale, v = y_ideal/scale, 0 ≤ p+q ≤ order)
/// Stacks x and y equations, solves each axis independently.
pub(super) fn fit_poly_ls(
    points: &[MatchedPoint],
    mask: &[bool],
    pairs: &[(u32, u32)],
    scale: f64,
    a_coeffs: &mut [f64],
    b_coeffs: &mut [f64],
) {
    let ncoeffs = pairs.len();
    let n_inliers: usize = mask.iter().filter(|&&m| m).count();
    if n_inliers < ncoeffs {
        return;
    }

    let mut a_mat = DynMatrix::<f64>::zeros(n_inliers, ncoeffs);
    let mut bx_vec = DynVector::<f64>::zeros(n_inliers);
    let mut by_vec = DynVector::<f64>::zeros(n_inliers);

    let mut row = 0;
    for (i, p) in points.iter().enumerate() {
        if !mask[i] {
            continue;
        }
        let u = p.x_ideal / scale;
        let v = p.y_ideal / scale;

        for (j, &(pp, qq)) in pairs.iter().enumerate() {
            a_mat[(row, j)] = u.powi(pp as i32) * v.powi(qq as i32);
        }
        bx_vec[row] = (p.x_obs - p.x_ideal) / scale;
        by_vec[row] = (p.y_obs - p.y_ideal) / scale;
        row += 1;
    }

    // A rank-deficient design matrix (e.g. collinear or too-few points) leaves
    // the coefficients at their previous values; log it so a silently-degraded
    // fit isn't reported as a normal one.
    match a_mat.solve_qr(&bx_vec) {
        Ok(cx) => {
            for j in 0..ncoeffs {
                a_coeffs[j] = cx[j];
            }
        }
        Err(_) => debug!("fit_poly_ls: x-axis QR solve failed; keeping prior coeffs"),
    }

    match a_mat.solve_qr(&by_vec) {
        Ok(cy) => {
            for j in 0..ncoeffs {
                b_coeffs[j] = cy[j];
            }
        }
        Err(_) => debug!("fit_poly_ls: y-axis QR solve failed; keeping prior coeffs"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    /// ~200 matched points on a jittered grid, distorted by `k1·r²` and
    /// perturbed with Gaussian centroid noise `sigma_px` per axis.
    fn noisy_points(k1: f64, sigma_px: f64, seed: u64) -> Vec<MatchedPoint> {
        let mut rng = StdRng::seed_from_u64(seed);
        let noise = Normal::new(0.0, sigma_px).unwrap();
        let true_d = RadialDistortion::new(k1, 0.0, 0.0);
        let mut points = Vec::new();
        let mut k = 0u32;
        for ix in -7..=7 {
            for iy in -6..=6 {
                let x_ideal = ix as f64 * 130.0 + (k % 11) as f64;
                let y_ideal = iy as f64 * 130.0 + (k % 7) as f64;
                k += 1;
                let (xd, yd) = true_d.distort(x_ideal, y_ideal);
                points.push(MatchedPoint {
                    x_obs: xd + noise.sample(&mut rng),
                    y_obs: yd + noise.sample(&mut rng),
                    x_ideal,
                    y_ideal,
                });
            }
        }
        points
    }

    /// The sigma-clip loop must keep essentially all points under ordinary
    /// Gaussian noise. (A threshold of `k·σ` without the median offset
    /// rejects ~14% per pass and collapses.)
    #[test]
    fn test_polynomial_sigma_clip_keeps_good_points_under_noise() {
        let points = noisy_points(-7e-9, 0.3, 7);
        let n = points.len();
        let config = DistortionFitConfig::default();
        let fit = fit_polynomial_sigma_clip(&points, 3, 1000.0, &config);
        let kept = fit.mask.iter().filter(|&&m| m).count();
        assert!(
            kept as f64 >= 0.95 * n as f64,
            "sigma-clip kept only {kept}/{n} good points"
        );
        // With σ over all points the mask must settle quickly rather than
        // ratchet toward max_iterations (20).
        assert!(
            fit.iterations <= 5,
            "clip loop took {} passes to converge",
            fit.iterations
        );
        let model = Distortion::Polynomial(PolynomialDistortion::new(
            3,
            1000.0,
            fit.a_coeffs,
            fit.b_coeffs,
        ));
        let all = vec![true; n];
        let rms = compute_corrected_rmse(&points, &all, &model);
        // 2-D RMS of σ = 0.3 px/axis noise is ≈ 0.42 px.
        assert!(rms < 0.5, "post-fit RMS over all points {rms:.3} px");
    }

    #[test]
    fn test_radial_sigma_clip_keeps_good_points_under_noise() {
        let true_k1 = -7e-9;
        let points = noisy_points(true_k1, 0.3, 11);
        let n = points.len();
        let config = DistortionFitConfig::default();
        let fit = fit_radial_centered_sigma_clip(&points, &config);
        let kept = fit.mask.iter().filter(|&&m| m).count();
        assert!(
            kept as f64 >= 0.95 * n as f64,
            "sigma-clip kept only {kept}/{n} good points"
        );
        // The joint 8-parameter fit trades k1 against γ/k2 under noise, so
        // judge it by prediction accuracy over the whole field instead.
        let resid = intrinsics_residuals(
            &points,
            &[
                fit.cx, fit.cy, fit.gamma, fit.k1, fit.k2, fit.k3, fit.p1, fit.p2,
            ],
        );
        let rms = masked_rms(&resid, &vec![true; n]);
        assert!(rms < 0.5, "post-fit RMS over all points {rms:.3} px");
    }

    /// Test that fitting recovers known radial distortion from synthetic data.
    #[test]
    fn test_fit_radial_synthetic() {
        let true_k1 = -7e-9;
        let true_k2 = 2e-15;
        let true_k3 = 0.0;
        let true_distortion = RadialDistortion::new(true_k1, true_k2, true_k3);

        let mut points = Vec::new();
        for ix in -5..=5 {
            for iy in -5..=5 {
                let x_ideal = ix as f64 * 100.0;
                let y_ideal = iy as f64 * 100.0;
                let (x_obs, y_obs) = true_distortion.distort(x_ideal, y_ideal);
                points.push(MatchedPoint {
                    x_obs,
                    y_obs,
                    x_ideal,
                    y_ideal,
                });
            }
        }

        let mask = vec![true; points.len()];
        let (k1, k2, k3) = fit_radial_ls(&points, &mask);

        assert!(
            (k1 - true_k1).abs() < 1e-12,
            "k1: fitted={:.6e}, true={:.6e}",
            k1,
            true_k1,
        );
        assert!(
            (k2 - true_k2).abs() < 1e-18,
            "k2: fitted={:.6e}, true={:.6e}",
            k2,
            true_k2,
        );
        assert!(k3.abs() < 1e-18, "k3: fitted={:.3e}, expected ~0", k3);
    }

    /// The joint fit must separate a pure focal-scale error (wrong anchor
    /// focal length) from lens distortion: `gamma` recovers the scale and
    /// the rescaled Brown-Conrady coefficients stay ~0.
    #[test]
    fn test_fit_radial_recovers_focal_scale() {
        let true_gamma = 1.009; // ~1% focal-length error, as seen on TESS

        let mut points = Vec::new();
        for ix in -5..=5 {
            for iy in -5..=5 {
                let x_ideal = ix as f64 * 100.0;
                let y_ideal = iy as f64 * 100.0;
                points.push(MatchedPoint {
                    x_obs: x_ideal * true_gamma,
                    y_obs: y_ideal * true_gamma,
                    x_ideal,
                    y_ideal,
                });
            }
        }

        let config = DistortionFitConfig::default();
        let fit = fit_radial_centered_sigma_clip(&points, &config);

        assert!(
            (fit.gamma - true_gamma).abs() < 1e-6,
            "gamma: fitted={:.8}, true={:.8}",
            fit.gamma,
            true_gamma,
        );
        let model = fit.rescaled_model();
        // Residual lens distortion in the rescaled model must be far below
        // a centroid width well into the field.
        let (xd, yd) = model.distort(500.0, 500.0);
        assert!(
            (xd - 500.0).abs() < 1e-3 && (yd - 500.0).abs() < 1e-3,
            "rescaled model not ~identity: distort(500, 500) = ({xd}, {yd})",
        );
    }

    /// Focal scale and genuine radial distortion fit jointly without
    /// trading off.
    #[test]
    fn test_fit_radial_scale_and_distortion_jointly() {
        let true_gamma = 0.995;
        let true_k1 = -7e-9;
        let true_distortion = RadialDistortion::new(true_k1, 0.0, 0.0);

        let mut points = Vec::new();
        for ix in -7..=7 {
            for iy in -7..=7 {
                let x_ideal = ix as f64 * 100.0;
                let y_ideal = iy as f64 * 100.0;
                let (xd, yd) = true_distortion.distort(x_ideal, y_ideal);
                points.push(MatchedPoint {
                    x_obs: xd * true_gamma,
                    y_obs: yd * true_gamma,
                    x_ideal,
                    y_ideal,
                });
            }
        }

        let config = DistortionFitConfig::default();
        let fit = fit_radial_centered_sigma_clip(&points, &config);

        assert!(
            (fit.gamma - true_gamma).abs() < 1e-5,
            "gamma: fitted={:.8}, true={:.8}",
            fit.gamma,
            true_gamma,
        );
        assert!(
            (fit.k1 - true_k1).abs() < 1e-11,
            "k1: fitted={:.6e}, true={:.6e}",
            fit.k1,
            true_k1,
        );
    }

    /// Mosaic-camera geometry: the optical axis (distortion center) sits
    /// near a CCD corner, ~1.5 field-radii from the image center — the TESS
    /// situation. The unregularized center must travel there, and the joint
    /// fit must still recover scale and distortion.
    #[test]
    fn test_fit_radial_mosaic_corner_center() {
        let true_gamma = 1.009;
        let (true_cx, true_cy) = (-1100.0, -1080.0);
        let true_d = RadialDistortion::with_tangential(-5e-9, 1e-15, -1e-21, 1e-7, -2e-7);

        let mut points = Vec::new();
        let mut k = 0u32;
        for ix in -31..=31 {
            for iy in -31..=31 {
                let x_ideal = ix as f64 * 33.0 + (k % 17) as f64; // break grid symmetry
                let y_ideal = iy as f64 * 33.0 + (k % 13) as f64;
                k += 1;
                let (dx, dy) = true_d.distort(x_ideal - true_cx, y_ideal - true_cy);
                points.push(MatchedPoint {
                    x_obs: true_cx + true_gamma * dx,
                    y_obs: true_cy + true_gamma * dy,
                    x_ideal,
                    y_ideal,
                });
            }
        }

        let config = DistortionFitConfig::default();
        let fit = fit_radial_centered_sigma_clip(&points, &config);

        let resid = intrinsics_residuals(
            &points,
            &[
                fit.cx, fit.cy, fit.gamma, fit.k1, fit.k2, fit.k3, fit.p1, fit.p2,
            ],
        );
        let rms = masked_rms(&resid, &fit.mask);
        assert!(rms < 0.01, "rms {rms:.4} px on noiseless synthetic data");
        assert!(
            (fit.gamma - true_gamma).abs() < 1e-4,
            "gamma: fitted={:.8}, true={:.8}",
            fit.gamma,
            true_gamma,
        );
        assert!(
            (fit.cx - true_cx).abs() < 20.0 && (fit.cy - true_cy).abs() < 20.0,
            "center: fitted=({:.1}, {:.1}), true=({true_cx}, {true_cy})",
            fit.cx,
            fit.cy,
        );
    }
}
