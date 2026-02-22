//! Distortion model fitting from solve results.
//!
//! Given one or more plate-solve results (each with matched catalog star IDs and
//! centroid indices), this module fits distortion models by comparing observed
//! centroid positions to their ideal (pinhole-projected) positions computed from
//! the catalog.
//!
//! The fitting is iterative with sigma-clipping to reject mismatched stars.

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector, Matrix3};
use tracing::debug;

use crate::centroid::Centroid;
use crate::solver::{SolveResult, SolveStatus, SolverDatabase};

use super::polynomial::{num_coeffs, term_pairs, PolynomialDistortion};
use super::radial::RadialDistortion;
use super::Distortion;

/// Configuration for distortion fitting.
#[derive(Debug, Clone)]
pub struct DistortionFitConfig {
    /// Sigma threshold for iterative outlier rejection. Default 3.0.
    pub sigma_clip: f64,
    /// Maximum iterations for iterative fitting. Default 20.
    pub max_iterations: u32,
    /// If provided, a second stage re-applies the model with this loose pixel
    /// threshold to recover stars rejected in the tight sigma-clip stage.
    /// Stars with residuals below this threshold are kept. Default 5.0 px.
    pub stage2_threshold_px: Option<f64>,
}

impl Default for DistortionFitConfig {
    fn default() -> Self {
        Self {
            sigma_clip: 3.0,
            max_iterations: 20,
            stage2_threshold_px: Some(5.0),
        }
    }
}

/// Result of a distortion fitting procedure.
#[derive(Debug, Clone)]
pub struct DistortionFitResult {
    /// The fitted distortion model.
    pub model: Distortion,
    /// RMS residual in pixels BEFORE distortion correction.
    pub rmse_before_px: f64,
    /// RMS residual in pixels AFTER distortion correction.
    pub rmse_after_px: f64,
    /// Number of inlier matches used in the final fit.
    pub n_inliers: usize,
    /// Number of outlier matches rejected.
    pub n_outliers: usize,
    /// Number of sigma-clip iterations performed.
    pub iterations: u32,
    /// Per-match inlier mask (true = inlier, false = outlier).
    /// Matches are ordered as they appear across all solve results
    /// (solve_results\[0\] matches first, then \[1\], etc.).
    pub inlier_mask: Vec<bool>,
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

// ── Radial distortion fitting ───────────────────────────────────────────────

/// Fit a radial distortion model (k1, k2, k3) from plate-solve results.
///
/// Same interface as [`fit_polynomial_distortion`].
pub fn fit_radial_distortion(
    solve_results: &[&SolveResult],
    centroids: &[&[Centroid]],
    database: &SolverDatabase,
    image_width: u32,
    config: &DistortionFitConfig,
) -> DistortionFitResult {
    assert_eq!(
        solve_results.len(),
        centroids.len(),
        "solve_results and centroids must have the same length"
    );

    let id_to_idx = build_id_lookup(database);
    let points = gather_matched_points(solve_results, centroids, database, &id_to_idx, image_width);

    if points.is_empty() {
        return DistortionFitResult {
            model: Distortion::None,
            rmse_before_px: 0.0,
            rmse_after_px: 0.0,
            n_inliers: 0,
            n_outliers: 0,
            iterations: 0,
            inlier_mask: Vec::new(),
        };
    }

    let n = points.len();

    // Stage 1: Iterative radial fit with sigma-clipping
    let mut mask = vec![true; n];
    let mut iterations = 0u32;
    let mut k1;
    let mut k2;
    let mut k3;

    (k1, k2, k3) = fit_radial_ls(&points, &mask);

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        let residuals: Vec<f64> = points
            .iter()
            .map(|p| {
                let r2 = p.x_ideal * p.x_ideal + p.y_ideal * p.y_ideal;
                let r4 = r2 * r2;
                let r6 = r2 * r4;
                let scale = k1 * r2 + k2 * r4 + k3 * r6;
                let dx_model = p.x_ideal * scale;
                let dy_model = p.y_ideal * scale;
                let rx = p.x_obs - p.x_ideal - dx_model;
                let ry = p.y_obs - p.y_ideal - dy_model;
                (rx * rx + ry * ry).sqrt()
            })
            .collect();

        let inlier_resids: Vec<f64> = residuals
            .iter()
            .zip(&mask)
            .filter(|(_, &m)| m)
            .map(|(&r, _)| r)
            .collect();

        if inlier_resids.is_empty() {
            break;
        }

        let median = percentile(&inlier_resids, 0.5);
        let mut abs_devs: Vec<f64> = inlier_resids.iter().map(|&r| (r - median).abs()).collect();
        abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = percentile(&abs_devs, 0.5);
        let sigma = mad * 1.4826;

        if sigma < 1e-12 {
            break;
        }

        let threshold = config.sigma_clip * sigma;
        let new_mask: Vec<bool> = residuals.iter().map(|&r| r <= threshold).collect();

        let changed = mask.iter().zip(&new_mask).any(|(&a, &b)| a != b);
        mask = new_mask;

        if !changed {
            break;
        }

        let n_inliers = mask.iter().filter(|&&m| m).count();
        if n_inliers < 3 {
            break;
        }

        (k1, k2, k3) = fit_radial_ls(&points, &mask);
    }

    // Stage 2
    if let Some(threshold_px) = config.stage2_threshold_px {
        let residuals: Vec<f64> = points
            .iter()
            .map(|p| {
                let r2 = p.x_ideal * p.x_ideal + p.y_ideal * p.y_ideal;
                let r4 = r2 * r2;
                let r6 = r2 * r4;
                let scale = k1 * r2 + k2 * r4 + k3 * r6;
                let dx_model = p.x_ideal * scale;
                let dy_model = p.y_ideal * scale;
                let rx = p.x_obs - p.x_ideal - dx_model;
                let ry = p.y_obs - p.y_ideal - dy_model;
                (rx * rx + ry * ry).sqrt()
            })
            .collect();

        let mask_s2: Vec<bool> = residuals.iter().map(|&r| r <= threshold_px).collect();
        let n_recovered = mask_s2
            .iter()
            .zip(&mask)
            .filter(|(&s2, &s1)| s2 && !s1)
            .count();

        if n_recovered > 0 {
            mask = mask_s2;
            let n_inliers = mask.iter().filter(|&&m| m).count();
            if n_inliers >= 3 {
                (k1, k2, k3) = fit_radial_ls(&points, &mask);
            }
        }
    }

    let model = RadialDistortion::new(k1, k2, k3);
    // Compute before/after RMSE on the SAME inlier set for a fair comparison
    let rmse_before = compute_corrected_rmse(&points, &mask, &Distortion::None);
    let rmse_after = compute_corrected_rmse(&points, &mask, &Distortion::Radial(model.clone()));
    let n_inliers = mask.iter().filter(|&&m| m).count();

    debug!(
        "Radial fit: k1={:.3e}, k2={:.3e}, k3={:.3e}, inliers={}/{}, RMSE {:.3} → {:.3} px",
        k1, k2, k3, n_inliers, n, rmse_before, rmse_after
    );

    DistortionFitResult {
        model: Distortion::Radial(model),
        rmse_before_px: rmse_before,
        rmse_after_px: rmse_after,
        n_inliers,
        n_outliers: n - n_inliers,
        iterations,
        inlier_mask: mask,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Polynomial (SIP-like) distortion fitting
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a sigma-clipped polynomial fit (forward + inverse).
pub(super) struct PolyFitResult {
    pub a_coeffs: Vec<f64>,
    pub b_coeffs: Vec<f64>,
    pub ap_coeffs: Vec<f64>,
    pub bp_coeffs: Vec<f64>,
    pub mask: Vec<bool>,
    pub iterations: u32,
}

/// Fit a polynomial distortion model with iterative sigma-clipping.
///
/// This is the core fitting loop extracted for reuse by multi-image calibration.
/// It performs:
/// 1. Initial forward polynomial LS fit.
/// 2. Iterative MAD-based sigma-clipping.
/// 3. Stage 2 outlier recovery (optional, controlled by `config.stage2_threshold_px`).
/// 4. Inverse polynomial fit on the final inlier set.
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

    // Initial fit
    fit_poly_ls(points, &mask, &pairs, scale, &mut a_coeffs, &mut b_coeffs);

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Compute residuals using current model
        let residuals: Vec<f64> = points
            .iter()
            .map(|p| {
                let u = p.x_ideal / scale;
                let v = p.y_ideal / scale;
                let dx_model: f64 = pairs
                    .iter()
                    .enumerate()
                    .map(|(i, &(pp, qq))| a_coeffs[i] * u.powi(pp as i32) * v.powi(qq as i32))
                    .sum();
                let dy_model: f64 = pairs
                    .iter()
                    .enumerate()
                    .map(|(i, &(pp, qq))| b_coeffs[i] * u.powi(pp as i32) * v.powi(qq as i32))
                    .sum();
                let rx = p.x_obs - p.x_ideal - dx_model * scale;
                let ry = p.y_obs - p.y_ideal - dy_model * scale;
                (rx * rx + ry * ry).sqrt()
            })
            .collect();

        // MAD-based robust clipping
        let inlier_resids: Vec<f64> = residuals
            .iter()
            .zip(&mask)
            .filter(|(_, &m)| m)
            .map(|(&r, _)| r)
            .collect();

        if inlier_resids.is_empty() {
            break;
        }

        let median = percentile(&inlier_resids, 0.5);
        let mut abs_devs: Vec<f64> = inlier_resids.iter().map(|&r| (r - median).abs()).collect();
        abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = percentile(&abs_devs, 0.5);
        let sigma = mad * 1.4826;

        if sigma < 1e-12 {
            break;
        }

        let threshold = config.sigma_clip * sigma;
        let new_mask: Vec<bool> = residuals.iter().map(|&r| r <= threshold).collect();

        let changed = mask.iter().zip(&new_mask).any(|(&a, &b)| a != b);
        mask = new_mask;

        if !changed {
            break;
        }

        let n_inliers = mask.iter().filter(|&&m| m).count();
        if n_inliers < ncoeffs {
            debug!(
                "Too few inliers ({}) for polynomial fit after sigma-clip",
                n_inliers
            );
            break;
        }

        fit_poly_ls(points, &mask, &pairs, scale, &mut a_coeffs, &mut b_coeffs);
    }

    // Stage 2: recover outliers below a threshold
    if let Some(threshold_px) = config.stage2_threshold_px {
        let residuals: Vec<f64> = points
            .iter()
            .map(|p| {
                let u = p.x_ideal / scale;
                let v = p.y_ideal / scale;
                let dx_model: f64 = pairs
                    .iter()
                    .enumerate()
                    .map(|(i, &(pp, qq))| a_coeffs[i] * u.powi(pp as i32) * v.powi(qq as i32))
                    .sum();
                let dy_model: f64 = pairs
                    .iter()
                    .enumerate()
                    .map(|(i, &(pp, qq))| b_coeffs[i] * u.powi(pp as i32) * v.powi(qq as i32))
                    .sum();
                let rx = p.x_obs - p.x_ideal - dx_model * scale;
                let ry = p.y_obs - p.y_ideal - dy_model * scale;
                (rx * rx + ry * ry).sqrt()
            })
            .collect();

        let mask_s2: Vec<bool> = residuals.iter().map(|&r| r <= threshold_px).collect();
        let n_recovered = mask_s2
            .iter()
            .zip(&mask)
            .filter(|(&s2, &s1)| s2 && !s1)
            .count();

        if n_recovered > 0 {
            mask = mask_s2;
            let n_inliers = mask.iter().filter(|&&m| m).count();
            if n_inliers >= ncoeffs {
                fit_poly_ls(points, &mask, &pairs, scale, &mut a_coeffs, &mut b_coeffs);
            }
        }
    }

    // Fit the inverse polynomial (distorted → ideal) from the same data
    let mut ap_coeffs = vec![0.0; ncoeffs];
    let mut bp_coeffs = vec![0.0; ncoeffs];
    fit_inverse_poly_ls(points, &mask, &pairs, scale, &mut ap_coeffs, &mut bp_coeffs);

    PolyFitResult {
        a_coeffs,
        b_coeffs,
        ap_coeffs,
        bp_coeffs,
        mask,
        iterations,
    }
}

/// Fit a polynomial (SIP-like) distortion model from plate-solve results.
///
/// This model fits arbitrary 2D polynomial correction terms:
///   x_obs = x_ideal + Σ A_pq · (x_ideal/s)^p · (y_ideal/s)^q   (s = scale, 0 ≤ p+q ≤ order)
///   y_obs = y_ideal + Σ B_pq · (x_ideal/s)^p · (y_ideal/s)^q
/// for 2 ≤ p+q ≤ order.
///
/// An inverse polynomial (distorted → ideal) is also fitted for efficient
/// undistortion in the solver.
///
/// The `order` parameter determines the polynomial complexity:
/// - order 2: 6 terms per axis (12 total) — offset + linear + quadratic
/// - order 3: 10 terms per axis (20 total) — + cubic
/// - order 4: 15 terms per axis (30 total) — + quartic (recommended for TESS)
/// - order 5: 21 terms per axis (42 total) — + quintic
///
/// Order-0 terms absorb optical center offset; order-1 terms absorb residual
/// plate scale and rotation errors.
pub fn fit_polynomial_distortion(
    solve_results: &[&SolveResult],
    centroids: &[&[Centroid]],
    database: &SolverDatabase,
    image_width: u32,
    order: u32,
    config: &DistortionFitConfig,
) -> DistortionFitResult {
    assert_eq!(
        solve_results.len(),
        centroids.len(),
        "solve_results and centroids must have the same length"
    );
    assert!(
        order >= 2 && order <= 6,
        "polynomial order must be in [2, 6]"
    );

    let id_to_idx = build_id_lookup(database);
    let points = gather_matched_points(solve_results, centroids, database, &id_to_idx, image_width);

    if points.is_empty() {
        return DistortionFitResult {
            model: Distortion::None,
            rmse_before_px: 0.0,
            rmse_after_px: 0.0,
            n_inliers: 0,
            n_outliers: 0,
            iterations: 0,
            inlier_mask: Vec::new(),
        };
    }

    let n = points.len();
    let ncoeffs = num_coeffs(order);
    let scale = image_width as f64 / 2.0;

    // Minimum data points: need at least ncoeffs matched pairs
    if n < ncoeffs {
        let rmse_raw = compute_rmse_px(&points);
        debug!(
            "Too few matched points ({}) for order-{} polynomial fit ({} coefficients needed)",
            n, order, ncoeffs
        );
        return DistortionFitResult {
            model: Distortion::None,
            rmse_before_px: rmse_raw,
            rmse_after_px: rmse_raw,
            n_inliers: n,
            n_outliers: 0,
            iterations: 0,
            inlier_mask: vec![true; n],
        };
    }

    // Delegate to the reusable sigma-clip helper
    let fit = fit_polynomial_sigma_clip(&points, order, scale, config);

    let model = PolynomialDistortion::new(
        order, scale,
        fit.a_coeffs, fit.b_coeffs, fit.ap_coeffs, fit.bp_coeffs,
    );
    let dist = Distortion::Polynomial(model.clone());
    // Compute before/after RMSE on the SAME inlier set for a fair comparison
    let rmse_before = compute_corrected_rmse(&points, &fit.mask, &Distortion::None);
    let rmse_after = compute_corrected_rmse(&points, &fit.mask, &dist);
    let n_inliers = fit.mask.iter().filter(|&&m| m).count();

    debug!(
        "Polynomial (order {}) fit: {} coefficients/axis, inliers={}/{}, RMSE {:.3} → {:.3} px",
        order, ncoeffs, n_inliers, n, rmse_before, rmse_after
    );

    DistortionFitResult {
        model: dist,
        rmse_before_px: rmse_before,
        rmse_after_px: rmse_after,
        n_inliers,
        n_outliers: n - n_inliers,
        iterations: fit.iterations,
        inlier_mask: fit.mask,
    }
}

// ── Internal helpers ────────────────────────────────────────────────────────

/// Build a HashMap from catalog_id → index into star_vectors.
pub(super) fn build_id_lookup(database: &SolverDatabase) -> HashMap<u64, usize> {
    database
        .star_catalog_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect()
}

/// Gather all matched points across multiple solve results.
///
/// For each matched pair, project the catalog star to pixel coordinates using
/// the solve result's rotation and FOV, and pair it with the observed centroid.
fn gather_matched_points(
    solve_results: &[&SolveResult],
    centroids: &[&[Centroid]],
    database: &SolverDatabase,
    id_to_idx: &HashMap<u64, usize>,
    image_width: u32,
) -> Vec<MatchedPoint> {
    let mut points = Vec::new();

    for (sr, cents) in solve_results.iter().zip(centroids.iter()) {
        if sr.status != SolveStatus::MatchFound {
            continue;
        }
        let quat = match &sr.qicrs2cam {
            Some(q) => q,
            None => continue,
        };
        let fov_rad = match sr.fov_rad {
            Some(f) => f,
            None => continue,
        };

        let pixel_scale = fov_rad / image_width as f32;
        let rot: Matrix3<f32> = *quat.to_rotation_matrix().matrix();

        let parity_sign: f64 = if sr.parity_flip { -1.0 } else { 1.0 };

        for (match_idx, &cat_id) in sr.matched_catalog_ids.iter().enumerate() {
            let cent_idx = sr.matched_centroid_indices[match_idx];
            if cent_idx >= cents.len() {
                continue;
            }

            let star_idx = match id_to_idx.get(&cat_id) {
                Some(&idx) => idx,
                None => continue,
            };

            let sv = &database.star_vectors[star_idx];
            let icrs_v = nalgebra::Vector3::new(sv[0], sv[1], sv[2]);
            let cam_v = rot * icrs_v;

            if cam_v.z <= 0.0 {
                continue;
            }

            // Project to pixel coordinates
            let x_ideal = parity_sign * (cam_v.x as f64) / (cam_v.z as f64) / (pixel_scale as f64);
            let y_ideal = (cam_v.y as f64) / (cam_v.z as f64) / (pixel_scale as f64);

            let x_obs = cents[cent_idx].x as f64;
            let y_obs = cents[cent_idx].y as f64;

            points.push(MatchedPoint {
                x_obs,
                y_obs,
                x_ideal,
                y_ideal,
            });
        }
    }

    points
}

/// Solve the radial least-squares fit.
///
/// Model: x_obs - x_ideal = x_ideal · (k1·r² + k2·r⁴ + k3·r⁶)
///        y_obs - y_ideal = y_ideal · (k1·r² + k2·r⁴ + k3·r⁶)
///
/// We stack both x and y equations into one system with 3 unknowns.
fn fit_radial_ls(points: &[MatchedPoint], mask: &[bool]) -> (f64, f64, f64) {
    let inlier_count: usize = mask.iter().filter(|&&m| m).count();

    if inlier_count < 3 {
        return (0.0, 0.0, 0.0);
    }

    // Each point contributes 2 rows (x and y equations)
    let nrows = inlier_count * 2;
    let mut a_mat = DMatrix::<f64>::zeros(nrows, 3);
    let mut b_vec = DVector::<f64>::zeros(nrows);

    let mut row = 0;
    for (i, p) in points.iter().enumerate() {
        if !mask[i] {
            continue;
        }
        let r2 = p.x_ideal * p.x_ideal + p.y_ideal * p.y_ideal;
        let r4 = r2 * r2;
        let r6 = r2 * r4;

        // x equation
        a_mat[(row, 0)] = p.x_ideal * r2;
        a_mat[(row, 1)] = p.x_ideal * r4;
        a_mat[(row, 2)] = p.x_ideal * r6;
        b_vec[row] = p.x_obs - p.x_ideal;
        row += 1;

        // y equation
        a_mat[(row, 0)] = p.y_ideal * r2;
        a_mat[(row, 1)] = p.y_ideal * r4;
        a_mat[(row, 2)] = p.y_ideal * r6;
        b_vec[row] = p.y_obs - p.y_ideal;
        row += 1;
    }

    let svd = a_mat.svd(true, true);
    let coeffs = svd
        .solve(&b_vec, 1e-12)
        .unwrap_or_else(|_| DVector::zeros(3));

    (coeffs[0], coeffs[1], coeffs[2])
}

/// Compute RMS pixel residual (uncorrected) across all points.
fn compute_rmse_px(points: &[MatchedPoint]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = points
        .iter()
        .map(|p| {
            let dx = p.x_obs - p.x_ideal;
            let dy = p.y_obs - p.y_ideal;
            dx * dx + dy * dy
        })
        .sum();
    (sum_sq / points.len() as f64).sqrt()
}

/// Compute RMS pixel residual after applying distortion correction to inliers.
pub(super) fn compute_corrected_rmse(points: &[MatchedPoint], mask: &[bool], distortion: &Distortion) -> f64 {
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

/// Compute the percentile of a sorted slice. `p` is in [0, 1].
pub(super) fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let mut values = sorted.to_vec();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (p * (values.len() - 1) as f64).round() as usize;
    values[idx.min(values.len() - 1)]
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

    // Fit x-axis: (x_obs - x_ideal) = Σ A_pq * u^p * v^q * scale
    let mut a_mat = DMatrix::<f64>::zeros(n_inliers, ncoeffs);
    let mut bx_vec = DVector::<f64>::zeros(n_inliers);
    let mut by_vec = DVector::<f64>::zeros(n_inliers);

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

    let svd = a_mat.svd(true, true);

    if let Ok(cx) = svd.solve(&bx_vec, 1e-12) {
        for j in 0..ncoeffs {
            a_coeffs[j] = cx[j];
        }
    }

    if let Ok(cy) = svd.solve(&by_vec, 1e-12) {
        for j in 0..ncoeffs {
            b_coeffs[j] = cy[j];
        }
    }
}

/// Fit the inverse polynomial (distorted → ideal) by least-squares.
///
/// Model: x_ideal = x_obs + Σ AP_pq · u_d^p · v_d^q   (u_d = x_obs/scale, v_d = y_obs/scale)
pub(super) fn fit_inverse_poly_ls(
    points: &[MatchedPoint],
    mask: &[bool],
    pairs: &[(u32, u32)],
    scale: f64,
    ap_coeffs: &mut [f64],
    bp_coeffs: &mut [f64],
) {
    let ncoeffs = pairs.len();
    let n_inliers: usize = mask.iter().filter(|&&m| m).count();
    if n_inliers < ncoeffs {
        return;
    }

    let mut a_mat = DMatrix::<f64>::zeros(n_inliers, ncoeffs);
    let mut bx_vec = DVector::<f64>::zeros(n_inliers);
    let mut by_vec = DVector::<f64>::zeros(n_inliers);

    let mut row = 0;
    for (i, p) in points.iter().enumerate() {
        if !mask[i] {
            continue;
        }
        let u = p.x_obs / scale;
        let v = p.y_obs / scale;

        for (j, &(pp, qq)) in pairs.iter().enumerate() {
            a_mat[(row, j)] = u.powi(pp as i32) * v.powi(qq as i32);
        }
        bx_vec[row] = (p.x_ideal - p.x_obs) / scale;
        by_vec[row] = (p.y_ideal - p.y_obs) / scale;
        row += 1;
    }

    let svd = a_mat.svd(true, true);

    if let Ok(cx) = svd.solve(&bx_vec, 1e-12) {
        for j in 0..ncoeffs {
            ap_coeffs[j] = cx[j];
        }
    }

    if let Ok(cy) = svd.solve(&by_vec, 1e-12) {
        for j in 0..ncoeffs {
            bp_coeffs[j] = cy[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            "k1: fitted={:.6e}, true={:.6e}, diff={:.3e}",
            k1,
            true_k1,
            (k1 - true_k1).abs()
        );
        assert!(
            (k2 - true_k2).abs() < 1e-18,
            "k2: fitted={:.6e}, true={:.6e}, diff={:.3e}",
            k2,
            true_k2,
            (k2 - true_k2).abs()
        );
        assert!(k3.abs() < 1e-18, "k3: fitted={:.3e}, expected ~0", k3);
    }
}
