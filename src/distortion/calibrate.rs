//! Camera calibration from plate-solve results.
//!
//! Given one or more plate-solve results, fits a [`CameraModel`] by fitting a
//! distortion model — either SIP polynomial or Brown-Conrady radial — to the
//! matched star pairs. Selected via [`CalibrateConfig::model`].
//!
//! For each model, single-image calibration delegates to the standalone fitter
//! ([`fit_polynomial_distortion`](super::fit::fit_polynomial_distortion) /
//! [`fit_radial_distortion`](super::fit::fit_radial_distortion)). Multi-image
//! calibration uses alternating per-image attitude refinement (via WCS refine)
//! and a global fit, which correctly handles different per-image pointings.

use numeris::Matrix3;
use tracing::debug;

use crate::camera_model::CameraModel;
use crate::centroid::Centroid;
use crate::distortion::fit::{
    fit_polynomial_distortion, fit_radial_distortion, DistortionFitConfig,
};
use crate::solver::wcs_refine;
use crate::solver::{SolveResult, SolveStatus, SolverDatabase};

use super::fit::{
    build_id_lookup, compute_corrected_rmse, fit_polynomial_sigma_clip,
    fit_radial_centered_sigma_clip, MatchedPoint,
};
use super::polynomial::{num_coeffs, PolynomialDistortion};
use super::radial::RadialDistortion;
use super::Distortion;

/// Distortion model selector for [`calibrate_camera`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistortionModelType {
    /// SIP-like polynomial of the given order (2..=6). Captures arbitrary 2D
    /// distortion including tangential/decentering — preferred for off-axis
    /// CCDs and astronomy WCS.
    Polynomial { order: u32 },
    /// Brown-Conrady radial `(k1, k2, k3)`. Three parameters total —
    /// well-conditioned with few matches; the standard model in computer-vision
    /// camera calibration. Assumes distortion is symmetric about the optical
    /// center.
    Radial,
}

impl Default for DistortionModelType {
    fn default() -> Self {
        DistortionModelType::Polynomial { order: 4 }
    }
}

/// Configuration for camera calibration.
#[derive(Debug, Clone)]
pub struct CalibrateConfig {
    /// Distortion model to fit. Default: `Polynomial { order: 4 }`.
    pub model: DistortionModelType,
    /// Maximum iterations for sigma-clipping. Default 20.
    pub max_iterations: u32,
    /// Sigma threshold for MAD-based outlier rejection. Default 3.0.
    pub sigma_clip: f64,
    /// Convergence threshold for multi-image outer loop RMSE change. Default 0.01.
    pub convergence_threshold_px: f64,
}

impl Default for CalibrateConfig {
    fn default() -> Self {
        Self {
            model: DistortionModelType::default(),
            max_iterations: 20,
            sigma_clip: 3.0,
            convergence_threshold_px: 0.01,
        }
    }
}

/// Result of camera calibration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CalibrateResult {
    /// The fitted camera model (focal length, crpix, distortion).
    pub camera_model: CameraModel,
    /// RMS residual in pixels before calibration.
    pub rmse_before_px: f64,
    /// RMS residual in pixels after calibration.
    pub rmse_after_px: f64,
    /// Number of inlier star matches used.
    pub n_inliers: usize,
    /// Number of outlier star matches rejected.
    pub n_outliers: usize,
    /// Number of sigma-clip iterations performed.
    pub iterations: u32,
}

/// Calibrate a camera model from one or more plate-solve results.
///
/// Each solve result must have `status == MatchFound` and provide matched catalog IDs
/// and centroid indices. The corresponding centroid arrays must be provided in the same
/// order.
///
/// The distortion model fit is controlled by [`CalibrateConfig::model`] —
/// SIP polynomial (default) or radial Brown-Conrady. For a single image, the
/// fitter pools all matched points and runs sigma-clipped LS in one pass. For
/// multiple images, alternating per-image attitude refinement and global
/// fitting separates per-image pointing from shared distortion.
///
/// For polynomial models, the resulting `CameraModel` has `crpix` extracted
/// from the polynomial's order-0 terms (representing the optical center
/// offset). For radial models, `crpix` is `[0, 0]` since the radial form has
/// no constant offset to absorb. `focal_length_px` is derived from the median
/// solve FOV in both cases.
pub fn calibrate_camera(
    solve_results: &[&SolveResult],
    centroids: &[&[Centroid]],
    database: &SolverDatabase,
    image_width: u32,
    image_height: u32,
    config: &CalibrateConfig,
) -> CalibrateResult {
    assert_eq!(
        solve_results.len(),
        centroids.len(),
        "solve_results and centroids must have the same length"
    );
    if let DistortionModelType::Polynomial { order } = config.model {
        assert!(
            (2..=6).contains(&order),
            "polynomial order must be in [2, 6]"
        );
    }

    // Count valid (MatchFound) solves
    let n_valid = solve_results
        .iter()
        .filter(|sr| sr.status == SolveStatus::MatchFound && sr.qicrs2cam.is_some())
        .count();

    if n_valid <= 1 {
        single_image_calibrate(solve_results, centroids, database, image_width, image_height, config)
    } else {
        multi_image_calibrate(solve_results, centroids, database, image_width, image_height, config)
    }
}

/// Extract optical center offset from polynomial order-0 terms into crpix.
///
/// The forward polynomial's constant terms A_00, B_00 give the observed pixel
/// position when the ideal pixel is at the origin — i.e., where the optical
/// center lands on the sensor. Since the pipeline is `pixel - crpix → undistort`,
/// we set `crpix = [A_00, B_00] * scale` and zero out the constant terms.
///
/// This separates the physical optical center offset (crpix) from the actual lens
/// distortion (order 2+), making the camera model more interpretable.
fn extract_crpix(distortion: Distortion) -> ([f64; 2], Distortion) {
    match distortion {
        Distortion::Polynomial(poly) => {
            // A_00 and B_00 are the forward polynomial's constant terms.
            // distort(0, 0) = (A_00, B_00) * scale = optical center on sensor.
            let crpix_x = poly.a_coeffs[0] * poly.scale;
            let crpix_y = poly.b_coeffs[0] * poly.scale;

            // Zero out order-0 terms in the forward polynomial. The inverse
            // (ap/bp) coefficients are no longer fit (Newton iteration on the
            // forward polynomial replaced separate-inverse evaluation); they
            // remain zero-valued for binary-format compatibility.
            let mut a = poly.a_coeffs.clone();
            let mut b = poly.b_coeffs.clone();
            a[0] = 0.0;
            b[0] = 0.0;

            let new_poly = PolynomialDistortion::new(
                poly.order,
                poly.scale,
                a,
                b,
                poly.ap_coeffs,
                poly.bp_coeffs,
            );
            ([crpix_x, crpix_y], Distortion::Polynomial(new_poly))
        }
        other => ([0.0, 0.0], other),
    }
}

/// Single-image calibration: pools matched points and runs the appropriate
/// sigma-clipped fitter.
fn single_image_calibrate(
    solve_results: &[&SolveResult],
    centroids: &[&[Centroid]],
    database: &SolverDatabase,
    image_width: u32,
    image_height: u32,
    config: &CalibrateConfig,
) -> CalibrateResult {
    let fit_config = DistortionFitConfig {
        sigma_clip: config.sigma_clip,
        max_iterations: config.max_iterations,
        stage2_threshold_px: Some(5.0),
    };

    let fit_result = match config.model {
        DistortionModelType::Polynomial { order } => fit_polynomial_distortion(
            solve_results,
            centroids,
            database,
            image_width,
            order,
            &fit_config,
        ),
        DistortionModelType::Radial => fit_radial_distortion(
            solve_results,
            centroids,
            database,
            image_width,
            &fit_config,
        ),
    };

    // Get FOV from first successful solve result
    let fov_rad = solve_results
        .iter()
        .find_map(|sr| sr.fov_rad)
        .unwrap_or(0.1);

    // Detect parity from solve results
    let parity_flip = solve_results
        .iter()
        .find(|sr| sr.status == SolveStatus::MatchFound)
        .map_or(false, |sr| sr.parity_flip);

    // Polynomial: extract crpix from polynomial order-0 terms.
    // Radial: fit_result.crpix already carries the fitted optical-center
    //         offset (jointly fit with k1/k2/k3 via Gauss-Newton).
    let (crpix, distortion) = match fit_result.crpix {
        Some(c) => (c, fit_result.model),
        None => extract_crpix(fit_result.model),
    };

    let cam = CameraModel {
        focal_length_px: image_width as f64 / fov_rad as f64,
        image_width,
        image_height,
        crpix,
        parity_flip,
        distortion,
    };

    debug!(
        "calibrate_camera (single, {:?}): crpix=[{:.2}, {:.2}], RMSE {:.3} -> {:.3} px, {}/{} inliers",
        config.model,
        crpix[0], crpix[1],
        fit_result.rmse_before_px,
        fit_result.rmse_after_px,
        fit_result.n_inliers,
        fit_result.n_inliers + fit_result.n_outliers,
    );

    CalibrateResult {
        camera_model: cam,
        rmse_before_px: fit_result.rmse_before_px,
        rmse_after_px: fit_result.rmse_after_px,
        n_inliers: fit_result.n_inliers,
        n_outliers: fit_result.n_outliers,
        iterations: fit_result.iterations,
    }
}

/// Multi-image calibration: alternating per-image attitude refinement + global fit.
///
/// Dispatches on `config.model` for the global-fit step (Phase 3).
fn multi_image_calibrate(
    solve_results: &[&SolveResult],
    centroids: &[&[Centroid]],
    database: &SolverDatabase,
    image_width: u32,
    image_height: u32,
    config: &CalibrateConfig,
) -> CalibrateResult {
    let scale = image_width as f64 / 2.0;

    // Build catalog ID -> star_vectors index lookup
    let id_to_idx = build_id_lookup(database);

    // Compute global properties from valid solves
    let mut fovs: Vec<f32> = Vec::new();
    let mut parity_flip = false;
    for sr in solve_results.iter() {
        if sr.status != SolveStatus::MatchFound {
            continue;
        }
        if let Some(fov) = sr.fov_rad {
            fovs.push(fov);
        }
        parity_flip = sr.parity_flip;
    }
    fovs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_fov = fovs[fovs.len() / 2];
    // True pinhole pixel scale (1/f) from median angular FOV.
    let global_pixel_scale = {
        let f = (image_width as f64 / 2.0) / (median_fov as f64 / 2.0).tan();
        1.0 / f
    };
    let parity_sign: f64 = if parity_flip { -1.0 } else { 1.0 };

    debug!(
        "calibrate_camera (multi): {} valid images, median FOV={:.3} deg, parity={}",
        fovs.len(),
        median_fov.to_degrees(),
        parity_flip,
    );

    // Current distortion model (starts as identity). For polynomial fits the
    // crpix offset is absorbed into the polynomial's order-0 terms and stays
    // [0, 0] until extracted at the end. For centered radial fits, the crpix
    // is fit jointly with the radial coefficients each iteration; subsequent
    // Phase-1 wcs_refines need it to undistort centroids correctly.
    let mut current_distortion = Distortion::None;
    let mut current_crpix = [0.0_f64, 0.0];
    let mut last_rmse = f64::MAX;
    let mut last_rmse_before = 0.0_f64;

    let fit_config = DistortionFitConfig {
        sigma_clip: config.sigma_clip,
        max_iterations: config.max_iterations,
        stage2_threshold_px: Some(5.0),
    };

    // Precompute per-image data that doesn't change across iterations
    struct ImageData {
        sr_idx: usize,
        rotation: Matrix3<f32>,
        fov_rad: f32,
    }

    let mut image_data: Vec<ImageData> = Vec::new();
    for (idx, sr) in solve_results.iter().enumerate() {
        if sr.status != SolveStatus::MatchFound {
            continue;
        }
        let quat = match &sr.qicrs2cam {
            Some(q) => q,
            None => continue,
        };
        let fov = match sr.fov_rad {
            Some(f) => f,
            None => continue,
        };
        let rot: Matrix3<f32> = quat.to_rotation_matrix();
        image_data.push(ImageData {
            sr_idx: idx,
            rotation: rot,
            fov_rad: fov,
        });
    }

    let mut total_iterations = 0u32;
    let mut final_mask = Vec::new();
    let mut final_n_points = 0usize;

    // ── Outer alternation loop ──
    for outer in 0..3 {
        // ── Phase 1: Per-image attitude refinement ──
        // For each image, undistort centroids with current model, then refine attitude.
        struct RefinedImage {
            sr_idx: usize,
            matches: Vec<(usize, usize)>,   // (centroid_idx_in_full_array, catalog_star_idx)
            crval_ra: f64,
            crval_dec: f64,
            cd_matrix: [[f64; 2]; 2],
        }

        let mut refined_images: Vec<RefinedImage> = Vec::new();

        for img in &image_data {
            let sr = solve_results[img.sr_idx];
            let cents = centroids[img.sr_idx];

            // Per-image true pinhole pixel scale (1/f) from angular FOV.
            let per_image_ps = {
                let f = (image_width as f64 / 2.0) / (img.fov_rad as f64 / 2.0).tan();
                1.0 / f
            };

            // Preprocess centroids: subtract crpix → undistort → re-add crpix → parity.
            // For polynomial models, current_crpix is [0, 0] (offset is in the
            // polynomial's order-0 terms). For centered radial, current_crpix
            // carries the optical-axis offset so the radial undistort sees
            // optical-axis-centered coordinates.
            let centroids_px: Vec<(f64, f64)> = cents
                .iter()
                .map(|c| {
                    let cx = c.x as f64 - current_crpix[0];
                    let cy = c.y as f64 - current_crpix[1];
                    let (ux, uy) = current_distortion.undistort(cx, cy);
                    let ux = ux + current_crpix[0];
                    let uy = uy + current_crpix[1];
                    (parity_sign * ux, uy)
                })
                .collect();

            // Build initial matches from SolveResult
            // matched_centroid_indices are indices into the original centroid array
            let mut initial_matches: Vec<(usize, usize)> = Vec::new();
            for (match_idx, &cat_id) in sr.matched_catalog_ids.iter().enumerate() {
                let cent_idx = sr.matched_centroid_indices[match_idx];
                if cent_idx >= cents.len() {
                    continue;
                }
                if let Some(&star_idx) = id_to_idx.get(&cat_id) {
                    initial_matches.push((cent_idx, star_idx));
                }
            }

            if initial_matches.len() < 4 {
                continue;
            }

            // Compute match radius from FOV
            let match_radius_rad = 0.01 * img.fov_rad;

            // Call wcs_refine for this image
            let wcs_result = wcs_refine::wcs_refine(
                &img.rotation,
                &initial_matches,
                &centroids_px,
                &database.star_vectors,
                &database.star_catalog,
                per_image_ps,
                parity_flip,
                match_radius_rad,
                cents.len().min(500),
                10,
            );

            if wcs_result.matches.len() < 4 {
                debug!(
                    "  multi-cal outer {}: image {} wcs_refine returned only {} matches, skipping",
                    outer, img.sr_idx, wcs_result.matches.len()
                );
                continue;
            }

            debug!(
                "  multi-cal outer {}: image {} refined: {} matches, RMSE={:.2}\"",
                outer,
                img.sr_idx,
                wcs_result.matches.len(),
                wcs_result.rmse_rad.to_degrees() * 3600.0,
            );

            refined_images.push(RefinedImage {
                sr_idx: img.sr_idx,
                matches: wcs_result.matches,
                crval_ra: wcs_result.crval_rad[0],
                crval_dec: wcs_result.crval_rad[1],
                cd_matrix: wcs_result.cd_matrix,
            });
        }

        if refined_images.is_empty() {
            debug!("  multi-cal outer {}: no refined images, aborting", outer);
            break;
        }

        // ── Phase 2: Gather refined matched points ──
        let mut all_points: Vec<MatchedPoint> = Vec::new();

        for ref_img in &refined_images {
            let cents = centroids[ref_img.sr_idx];

            // Derive rotation matrix from refined WCS
            let (rot, _fov, _parity) = wcs_refine::wcs_to_rotation(
                &ref_img.cd_matrix,
                ref_img.crval_ra,
                ref_img.crval_dec,
                image_width,
            );

            for &(cent_idx, cat_idx) in &ref_img.matches {
                let sv = &database.star_vectors[cat_idx];
                let icrs_v = numeris::Vector3::from_array([sv[0], sv[1], sv[2]]);
                let cam_v = rot * icrs_v;

                if cam_v[2] <= 0.0 {
                    continue;
                }

                // Ideal position using global pixel scale (consistent across all images)
                let x_ideal = parity_sign * (cam_v[0] as f64) / (cam_v[2] as f64) / global_pixel_scale;
                let y_ideal = (cam_v[1] as f64) / (cam_v[2] as f64) / global_pixel_scale;

                // Observed position: raw centroid (no undistortion applied)
                let x_obs = cents[cent_idx].x as f64;
                let y_obs = cents[cent_idx].y as f64;

                all_points.push(MatchedPoint {
                    x_obs,
                    y_obs,
                    x_ideal,
                    y_ideal,
                });
            }
        }

        let min_points = match config.model {
            DistortionModelType::Polynomial { order } => num_coeffs(order),
            DistortionModelType::Radial => 3,
        };
        if all_points.len() < min_points {
            debug!(
                "  multi-cal outer {}: too few points ({}) for {:?} fit",
                outer,
                all_points.len(),
                config.model,
            );
            break;
        }

        debug!(
            "  multi-cal outer {}: {} total matched points from {} images",
            outer,
            all_points.len(),
            refined_images.len(),
        );

        // ── Phase 3: Global model fit ──
        // Polynomial: fit absorbs optical-center offset into the order-0
        // (constant) terms; current_crpix stays [0, 0] until extract_crpix
        // pulls it out at the end.
        // Radial: nonlinear LS jointly fits (cx, cy, k1, k2, k3); the fitted
        // (cx, cy) becomes current_crpix and the radial coefficients stay
        // pure (k1, k2, k3).
        let (dist, fit_crpix, mask, iters) = match config.model {
            DistortionModelType::Polynomial { order } => {
                let fit = fit_polynomial_sigma_clip(&all_points, order, scale, &fit_config);
                let model = PolynomialDistortion::new(
                    order,
                    scale,
                    fit.a_coeffs,
                    fit.b_coeffs,
                    fit.ap_coeffs,
                    fit.bp_coeffs,
                );
                (
                    Distortion::Polynomial(model),
                    [0.0, 0.0],
                    fit.mask,
                    fit.iterations,
                )
            }
            DistortionModelType::Radial => {
                let fit = fit_radial_centered_sigma_clip(&all_points, &fit_config);
                let model =
                    RadialDistortion::with_tangential(fit.k1, fit.k2, fit.k3, fit.p1, fit.p2);
                (
                    Distortion::Radial(model),
                    [fit.cx, fit.cy],
                    fit.mask,
                    fit.iterations,
                )
            }
        };

        let n_inliers = mask.iter().filter(|&&m| m).count();
        // Compute RMSE in the appropriate frame:
        // - polynomial absorbs crpix internally, so call the existing helper
        // - radial centers on fit_crpix, so shift before calling distort
        let rmse_after = if fit_crpix == [0.0, 0.0] {
            compute_corrected_rmse(&all_points, &mask, &dist)
        } else {
            // Centered radial: distort on (x_ideal - cx, y_ideal - cy)
            // and shift result back. Use a local helper closure.
            let mut sum_sq = 0.0_f64;
            let mut nn = 0usize;
            for (i, p) in all_points.iter().enumerate() {
                if !mask[i] {
                    continue;
                }
                let (dx, dy) = dist.distort(p.x_ideal - fit_crpix[0], p.y_ideal - fit_crpix[1]);
                let pred_x = dx + fit_crpix[0];
                let pred_y = dy + fit_crpix[1];
                let rx = p.x_obs - pred_x;
                let ry = p.y_obs - pred_y;
                sum_sq += rx * rx + ry * ry;
                nn += 1;
            }
            if nn == 0 { 0.0 } else { (sum_sq / nn as f64).sqrt() }
        };
        let rmse_before = compute_corrected_rmse(&all_points, &mask, &Distortion::None);

        debug!(
            "  multi-cal outer {}: {:?} fit: {}/{} inliers, RMSE {:.3} -> {:.3} px",
            outer, config.model, n_inliers, all_points.len(), rmse_before, rmse_after,
        );

        total_iterations += iters;
        final_mask = mask;
        final_n_points = all_points.len();
        current_distortion = dist;
        current_crpix = fit_crpix;
        last_rmse_before = rmse_before;

        // Check convergence
        let rmse_change = (last_rmse - rmse_after).abs();
        let rmse_frac_change = if last_rmse > 1e-12 {
            rmse_change / last_rmse
        } else {
            0.0
        };

        last_rmse = rmse_after;

        if rmse_frac_change < 0.01 || rmse_change < config.convergence_threshold_px {
            debug!(
                "  multi-cal: converged at outer iteration {} (RMSE change={:.4} px, {:.2}%)",
                outer, rmse_change, rmse_frac_change * 100.0,
            );
            break;
        }
    }

    // Build final CameraModel.
    // Polynomial: extract crpix from order-0 terms via extract_crpix.
    // Radial: current_crpix already holds the fitted optical-center offset,
    //         and the distortion is pure (k1, k2, k3) — no extraction needed.
    let (crpix, distortion) = match current_distortion {
        Distortion::Polynomial(_) => extract_crpix(current_distortion),
        _ => (current_crpix, current_distortion),
    };

    let cam = CameraModel {
        focal_length_px: image_width as f64 / median_fov as f64,
        image_width,
        image_height,
        crpix,
        parity_flip,
        distortion,
    };

    let n_inliers = final_mask.iter().filter(|&&m| m).count();

    debug!(
        "calibrate_camera (multi, {:?}): crpix=[{:.2}, {:.2}], RMSE {:.3} -> {:.3} px, {}/{} inliers",
        config.model, crpix[0], crpix[1], last_rmse_before, last_rmse, n_inliers, final_n_points,
    );

    CalibrateResult {
        camera_model: cam,
        rmse_before_px: last_rmse_before,
        rmse_after_px: last_rmse,
        n_inliers,
        n_outliers: final_n_points - n_inliers,
        iterations: total_iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibrate_config_defaults() {
        let cfg = CalibrateConfig::default();
        assert!(matches!(
            cfg.model,
            DistortionModelType::Polynomial { order: 4 }
        ));
        assert_eq!(cfg.max_iterations, 20);
        assert!((cfg.sigma_clip - 3.0).abs() < 1e-12);
        assert!((cfg.convergence_threshold_px - 0.01).abs() < 1e-12);
    }
}
