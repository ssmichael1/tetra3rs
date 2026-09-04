//! Camera calibration from plate-solve results.
//!
//! Given one or more plate-solve results, fits a [`CameraModel`] by fitting a
//! distortion model — either SIP polynomial or Brown-Conrady radial — to the
//! matched star pairs. Selected via [`CalibrateConfig::model`].
//!
//! Both paths pool matched `(observed, ideal)` points and run the shared
//! sigma-clipped fitter ([`fit_pooled`](super::fit::fit_pooled)). Single-image
//! calibration projects the ideal points with the solve's own attitude;
//! multi-image calibration alternates per-image attitude refinement (via WCS
//! refine) with the global fit, which correctly handles different per-image
//! pointings.

use numeris::Matrix3;
use tracing::debug;

use crate::camera_model::CameraModel;
use crate::centroid::Centroid;
use crate::solver::wcs_refine;
use crate::solver::{focal_length_from_fov, pixel_scale_from_fov, SolveResult, SolverDatabase};

use super::fit::{
    build_id_lookup, fit_pooled, matched_pairs, min_points, project_matches, DistortionFitConfig,
    MatchedPoint,
};
use super::polynomial::PolynomialDistortion;
use super::Distortion;

/// Distortion model selector for [`calibrate_camera`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistortionModelType {
    /// SIP-like polynomial of the given order (2..=6). Captures arbitrary 2D
    /// distortion including tangential/decentering — preferred for off-axis
    /// CCDs and astronomy WCS.
    Polynomial { order: u32 },
    /// Brown-Conrady radial `(k1, k2, k3)` + tangential `(p1, p2)`, with the
    /// optical center and a focal-scale factor fit jointly (8 parameters) —
    /// the standard model in computer-vision camera calibration. Assumes the
    /// radial term is symmetric about the fitted optical axis.
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

impl From<&CalibrateConfig> for DistortionFitConfig {
    /// The fitter settings a calibration run passes down.
    fn from(config: &CalibrateConfig) -> Self {
        Self {
            sigma_clip: config.sigma_clip,
            max_iterations: config.max_iterations,
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
/// Failed solves (`Err`) in `solve_results` are skipped; each successful one
/// provides matched catalog IDs and centroid indices into the corresponding
/// entry of `centroids` (same order).
///
/// The distortion model fit is controlled by [`CalibrateConfig::model`] —
/// SIP polynomial (default) or radial Brown-Conrady. For a single image, the
/// fitter pools all matched points and runs sigma-clipped LS in one pass. For
/// multiple images, alternating per-image attitude refinement and global
/// fitting separates per-image pointing from shared distortion.
///
/// For polynomial models, the resulting `CameraModel` has `crpix` extracted
/// from the polynomial's order-0 terms (representing the optical center
/// offset), and `focal_length_px` derived from the median solve FOV. For
/// radial models, the jointly-fit optical-axis position is carried inside
/// the distortion model itself ([`RadialDistortion::center`]) — `crpix`
/// (the projection origin) stays `[0, 0]` — and `focal_length_px` folds in
/// the jointly-fit focal-scale factor (the solve FOV is a whole-field
/// average biased by distortion, and the Brown-Conrady form has no linear
/// term of its own).
///
/// [`RadialDistortion::center`]: super::radial::RadialDistortion::center
///
/// # Errors
///
/// Returns [`Error::InvalidInput`](crate::Error::InvalidInput) when the input
/// carries no successful solves, when no solves agree on parity, or when too
/// few matched points remain for any distortion fit to complete — cases that
/// would otherwise yield a fabricated (identity / infinite-RMSE) camera model.
pub fn calibrate_camera(
    solve_results: &[&SolveResult],
    centroids: &[&[Centroid]],
    database: &SolverDatabase,
    image_width: u32,
    image_height: u32,
    config: &CalibrateConfig,
) -> crate::Result<CalibrateResult> {
    if solve_results.len() != centroids.len() {
        return Err(crate::Error::InvalidInput(format!(
            "calibrate_camera: solve_results ({}) and centroids ({}) must have the same length",
            solve_results.len(),
            centroids.len()
        )));
    }
    if let DistortionModelType::Polynomial { order } = config.model {
        if !(2..=6).contains(&order) {
            return Err(crate::Error::InvalidInput(format!(
                "calibrate_camera: polynomial order must be in [2, 6], got {order}"
            )));
        }
    }

    // Count valid (successful) solves. With none, there is no attitude/FOV to
    // anchor the fit — returning a fabricated model would be silent garbage.
    let n_valid = solve_results.iter().filter(|sr| sr.is_ok()).count();
    if n_valid == 0 {
        return Err(crate::Error::InvalidInput(
            "calibrate_camera: no successful solves in input".into(),
        ));
    }

    let result = if n_valid == 1 {
        single_image_calibrate(
            solve_results,
            centroids,
            database,
            image_width,
            image_height,
            config,
        )?
    } else {
        multi_image_calibrate(
            solve_results,
            centroids,
            database,
            image_width,
            image_height,
            config,
        )?
    };

    // The fitters guard their own inputs, but a model that fails the same
    // invariants every load path enforces must never be handed back as `Ok`.
    result.camera_model.validate().map_err(|e| {
        crate::Error::InvalidInput(format!(
            "calibrate_camera: fitted camera model is invalid ({e})"
        ))
    })?;
    Ok(result)
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

            let new_poly = PolynomialDistortion::new(poly.order, poly.scale, a, b);
            ([crpix_x, crpix_y], Distortion::Polynomial(new_poly))
        }
        // Matched explicitly (no catch-all) so a future variant carrying its
        // own optical-center offset is caught by the compiler here, like the
        // multi-image path's match does.
        Distortion::None => ([0.0, 0.0], Distortion::None),
        Distortion::Radial(r) => ([0.0, 0.0], Distortion::Radial(r)),
    }
}

/// Single-image calibration: projects the solve's matched stars with its own
/// attitude and FOV, then runs the shared pooled fitter. The caller guarantees
/// exactly one successful solve.
fn single_image_calibrate(
    solve_results: &[&SolveResult],
    centroids: &[&[Centroid]],
    database: &SolverDatabase,
    image_width: u32,
    image_height: u32,
    config: &CalibrateConfig,
) -> crate::Result<CalibrateResult> {
    let (solution, cents) = solve_results
        .iter()
        .zip(centroids)
        .find_map(|(sr, cents)| sr.as_ref().ok().map(|sol| (sol, *cents)))
        .ok_or_else(|| {
            crate::Error::InvalidInput("calibrate_camera: no successful solves in input".into())
        })?;
    let fov_rad = solution.fov_rad;
    let parity_flip = solution.parity_flip;
    let parity_sign: f64 = if parity_flip { -1.0 } else { 1.0 };

    // Ideal points from the solve's own attitude at the true pinhole pixel
    // scale (1/f) implied by its FOV.
    let id_to_idx = build_id_lookup(database);
    let mut points: Vec<MatchedPoint> = Vec::new();
    project_matches(
        solution.qicrs2cam.to_rotation_matrix(),
        matched_pairs(solution, cents.len(), &id_to_idx),
        cents,
        &database.star_vectors,
        parity_sign,
        pixel_scale_from_fov(image_width, fov_rad as f64),
        &mut points,
    );

    let scale = image_width as f64 / 2.0;
    let fit_config = DistortionFitConfig::from(config);
    let fit = fit_pooled(&points, config.model, scale, &fit_config).ok_or_else(|| {
        crate::Error::InvalidInput(format!(
            "calibrate_camera: too few matched points ({}) for a {:?} distortion fit",
            points.len(),
            config.model
        ))
    })?;

    // Anchor focal length implied by the solve FOV (tan-consistent with the
    // ideal-point projection above), corrected by the jointly-fit linear
    // scale term (1.0 for polynomial fits).
    let f_anchor = focal_length_from_fov(image_width, fov_rad as f64);
    let focal_length_px = f_anchor * fit.focal_scale;

    // Polynomial: extract crpix from the polynomial's order-0 terms.
    // Radial: the jointly-fit optical-axis position stays inside the model
    //         (RadialDistortion::center), so extract_crpix leaves crpix [0, 0].
    let n_inliers = fit.n_inliers();
    let (crpix, distortion) = extract_crpix(fit.model);

    debug!(
        "calibrate_camera (single, {:?}): crpix=[{:.2}, {:.2}], RMSE {:.3} -> {:.3} px, {}/{} inliers",
        config.model,
        crpix[0],
        crpix[1],
        fit.rmse_before_px,
        fit.rmse_after_px,
        n_inliers,
        points.len(),
    );

    Ok(CalibrateResult {
        camera_model: CameraModel {
            focal_length_px,
            image_width,
            image_height,
            crpix,
            parity_flip,
            distortion,
        },
        rmse_before_px: fit.rmse_before_px,
        rmse_after_px: fit.rmse_after_px,
        n_inliers,
        n_outliers: points.len() - n_inliers,
        iterations: fit.iterations,
    })
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
) -> crate::Result<CalibrateResult> {
    let scale = image_width as f64 / 2.0;

    // Build catalog ID -> star_vectors index lookup
    let id_to_idx = build_id_lookup(database);

    // Compute global properties from valid solves. Parity is a physical
    // property of the camera, so all images must agree; take the majority
    // vote, and below exclude any image that disagrees — a lone
    // opposite-parity solve is a false (mirror-image) match whose star
    // correspondences would poison the pooled distortion fit.
    let n_valid = solve_results.iter().filter(|sr| sr.is_ok()).count();
    let n_flipped = solve_results
        .iter()
        .copied()
        .flatten()
        .filter(|sol| sol.parity_flip)
        .count();
    let parity_flip = 2 * n_flipped > n_valid;

    let mut fovs: Vec<f32> = solve_results
        .iter()
        .copied()
        .flatten()
        .filter(|sol| sol.parity_flip == parity_flip)
        .map(|sol| sol.fov_rad)
        .collect();
    fovs.sort_by(f32::total_cmp);
    // With n_valid >= 2 the majority-parity set is non-empty, but guard the
    // index anyway rather than panic on a pathological all-disagree input.
    if fovs.is_empty() {
        return Err(crate::Error::InvalidInput(
            "calibrate_camera: no solves agree on parity".into(),
        ));
    }
    let median_fov = fovs[fovs.len() / 2];
    // True pinhole pixel scale (1/f) from median angular FOV. The median
    // solve FOV is a whole-field average biased by distortion, so this is
    // only an anchor: radial fits jointly fit a linear scale correction
    // that gets folded in below (the final focal length no longer depends
    // on the median FOV estimate).
    let mut global_pixel_scale = pixel_scale_from_fov(image_width, median_fov as f64);
    let parity_sign: f64 = if parity_flip { -1.0 } else { 1.0 };

    debug!(
        "calibrate_camera (multi): {} valid images, median FOV={:.3} deg, parity={}",
        fovs.len(),
        median_fov.to_degrees(),
        parity_flip,
    );

    // Current distortion model (starts as identity). The projection origin
    // (crpix) stays at the image center throughout: polynomial fits keep the
    // optical-center offset in their order-0 terms (extracted at the very end),
    // and radial fits keep the optical-axis position inside the model
    // (RadialDistortion::center).
    let mut current_distortion = Distortion::None;
    // Cumulative focal-length correction from the radial fits' linear scale
    // term (γ product). Applied to both the global and per-image scales so
    // every phase of the alternation agrees on the corrected frame.
    let mut scale_correction = 1.0_f64;
    let mut last_rmse = f64::MAX;
    let mut last_rmse_before = 0.0_f64;

    let fit_config = DistortionFitConfig::from(config);

    // Precompute per-image data that doesn't change across iterations.
    // Holding the &Solution (rather than an index back into solve_results)
    // makes "image_data only contains successful solves" structural.
    struct ImageData<'a> {
        /// Index in the caller's input slices — used only for log messages.
        idx: usize,
        sol: &'a crate::solver::Solution,
        centroids: &'a [Centroid],
        rotation: Matrix3<f32>,
        fov_rad: f32,
    }

    let mut image_data: Vec<ImageData> = Vec::new();
    for (idx, sr) in solve_results.iter().enumerate() {
        let Ok(sol) = sr else {
            continue;
        };
        if sol.parity_flip != parity_flip {
            debug!(
                "calibrate_camera (multi): image {} parity_flip={} disagrees with consensus {} — \
                 likely a false mirror-image solve; excluding from calibration",
                idx, sol.parity_flip, parity_flip,
            );
            continue;
        }
        image_data.push(ImageData {
            idx,
            sol,
            centroids: centroids[idx],
            rotation: sol.qicrs2cam.to_rotation_matrix(),
            fov_rad: sol.fov_rad,
        });
    }

    let mut total_iterations = 0u32;
    let mut final_mask = Vec::new();
    let mut final_n_points = 0usize;

    // ── Outer alternation loop ──
    for outer in 0..3 {
        // ── Phase 1: Per-image attitude refinement ──
        // For each image, undistort centroids with current model, then refine attitude.
        struct RefinedImage<'a> {
            centroids: &'a [Centroid],
            matches: Vec<(usize, usize)>, // (centroid_idx_in_full_array, catalog_star_idx)
            crval_ra: f64,
            crval_dec: f64,
            cd_matrix: [[f64; 2]; 2],
        }

        let mut refined_images: Vec<RefinedImage> = Vec::new();

        for img in &image_data {
            let cents = img.centroids;

            // Per-image true pinhole pixel scale (1/f) from angular FOV,
            // with the cumulative linear scale correction from prior radial
            // fits applied. wcs_refine locks the pixel scale, so without
            // this the per-image refines would fight the corrected global
            // frame and the alternation drifts instead of converging.
            let per_image_ps = {
                let f = focal_length_from_fov(image_width, img.fov_rad as f64);
                1.0 / (f * scale_correction)
            };

            // Preprocess centroids: undistort → parity. crpix is at the image
            // center (both models re-center internally), so there is no crpix
            // shift to apply here.
            let centroids_px: Vec<(f64, f64)> = cents
                .iter()
                .map(|c| {
                    let (ux, uy) = current_distortion.undistort(c.x as f64, c.y as f64);
                    (parity_sign * ux, uy)
                })
                .collect();

            // Build initial matches from the solution's matched pairs
            // (centroid indices are into the original centroid array).
            // Non-finite centroids (e.g. a caller passing a different array
            // than the one solved) would poison the 3×3 normal equations.
            let initial_matches: Vec<(usize, usize)> =
                matched_pairs(img.sol, cents.len(), &id_to_idx)
                    .filter(|&(ci, _)| cents[ci].x.is_finite() && cents[ci].y.is_finite())
                    .collect();

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
                crate::solver::solve::StarVectors::raw(&database.star_vectors),
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
                    outer,
                    img.idx,
                    wcs_result.matches.len()
                );
                continue;
            }

            debug!(
                "  multi-cal outer {}: image {} refined: {} matches, RMSE={:.2}\"",
                outer,
                img.idx,
                wcs_result.matches.len(),
                wcs_result.rmse_rad.to_degrees() * 3600.0,
            );

            refined_images.push(RefinedImage {
                centroids: cents,
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
            let cents = ref_img.centroids;

            // Derive rotation matrix from refined WCS
            let (rot, _fov, _parity) = wcs_refine::wcs_to_rotation(
                &ref_img.cd_matrix,
                ref_img.crval_ra,
                ref_img.crval_dec,
                image_width,
            );

            // Observed position: raw centroid (no undistortion applied).
            // Ideal position uses the global pixel scale (consistent across images).
            project_matches(
                rot,
                ref_img.matches.iter().copied(),
                cents,
                &database.star_vectors,
                parity_sign,
                global_pixel_scale,
                &mut all_points,
            );
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
        // Radial: nonlinear LS jointly fits (cx, cy, γ, k1, k2, k3, p1, p2);
        // the optical-axis position is carried inside the returned model and
        // γ is folded into the global focal length.
        let Some(fit) = fit_pooled(&all_points, config.model, scale, &fit_config) else {
            debug!(
                "  multi-cal outer {}: too few points ({} < {}) for {:?} fit",
                outer,
                all_points.len(),
                min_points(config.model),
                config.model,
            );
            break;
        };
        if let Distortion::Radial(_) = fit.model {
            // Fold the jointly-fit focal-scale factor into the global anchor
            // focal length (f ← γ·f). The rescaled model is expressed in that
            // corrected frame, so the next outer iteration projects ideal
            // points consistently.
            global_pixel_scale /= fit.focal_scale;
            scale_correction *= fit.focal_scale;
            debug!(
                "  multi-cal outer {}: radial fit gamma={:.6} folded into focal length",
                outer, fit.focal_scale,
            );
        }

        let n_inliers = fit.n_inliers();
        let rmse_after = fit.rmse_after_px;
        debug!(
            "  multi-cal outer {}: {:?} fit: {}/{} inliers, RMSE {:.3} -> {:.3} px",
            outer,
            config.model,
            n_inliers,
            all_points.len(),
            fit.rmse_before_px,
            rmse_after,
        );

        total_iterations += fit.iterations;
        final_mask = fit.mask;
        final_n_points = all_points.len();
        current_distortion = fit.model;
        last_rmse_before = fit.rmse_before_px;

        // Check convergence. Two independent criteria: an absolute per-outer
        // RMSE change (`convergence_threshold_px`, user-configurable) and a
        // relative change — the latter stops runs whose RMSE has plateaued at a
        // level well above the absolute floor (e.g. hard multi-sector fits).
        const RMSE_REL_CONVERGENCE: f64 = 0.01; // 1% change between outers
        let rmse_change = (last_rmse - rmse_after).abs();
        let rmse_frac_change = if last_rmse > 1e-12 {
            rmse_change / last_rmse
        } else {
            0.0
        };

        last_rmse = rmse_after;

        if rmse_frac_change < RMSE_REL_CONVERGENCE || rmse_change < config.convergence_threshold_px
        {
            debug!(
                "  multi-cal: converged at outer iteration {} (RMSE change={:.4} px, {:.2}%)",
                outer,
                rmse_change,
                rmse_frac_change * 100.0,
            );
            break;
        }
    }

    // Build final CameraModel.
    // Polynomial: extract crpix from order-0 terms via extract_crpix.
    // Radial: crpix stays [0, 0] (the optical-axis position lives inside the
    //         model) — no extraction needed.
    // Match every variant explicitly (no `_ =>`) so a future variant carrying
    // its own optical-center offset is caught by the compiler rather than
    // silently defaulting crpix to [0, 0].
    let (crpix, distortion) = match current_distortion {
        Distortion::Polynomial(_) => extract_crpix(current_distortion),
        Distortion::None | Distortion::Radial(_) => ([0.0, 0.0], current_distortion),
    };

    // Tan-consistent with the ideal-point projection above, including any
    // linear scale corrections folded in by radial fits.
    let cam = CameraModel {
        focal_length_px: 1.0 / global_pixel_scale,
        image_width,
        image_height,
        crpix,
        parity_flip,
        distortion,
    };

    // `last_rmse` is still its f64::MAX sentinel if no global fit ever
    // completed (e.g. every outer iteration bailed on too-few pooled points):
    // the camera model above is the identity anchor, not a real calibration.
    if !last_rmse.is_finite() {
        return Err(crate::Error::InvalidInput(
            "calibrate_camera: no distortion fit completed (too few matched points)".into(),
        ));
    }

    let n_inliers = final_mask.iter().filter(|&&m| m).count();

    debug!(
        "calibrate_camera (multi, {:?}): crpix=[{:.2}, {:.2}], RMSE {:.3} -> {:.3} px, {}/{} inliers",
        config.model, crpix[0], crpix[1], last_rmse_before, last_rmse, n_inliers, final_n_points,
    );

    Ok(CalibrateResult {
        camera_model: cam,
        rmse_before_px: last_rmse_before,
        rmse_after_px: last_rmse,
        n_inliers,
        n_outliers: final_n_points - n_inliers,
        iterations: total_iterations,
    })
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
