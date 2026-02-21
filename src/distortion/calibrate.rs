//! Camera calibration from plate-solve results.
//!
//! Given one or more plate-solve results, fits a [`CameraModel`] by fitting a
//! polynomial distortion model to the matched star pairs. The polynomial includes
//! all terms from order 0 through the requested order, with order-0 terms absorbing
//! optical center offset, order-1 terms absorbing scale/rotation/shear corrections,
//! and order 2+ capturing lens distortion.
//!
//! Internally delegates to [`fit_polynomial_distortion`](super::fit::fit_polynomial_distortion)
//! for the actual fitting, then wraps the result in a [`CameraModel`].

use tracing::debug;

use crate::camera_model::CameraModel;
use crate::centroid::Centroid;
use crate::distortion::fit::{fit_polynomial_distortion, DistortionFitConfig};
use crate::solver::{SolveResult, SolveStatus, SolverDatabase};

/// Configuration for camera calibration.
#[derive(Debug, Clone)]
pub struct CalibrateConfig {
    /// Polynomial distortion order (2–6). Default 4.
    pub polynomial_order: u32,
    /// Maximum iterations for sigma-clipping. Default 20.
    pub max_iterations: u32,
    /// Sigma threshold for MAD-based outlier rejection. Default 3.0.
    pub sigma_clip: f64,
    /// Convergence threshold (unused, kept for API compatibility). Default 0.01.
    pub convergence_threshold_px: f64,
}

impl Default for CalibrateConfig {
    fn default() -> Self {
        Self {
            polynomial_order: 4,
            max_iterations: 20,
            sigma_clip: 3.0,
            convergence_threshold_px: 0.01,
        }
    }
}

/// Result of camera calibration.
#[derive(Debug, Clone)]
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
/// Internally, this fits a full polynomial distortion model (order 0 through `order`)
/// using [`fit_polynomial_distortion`], then wraps the result in a [`CameraModel`].
/// The polynomial's order-0 terms absorb optical center offset, order-1 terms absorb
/// residual scale/rotation/shear, and order 2+ terms capture lens distortion.
///
/// The resulting `CameraModel` has `crpix = [0, 0]` (offset absorbed by polynomial)
/// and `focal_length_px` derived from the solve's FOV (though the solver uses its own
/// FOV-based pixel scale, not the camera model's focal length).
pub fn calibrate_camera(
    solve_results: &[&SolveResult],
    centroids: &[&[Centroid]],
    database: &SolverDatabase,
    image_width: u32,
    config: &CalibrateConfig,
) -> CalibrateResult {
    assert_eq!(
        solve_results.len(),
        centroids.len(),
        "solve_results and centroids must have the same length"
    );
    assert!(
        config.polynomial_order >= 2 && config.polynomial_order <= 6,
        "polynomial order must be in [2, 6]"
    );

    // Delegate to fit_polynomial_distortion for the actual fitting
    let fit_config = DistortionFitConfig {
        sigma_clip: config.sigma_clip,
        max_iterations: config.max_iterations,
        stage2_threshold_px: Some(5.0),
    };

    let fit_result = fit_polynomial_distortion(
        solve_results,
        centroids,
        database,
        image_width,
        config.polynomial_order,
        &fit_config,
    );

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

    // Build CameraModel wrapping the fitted distortion.
    // crpix = [0, 0] because the polynomial's order-0 terms absorb the offset.
    // focal_length_px is set to match the solver's pixel_scale = fov/width convention.
    let cam = CameraModel {
        focal_length_px: image_width as f64 / fov_rad as f64,
        crpix: [0.0, 0.0],
        parity_flip,
        distortion: fit_result.model,
    };

    debug!(
        "calibrate_camera: order {}, RMSE {:.3} → {:.3} px, {}/{} inliers",
        config.polynomial_order,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibrate_config_defaults() {
        let cfg = CalibrateConfig::default();
        assert_eq!(cfg.polynomial_order, 4);
        assert_eq!(cfg.max_iterations, 20);
        assert!((cfg.sigma_clip - 3.0).abs() < 1e-12);
        assert!((cfg.convergence_threshold_px - 0.01).abs() < 1e-12);
    }
}
