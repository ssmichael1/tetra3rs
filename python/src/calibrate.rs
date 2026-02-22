use pyo3::prelude::*;

use crate::camera_model::PyCameraModel;

/// Result of camera calibration.
///
/// Returned by ``SolverDatabase.calibrate_camera``.
///
/// Attributes:
///     camera_model: The fitted CameraModel (focal length, crpix, distortion).
///     rmse_before_px: RMS residual in pixels before calibration.
///     rmse_after_px: RMS residual in pixels after calibration.
///     n_inliers: Number of inlier star matches used in the fit.
///     n_outliers: Number of outlier star matches rejected by sigma clipping.
///     iterations: Number of sigma-clip iterations performed.
#[pyclass(name = "CalibrateResult", frozen)]
pub(crate) struct PyCalibrateResult {
    pub(crate) camera_model: PyCameraModel,
    pub(crate) rmse_before_px: f64,
    pub(crate) rmse_after_px: f64,
    pub(crate) n_inliers: usize,
    pub(crate) n_outliers: usize,
    pub(crate) iterations: u32,
}

#[pymethods]
impl PyCalibrateResult {
    /// The fitted CameraModel (focal length, crpix, distortion).
    #[getter]
    fn camera_model(&self) -> PyCameraModel {
        self.camera_model.clone()
    }

    /// RMS residual in pixels before calibration.
    #[getter]
    fn rmse_before_px(&self) -> f64 {
        self.rmse_before_px
    }

    /// RMS residual in pixels after calibration.
    #[getter]
    fn rmse_after_px(&self) -> f64 {
        self.rmse_after_px
    }

    /// Number of inlier star matches used in the fit.
    #[getter]
    fn n_inliers(&self) -> usize {
        self.n_inliers
    }

    /// Number of outlier star matches rejected by sigma clipping.
    #[getter]
    fn n_outliers(&self) -> usize {
        self.n_outliers
    }

    /// Number of sigma-clip iterations performed.
    #[getter]
    fn iterations(&self) -> u32 {
        self.iterations
    }

    fn __repr__(&self) -> String {
        format!(
            "CalibrateResult(rmse={:.3}->{:.3} px, inliers={}, outliers={}, iterations={})",
            self.rmse_before_px,
            self.rmse_after_px,
            self.n_inliers,
            self.n_outliers,
            self.iterations,
        )
    }
}
