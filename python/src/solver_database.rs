use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use tetra3::camera_model::CameraModel;
use tetra3::distortion::calibrate::{calibrate_camera, CalibrateConfig};
use tetra3::solver::{
    GenerateDatabaseConfig, SolveConfig, SolveStatus, SolverDatabase,
};
use tetra3::Centroid;

use crate::calibrate::PyCalibrateResult;
use crate::camera_model::PyCameraModel;
use crate::catalog_star::PyCatalogStar;
use crate::centroid::PyCentroid;
use crate::helpers::parse_solve_results_and_centroids;
use crate::solve_result::PySolveResult;

/// A star pattern database for plate solving.
///
/// Generate from the Hipparcos catalog, or load a previously saved database.
///
/// Example:
///     db = tetra3rs.SolverDatabase.generate_from_hipparcos("data/hip2.dat")
///     db.save_to_file("my_db.bin")
///     db = tetra3rs.SolverDatabase.load_from_file("my_db.bin")
#[pyclass(name = "SolverDatabase")]
pub(crate) struct PySolverDatabase {
    inner: SolverDatabase,
}

#[pymethods]
impl PySolverDatabase {
    /// Generate a database from the Hipparcos catalog file.
    ///
    /// Args:
    ///     catalog_path: Path to the hip2.dat file.
    ///     max_fov_deg: Maximum field of view in degrees. Default 30.
    ///     min_fov_deg: Minimum field of view in degrees. None = same as max (single-scale).
    ///     star_max_magnitude: Faintest star to include. None = auto.
    ///     pattern_max_error: Maximum edge-ratio error. Default 0.001.
    ///     epoch_proper_motion_year: Year for proper motion propagation. Default 2025.
    #[staticmethod]
    #[pyo3(signature = (
        catalog_path,
        max_fov_deg = 30.0,
        min_fov_deg = None,
        star_max_magnitude = None,
        pattern_max_error = 0.001,
        lattice_field_oversampling = 100,
        patterns_per_lattice_field = 50,
        verification_stars_per_fov = 150,
        multiscale_step = 1.5,
        epoch_proper_motion_year = Some(2025.0),
        catalog_nside = 16,
    ))]
    fn generate_from_hipparcos(
        catalog_path: &str,
        max_fov_deg: f32,
        min_fov_deg: Option<f32>,
        star_max_magnitude: Option<f32>,
        pattern_max_error: f32,
        lattice_field_oversampling: u32,
        patterns_per_lattice_field: u32,
        verification_stars_per_fov: u32,
        multiscale_step: f32,
        epoch_proper_motion_year: Option<f64>,
        catalog_nside: u32,
    ) -> PyResult<Self> {
        let config = GenerateDatabaseConfig {
            max_fov_deg,
            min_fov_deg,
            star_max_magnitude,
            pattern_max_error,
            lattice_field_oversampling,
            patterns_per_lattice_field,
            verification_stars_per_fov,
            multiscale_step,
            epoch_proper_motion_year,
            catalog_nside,
        };
        let db = SolverDatabase::generate_from_hipparcos(catalog_path, &config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PySolverDatabase { inner: db })
    }

    /// Save the database to a file.
    fn save_to_file(&self, path: &str) -> PyResult<()> {
        self.inner
            .save_to_file(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Load a database from a file.
    #[staticmethod]
    fn load_from_file(path: &str) -> PyResult<Self> {
        let db = SolverDatabase::load_from_file(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(PySolverDatabase { inner: db })
    }

    /// Solve for camera attitude given star centroids.
    ///
    /// Args:
    ///     centroids: Either a list of Centroid objects (from extract_centroids),
    ///         or an Nx2/Nx3 numpy array of centroid positions in pixels.
    ///         Columns are (x, y) or (x, y, brightness).
    ///         Origin is at the image center, +X right, +Y down.
    ///     fov_estimate_deg: Estimated horizontal field of view in degrees.
    ///     fov_estimate_rad: Estimated horizontal field of view in radians.
    ///         Exactly one of fov_estimate_deg or fov_estimate_rad must be provided.
    ///     image_width: Image width in pixels.
    ///     image_height: Image height in pixels.
    ///     image_shape: Image shape as (height, width) tuple (numpy convention).
    ///         Can be used instead of image_width/image_height.
    ///     fov_max_error: Maximum FOV error in degrees. None = no filtering.
    ///     match_radius: Match distance as fraction of FOV. Default 0.01.
    ///     match_threshold: False-positive probability threshold. Default 1e-5.
    ///     solve_timeout_ms: Timeout in milliseconds. None = no timeout.
    ///     match_max_error: Maximum edge-ratio error. None = use database value.
    ///     refine_iterations: Number of iterative SVD refinement passes. Default 2.
    ///     camera_model: A CameraModel specifying optical center, distortion, and parity.
    ///         None = simple pinhole model with no distortion.
    ///
    /// Returns:
    ///     SolveResult on success, None if no match was found.
    #[pyo3(signature = (
        centroids,
        fov_estimate_deg = None,
        fov_estimate_rad = None,
        image_width = None,
        image_height = None,
        image_shape = None,
        fov_max_error_deg = None,
        fov_max_error_rad = None,
        match_radius = 0.01,
        match_threshold = 1e-5,
        solve_timeout_ms = Some(5000),
        match_max_error = None,
        refine_iterations = 2,
        camera_model = None,
    ))]
    fn solve_from_centroids<'py>(
        &self,
        _py: Python<'py>,
        centroids: &Bound<'py, pyo3::PyAny>,
        fov_estimate_deg: Option<f64>,
        fov_estimate_rad: Option<f64>,
        image_width: Option<u32>,
        image_height: Option<u32>,
        image_shape: Option<(u32, u32)>,
        fov_max_error_deg: Option<f64>,
        fov_max_error_rad: Option<f64>,
        match_radius: f32,
        match_threshold: f64,
        solve_timeout_ms: Option<u64>,
        match_max_error: Option<f32>,
        refine_iterations: u32,
        camera_model: Option<PyCameraModel>,
    ) -> PyResult<Option<PySolveResult>> {
        // Resolve FOV estimate: exactly one of deg or rad must be provided
        let fov_rad = match (fov_estimate_deg, fov_estimate_rad) {
            (Some(deg), None) => (deg as f32).to_radians(),
            (None, Some(rad)) => rad as f32,
            (Some(_), Some(_)) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Specify exactly one of fov_estimate_deg or fov_estimate_rad, not both",
                ));
            }
            (None, None) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Must specify either fov_estimate_deg or fov_estimate_rad",
                ));
            }
        };

        // Resolve image dimensions: image_shape=(h, w) or image_width + image_height
        let (img_width, img_height) = match (image_shape, image_width, image_height) {
            (Some((h, w)), None, None) => (w, h),
            (None, Some(w), Some(h)) => (w, h),
            (Some(_), Some(_), _) | (Some(_), _, Some(_)) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Specify either image_shape or image_width/image_height, not both",
                ));
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Must specify image dimensions via image_shape=(height, width) or image_width + image_height",
                ));
            }
        };

        // Accept either a list of Centroid objects or an Nx2/Nx3 numpy array
        let centroid_vec: Vec<Centroid> = if let Ok(list) = centroids.cast::<pyo3::types::PyList>()
        {
            list.iter()
                .map(|item| {
                    let c: PyCentroid = item.extract()?;
                    Ok(c.inner)
                })
                .collect::<PyResult<Vec<Centroid>>>()?
        } else if let Ok(arr) = centroids.extract::<PyReadonlyArray2<f64>>() {
            let a = arr.as_array();
            let ncols = a.shape()[1];
            if ncols < 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "centroids array must have at least 2 columns (x, y)",
                ));
            }
            (0..a.shape()[0])
                .map(|i| Centroid {
                    x: a[[i, 0]] as f32,
                    y: a[[i, 1]] as f32,
                    mass: if ncols >= 3 {
                        Some(a[[i, 2]] as f32)
                    } else {
                        None
                    },
                    cov: None,
                })
                .collect()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "centroids must be a list of Centroid objects or an Nx2/Nx3 numpy array of float64",
            ));
        };

        // Resolve FOV max error: at most one of deg or rad
        let fov_max_err = match (fov_max_error_deg, fov_max_error_rad) {
            (Some(deg), None) => Some((deg as f32).to_radians()),
            (None, Some(rad)) => Some(rad as f32),
            (Some(_), Some(_)) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Specify at most one of fov_max_error_deg or fov_max_error_rad, not both",
                ));
            }
            (None, None) => None,
        };

        // Use provided camera model, or create a default pinhole model from FOV
        let cam = match camera_model {
            Some(py_cam) => py_cam.inner,
            None => CameraModel::from_fov(fov_rad as f64, img_width, img_height),
        };

        let config = SolveConfig {
            fov_estimate_rad: fov_rad,
            image_width: img_width,
            image_height: img_height,
            fov_max_error_rad: fov_max_err,
            match_radius,
            match_threshold,
            solve_timeout_ms,
            match_max_error,
            refine_iterations,
            camera_model: cam,
        };

        let result = self.inner.solve_from_centroids(&centroid_vec, &config);

        match result.status {
            SolveStatus::MatchFound => Ok(Some(PySolveResult::from_result(result))),
            _ => Ok(None),
        }
    }

    /// Number of stars in the catalog.
    #[getter]
    fn num_stars(&self) -> usize {
        self.inner.star_catalog.len()
    }

    /// Number of patterns in the database.
    #[getter]
    fn num_patterns(&self) -> u32 {
        self.inner.props.num_patterns
    }

    /// Maximum FOV the database was built for (degrees).
    #[getter]
    fn max_fov_deg(&self) -> f32 {
        self.inner.props.max_fov_rad.to_degrees()
    }

    /// Minimum FOV the database was built for (degrees).
    #[getter]
    fn min_fov_deg(&self) -> f32 {
        self.inner.props.min_fov_rad.to_degrees()
    }

    fn __repr__(&self) -> String {
        format!(
            "SolverDatabase(stars={}, patterns={}, fov={:.1}°–{:.1}°)",
            self.inner.star_catalog.len(),
            self.inner.props.num_patterns,
            self.inner.props.min_fov_rad.to_degrees(),
            self.inner.props.max_fov_rad.to_degrees(),
        )
    }

    fn __str__(&self) -> String {
        format!(
            "SolverDatabase: {} stars, {} patterns, FOV {:.1}°–{:.1}°",
            self.inner.star_catalog.len(),
            self.inner.props.num_patterns,
            self.inner.props.min_fov_rad.to_degrees(),
            self.inner.props.max_fov_rad.to_degrees(),
        )
    }

    // ── Catalog access ──────────────────────────────────────────────────

    /// Get a catalog star by its internal index (0-based, brightness order).
    ///
    /// Args:
    ///     index: Star index in [0, num_stars).
    ///
    /// Returns:
    ///     CatalogStar at that index.
    fn get_star(&self, index: usize) -> PyResult<PyCatalogStar> {
        let stars = &self.inner.star_catalog.stars;
        if index >= stars.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "star index {} out of range (catalog has {} stars)",
                index,
                stars.len()
            )));
        }
        Ok(PyCatalogStar {
            inner: stars[index].clone(),
        })
    }

    /// Get a catalog star by its catalog ID (e.g. Hipparcos number).
    ///
    /// Args:
    ///     catalog_id: The catalog identifier to search for.
    ///
    /// Returns:
    ///     CatalogStar with that ID, or None if not found.
    fn get_star_by_id(&self, catalog_id: u64) -> Option<PyCatalogStar> {
        self.inner
            .star_catalog_ids
            .iter()
            .position(|&id| id == catalog_id)
            .map(|idx| PyCatalogStar {
                inner: self.inner.star_catalog.stars[idx].clone(),
            })
    }

    /// Query catalog stars within an angular radius of a sky position.
    ///
    /// Args:
    ///     ra_deg: Right ascension of cone center in degrees.
    ///     dec_deg: Declination of cone center in degrees.
    ///     radius_deg: Search radius in degrees.
    ///
    /// Returns:
    ///     List of CatalogStar objects within the cone, sorted by brightness.
    fn cone_search(&self, ra_deg: f64, dec_deg: f64, radius_deg: f64) -> Vec<PyCatalogStar> {
        let ra_rad = (ra_deg as f32).to_radians();
        let dec_rad = (dec_deg as f32).to_radians();
        let radius_rad = (radius_deg as f32).to_radians();
        let indices = self
            .inner
            .star_catalog
            .query_indices(ra_rad, dec_rad, radius_rad);
        // Indices are already in brightness order (catalog is brightness-sorted)
        indices
            .into_iter()
            .map(|idx| PyCatalogStar {
                inner: self.inner.star_catalog.stars[idx].clone(),
            })
            .collect()
    }

    // ── Camera calibration ────────────────────────────────────────────────

    /// Calibrate a camera model from one or more plate-solve results.
    ///
    /// Fits a global CameraModel (focal length, optical center, polynomial distortion)
    /// by alternating per-image constrained WCS refinement with a global linear
    /// least-squares fit. Distortion terms start at order 2 (SIP convention).
    ///
    /// Args:
    ///     solve_results: A SolveResult or list of SolveResult objects.
    ///     centroids: Matching centroids (list of Centroid lists, or single list).
    ///     image_width: Image width in pixels.
    ///     image_height: Image height in pixels.
    ///     image_shape: Image shape as (height, width) tuple (numpy convention).
    ///         Can be used instead of image_width/image_height.
    ///     order: Polynomial distortion order (2-6). Default 4.
    ///     max_iterations: Maximum outer iterations. Default 10.
    ///     sigma_clip: Sigma threshold for outlier rejection. Default 3.0.
    ///     convergence_threshold_px: Stop when max update < this (pixels). Default 0.01.
    ///
    /// Returns:
    ///     CalibrateResult with camera_model, rmse_before_px, rmse_after_px,
    ///     n_inliers, n_outliers, and iterations.
    #[pyo3(signature = (
        solve_results,
        centroids,
        image_width = None,
        image_height = None,
        image_shape = None,
        order = 4,
        max_iterations = 10,
        sigma_clip = 3.0,
        convergence_threshold_px = 0.01,
    ))]
    fn calibrate_camera<'py>(
        &self,
        _py: Python<'py>,
        solve_results: &Bound<'py, pyo3::PyAny>,
        centroids: &Bound<'py, pyo3::PyAny>,
        image_width: Option<u32>,
        image_height: Option<u32>,
        image_shape: Option<(u32, u32)>,
        order: u32,
        max_iterations: u32,
        sigma_clip: f64,
        convergence_threshold_px: f64,
    ) -> PyResult<PyCalibrateResult> {
        if order < 2 || order > 6 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "polynomial order must be in [2, 6]",
            ));
        }

        // Resolve image dimensions: image_shape=(h, w) or image_width + image_height
        let (img_width, img_height) = match (image_shape, image_width, image_height) {
            (Some((h, w)), None, None) => (w, h),
            (None, Some(w), Some(h)) => (w, h),
            (Some(_), Some(_), _) | (Some(_), _, Some(_)) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Specify either image_shape or image_width/image_height, not both",
                ));
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Must specify image dimensions via image_shape=(height, width) or image_width + image_height",
                ));
            }
        };

        let (sr_vec, cent_vec) = parse_solve_results_and_centroids(solve_results, centroids)?;

        let sr_refs: Vec<&tetra3::solver::SolveResult> = sr_vec.iter().collect();
        let cent_refs: Vec<&[Centroid]> = cent_vec.iter().map(|v| v.as_slice()).collect();

        let config = CalibrateConfig {
            polynomial_order: order,
            max_iterations,
            sigma_clip,
            convergence_threshold_px,
        };

        let result = calibrate_camera(&sr_refs, &cent_refs, &self.inner, img_width, img_height, &config);

        Ok(PyCalibrateResult {
            camera_model: PyCameraModel {
                inner: result.camera_model,
            },
            rmse_before_px: result.rmse_before_px,
            rmse_after_px: result.rmse_after_px,
            n_inliers: result.n_inliers,
            n_outliers: result.n_outliers,
            iterations: result.iterations,
        })
    }
}
