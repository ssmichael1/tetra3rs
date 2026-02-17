//! Python bindings for tetra3rs via PyO3.
//!
//! Exposes the star plate solver to Python as the `tetra3rs` module.

use numpy::ndarray;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use tetra3::centroid_extraction::CentroidExtractionConfig;
use tetra3::solver::{GenerateDatabaseConfig, SolveConfig, SolveStatus, SolverDatabase};
use tetra3::Centroid;

// ═══════════════════════════════════════════════════════════════════════════
// PySolverDatabase — wraps SolverDatabase
// ═══════════════════════════════════════════════════════════════════════════

/// A star pattern database for plate solving.
///
/// Generate from the Hipparcos catalog, or load a previously saved database.
///
/// Example:
///     db = tetra3rs.SolverDatabase.generate_from_hipparcos("data/hip2.dat")
///     db.save_to_file("my_db.bin")
///     db = tetra3rs.SolverDatabase.load_from_file("my_db.bin")
#[pyclass(name = "SolverDatabase")]
struct PySolverDatabase {
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
    ///     centroids: Nx2 or Nx3 numpy array of centroid positions in pixels.
    ///         Columns are (x, y) or (x, y, brightness).
    ///         Origin is at the image center, +X right, +Y down.
    ///     fov_estimate: Estimated horizontal field of view in degrees.
    ///     image_width: Image width in pixels.
    ///     image_height: Image height in pixels.
    ///     fov_max_error: Maximum FOV error in degrees. None = no filtering.
    ///     match_radius: Match distance as fraction of FOV. Default 0.01.
    ///     match_threshold: False-positive probability threshold. Default 1e-5.
    ///     solve_timeout_ms: Timeout in milliseconds. None = no timeout.
    ///     match_max_error: Maximum edge-ratio error. None = use database value.
    ///
    /// Returns:
    ///     dict with keys: 'rotation_matrix', 'fov_deg', 'num_matches',
    ///     'rmse_arcsec', 'p90e_arcsec', 'max_err_arcsec', 'probability',
    ///     'solve_time_ms', 'matched_centroids', 'matched_catalog_ids'.
    ///     Returns None if no match was found.
    #[pyo3(signature = (
        centroids,
        fov_estimate,
        image_width,
        image_height,
        fov_max_error = None,
        match_radius = 0.01,
        match_threshold = 1e-5,
        solve_timeout_ms = Some(5000),
        match_max_error = None,
    ))]
    fn solve_from_centroids<'py>(
        &self,
        py: Python<'py>,
        centroids: PyReadonlyArray2<f64>,
        fov_estimate: f64,
        image_width: u32,
        image_height: u32,
        fov_max_error: Option<f64>,
        match_radius: f32,
        match_threshold: f64,
        solve_timeout_ms: Option<u64>,
        match_max_error: Option<f32>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let arr = centroids.as_array();
        let ncols = arr.shape()[1];
        if ncols < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "centroids must have at least 2 columns (x, y)",
            ));
        }

        // Convert numpy array to Vec<Centroid>
        let centroid_vec: Vec<Centroid> = (0..arr.shape()[0])
            .map(|i| Centroid {
                x: arr[[i, 0]] as f32,
                y: arr[[i, 1]] as f32,
                mass: if ncols >= 3 {
                    Some(arr[[i, 2]] as f32)
                } else {
                    None
                },
                cov: None,
            })
            .collect();

        let config = SolveConfig {
            fov_estimate_rad: (fov_estimate as f32).to_radians(),
            image_width,
            image_height,
            fov_max_error_rad: fov_max_error.map(|e| (e as f32).to_radians()),
            match_radius,
            match_threshold,
            solve_timeout_ms,
            match_max_error,
        };

        let result = self.inner.solve_from_centroids(&centroid_vec, &config);

        match result.status {
            SolveStatus::MatchFound => {
                let dict = PyDict::new(py);

                // Convert quaternion to 3x3 rotation matrix
                if let Some(q) = &result.qicrs2cam {
                    let rot = q.to_rotation_matrix();
                    let m = rot.matrix();
                    let rot_array = PyArray2::from_owned_array(
                        py,
                        ndarray::array![
                            [m[(0, 0)] as f64, m[(0, 1)] as f64, m[(0, 2)] as f64],
                            [m[(1, 0)] as f64, m[(1, 1)] as f64, m[(1, 2)] as f64],
                            [m[(2, 0)] as f64, m[(2, 1)] as f64, m[(2, 2)] as f64],
                        ],
                    );
                    dict.set_item("rotation_matrix", rot_array)?;
                }

                dict.set_item(
                    "fov_deg",
                    result.fov_rad.map(|f| f.to_degrees() as f64),
                )?;
                dict.set_item("num_matches", result.num_matches)?;
                dict.set_item(
                    "rmse_arcsec",
                    result.rmse_rad.map(|r| r.to_degrees() as f64 * 3600.0),
                )?;
                dict.set_item(
                    "p90e_arcsec",
                    result.p90e_rad.map(|r| r.to_degrees() as f64 * 3600.0),
                )?;
                dict.set_item(
                    "max_err_arcsec",
                    result.max_err_rad.map(|r| r.to_degrees() as f64 * 3600.0),
                )?;
                dict.set_item("probability", result.prob)?;
                dict.set_item("solve_time_ms", result.solve_time_ms as f64)?;
                dict.set_item(
                    "matched_centroids",
                    PyArray1::from_vec(
                        py,
                        result
                            .matched_centroid_indices
                            .iter()
                            .map(|&i| i as u64)
                            .collect(),
                    ),
                )?;
                dict.set_item(
                    "matched_catalog_ids",
                    PyArray1::from_vec(py, result.matched_catalog_ids.clone()),
                )?;
                dict.set_item("status", "match_found")?;

                Ok(Some(dict))
            }
            _ => {
                // Return None for no match / timeout / too few
                Ok(None)
            }
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
}

// ═══════════════════════════════════════════════════════════════════════════
// extract_centroids — wraps centroid extraction for numpy arrays
// ═══════════════════════════════════════════════════════════════════════════

/// Extract star centroids from a 2D image array.
///
/// Args:
///     image: 2D numpy array (height x width) of pixel values (float32 or float64).
///     sigma_threshold: Detection threshold in sigma above background. Default 5.0.
///     min_pixels: Minimum blob size. Default 3.
///     max_pixels: Maximum blob size. Default 10000.
///     max_centroids: Maximum number of centroids to return. None = all.
///     local_bg_block_size: Block size for local background estimation. None = global only.
///     max_elongation: Maximum blob elongation ratio. None = disabled.
///
/// Returns:
///     dict with keys:
///         'centroids': Nx3 numpy array of (x, y, brightness) in centered pixel coords.
///         'image_width': int
///         'image_height': int
///         'background_mean': float
///         'background_sigma': float
///         'threshold': float
///         'num_blobs_raw': int
#[pyfunction]
#[pyo3(signature = (
    image,
    sigma_threshold = 5.0,
    min_pixels = 3,
    max_pixels = 10000,
    max_centroids = None,
    local_bg_block_size = Some(64),
    max_elongation = Some(3.0),
))]
fn extract_centroids<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<f64>,
    sigma_threshold: f32,
    min_pixels: usize,
    max_pixels: usize,
    max_centroids: Option<usize>,
    local_bg_block_size: Option<u32>,
    max_elongation: Option<f32>,
) -> PyResult<Bound<'py, PyDict>> {
    let arr = image.as_array();
    let height = arr.shape()[0] as u32;
    let width = arr.shape()[1] as u32;

    // Convert to f32 row-major Vec
    let pixels: Vec<f32> = arr.iter().map(|&v| v as f32).collect();

    let config = CentroidExtractionConfig {
        sigma_threshold,
        min_pixels,
        max_pixels,
        max_centroids,
        sigma_clip_iterations: 5,
        sigma_clip_factor: 3.0,
        use_8_connectivity: true,
        local_bg_block_size,
        max_elongation,
    };

    let result = tetra3::centroid_extraction::extract_centroids_from_raw(&pixels, width, height, &config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Build Nx3 centroid array: (x, y, brightness)
    let n = result.centroids.len();
    let mut centroid_data = ndarray::Array2::<f64>::zeros((n, 3));
    for (i, c) in result.centroids.iter().enumerate() {
        centroid_data[[i, 0]] = c.x as f64;
        centroid_data[[i, 1]] = c.y as f64;
        centroid_data[[i, 2]] = c.mass.unwrap_or(0.0) as f64;
    }

    let dict = PyDict::new(py);
    dict.set_item("centroids", PyArray2::from_owned_array(py, centroid_data))?;
    dict.set_item("image_width", width)?;
    dict.set_item("image_height", height)?;
    dict.set_item("background_mean", result.background_mean as f64)?;
    dict.set_item("background_sigma", result.background_sigma as f64)?;
    dict.set_item("threshold", result.threshold as f64)?;
    dict.set_item("num_blobs_raw", result.num_blobs_raw)?;

    Ok(dict)
}

// ═══════════════════════════════════════════════════════════════════════════
// Module definition
// ═══════════════════════════════════════════════════════════════════════════

/// tetra3rs: Fast star plate solver
///
/// A Rust implementation of the tetra3 star plate solving algorithm,
/// exposed to Python via PyO3.
#[pymodule]
fn tetra3rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySolverDatabase>()?;
    m.add_function(wrap_pyfunction!(extract_centroids, m)?)?;
    Ok(())
}
