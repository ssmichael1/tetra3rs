//! Python bindings for tetra3rs via PyO3.
//!
//! Exposes the star plate solver to Python as the `tetra3rs` module.

use numpy::ndarray;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use tetra3::centroid_extraction::CentroidExtractionConfig;
use tetra3::solver::{
    GenerateDatabaseConfig, SolveConfig, SolveResult, SolveStatus, SolverDatabase,
};
use tetra3::Centroid;
use tetra3::Star;

// ═══════════════════════════════════════════════════════════════════════════
// CatalogStar — lightweight wrapper for Star
// ═══════════════════════════════════════════════════════════════════════════

/// A star from the solver catalog.
///
/// Attributes:
///     id: Catalog identifier (e.g. Hipparcos number).
///     ra_deg: Right ascension in degrees [0, 360).
///     dec_deg: Declination in degrees [-90, 90].
///     magnitude: Visual magnitude.
#[pyclass(name = "CatalogStar", frozen, from_py_object)]
#[derive(Clone)]
struct PyCatalogStar {
    inner: Star,
}

#[pymethods]
impl PyCatalogStar {
    /// Catalog identifier (e.g. Hipparcos number).
    #[getter]
    fn id(&self) -> u64 {
        self.inner.id
    }

    /// Right ascension in degrees [0, 360).
    #[getter]
    fn ra_deg(&self) -> f64 {
        self.inner.ra_rad.to_degrees() as f64
    }

    /// Declination in degrees [-90, 90].
    #[getter]
    fn dec_deg(&self) -> f64 {
        self.inner.dec_rad.to_degrees() as f64
    }

    /// Visual magnitude.
    #[getter]
    fn magnitude(&self) -> f32 {
        self.inner.mag
    }

    fn __repr__(&self) -> String {
        format!(
            "CatalogStar(id={}, ra={:.4}°, dec={:.4}°, mag={:.2})",
            self.inner.id,
            self.inner.ra_rad.to_degrees(),
            self.inner.dec_rad.to_degrees(),
            self.inner.mag,
        )
    }

    fn __str__(&self) -> String {
        format!(
            "HIP {} at RA {:.4}°, Dec {:.4}°, mag {:.2}",
            self.inner.id,
            self.inner.ra_rad.to_degrees(),
            self.inner.dec_rad.to_degrees(),
            self.inner.mag,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PyCentroid — wraps Centroid with covariance
// ═══════════════════════════════════════════════════════════════════════════

/// A detected star centroid with position, brightness, and shape.
///
/// Attributes:
///     x: X position in pixels (origin at image center, +X right).
///     y: Y position in pixels (origin at image center, +Y down).
///     brightness: Integrated intensity above background.
///     cov: 2x2 intensity-weighted covariance matrix [[σxx, σxy], [σxy, σyy]] in pixels².
#[pyclass(name = "Centroid", frozen, from_py_object)]
#[derive(Clone)]
struct PyCentroid {
    inner: tetra3::Centroid,
}

#[pymethods]
impl PyCentroid {
    /// X position in pixels (origin at image center, +X right).
    #[getter]
    fn x(&self) -> f32 {
        self.inner.x
    }

    /// Y position in pixels (origin at image center, +Y down).
    #[getter]
    fn y(&self) -> f32 {
        self.inner.y
    }

    /// Integrated intensity above background.
    #[getter]
    fn brightness(&self) -> Option<f32> {
        self.inner.mass
    }

    /// 2x2 intensity-weighted covariance matrix as a numpy array.
    /// [[σxx, σxy], [σxy, σyy]] in pixels².
    #[getter]
    fn cov<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner.cov.map(|m| {
            PyArray2::from_owned_array(
                py,
                ndarray::array![
                    [m[(0, 0)] as f64, m[(0, 1)] as f64],
                    [m[(1, 0)] as f64, m[(1, 1)] as f64],
                ],
            )
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Centroid(x={:.2}, y={:.2}, brightness={:.1})",
            self.inner.x,
            self.inner.y,
            self.inner.mass.unwrap_or(0.0),
        )
    }

    fn __str__(&self) -> String {
        if let Some(cov) = &self.inner.cov {
            format!(
                "Centroid at ({:.2}, {:.2}) px, brightness={:.1}, cov=[[{:.3}, {:.3}], [{:.3}, {:.3}]]",
                self.inner.x,
                self.inner.y,
                self.inner.mass.unwrap_or(0.0),
                cov[(0, 0)], cov[(0, 1)],
                cov[(1, 0)], cov[(1, 1)],
            )
        } else {
            format!(
                "Centroid at ({:.2}, {:.2}) px, brightness={:.1}",
                self.inner.x,
                self.inner.y,
                self.inner.mass.unwrap_or(0.0),
            )
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PySolveResult — wraps SolveResult
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a successful plate-solve.
///
/// Returned by ``SolverDatabase.solve_from_centroids`` on a successful match.
/// Contains the camera attitude, matched stars, and error statistics.
#[pyclass(name = "SolveResult", frozen)]
struct PySolveResult {
    inner: SolveResult,
    /// Cached derived quantities (computed once at construction).
    ra_deg: f64,
    dec_deg: f64,
    roll_deg: f64,
    /// 3x3 rotation matrix elements (row-major), stored to avoid recomputation.
    rot_elements: [f64; 9],
}

impl PySolveResult {
    /// Construct from a successful `SolveResult`.
    fn from_result(result: SolveResult) -> Self {
        let q = result
            .qicrs2cam
            .as_ref()
            .expect("MatchFound must have quaternion");
        let rot = q.to_rotation_matrix();
        let m = rot.matrix();
        let rot_elements = [
            m[(0, 0)] as f64,
            m[(0, 1)] as f64,
            m[(0, 2)] as f64,
            m[(1, 0)] as f64,
            m[(1, 1)] as f64,
            m[(1, 2)] as f64,
            m[(2, 0)] as f64,
            m[(2, 1)] as f64,
            m[(2, 2)] as f64,
        ];

        // Boresight direction in ICRS: R^T * [0, 0, 1] = third row of R
        let bx = rot_elements[6];
        let by = rot_elements[7];
        let bz = rot_elements[8];
        let dec_rad = bz.asin();
        let ra_rad = by.atan2(bx);
        let ra_deg = ra_rad.to_degrees().rem_euclid(360.0);
        let dec_deg = dec_rad.to_degrees();

        // Roll angle: position angle of camera +Y, measured East of North.
        let cam_y_icrs = [rot_elements[3], rot_elements[4], rot_elements[5]];
        let sin_ra = ra_rad.sin();
        let cos_ra = ra_rad.cos();
        let sin_dec = dec_rad.sin();
        let cos_dec = dec_rad.cos();
        let north = [-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec];
        let east = [-sin_ra, cos_ra, 0.0];
        let dot_north: f64 = cam_y_icrs
            .iter()
            .zip(north.iter())
            .map(|(a, b)| a * b)
            .sum();
        let dot_east: f64 = cam_y_icrs.iter().zip(east.iter()).map(|(a, b)| a * b).sum();
        let roll_deg = dot_east.atan2(dot_north).to_degrees();

        // Avoid using `_` prefix; suppress unused warning via allow attribute is not needed
        // since all fields are used by getters.
        PySolveResult {
            inner: result,
            ra_deg,
            dec_deg,
            roll_deg,
            rot_elements,
        }
    }
}

#[pymethods]
impl PySolveResult {
    /// 3x3 rotation matrix from ICRS to camera frame as a numpy array.
    #[getter]
    fn rotation_matrix_icrs_to_camera<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let e = &self.rot_elements;
        PyArray2::from_owned_array(
            py,
            ndarray::array![[e[0], e[1], e[2]], [e[3], e[4], e[5]], [e[6], e[7], e[8]],],
        )
    }

    /// Right ascension of the boresight in degrees [0, 360).
    #[getter]
    fn ra_deg(&self) -> f64 {
        self.ra_deg
    }

    /// Declination of the boresight in degrees [-90, 90].
    #[getter]
    fn dec_deg(&self) -> f64 {
        self.dec_deg
    }

    /// Roll angle: position angle of camera +Y measured East of North, in degrees.
    #[getter]
    fn roll_deg(&self) -> f64 {
        self.roll_deg
    }

    /// Solved horizontal field of view in degrees.
    #[getter]
    fn fov_deg(&self) -> Option<f64> {
        self.inner.fov_rad.map(|f| f.to_degrees() as f64)
    }

    /// Number of matched star pairs.
    #[getter]
    fn num_matches(&self) -> Option<u32> {
        self.inner.num_matches
    }

    /// Root mean square error of matched stars in arcseconds.
    #[getter]
    fn rmse_arcsec(&self) -> Option<f64> {
        self.inner.rmse_rad.map(|r| r.to_degrees() as f64 * 3600.0)
    }

    /// 90th percentile error in arcseconds.
    #[getter]
    fn p90e_arcsec(&self) -> Option<f64> {
        self.inner.p90e_rad.map(|r| r.to_degrees() as f64 * 3600.0)
    }

    /// Maximum match error in arcseconds.
    #[getter]
    fn max_err_arcsec(&self) -> Option<f64> {
        self.inner
            .max_err_rad
            .map(|r| r.to_degrees() as f64 * 3600.0)
    }

    /// False-positive probability (lower is better).
    #[getter]
    fn probability(&self) -> Option<f64> {
        self.inner.prob
    }

    /// Time taken to solve in milliseconds.
    #[getter]
    fn solve_time_ms(&self) -> f64 {
        self.inner.solve_time_ms as f64
    }

    /// Indices of matched centroids in the input array.
    #[getter]
    fn matched_centroids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        PyArray1::from_vec(
            py,
            self.inner
                .matched_centroid_indices
                .iter()
                .map(|&i| i as u64)
                .collect(),
        )
    }

    /// Catalog IDs of matched stars.
    #[getter]
    fn matched_catalog_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        PyArray1::from_vec(py, self.inner.matched_catalog_ids.clone())
    }

    /// Status string (always 'match_found').
    #[getter]
    fn status(&self) -> &'static str {
        "match_found"
    }

    /// Whether the image x-axis was flipped to achieve a proper rotation.
    ///
    /// When ``True``, the rotation matrix assumes negated x-coordinates.
    /// Pixel-to-sky and sky-to-pixel conversions must account for this.
    #[getter]
    fn parity_flip(&self) -> bool {
        self.inner.parity_flip
    }

    fn __repr__(&self) -> String {
        format!(
            "SolveResult(ra={:.4}°, dec={:.4}°, roll={:.2}°, matches={}, rmse={:.2}\", parity_flip={})",
            self.ra_deg,
            self.dec_deg,
            self.roll_deg,
            self.inner.num_matches.unwrap_or(0),
            self.inner
                .rmse_rad
                .map(|r| r.to_degrees() as f64 * 3600.0)
                .unwrap_or(0.0),
            self.inner.parity_flip,
        )
    }

    fn __str__(&self) -> String {
        let flip_str = if self.inner.parity_flip {
            ", parity flipped"
        } else {
            ""
        };
        format!(
            "SolveResult: RA {:.4}°, Dec {:.4}°, Roll {:.2}°, {} matches, RMSE {:.2}\", prob {:.2e}{}",
            self.ra_deg,
            self.dec_deg,
            self.roll_deg,
            self.inner.num_matches.unwrap_or(0),
            self.inner.rmse_rad.map(|r| r.to_degrees() as f64 * 3600.0).unwrap_or(0.0),
            self.inner.prob.unwrap_or(0.0),
            flip_str,
        )
    }
}

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

        let config = SolveConfig {
            fov_estimate_rad: fov_rad,
            image_width: img_width,
            image_height: img_height,
            fov_max_error_rad: fov_max_err,
            match_radius,
            match_threshold,
            solve_timeout_ms,
            match_max_error,
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
}

// ═══════════════════════════════════════════════════════════════════════════
// extract_centroids — wraps centroid extraction for numpy arrays
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a 2D numpy array of any supported dtype to Vec<f32>.
///
/// Supported dtypes: float64, float32, uint8, uint16, int16.
fn image_to_f32(image: &Bound<'_, pyo3::PyAny>) -> PyResult<(Vec<f32>, u32, u32)> {
    // Get the dtype string to dispatch
    let dtype = image.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;

    match (kind.as_str(), itemsize) {
        ("f", 8) => {
            let arr: PyReadonlyArray2<f64> = image.extract()?;
            let a = arr.as_array();
            let h = a.shape()[0] as u32;
            let w = a.shape()[1] as u32;
            Ok((a.iter().map(|&v| v as f32).collect(), w, h))
        }
        ("f", 4) => {
            let arr: PyReadonlyArray2<f32> = image.extract()?;
            let a = arr.as_array();
            let h = a.shape()[0] as u32;
            let w = a.shape()[1] as u32;
            Ok((a.iter().copied().collect(), w, h))
        }
        ("u", 1) => {
            let arr: PyReadonlyArray2<u8> = image.extract()?;
            let a = arr.as_array();
            let h = a.shape()[0] as u32;
            let w = a.shape()[1] as u32;
            Ok((a.iter().map(|&v| v as f32).collect(), w, h))
        }
        ("u", 2) => {
            let arr: PyReadonlyArray2<u16> = image.extract()?;
            let a = arr.as_array();
            let h = a.shape()[0] as u32;
            let w = a.shape()[1] as u32;
            Ok((a.iter().map(|&v| v as f32).collect(), w, h))
        }
        ("i", 2) => {
            let arr: PyReadonlyArray2<i16> = image.extract()?;
            let a = arr.as_array();
            let h = a.shape()[0] as u32;
            let w = a.shape()[1] as u32;
            Ok((a.iter().map(|&v| v as f32).collect(), w, h))
        }
        _ => {
            let dtype_str: String = dtype.str()?.extract()?;
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported image dtype '{}'. Expected float64, float32, uint16, int16, or uint8.",
                dtype_str,
            )))
        }
    }
}

/// Extract star centroids from a 2D image array.
///
/// Args:
///     image: 2D numpy array (height x width) of pixel values.
///         Supported dtypes: float64, float32, uint16, int16, uint8.
///     sigma_threshold: Detection threshold in sigma above background. Default 5.0.
///     min_pixels: Minimum blob size. Default 3.
///     max_pixels: Maximum blob size. Default 10000.
///     max_centroids: Maximum number of centroids to return. None = all.
///     local_bg_block_size: Block size for local background estimation. None = global only.
///     max_elongation: Maximum blob elongation ratio. None = disabled.
///
/// Returns:
///     dict with keys:
///         'centroids': list of Centroid objects, sorted by brightness (brightest first).
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
    image: &Bound<'py, pyo3::PyAny>,
    sigma_threshold: f32,
    min_pixels: usize,
    max_pixels: usize,
    max_centroids: Option<usize>,
    local_bg_block_size: Option<u32>,
    max_elongation: Option<f32>,
) -> PyResult<Bound<'py, PyDict>> {
    let (pixels, width, height) = image_to_f32(image)?;

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

    let result =
        tetra3::centroid_extraction::extract_centroids_from_raw(&pixels, width, height, &config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Convert to list of PyCentroid objects
    let py_centroids: Vec<PyCentroid> = result
        .centroids
        .into_iter()
        .map(|c| PyCentroid { inner: c })
        .collect();

    let dict = PyDict::new(py);
    dict.set_item("centroids", py_centroids.into_pyobject(py)?)?;
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
    m.add_class::<PyCentroid>()?;
    m.add_class::<PyCatalogStar>()?;
    m.add_class::<PySolveResult>()?;
    m.add_class::<PySolverDatabase>()?;
    m.add_function(wrap_pyfunction!(extract_centroids, m)?)?;
    Ok(())
}
