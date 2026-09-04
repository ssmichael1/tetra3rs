use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use tetra3::camera_model::CameraModel;
use tetra3::solver::{GenerateDatabaseConfig, SolveConfig, SolverDatabase};
use tetra3::Centroid;
use tetra3::{calibrate_camera, CalibrateConfig, DistortionModelType};

use crate::calibrate::PyCalibrateResult;
use crate::camera_model::PyCameraModel;
use crate::catalog_star::PyCatalogStar;
use crate::helpers::{
    at_most_one_angle_rad, exactly_one_angle_rad, parse_centroids_single,
    parse_solve_results_and_centroids, resolve_image_dims,
};
use crate::solve_result::{PySolveFailure, PySolveResult};

/// A star pattern database for plate solving.
///
/// Generate from the Gaia catalog, or load a previously saved database.
///
/// Example:
///     db = tetra3rs.SolverDatabase.generate_from_gaia()  # uses bundled gaia-catalog
///     db.save_to_file("my_db.bin")
///     db = tetra3rs.SolverDatabase.load_from_file("my_db.bin")
#[pyclass(name = "SolverDatabase", module = "tetra3rs")]
pub(crate) struct PySolverDatabase {
    inner: SolverDatabase,
}

#[pymethods]
impl PySolverDatabase {
    /// Generate a database from a Gaia DR3 binary catalog file.
    ///
    /// Expects a binary file (``.bin``) in the compact GDR3 format from the
    /// ``gaia-catalog`` package (header magic ``GDR3``). If ``catalog_path`` is
    /// None, uses the bundled catalog from the ``gaia-catalog`` package
    /// (installed as a dependency).
    ///
    /// Args:
    ///     catalog_path: Path to the Gaia catalog file. If None, uses the
    ///         bundled catalog from the ``gaia-catalog`` package.
    ///     max_fov_deg: Maximum field of view in degrees. Default 30.
    ///     min_fov_deg: Minimum field of view in degrees. None = same as max (single-scale).
    ///     star_max_magnitude: Faintest star to include (G-band). None = auto.
    ///     pattern_max_error: Maximum edge-ratio error. Default 0.001.
    ///     epoch_proper_motion_year: Year for proper motion propagation. Default 2025.
    #[staticmethod]
    #[pyo3(signature = (
        catalog_path = None,
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
    #[allow(clippy::too_many_arguments)]
    fn generate_from_gaia(
        py: Python<'_>,
        catalog_path: Option<&str>,
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
        // Resolve catalog path: use provided path, or fall back to gaia_catalog package
        let resolved_path: String = match catalog_path {
            Some(p) => p.to_string(),
            None => {
                let module = py.import("gaia_catalog")?;
                let path_obj = module.call_method0("catalog_path")?;
                path_obj.str()?.to_string()
            }
        };

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
        // Generation runs for seconds to minutes on pure-Rust data; release
        // the GIL so other Python threads keep running.
        let db = py
            .detach(|| SolverDatabase::generate_from_gaia(&resolved_path, &config))
            .map_err(crate::helpers::map_tetra3_err)?;
        Ok(PySolverDatabase { inner: db })
    }

    /// Save the database to a file.
    fn save_to_file(&self, path: &str) -> PyResult<()> {
        self.inner
            .save_to_file(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Load a database from a file.
    ///
    /// Raises OSError for file problems and ValueError when the file decodes
    /// but fails the database's consistency validation (corrupt/tampered data
    /// that would otherwise crash mid-solve).
    #[staticmethod]
    fn load_from_file(py: Python<'_>, path: &str) -> PyResult<Self> {
        let path = path.to_string();
        let db = py
            .detach(move || SolverDatabase::load_from_file(&path))
            .map_err(crate::helpers::map_tetra3_err)?;
        Ok(PySolverDatabase { inner: db })
    }

    /// Solve for camera attitude given star centroids.
    ///
    /// Runs **lost-in-space** (pattern-hash search) by default, or **tracking**
    /// (direct correspondence from a prior estimate) when ``attitude_hint`` is
    /// given — with automatic fallback to lost-in-space. Arguments group by
    /// which mode reads them; arguments for one mode are ignored in the other:
    ///
    /// * **Both modes:** ``camera_model`` / ``fov_estimate_*`` / ``image_*``,
    ///   ``match_radius``, ``match_threshold``, ``solve_timeout_ms``,
    ///   ``max_patterns_checked``,
    ///   ``observer_velocity_km_s``.
    /// * **Lost-in-space only:** ``fov_max_error``, ``match_max_error``,
    ///   ``pattern_checking_stars``.
    /// * **Tracking only** (ignored unless ``attitude_hint`` is set):
    ///   ``attitude_hint``, ``hint_uncertainty_*``, ``strict_hint``.
    ///
    /// Args:
    ///     centroids: Either a list of Centroid objects (from extract_centroids),
    ///         or an Nx2/Nx3 numpy array of centroid positions in pixels.
    ///         Columns are (x, y) or (x, y, brightness).
    ///         Origin is at the image center, +X right, +Y down.
    ///     fov_estimate_deg: Estimated horizontal field of view in degrees.
    ///     fov_estimate_rad: Estimated horizontal field of view in radians.
    ///         Used (with the image dimensions) to build a pinhole camera model
    ///         when camera_model is not given — in that case exactly one of
    ///         fov_estimate_deg / fov_estimate_rad is required. When camera_model
    ///         is given, both are optional and ignored (the model's focal length
    ///         defines the FOV estimate).
    ///     image_width: Image width in pixels. Required only when camera_model is
    ///         not given (the model carries its own dimensions).
    ///     image_height: Image height in pixels. See image_width.
    ///     image_shape: Image shape as (height, width) tuple (numpy convention).
    ///         Can be used instead of image_width/image_height.
    ///     fov_max_error: Maximum FOV error in degrees. None = no filtering.
    ///     match_radius: Match distance as fraction of FOV. Default 0.01.
    ///     match_threshold: False-positive probability budget for accepting a
    ///         solution (candidate p-values are tested against this with a
    ///         sequential multiple-comparison correction). Raising it (e.g. 1e-3)
    ///         accepts weaker evidence — useful for very sparse fields at
    ///         increased false-positive risk. Default 1e-5.
    ///     solve_timeout_ms: Wall-clock timeout in milliseconds. None = no
    ///         timeout. Default 5000.
    ///     max_patterns_checked: Maximum number of image 4-star patterns the
    ///         lost-in-space search tests (summed over the FOV sweep) before
    ///         giving up with ``'timeout'``. Bounds the search by work rather
    ///         than wall time, so the outcome is the same on every machine.
    ///         Composes with ``solve_timeout_ms`` — whichever trips first
    ///         ends the search. None = unbounded. Default: sized so the
    ///         5 s timeout normally trips first. Lost-in-space only.
    ///     pattern_checking_stars: Number of brightest (well-separated)
    ///         centroids the lost-in-space search forms 4-star patterns from;
    ///         fainter centroids still take part in verification. Bounds a
    ///         no-match search at C(N, 4) patterns per FOV value. Raise it
    ///         when many of the brightest detections are not catalog stars.
    ///         Must be >= 4. Default 24. Lost-in-space only.
    ///     match_max_error: Maximum edge-ratio error. None = use database value.
    ///         Values below the database's pattern quantization error are clamped up to it.
    ///     camera_model: A CameraModel specifying focal length, image dimensions,
    ///         optical center, distortion, and parity. When provided it is the
    ///         single source of camera geometry (fov_estimate and image
    ///         dimensions are ignored). None = simple pinhole model built from
    ///         fov_estimate and the image dimensions.
    ///     observer_velocity_km_s: Observer's barycentric velocity as [vx, vy, vz] in km/s
    ///         (ICRS/GCRF frame). When set, catalog positions are aberration-corrected
    ///         to apparent positions, removing ~20" bias from Earth's orbital velocity.
    ///         None = no correction (default).
    ///     attitude_hint: Optional attitude hint. Accepts either:
    ///
    ///         * a 4-element ``[w, x, y, z]`` quaternion (list or 1D ndarray),
    ///           using the Hamilton, scalar-first convention — same as
    ///           ``SolveResult.quaternion``. This matches scipy's
    ///           ``Rotation.as_quat(scalar_first=True)``; it does **not** match
    ///           scipy's default (scalar-last) ordering.
    ///         * a 3×3 rotation matrix (2D ndarray) — same as
    ///           ``SolveResult.rotation_matrix_icrs_to_camera``.
    ///
    ///         Either form must rotate a vector from the ICRS frame into the
    ///         camera frame. When provided, the solver skips the 4-star
    ///         pattern-hash phase and instead projects nearby catalog stars via
    ///         the hint, nearest-neighbor matches them to centroids, and runs the
    ///         same verification + WCS refine path as lost-in-space. Typical use:
    ///         video-rate tracking where each frame's solve seeds the next.
    ///         Succeeds with as few as 3 matched stars (vs. 4 for LIS). On
    ///         failure falls back to lost-in-space unless ``strict_hint`` is True.
    ///     hint_uncertainty_deg: Angular uncertainty of the attitude hint, in degrees.
    ///     hint_uncertainty_rad: Angular uncertainty of the attitude hint, in radians.
    ///         At most one of the two may be provided; default 1° if neither is set.
    ///         Used to size the catalog cone search and the initial pixel match radius.
    ///         Ignored unless ``attitude_hint`` is set.
    ///     strict_hint: If True, do not fall back to lost-in-space if the hinted
    ///         solve fails. Default False. Ignored unless ``attitude_hint`` is set.
    ///
    /// Returns:
    ///     SolveResult on success, or a (falsy) SolveFailure carrying the
    ///     failure reason (``status``: ``'no_match'`` / ``'timeout'`` /
    ///     ``'too_few'`` / ``'invalid_config'``) and ``solve_time_ms``. Use
    ///     ``if result:`` to distinguish the two.
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
        max_patterns_checked = Some(SolveConfig::DEFAULT_MAX_PATTERNS_CHECKED),
        pattern_checking_stars = SolveConfig::DEFAULT_PATTERN_CHECKING_STARS,
        match_max_error = None,
        camera_model = None,
        observer_velocity_km_s = None,
        attitude_hint = None,
        hint_uncertainty_deg = None,
        hint_uncertainty_rad = None,
        strict_hint = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn solve_from_centroids<'py>(
        &self,
        py: Python<'py>,
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
        max_patterns_checked: Option<u64>,
        pattern_checking_stars: u32,
        match_max_error: Option<f32>,
        camera_model: Option<PyCameraModel>,
        observer_velocity_km_s: Option<[f64; 3]>,
        attitude_hint: Option<&Bound<'py, pyo3::PyAny>>,
        hint_uncertainty_deg: Option<f64>,
        hint_uncertainty_rad: Option<f64>,
        strict_hint: bool,
    ) -> PyResult<Py<pyo3::PyAny>> {
        // Accept either a list of Centroid objects or an Nx2/Nx3 numpy array.
        let centroid_vec: Vec<Centroid> = parse_centroids_single(centroids)?;

        // Resolve FOV max error: at most one of deg or rad
        let fov_max_err = at_most_one_angle_rad(
            fov_max_error_deg,
            fov_max_error_rad,
            "Specify at most one of fov_max_error_deg or fov_max_error_rad, not both",
        )?;

        // A camera_model is the single source of geometry. Only when it is
        // absent do we build a pinhole model — and only then are fov_estimate_*
        // and the image dimensions needed (and required).
        let cam = match camera_model {
            Some(py_cam) => {
                // A degenerate model (zero/NaN focal length, zero dimensions,
                // inconsistent distortion) can only produce a guaranteed
                // NoMatch or a panic deep in the solver — reject it here.
                py_cam
                    .inner
                    .validate()
                    .map_err(crate::helpers::map_tetra3_err)?;
                py_cam.inner
            }
            None => {
                let fov_rad = exactly_one_angle_rad(
                    fov_estimate_deg,
                    fov_estimate_rad,
                    "Specify exactly one of fov_estimate_deg or fov_estimate_rad, not both",
                    "Must specify fov_estimate_deg or fov_estimate_rad (or pass camera_model)",
                )?;
                // CameraModel::from_fov panics (by contract) outside (0, π);
                // from Python that must be a ValueError instead.
                if !(fov_rad.is_finite()
                    && fov_rad > 0.0
                    && (fov_rad as f64) < std::f64::consts::PI)
                {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "fov_estimate must be in (0, 180) degrees, got {} rad",
                        fov_rad
                    )));
                }
                let (img_width, img_height) =
                    resolve_image_dims(image_shape, image_width, image_height)?;
                CameraModel::from_fov(fov_rad as f64, img_width, img_height)
            }
        };

        // Resolve hint uncertainty: at most one of deg or rad.
        let hint_uncertainty = at_most_one_angle_rad(
            hint_uncertainty_deg,
            hint_uncertainty_rad,
            "Specify at most one of hint_uncertainty_deg or hint_uncertainty_rad",
        )?;

        // Build quaternion from the hint, accepting either a 4-element [w, x, y, z]
        // quaternion or a 3×3 rotation matrix.
        let attitude_hint_q = match attitude_hint {
            None => None,
            Some(obj) => Some(parse_attitude_hint(obj)?),
        };

        let default_config = SolveConfig::default();
        let config = SolveConfig {
            fov_max_error_rad: fov_max_err,
            match_radius,
            match_threshold,
            solve_timeout_ms,
            max_patterns_checked,
            pattern_checking_stars,
            match_max_error,
            camera_model: cam,
            observer_velocity_km_s,
            attitude_hint: attitude_hint_q,
            hint_uncertainty_rad: hint_uncertainty.unwrap_or(default_config.hint_uncertainty_rad),
            strict_hint,
        };

        // The solve can run up to its timeout (default 5 s) on pure-Rust data;
        // release the GIL so other Python threads keep running.
        let result = py.detach(|| self.inner.solve_from_centroids(&centroid_vec, &config));

        match result {
            Ok(solution) => Ok(PySolveResult::from_solution(solution)
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            Err(failure) => Ok(PySolveFailure { inner: failure }
                .into_pyobject(py)?
                .into_any()
                .unbind()),
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

    /// Database generation parameters as a dict.
    ///
    /// Returns the settings baked into this database when it was generated
    /// (stars per FOV, lattice field oversampling, pattern quantization, FOV
    /// range, magnitude limit, epochs, …). These are read from the stored
    /// [`DatabaseProperties`], so they reflect the actual database on disk —
    /// not solve-time options.
    ///
    /// Returns:
    ///     dict mapping parameter name to value.
    #[getter]
    fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let p = &self.inner.props;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("num_stars", self.inner.star_catalog.len())?;
        d.set_item("num_patterns", p.num_patterns)?;
        d.set_item("min_fov_deg", p.min_fov_rad.to_degrees())?;
        d.set_item("max_fov_deg", p.max_fov_rad.to_degrees())?;
        d.set_item("star_max_magnitude", p.star_max_magnitude)?;
        d.set_item("pattern_bins", p.pattern_bins)?;
        d.set_item("pattern_max_error", p.pattern_max_error)?;
        d.set_item("verification_stars_per_fov", p.verification_stars_per_fov)?;
        d.set_item("lattice_field_oversampling", p.lattice_field_oversampling)?;
        d.set_item("patterns_per_lattice_field", p.patterns_per_lattice_field)?;
        d.set_item("epoch_equinox", p.epoch_equinox)?;
        d.set_item("epoch_proper_motion_year", p.epoch_proper_motion_year)?;
        Ok(d)
    }

    /// Print the database generation parameters to stdout.
    ///
    /// Human-readable dump of the settings this database was built with —
    /// stars per FOV, lattice field oversampling, pattern quantization, FOV
    /// range, magnitude limit, and epochs. Useful for inspecting an opaque
    /// `.bin` database. See [`parameters`] for the same data as a dict.
    fn print_parameters(&self) {
        let p = &self.inner.props;
        println!("Solver database parameters:");
        println!("  catalog");
        println!(
            "    num_stars                  : {}",
            self.inner.star_catalog.len()
        );
        println!(
            "    star_max_magnitude         : {:.2}",
            p.star_max_magnitude
        );
        println!("    epoch_equinox              : {}", p.epoch_equinox);
        println!(
            "    epoch_proper_motion_year   : {:.1}",
            p.epoch_proper_motion_year
        );
        println!("  field of view");
        println!(
            "    min_fov_deg                : {:.3}",
            p.min_fov_rad.to_degrees()
        );
        println!(
            "    max_fov_deg                : {:.3}",
            p.max_fov_rad.to_degrees()
        );
        println!("  pattern hash");
        println!("    num_patterns               : {}", p.num_patterns);
        println!("    pattern_bins               : {}", p.pattern_bins);
        println!("    pattern_max_error          : {}", p.pattern_max_error);
        println!("  generation");
        println!(
            "    verification_stars_per_fov : {}",
            p.verification_stars_per_fov
        );
        println!(
            "    lattice_field_oversampling : {}",
            p.lattice_field_oversampling
        );
        println!(
            "    patterns_per_lattice_field : {}",
            p.patterns_per_lattice_field
        );
    }

    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<(Py<PyAny>, (Vec<u8>,))> {
        let bytes = slf
            .borrow()
            .inner
            .to_bytes()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let from_bytes = slf.get_type().getattr("_from_pickle_bytes")?;
        Ok((from_bytes.unbind(), (bytes,)))
    }

    #[staticmethod]
    fn _from_pickle_bytes(data: &[u8]) -> PyResult<Self> {
        let inner = postcard::from_bytes::<SolverDatabase>(data)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        // Corrupt pickle bytes can decode into a structurally-valid database
        // whose indices panic mid-solve; enforce the same invariants as
        // load_from_file (ValueError instead of a deferred PanicException).
        inner.validate().map_err(crate::helpers::map_tetra3_err)?;
        Ok(Self { inner })
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
        let stars = self.inner.star_catalog.stars();
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
    fn get_star_by_id(&self, catalog_id: i64) -> Option<PyCatalogStar> {
        self.inner
            .star_catalog_ids
            .iter()
            .position(|&id| id == catalog_id)
            .map(|idx| PyCatalogStar {
                inner: self.inner.star_catalog.stars()[idx].clone(),
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
                inner: self.inner.star_catalog.stars()[idx].clone(),
            })
            .collect()
    }

    // ── Camera calibration ────────────────────────────────────────────────

    /// Calibrate a camera model from one or more plate-solve results.
    ///
    /// Fits a global CameraModel (focal length, optical center, distortion)
    /// by alternating per-image constrained WCS refinement with a global
    /// least-squares fit.
    ///
    /// The distortion model is selected by ``model``:
    ///
    /// - ``"polynomial"`` (default) — SIP-like polynomial of the given
    ///   ``order``. Captures arbitrary 2D distortion including tangential
    ///   and decentering terms; preferred for off-axis CCDs (e.g. TESS).
    /// - ``"radial"`` — Brown-Conrady ``(k1, k2, k3)`` radial distortion.
    ///   Three parameters total — well-conditioned with few matches and the
    ///   standard model in computer-vision camera calibration. Assumes
    ///   distortion is symmetric about the optical center.
    ///
    /// Args:
    ///     solve_results: A SolveResult, or a list that may mix SolveResult
    ///         and SolveFailure objects — failures are skipped, so solve
    ///         outputs can be passed straight through without filtering.
    ///     centroids: Matching centroids (list of Centroid lists, or single list).
    ///     image_width: Image width in pixels.
    ///     image_height: Image height in pixels.
    ///     image_shape: Image shape as (height, width) tuple (numpy convention).
    ///         Can be used instead of image_width/image_height.
    ///     model: Distortion model to fit — ``"polynomial"`` or ``"radial"``.
    ///         Default ``"polynomial"``.
    ///     order: Polynomial distortion order (2-6). Ignored unless
    ///         ``model="polynomial"``. Default 4.
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
        model = "polynomial",
        order = 4,
        max_iterations = 10,
        sigma_clip = 3.0,
        convergence_threshold_px = 0.01,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn calibrate_camera<'py>(
        &self,
        py: Python<'py>,
        solve_results: &Bound<'py, pyo3::PyAny>,
        centroids: &Bound<'py, pyo3::PyAny>,
        image_width: Option<u32>,
        image_height: Option<u32>,
        image_shape: Option<(u32, u32)>,
        model: &str,
        order: u32,
        max_iterations: u32,
        sigma_clip: f64,
        convergence_threshold_px: f64,
    ) -> PyResult<PyCalibrateResult> {
        let model_type = match model {
            "polynomial" => {
                if !(2..=6).contains(&order) {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "polynomial order must be in [2, 6]",
                    ));
                }
                DistortionModelType::Polynomial { order }
            }
            "radial" => DistortionModelType::Radial,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "model must be 'polynomial' or 'radial', got '{}'",
                    other
                )));
            }
        };

        // Resolve image dimensions: image_shape=(h, w) or image_width + image_height
        let (img_width, img_height) = resolve_image_dims(image_shape, image_width, image_height)?;

        let (sr_vec, cent_vec) = parse_solve_results_and_centroids(solve_results, centroids)?;

        let sr_refs: Vec<&tetra3::solver::SolveResult> = sr_vec.iter().collect();
        let cent_refs: Vec<&[Centroid]> = cent_vec.iter().map(|v| v.as_slice()).collect();

        let config = CalibrateConfig {
            model: model_type,
            max_iterations,
            sigma_clip,
            convergence_threshold_px,
        };

        // Calibration iterates WCS refinement over all images on pure-Rust
        // data; release the GIL for the duration.
        let result = py
            .detach(|| {
                calibrate_camera(
                    &sr_refs,
                    &cent_refs,
                    &self.inner,
                    img_width,
                    img_height,
                    &config,
                )
            })
            .map_err(crate::helpers::map_tetra3_err)?;

        Ok(PyCalibrateResult { inner: result })
    }
}

/// Parse a Python `attitude_hint` argument into a tetra3 Quaternion.
///
/// Accepts either a 4-element quaternion `[w, x, y, z]` (list or 1D ndarray)
/// or a 3×3 rotation matrix (2D ndarray).
fn parse_attitude_hint(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<tetra3::Quaternion> {
    use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
    use pyo3::exceptions::PyValueError;

    // Build a unit quaternion from raw [w, x, y, z], rejecting a degenerate
    // (near-zero / non-finite) norm. Unnormalized hints would otherwise become
    // silently-wrong or NaN rotations and a confusing solve failure.
    let quat_from_wxyz = |w: f64, x: f64, y: f64, z: f64| -> PyResult<tetra3::Quaternion> {
        let n = (w * w + x * x + y * y + z * z).sqrt();
        if !(n.is_finite() && n > 1e-9) {
            return Err(PyValueError::new_err(
                "attitude_hint quaternion has a near-zero or non-finite norm",
            ));
        }
        Ok(tetra3::Quaternion::new(
            (w / n) as f32,
            (x / n) as f32,
            (y / n) as f32,
            (z / n) as f32,
        ))
    };

    // Try a 4-element [w, x, y, z] sequence: list, tuple, or 1D ndarray (the
    // Vec extraction accepts any f64 sequence, so no separate ndarray branch).
    if let Ok(vec) = obj.extract::<Vec<f64>>() {
        if vec.len() != 4 {
            return Err(PyValueError::new_err(format!(
                "attitude_hint sequence must have 4 elements [w, x, y, z], got {}",
                vec.len()
            )));
        }
        return quat_from_wxyz(vec[0], vec[1], vec[2], vec[3]);
    }
    // Try a 3×3 rotation matrix.
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f64>>() {
        let shape = arr.shape();
        if shape != [3, 3] {
            return Err(PyValueError::new_err(format!(
                "attitude_hint 2D array must have shape (3, 3), got {:?}",
                shape
            )));
        }
        let view = arr.as_array();
        // Reject a matrix that isn't (close to) a rotation: Mᵀ·M ≈ I. Catches
        // unnormalized / garbage matrices before they yield a bogus attitude.
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|k| view[(k, i)] * view[(k, j)]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                if !dot.is_finite() || (dot - expected).abs() > 1e-3 {
                    return Err(PyValueError::new_err(
                        "attitude_hint 3x3 matrix is not orthonormal (columns must be \
                         orthonormal; expected a rotation matrix)",
                    ));
                }
            }
        }
        let m = numeris::Matrix3::<f32>::new([
            [
                view[(0, 0)] as f32,
                view[(0, 1)] as f32,
                view[(0, 2)] as f32,
            ],
            [
                view[(1, 0)] as f32,
                view[(1, 1)] as f32,
                view[(1, 2)] as f32,
            ],
            [
                view[(2, 0)] as f32,
                view[(2, 1)] as f32,
                view[(2, 2)] as f32,
            ],
        ]);
        return Ok(tetra3::Quaternion::from_rotation_matrix(&m));
    }
    Err(PyValueError::new_err(
        "attitude_hint must be a 4-element [w, x, y, z] quaternion or a 3x3 rotation matrix",
    ))
}
