use numpy::ndarray;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use tetra3::solver::{Solution, SolveFailure, SolveStatus};

use crate::camera_model::PyCameraModel;

/// Result of a successful plate-solve.
///
/// Returned by ``SolverDatabase.solve_from_centroids`` on a successful match.
/// Contains the camera attitude, matched stars, and error statistics.
#[pyclass(name = "SolveResult", module = "tetra3rs", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PySolveResult {
    pub(crate) inner: Solution,
    /// Cached derived quantities (computed once at construction).
    ra_deg: f64,
    dec_deg: f64,
    roll_deg: f64,
    /// 3x3 rotation matrix elements (row-major), stored to avoid recomputation.
    rot_elements: [f64; 9],
}

impl PySolveResult {
    /// Construct from a plate-solve `Solution`.
    pub(crate) fn from_solution(solution: Solution) -> Self {
        let m = solution.qicrs2cam.to_rotation_matrix();
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

        PySolveResult {
            inner: solution,
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

    /// Attitude quaternion as a 4-element ``[w, x, y, z]`` array.
    ///
    /// **Convention.** Hamilton, scalar first: ``q = w + x·i + y·j + z·k``
    /// with ``w² + x² + y² + z² = 1``. This matches the
    /// ``scalar_first=True`` convention in scipy's ``Rotation.as_quat()``
    /// and is the usual convention in aerospace / attitude literature.
    /// (It does **not** match scipy's default scalar-last ordering.)
    ///
    /// **Sense.** Rotates a vector from the ICRS frame into the camera frame:
    /// ``camera_vec = q ⊗ icrs_vec ⊗ q*``. Equivalently,
    /// ``rotation_matrix_icrs_to_camera @ icrs_vec == camera_vec``.
    ///
    /// Suitable for feeding back as ``attitude_hint`` on the next frame's
    /// ``solve_from_centroids`` call (tracking mode). See
    /// ``concepts/tracking.md`` for more on the convention.
    #[getter]
    fn quaternion<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let q = &self.inner.qicrs2cam;
        PyArray1::from_vec(py, vec![q.w as f64, q.x as f64, q.y as f64, q.z as f64])
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
    fn fov_deg(&self) -> f64 {
        self.inner.fov_rad.to_degrees() as f64
    }

    /// Number of matched star pairs.
    #[getter]
    fn num_matches(&self) -> u32 {
        self.inner.num_matches
    }

    /// Root mean square error of matched stars in arcseconds.
    #[getter]
    fn rmse_arcsec(&self) -> f64 {
        self.inner.rmse_rad.to_degrees() as f64 * 3600.0
    }

    /// 90th percentile error in arcseconds.
    #[getter]
    fn p90e_arcsec(&self) -> f64 {
        self.inner.p90e_rad.to_degrees() as f64 * 3600.0
    }

    /// Maximum match error in arcseconds.
    #[getter]
    fn max_err_arcsec(&self) -> f64 {
        self.inner.max_err_rad.to_degrees() as f64 * 3600.0
    }

    /// False-positive probability (lower is better).
    #[getter]
    fn probability(&self) -> f64 {
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
    fn matched_catalog_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
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

    /// The camera model used during solving, with the refined focal length
    /// and detected parity.
    #[getter]
    fn camera_model(&self) -> PyCameraModel {
        PyCameraModel {
            inner: self.inner.camera_model.clone(),
        }
    }

    /// Fitted rotation angle in degrees (camera roll in tangent plane).
    ///
    /// The angle from the tangent-plane ξ (East) axis to the camera +X axis,
    /// measured counter-clockwise. When ``parity_flip`` is ``True``,
    /// "camera +X" means the x-negated (mirror-corrected) axis — the same
    /// frame the quaternion rotates into.
    #[getter]
    fn theta_deg(&self) -> f64 {
        self.inner.theta_rad.to_degrees()
    }

    /// WCS CD matrix as a 2x2 numpy array (tangent-plane radians per pixel).
    ///
    /// Maps pixel offsets from CRPIX to gnomonic tangent-plane coordinates
    /// at CRVAL.
    #[getter]
    fn cd_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let cd = self.inner.cd_matrix;
        PyArray2::from_owned_array(
            py,
            ndarray::array![[cd[0][0], cd[0][1]], [cd[1][0], cd[1][1]]],
        )
    }

    /// Covariance of the refined attitude parameters ``[theta, xi0, eta0]``
    /// as a 3x3 numpy array in rad²: roll about the boresight and the
    /// tangent-plane offsets of the boresight (East, North at CRVAL).
    ///
    /// Estimated from the refinement's normal equations and the observed
    /// centroid scatter (``sigma² · (JᵀJ)⁻¹``, ``sigma² = Σ residual² /
    /// (2n − 3)``). The diagonal is ``inf`` when the fit is unconstrained.
    #[getter]
    fn attitude_cov_rad2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let c = self.inner.attitude_cov_rad2;
        PyArray2::from_owned_array(
            py,
            ndarray::array![
                [c[0][0], c[0][1], c[0][2]],
                [c[1][0], c[1][1], c[1][2]],
                [c[2][0], c[2][1], c[2][2]]
            ],
        )
    }

    /// 1-sigma uncertainties ``[sigma_theta, sigma_xi, sigma_eta]`` of the
    /// refined attitude in radians (square roots of the diagonal of
    /// ``attitude_cov_rad2``). The boresight pointing uncertainty is
    /// ``hypot(sigma_xi, sigma_eta)``.
    #[getter]
    fn attitude_sigma_rad<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.attitude_sigma_rad().to_vec())
    }

    /// WCS reference point RA in degrees.
    ///
    /// The tangent point of the gnomonic (TAN) projection, close to the boresight.
    #[getter]
    fn crval_ra_deg(&self) -> f64 {
        self.inner.crval_rad[0].to_degrees().rem_euclid(360.0)
    }

    /// WCS reference point Dec in degrees.
    #[getter]
    fn crval_dec_deg(&self) -> f64 {
        self.inner.crval_rad[1].to_degrees()
    }

    /// Observer velocity (km/s, ICRS) the solve corrected stellar aberration
    /// for — the ``observer_velocity_km_s`` it was called with — or ``None``.
    /// ``SolverDatabase.calibrate_camera`` reuses it per image.
    #[getter]
    fn observer_velocity_km_s<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .observer_velocity_km_s
            .map(|v| PyArray1::from_vec(py, v.to_vec()))
    }

    /// Optical center offset from the geometric image center, in pixels [x, y].
    #[getter]
    fn crpix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let crpix = self.inner.camera_model.crpix;
        PyArray1::from_vec(py, vec![crpix[0], crpix[1]])
    }

    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<(Py<PyAny>, (Vec<u8>,))> {
        crate::helpers::pickle_reduce(slf, &slf.borrow().inner)
    }

    #[staticmethod]
    fn _from_pickle_bytes(data: &[u8]) -> PyResult<Self> {
        let solution = crate::helpers::from_postcard_bytes::<Solution>(data)?;
        // The embedded camera model drives pixel_to_world / world_to_pixel;
        // tampered bytes could give it an inconsistent distortion that
        // panics on first use.
        solution
            .camera_model
            .validate()
            .map_err(crate::helpers::map_tetra3_err)?;
        Ok(Self::from_solution(solution))
    }

    /// Always ``True`` — lets ``if result:`` distinguish success from a
    /// (falsy) ``SolveFailure``.
    fn __bool__(&self) -> bool {
        true
    }

    fn __repr__(&self) -> String {
        format!(
            "SolveResult(ra={:.4}°, dec={:.4}°, roll={:.2}°, matches={}, rmse={:.2}\", parity_flip={})",
            self.ra_deg,
            self.dec_deg,
            self.roll_deg,
            self.inner.num_matches,
            self.inner.rmse_rad.to_degrees() as f64 * 3600.0,
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
            self.inner.num_matches,
            self.inner.rmse_rad.to_degrees() as f64 * 3600.0,
            self.inner.prob,
            flip_str,
        )
    }

    /// Convert centered pixel coordinates to world coordinates (RA, Dec in degrees).
    ///
    /// Pixel coordinates use the same convention as solver centroids:
    /// origin at the image center, +X right, +Y down.
    ///
    /// Args:
    ///     x: X pixel coordinate(s). Scalar or 1D numpy array.
    ///     y: Y pixel coordinate(s). Scalar or 1D numpy array.
    ///
    /// Returns:
    ///     (ra_deg, dec_deg): Tuple of RA and Dec in degrees.
    ///         Scalars if input is scalar, numpy arrays if input is array.
    ///         Array elements are NaN where the transform is undefined.
    #[pyo3(signature = (x, y))]
    fn pixel_to_world<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        // Try array path first
        if let (Ok(x_arr), Ok(y_arr)) = (
            x.extract::<PyReadonlyArray1<f64>>(),
            y.extract::<PyReadonlyArray1<f64>>(),
        ) {
            let xa = x_arr.as_array();
            let ya = y_arr.as_array();
            let n = xa.len();
            if ya.len() != n {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "x and y arrays must have the same length",
                ));
            }
            let mut ra_vec = Vec::with_capacity(n);
            let mut dec_vec = Vec::with_capacity(n);
            for i in 0..n {
                let (r, d) = self.inner.pixel_to_world(xa[i], ya[i]);
                ra_vec.push(r);
                dec_vec.push(d);
            }
            let ra_out = PyArray1::from_vec(py, ra_vec);
            let dec_out = PyArray1::from_vec(py, dec_vec);
            Ok((ra_out, dec_out).into_pyobject(py)?.into_any().unbind())
        } else if let (Ok(xf), Ok(yf)) = (x.extract::<f64>(), y.extract::<f64>()) {
            // Scalar path
            let (ra, dec) = self.inner.pixel_to_world(xf, yf);
            Ok((ra, dec).into_pyobject(py)?.into_any().unbind())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "x and y must be scalars or 1D numpy arrays of float64",
            ))
        }
    }

    /// Convert world coordinates (RA, Dec in degrees) to centered pixel coordinates.
    ///
    /// Returns pixel coordinates in the same convention as solver centroids:
    /// origin at the image center, +X right, +Y down.
    ///
    /// Args:
    ///     ra_deg: Right ascension in degrees. Scalar or 1D numpy array.
    ///     dec_deg: Declination in degrees. Scalar or 1D numpy array.
    ///
    /// Returns:
    ///     (x, y): Tuple of pixel coordinates.
    ///         Scalars if input is scalar, numpy arrays if input is array.
    ///         Array elements are NaN for points behind the camera.
    #[pyo3(signature = (ra_deg, dec_deg))]
    fn world_to_pixel<'py>(
        &self,
        py: Python<'py>,
        ra_deg: &Bound<'py, PyAny>,
        dec_deg: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        // Try array path first
        if let (Ok(ra_arr), Ok(dec_arr)) = (
            ra_deg.extract::<PyReadonlyArray1<f64>>(),
            dec_deg.extract::<PyReadonlyArray1<f64>>(),
        ) {
            let ra_a = ra_arr.as_array();
            let dec_a = dec_arr.as_array();
            let n = ra_a.len();
            if dec_a.len() != n {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "ra_deg and dec_deg arrays must have the same length",
                ));
            }
            let mut x_vec = Vec::with_capacity(n);
            let mut y_vec = Vec::with_capacity(n);
            for i in 0..n {
                match self.inner.world_to_pixel(ra_a[i], dec_a[i]) {
                    Some((px, py_val)) => {
                        x_vec.push(px);
                        y_vec.push(py_val);
                    }
                    None => {
                        x_vec.push(f64::NAN);
                        y_vec.push(f64::NAN);
                    }
                }
            }
            let x_out = PyArray1::from_vec(py, x_vec);
            let y_out = PyArray1::from_vec(py, y_vec);
            Ok((x_out, y_out).into_pyobject(py)?.into_any().unbind())
        } else if let (Ok(ra_f), Ok(dec_f)) = (ra_deg.extract::<f64>(), dec_deg.extract::<f64>()) {
            // Scalar path
            match self.inner.world_to_pixel(ra_f, dec_f) {
                Some((x, y)) => Ok((x, y).into_pyobject(py)?.into_any().unbind()),
                None => Ok(py.None()),
            }
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "ra_deg and dec_deg must be scalars or 1D numpy arrays of float64",
            ))
        }
    }
}

/// A failed plate-solve attempt: why it failed and how long it took.
///
/// Returned by ``SolverDatabase.solve_from_centroids`` when no solution was
/// found. Falsy, so ``if result:`` cleanly separates success from failure::
///
///     result = db.solve_from_centroids(centroids, ...)
///     if result:
///         print(result.ra_deg, result.dec_deg)
///     else:
///         print(f"solve failed: {result.status} after {result.solve_time_ms:.0f} ms")
#[pyclass(name = "SolveFailure", module = "tetra3rs", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PySolveFailure {
    pub(crate) inner: SolveFailure,
}

impl PySolveFailure {
    fn status_str(&self) -> &'static str {
        match self.inner.status {
            SolveStatus::NoMatch => "no_match",
            SolveStatus::Timeout => "timeout",
            SolveStatus::TooFew => "too_few",
            SolveStatus::InvalidConfig => "invalid_config",
        }
    }
}

#[pymethods]
impl PySolveFailure {
    /// Why the solve produced no solution: ``'no_match'`` (all pattern
    /// combinations exhausted), ``'timeout'`` (``solve_timeout_ms`` reached),
    /// ``'too_few'`` (fewer than 4 usable centroids), or
    /// ``'invalid_config'`` (degenerate camera model or non-finite matching
    /// parameters — nothing was searched).
    #[getter]
    fn status(&self) -> &'static str {
        self.status_str()
    }

    /// Wall-clock time spent before giving up, in milliseconds.
    #[getter]
    fn solve_time_ms(&self) -> f64 {
        self.inner.solve_time_ms as f64
    }

    /// Always ``False`` — lets ``if result:`` distinguish a (truthy)
    /// ``SolveResult`` from a failure.
    fn __bool__(&self) -> bool {
        false
    }

    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<(Py<PyAny>, (Vec<u8>,))> {
        crate::helpers::pickle_reduce(slf, &slf.borrow().inner)
    }

    #[staticmethod]
    fn _from_pickle_bytes(data: &[u8]) -> PyResult<Self> {
        let inner = crate::helpers::from_postcard_bytes::<SolveFailure>(data)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "SolveFailure(status='{}', solve_time_ms={:.1})",
            self.status_str(),
            self.inner.solve_time_ms,
        )
    }

    fn __str__(&self) -> String {
        format!(
            "SolveFailure: {} after {:.1} ms",
            self.status_str(),
            self.inner.solve_time_ms,
        )
    }
}
