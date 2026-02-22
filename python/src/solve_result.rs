use numpy::ndarray;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use tetra3::solver::SolveResult;

use crate::camera_model::PyCameraModel;

/// Result of a successful plate-solve.
///
/// Returned by ``SolverDatabase.solve_from_centroids`` on a successful match.
/// Contains the camera attitude, matched stars, and error statistics.
#[pyclass(name = "SolveResult", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PySolveResult {
    pub(crate) inner: SolveResult,
    /// Cached derived quantities (computed once at construction).
    ra_deg: f64,
    dec_deg: f64,
    roll_deg: f64,
    /// 3x3 rotation matrix elements (row-major), stored to avoid recomputation.
    rot_elements: [f64; 9],
}

impl PySolveResult {
    /// Construct from a successful `SolveResult`.
    pub(crate) fn from_result(result: SolveResult) -> Self {
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

    /// The camera model used during solving, if any.
    ///
    /// Returns a ``CameraModel`` instance, or ``None`` if the solve failed.
    #[getter]
    fn camera_model(&self) -> Option<PyCameraModel> {
        self.inner
            .camera_model
            .as_ref()
            .map(|cam| PyCameraModel { inner: cam.clone() })
    }

    /// Fitted rotation angle in degrees (camera roll in tangent plane).
    ///
    /// The angle from the tangent-plane ξ (East) axis to the camera +X axis,
    /// measured counter-clockwise. ``None`` if the solve failed.
    #[getter]
    fn theta_deg(&self) -> Option<f64> {
        self.inner.theta_rad.map(|t| t.to_degrees())
    }

    /// WCS CD matrix as a 2x2 numpy array (tangent-plane radians per pixel).
    ///
    /// Maps pixel offsets from CRPIX to gnomonic tangent-plane coordinates
    /// at CRVAL. ``None`` if the solve failed.
    #[getter]
    fn cd_matrix<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner.cd_matrix.map(|cd| {
            PyArray2::from_owned_array(
                py,
                ndarray::array![[cd[0][0], cd[0][1]], [cd[1][0], cd[1][1]]],
            )
        })
    }

    /// WCS reference point RA in degrees.
    ///
    /// The tangent point of the gnomonic (TAN) projection, close to the boresight.
    #[getter]
    fn crval_ra_deg(&self) -> Option<f64> {
        self.inner
            .crval_rad
            .map(|c| c[0].to_degrees().rem_euclid(360.0))
    }

    /// WCS reference point Dec in degrees.
    #[getter]
    fn crval_dec_deg(&self) -> Option<f64> {
        self.inner.crval_rad.map(|c| c[1].to_degrees())
    }

    /// Optical center offset from the geometric image center, in pixels [x, y].
    #[getter]
    fn crpix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let crpix = self
            .inner
            .camera_model
            .as_ref()
            .map(|cam| cam.crpix)
            .unwrap_or([0.0, 0.0]);
        PyArray1::from_vec(py, vec![crpix[0], crpix[1]])
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
                match self.inner.pixel_to_world(xa[i], ya[i]) {
                    Some((r, d)) => {
                        ra_vec.push(r);
                        dec_vec.push(d);
                    }
                    None => {
                        ra_vec.push(f64::NAN);
                        dec_vec.push(f64::NAN);
                    }
                }
            }
            let ra_out = PyArray1::from_vec(py, ra_vec);
            let dec_out = PyArray1::from_vec(py, dec_vec);
            Ok((ra_out, dec_out).into_pyobject(py)?.into_any().unbind())
        } else if let (Ok(xf), Ok(yf)) = (x.extract::<f64>(), y.extract::<f64>()) {
            // Scalar path
            match self.inner.pixel_to_world(xf, yf) {
                Some((ra, dec)) => Ok((ra, dec).into_pyobject(py)?.into_any().unbind()),
                None => Ok(py.None()),
            }
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
