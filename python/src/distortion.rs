use numpy::PyArray1;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use tetra3::distortion::polynomial::num_coeffs as poly_num_coeffs;
use tetra3::distortion::{Distortion, PolynomialDistortion, RadialDistortion};

/// Helper: extract a Distortion enum from a Python RadialDistortion or PolynomialDistortion.
pub(crate) fn extract_distortion(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<Distortion> {
    if let Ok(r) = obj.extract::<PyRadialDistortion>() {
        Ok(Distortion::Radial(r.inner))
    } else if let Ok(p) = obj.extract::<PyPolynomialDistortion>() {
        Ok(Distortion::Polynomial(p.inner))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "distortion must be a RadialDistortion or PolynomialDistortion",
        ))
    }
}

/// Brown-Conrady radial+tangential distortion model.
///
/// ```text
/// x_d = x · (1 + k1·r² + k2·r⁴ + k3·r⁶) + 2·p1·x·y + p2·(r² + 2·x²)
/// y_d = y · (1 + k1·r² + k2·r⁴ + k3·r⁶) + p1·(r² + 2·y²) + 2·p2·x·y
/// ```
///
/// Coordinates are in pixels relative to the optical center (image center
/// minus CRPIX). Tangential coefficients ``p1, p2`` default to 0 — set them
/// to model lens decentering, sensor tilt, or off-axis CCD placement.
///
/// Example:
///     d = tetra3rs.RadialDistortion(k1=-7e-9, k2=2e-15)
///     d = tetra3rs.RadialDistortion(k1=-7e-9, p1=5e-7, p2=-3e-7)
///     x_undistorted, y_undistorted = d.undistort(100.0, 200.0)
///
/// References:
///   * Conrady, A. E. (1919). "Decentred Lens-Systems."
///     *Monthly Notices of the Royal Astronomical Society*, 79(5): 384-390.
///     — Original derivation of the tangential / decentering form.
///     https://doi.org/10.1093/mnras/79.5.384
///   * Brown, D. C. (1966). "Decentering Distortion of Lenses."
///     *Photogrammetric Engineering*, 32(3): 444-462. — Modernized form.
///   * Brown, D. C. (1971). "Close-Range Camera Calibration."
///     *Photogrammetric Engineering*, 37(8): 855-866. — Calibration procedure.
///   * Zhang, Z. (2000). "A Flexible New Technique for Camera Calibration."
///     *IEEE TPAMI*, 22(11): 1330-1334. — Multi-image planar calibration.
///     https://doi.org/10.1109/34.888718
///   * OpenCV documentation for the equivalent ``(k1, k2, k3, p1, p2)``
///     formulation: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
#[pyclass(name = "RadialDistortion", module = "tetra3rs", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyRadialDistortion {
    pub(crate) inner: RadialDistortion,
}

#[pymethods]
impl PyRadialDistortion {
    /// Create a Brown-Conrady distortion model.
    ///
    /// Args:
    ///     k1: First radial coefficient (barrel < 0, pincushion > 0). Default 0.
    ///     k2: Second radial coefficient. Default 0.
    ///     k3: Third radial coefficient. Default 0.
    ///     p1: First tangential / decentering coefficient. Default 0.
    ///     p2: Second tangential / decentering coefficient. Default 0.
    #[new]
    #[pyo3(signature = (k1 = 0.0, k2 = 0.0, k3 = 0.0, p1 = 0.0, p2 = 0.0))]
    fn new(k1: f64, k2: f64, k3: f64, p1: f64, p2: f64) -> Self {
        Self {
            inner: RadialDistortion::with_tangential(k1, k2, k3, p1, p2),
        }
    }

    #[getter]
    fn k1(&self) -> f64 {
        self.inner.k1
    }

    #[getter]
    fn k2(&self) -> f64 {
        self.inner.k2
    }

    #[getter]
    fn k3(&self) -> f64 {
        self.inner.k3
    }

    #[getter]
    fn p1(&self) -> f64 {
        self.inner.p1
    }

    #[getter]
    fn p2(&self) -> f64 {
        self.inner.p2
    }

    /// Forward distortion: ideal → distorted.
    fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        self.inner.distort(x, y)
    }

    /// Inverse distortion: distorted → ideal.
    fn undistort(&self, x: f64, y: f64) -> (f64, f64) {
        self.inner.undistort(x, y)
    }

    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<(Py<PyAny>, (Vec<u8>,))> {
        let inner = &slf.borrow().inner;
        let bytes = postcard::to_allocvec(inner)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let from_bytes = slf.get_type().getattr("_from_pickle_bytes")?;
        Ok((from_bytes.unbind(), (bytes,)))
    }

    #[staticmethod]
    fn _from_pickle_bytes(data: &[u8]) -> PyResult<Self> {
        let inner = postcard::from_bytes::<RadialDistortion>(data)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        if self.inner.p1 == 0.0 && self.inner.p2 == 0.0 {
            format!(
                "RadialDistortion(k1={:.3e}, k2={:.3e}, k3={:.3e})",
                self.inner.k1, self.inner.k2, self.inner.k3
            )
        } else {
            format!(
                "RadialDistortion(k1={:.3e}, k2={:.3e}, k3={:.3e}, p1={:.3e}, p2={:.3e})",
                self.inner.k1, self.inner.k2, self.inner.k3, self.inner.p1, self.inner.p2
            )
        }
    }
}

/// SIP-like polynomial distortion model with independent x,y correction terms.
///
/// Forward:  x_d = x + Σ A_pq · (x/s)^p · (y/s)^q   (2 ≤ p+q ≤ order)
/// Inverse:  x_i = x_d + Σ AP_pq · (x_d/s)^p · (y_d/s)^q
///
/// Where s = scale = image_width/2.
///
/// This model captures radial, tangential, and cross-term distortion — suitable
/// for cameras where the optical center is offset from the CCD center (e.g. TESS).
///
/// Typically produced by ``SolverDatabase.calibrate_camera()`` (the fitted polynomial
/// is returned as part of the camera model's ``distortion`` field), or constructed
/// directly from coefficient arrays.
///
/// References:
///   * Shupe, D. L.; Moshir, M.; Li, J.; Makovoz, D.; Narron, R.;
///     Hook, R. N. (2005). "The SIP Convention for Representing Distortion
///     in FITS Image Headers." *Astronomical Data Analysis Software and
///     Systems XIV*, ASP Conference Series, 347: 491. — Original SIP
///     specification.
///     https://www.adass.org/adass/proceedings/adass04/reprints/P3-1-3.pdf
///   * FITS WCS SIP convention registry entry:
///     https://fits.gsfc.nasa.gov/registry/sip.html
#[pyclass(name = "PolynomialDistortion", module = "tetra3rs", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyPolynomialDistortion {
    pub(crate) inner: PolynomialDistortion,
}

#[pymethods]
impl PyPolynomialDistortion {
    /// Create a polynomial distortion model from coefficient arrays.
    ///
    /// Args:
    ///     order: Polynomial order (2–6 typical).
    ///     scale: Normalization scale (typically image_width / 2).
    ///     a_coeffs: Forward A coefficients (x correction, ideal → distorted).
    ///     b_coeffs: Forward B coefficients (y correction, ideal → distorted).
    ///     ap_coeffs: Inverse AP coefficients (x correction, distorted → ideal).
    ///     bp_coeffs: Inverse BP coefficients (y correction, distorted → ideal).
    #[new]
    fn new(
        order: u32,
        scale: f64,
        a_coeffs: Vec<f64>,
        b_coeffs: Vec<f64>,
        ap_coeffs: Vec<f64>,
        bp_coeffs: Vec<f64>,
    ) -> PyResult<Self> {
        let n = poly_num_coeffs(order);
        if a_coeffs.len() != n
            || b_coeffs.len() != n
            || ap_coeffs.len() != n
            || bp_coeffs.len() != n
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Each coefficient array must have {} elements for order {} (got a={}, b={}, ap={}, bp={})",
                n, order, a_coeffs.len(), b_coeffs.len(), ap_coeffs.len(), bp_coeffs.len()
            )));
        }
        Ok(Self {
            inner: PolynomialDistortion::new(
                order, scale, a_coeffs, b_coeffs, ap_coeffs, bp_coeffs,
            ),
        })
    }

    #[getter]
    fn order(&self) -> u32 {
        self.inner.order
    }

    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }

    /// Number of polynomial coefficients per axis.
    #[getter]
    fn num_coeffs(&self) -> usize {
        self.inner.a_coeffs.len()
    }

    /// Forward A coefficients (x correction, ideal → distorted) as a numpy array.
    #[getter]
    fn a_coeffs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.a_coeffs.clone())
    }

    /// Forward B coefficients (y correction, ideal → distorted) as a numpy array.
    #[getter]
    fn b_coeffs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.b_coeffs.clone())
    }

    /// Inverse AP coefficients (x correction, distorted → ideal) as a numpy array.
    #[getter]
    fn ap_coeffs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.ap_coeffs.clone())
    }

    /// Inverse BP coefficients (y correction, distorted → ideal) as a numpy array.
    #[getter]
    fn bp_coeffs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.bp_coeffs.clone())
    }

    /// Forward distortion: ideal → distorted (pixel coords, relative to image center).
    fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        self.inner.distort(x, y)
    }

    /// Inverse distortion: distorted → ideal (pixel coords, relative to image center).
    fn undistort(&self, x: f64, y: f64) -> (f64, f64) {
        self.inner.undistort(x, y)
    }

    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<(Py<PyAny>, (Vec<u8>,))> {
        let inner = &slf.borrow().inner;
        let bytes = postcard::to_allocvec(inner)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let from_bytes = slf.get_type().getattr("_from_pickle_bytes")?;
        Ok((from_bytes.unbind(), (bytes,)))
    }

    #[staticmethod]
    fn _from_pickle_bytes(data: &[u8]) -> PyResult<Self> {
        let inner = postcard::from_bytes::<PolynomialDistortion>(data)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "PolynomialDistortion(order={}, scale={:.1}, num_coeffs={})",
            self.inner.order,
            self.inner.scale,
            self.inner.a_coeffs.len(),
        )
    }
}
