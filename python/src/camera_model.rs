use pyo3::prelude::*;
use pyo3::types::PyAny;

use tetra3::camera_model::CameraModel;
use tetra3::distortion::Distortion;

use crate::distortion::{extract_distortion, PyPolynomialDistortion, PyRadialDistortion};

/// Camera intrinsics model: focal length, optical center, parity, and distortion.
///
/// Encapsulates the mapping between pixel coordinates and tangent-plane coordinates.
///
/// Example:
///     cam = tetra3rs.CameraModel.from_fov(fov_deg=10.0, image_width=2048)
///     xi, eta = cam.pixel_to_tanplane(100.0, 200.0)
#[pyclass(name = "CameraModel", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyCameraModel {
    pub(crate) inner: CameraModel,
}

#[pymethods]
impl PyCameraModel {
    /// Create a camera model with explicit parameters.
    ///
    /// Args:
    ///     focal_length_px: Focal length in pixels.
    ///     image_width: Image width in pixels.
    ///     image_height: Image height in pixels.
    ///     crpix: Optical center offset from image center [x, y]. Default [0, 0].
    ///     parity_flip: Whether x-axis is flipped. Default False.
    ///     distortion: A RadialDistortion or PolynomialDistortion. Default None.
    #[new]
    #[pyo3(signature = (focal_length_px, image_width, image_height, crpix = None, parity_flip = false, distortion = None))]
    fn new(
        focal_length_px: f64,
        image_width: u32,
        image_height: u32,
        crpix: Option<[f64; 2]>,
        parity_flip: bool,
        distortion: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let dist = match distortion {
            Some(obj) => extract_distortion(obj)?,
            None => Distortion::None,
        };
        Ok(Self {
            inner: CameraModel {
                focal_length_px,
                image_width,
                image_height,
                crpix: crpix.unwrap_or([0.0, 0.0]),
                parity_flip,
                distortion: dist,
            },
        })
    }

    /// Create a camera model from a horizontal field of view and image dimensions.
    ///
    /// Args:
    ///     fov_deg: Horizontal field of view in degrees.
    ///     image_width: Image width in pixels.
    ///     image_height: Image height in pixels.
    ///
    /// Returns:
    ///     CameraModel with no distortion, crpix=[0,0], parity_flip=False.
    #[staticmethod]
    #[pyo3(signature = (fov_deg, image_width, image_height))]
    fn from_fov(fov_deg: f64, image_width: u32, image_height: u32) -> Self {
        Self {
            inner: CameraModel::from_fov(fov_deg.to_radians(), image_width, image_height),
        }
    }

    /// Focal length in pixels.
    #[getter]
    fn focal_length_px(&self) -> f64 {
        self.inner.focal_length_px
    }

    /// Image width in pixels.
    #[getter]
    fn image_width(&self) -> u32 {
        self.inner.image_width
    }

    /// Image height in pixels.
    #[getter]
    fn image_height(&self) -> u32 {
        self.inner.image_height
    }

    /// Optical center offset from image center [x, y] in pixels.
    #[getter]
    fn crpix(&self) -> [f64; 2] {
        self.inner.crpix
    }

    /// Whether the image x-axis is flipped.
    #[getter]
    fn parity_flip(&self) -> bool {
        self.inner.parity_flip
    }

    /// The distortion model, or None if no distortion.
    #[getter]
    fn distortion<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
        match &self.inner.distortion {
            Distortion::Radial(r) => {
                let obj = PyRadialDistortion { inner: r.clone() };
                Ok(Some(obj.into_pyobject(py)?.into_any().unbind()))
            }
            Distortion::Polynomial(p) => {
                let obj = PyPolynomialDistortion { inner: p.clone() };
                Ok(Some(obj.into_pyobject(py)?.into_any().unbind()))
            }
            Distortion::None => Ok(None),
        }
    }

    /// Horizontal field of view in degrees.
    #[getter]
    fn fov_deg(&self) -> f64 {
        self.inner.fov_deg()
    }

    /// Pixel scale in radians per pixel.
    fn pixel_scale(&self) -> f64 {
        self.inner.pixel_scale()
    }

    /// Convert pixel coordinates to tangent-plane coordinates.
    ///
    /// Args:
    ///     px: X pixel coordinate (from image center).
    ///     py: Y pixel coordinate (from image center).
    ///
    /// Returns:
    ///     (xi, eta) in radians on the tangent plane.
    fn pixel_to_tanplane(&self, px: f64, py: f64) -> (f64, f64) {
        self.inner.pixel_to_tanplane(px, py)
    }

    /// Convert tangent-plane coordinates to pixel coordinates.
    ///
    /// Args:
    ///     xi: Tangent-plane ξ coordinate in radians.
    ///     eta: Tangent-plane η coordinate in radians.
    ///
    /// Returns:
    ///     (px, py) in pixels from image center.
    fn tanplane_to_pixel(&self, xi: f64, eta: f64) -> (f64, f64) {
        self.inner.tanplane_to_pixel(xi, eta)
    }

    fn __repr__(&self) -> String {
        format!(
            "CameraModel(f={:.1}px, crpix=[{:.1}, {:.1}], parity={}, distortion={})",
            self.inner.focal_length_px,
            self.inner.crpix[0],
            self.inner.crpix[1],
            self.inner.parity_flip,
            if self.inner.distortion.is_none() {
                "None"
            } else {
                "Some"
            },
        )
    }
}
