use numpy::ndarray;
use numpy::PyArray2;
use pyo3::prelude::*;

use crate::distortion::extract_distortion;

/// A detected star centroid with position, brightness, and shape.
///
/// Attributes:
///     x: X position in pixels (origin at image center, +X right).
///     y: Y position in pixels (origin at image center, +Y down).
///     brightness: Integrated intensity above background.
///     cov: 2x2 intensity-weighted covariance matrix [[σxx, σxy], [σxy, σyy]] in pixels².
#[pyclass(name = "Centroid", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyCentroid {
    pub(crate) inner: tetra3::Centroid,
}

#[pymethods]
impl PyCentroid {
    /// Create a new Centroid.
    ///
    /// Args:
    ///     x: X position in pixels (origin at image center, +X right).
    ///     y: Y position in pixels (origin at image center, +Y down).
    ///     brightness: Integrated intensity above background (optional).
    #[new]
    #[pyo3(signature = (x, y, brightness = None))]
    fn new(x: f32, y: f32, brightness: Option<f32>) -> Self {
        Self {
            inner: tetra3::Centroid {
                x,
                y,
                mass: brightness,
                cov: None,
            },
        }
    }

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

    /// Return a new Centroid with position shifted by (dx, dy).
    ///
    /// Preserves brightness and covariance.
    ///
    /// Args:
    ///     dx: X offset in pixels.
    ///     dy: Y offset in pixels.
    ///
    /// Returns:
    ///     A new Centroid with shifted position.
    fn with_offset(&self, dx: f32, dy: f32) -> Self {
        Self {
            inner: tetra3::Centroid {
                x: self.inner.x + dx,
                y: self.inner.y + dy,
                mass: self.inner.mass,
                cov: self.inner.cov,
            },
        }
    }

    /// Remove lens distortion from this centroid's position (distorted → ideal).
    ///
    /// Takes a distortion model (RadialDistortion)
    /// and returns a new Centroid at the corrected position.
    /// Brightness and covariance are preserved.
    ///
    /// Args:
    ///     distortion: A RadialDistortion model.
    ///
    /// Returns:
    ///     A new Centroid with undistorted (ideal) position.
    fn undistort(&self, distortion: &Bound<'_, pyo3::PyAny>) -> PyResult<Self> {
        let dist = extract_distortion(distortion)?;
        let (xu, yu) = dist.undistort(self.inner.x as f64, self.inner.y as f64);
        Ok(Self {
            inner: tetra3::Centroid {
                x: xu as f32,
                y: yu as f32,
                mass: self.inner.mass,
                cov: self.inner.cov,
            },
        })
    }

    /// Apply lens distortion to this centroid's position (ideal → distorted).
    ///
    /// Takes a distortion model (RadialDistortion)
    /// and returns a new Centroid at the distorted position.
    /// Brightness and covariance are preserved.
    ///
    /// Args:
    ///     distortion: A RadialDistortion model.
    ///
    /// Returns:
    ///     A new Centroid with distorted position.
    fn distort(&self, distortion: &Bound<'_, pyo3::PyAny>) -> PyResult<Self> {
        let dist = extract_distortion(distortion)?;
        let (xd, yd) = dist.distort(self.inner.x as f64, self.inner.y as f64);
        Ok(Self {
            inner: tetra3::Centroid {
                x: xd as f32,
                y: yd as f32,
                mass: self.inner.mass,
                cov: self.inner.cov,
            },
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
