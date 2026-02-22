use pyo3::prelude::*;
use tetra3::Star;

/// A star from the solver catalog.
///
/// Attributes:
///     id: Catalog identifier (e.g. Hipparcos number).
///     ra_deg: Right ascension in degrees [0, 360).
///     dec_deg: Declination in degrees [-90, 90].
///     magnitude: Visual magnitude.
#[pyclass(name = "CatalogStar", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyCatalogStar {
    pub(crate) inner: Star,
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
