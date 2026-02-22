//! Python bindings for tetra3rs via PyO3.
//!
//! Exposes the star plate solver to Python as the `tetra3rs` module.

mod calibrate;
mod camera_model;
mod catalog_star;
mod centroid;
mod distortion;
mod extraction;
mod helpers;
mod solve_result;
mod solver_database;

use pyo3::prelude::*;

/// tetra3rs: Fast star plate solver
///
/// A Rust implementation of the tetra3 star plate solving algorithm,
/// exposed to Python via PyO3.
#[pymodule]
fn tetra3rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<centroid::PyCentroid>()?;
    m.add_class::<catalog_star::PyCatalogStar>()?;
    m.add_class::<camera_model::PyCameraModel>()?;
    m.add_class::<calibrate::PyCalibrateResult>()?;
    m.add_class::<extraction::PyExtractionResult>()?;
    m.add_class::<solve_result::PySolveResult>()?;
    m.add_class::<solver_database::PySolverDatabase>()?;
    m.add_class::<distortion::PyRadialDistortion>()?;
    m.add_class::<distortion::PyPolynomialDistortion>()?;
    m.add_function(wrap_pyfunction!(extraction::extract_centroids, m)?)?;
    Ok(())
}
