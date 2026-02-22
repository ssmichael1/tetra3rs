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
use pyo3::types::PyDateTime;

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
    m.add_function(wrap_pyfunction!(earth_barycentric_velocity, m)?)?;
    Ok(())
}

/// Approximate Earth barycentric velocity in km/s (ICRS equatorial frame).
///
/// Uses a circular-orbit approximation. Accuracy ~0.5 km/s (~1.7%),
/// sufficient for stellar aberration correction (~20" effect, ~0.3" error).
///
/// Args:
///     dt: Observation time as a ``datetime.datetime`` (UTC).
///         The ~69 s offset between UTC and TT is negligible for this
///         approximation.
///
/// Returns:
///     [vx, vy, vz] in km/s, ICRS equatorial frame. Pass directly to
///     ``solve_from_centroids(observer_velocity_km_s=...)``.
///
/// Example::
///
///     from datetime import datetime
///     v = tetra3rs.earth_barycentric_velocity(datetime(2025, 7, 10))
///     result = db.solve_from_centroids(centroids, ..., observer_velocity_km_s=v)
#[pyfunction]
fn earth_barycentric_velocity<'py>(dt: &Bound<'py, PyDateTime>) -> PyResult<[f64; 3]> {
    // J2000.0 = 2000 Jan 1 12:00:00 UTC (close enough for this approximation)
    // Convert datetime → days since J2000.0
    let year = dt.getattr("year")?.extract::<i32>()?;
    let month = dt.getattr("month")?.extract::<u32>()?;
    let day = dt.getattr("day")?.extract::<u32>()?;
    let hour = dt.getattr("hour")?.extract::<u32>()?;
    let minute = dt.getattr("minute")?.extract::<u32>()?;
    let second = dt.getattr("second")?.extract::<u32>()?;
    let microsecond = dt.getattr("microsecond")?.extract::<u32>()?;

    let jd = datetime_to_jd(year, month, day, hour, minute, second, microsecond);
    let days_since_j2000 = jd - 2_451_545.0;

    Ok(tetra3::earth_barycentric_velocity(days_since_j2000))
}

/// Convert a calendar date/time to Julian Date.
fn datetime_to_jd(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    microsecond: u32,
) -> f64 {
    // Algorithm from Meeus, "Astronomical Algorithms", ch. 7
    let (y, m) = if month <= 2 {
        (year - 1, month + 12)
    } else {
        (year, month)
    };
    let a = y / 100;
    let b = 2 - a + a / 4; // Gregorian correction
    let jd_day = (365.25 * (y + 4716) as f64).floor()
        + (30.6001 * (m + 1) as f64).floor()
        + day as f64
        + b as f64
        - 1524.5;
    let frac = (hour as f64 + minute as f64 / 60.0 + (second as f64 + microsecond as f64 * 1e-6) / 3600.0) / 24.0;
    jd_day + frac
}
