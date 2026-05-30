use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;

use tetra3::solver::SolveResult;
use tetra3::Centroid;

use crate::centroid::PyCentroid;
use crate::solve_result::PySolveResult;

/// Parse solve_results and centroids from Python objects.
///
/// Accepts either a single SolveResult + centroids, or lists of each.
pub(crate) fn parse_solve_results_and_centroids(
    solve_results: &Bound<'_, pyo3::PyAny>,
    centroids: &Bound<'_, pyo3::PyAny>,
) -> PyResult<(Vec<SolveResult>, Vec<Vec<Centroid>>)> {
    // Try to extract as a single SolveResult first
    let sr_vec: Vec<SolveResult> = if let Ok(single) = solve_results.extract::<PySolveResult>() {
        vec![single.inner]
    } else if let Ok(list) = solve_results.cast::<pyo3::types::PyList>() {
        list.iter()
            .map(|item| {
                let sr: PySolveResult = item.extract()?;
                Ok(sr.inner)
            })
            .collect::<PyResult<Vec<SolveResult>>>()?
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "solve_results must be a SolveResult or list of SolveResult objects",
        ));
    };

    // Parse centroids: if a single solve result, wrap in a list
    let cent_vec: Vec<Vec<Centroid>> = if sr_vec.len() == 1 {
        vec![parse_centroids_single(centroids)?]
    } else if let Ok(list) = centroids.cast::<pyo3::types::PyList>() {
        if list.len() != sr_vec.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "centroids list has {} elements but solve_results has {}",
                list.len(),
                sr_vec.len()
            )));
        }
        list.iter()
            .map(|item| parse_centroids_single(&item))
            .collect::<PyResult<Vec<Vec<Centroid>>>>()?
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "When solve_results is a list, centroids must also be a list of the same length",
        ));
    };

    Ok((sr_vec, cent_vec))
}

/// Parse a single set of centroids from Python (list of Centroid or Nx2/Nx3 array).
pub(crate) fn parse_centroids_single(
    centroids: &Bound<'_, pyo3::PyAny>,
) -> PyResult<Vec<Centroid>> {
    if let Ok(list) = centroids.cast::<pyo3::types::PyList>() {
        list.iter()
            .map(|item| {
                let c: PyCentroid = item.extract()?;
                Ok(c.inner)
            })
            .collect()
    } else if let Ok(arr) = centroids.extract::<PyReadonlyArray2<f64>>() {
        let a = arr.as_array();
        let ncols = a.shape()[1];
        if ncols < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "centroids array must have at least 2 columns (x, y)",
            ));
        }
        Ok((0..a.shape()[0])
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
            .collect())
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "centroids must be a list of Centroid objects or an Nx2/Nx3 numpy array",
        ))
    }
}

/// Serialize a value with postcard, mapping any error to a Python `RuntimeError`.
///
/// Shared by the pickle (`__reduce__`) implementations so the postcard call and
/// its error conversion are written once rather than per type.
pub(crate) fn to_postcard_bytes<T: Serialize>(value: &T) -> PyResult<Vec<u8>> {
    postcard::to_allocvec(value).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Deserialize a value from postcard bytes, mapping any error to a Python
/// `RuntimeError`. Counterpart to [`to_postcard_bytes`].
pub(crate) fn from_postcard_bytes<T: DeserializeOwned>(data: &[u8]) -> PyResult<T> {
    postcard::from_bytes::<T>(data).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Resolve image dimensions from either `image_shape=(height, width)` (numpy
/// convention) or separate `image_width`/`image_height`. Exactly one form must
/// be supplied. Returns `(width, height)`.
pub(crate) fn resolve_image_dims(
    image_shape: Option<(u32, u32)>,
    image_width: Option<u32>,
    image_height: Option<u32>,
) -> PyResult<(u32, u32)> {
    match (image_shape, image_width, image_height) {
        (Some((h, w)), None, None) => Ok((w, h)),
        (None, Some(w), Some(h)) => Ok((w, h)),
        (Some(_), Some(_), _) | (Some(_), _, Some(_)) => Err(
            pyo3::exceptions::PyValueError::new_err(
                "Specify either image_shape or image_width/image_height, not both",
            ),
        ),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "Must specify image dimensions via image_shape=(height, width) or image_width + image_height",
        )),
    }
}

/// Resolve an angle given as **exactly one** of a `*_deg` or `*_rad` option,
/// returning radians (f32). The error messages are passed in so each caller
/// keeps its own exact wording.
pub(crate) fn exactly_one_angle_rad(
    deg: Option<f64>,
    rad: Option<f64>,
    both_msg: &str,
    neither_msg: &str,
) -> PyResult<f32> {
    match (deg, rad) {
        (Some(d), None) => Ok((d as f32).to_radians()),
        (None, Some(r)) => Ok(r as f32),
        (Some(_), Some(_)) => Err(pyo3::exceptions::PyValueError::new_err(both_msg.to_string())),
        (None, None) => Err(pyo3::exceptions::PyValueError::new_err(neither_msg.to_string())),
    }
}

/// Resolve an optional angle given as **at most one** of a `*_deg` or `*_rad`
/// option, returning `Some(radians)` or `None` when neither is given.
pub(crate) fn at_most_one_angle_rad(
    deg: Option<f64>,
    rad: Option<f64>,
    both_msg: &str,
) -> PyResult<Option<f32>> {
    match (deg, rad) {
        (Some(d), None) => Ok(Some((d as f32).to_radians())),
        (None, Some(r)) => Ok(Some(r as f32)),
        (Some(_), Some(_)) => Err(pyo3::exceptions::PyValueError::new_err(both_msg.to_string())),
        (None, None) => Ok(None),
    }
}
