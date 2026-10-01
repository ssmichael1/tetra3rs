use numpy::ndarray::{Dimension, Ix2};
use numpy::PyReadonlyArray;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;

use tetra3::solver::SolveResult;
use tetra3::Centroid;

use crate::centroid::PyCentroid;
use crate::solve_result::{PySolveFailure, PySolveResult};

/// Parse solve_results and centroids from Python objects.
///
/// Accepts either a single SolveResult + one centroid set, or a list of
/// solves + a list of centroid sets of the same length (a one-element list
/// is still the list form).
pub(crate) fn parse_solve_results_and_centroids(
    solve_results: &Bound<'_, pyo3::PyAny>,
    centroids: &Bound<'_, pyo3::PyAny>,
) -> PyResult<(Vec<SolveResult>, Vec<Vec<Centroid>>)> {
    // Try to extract as a single SolveResult first. Lists may mix
    // SolveResult and SolveFailure items — the Rust calibrate API accepts
    // failures and skips them, so a caller can pass its solve outputs
    // straight through without filtering.
    if let Ok(single) = solve_results.extract::<PySolveResult>() {
        return Ok((
            vec![Ok(single.inner)],
            vec![parse_centroids_single(centroids)?],
        ));
    }
    let sr_vec: Vec<SolveResult> = if let Ok(list) = solve_results.cast::<pyo3::types::PyList>() {
        list.iter()
            .map(|item| {
                if let Ok(sr) = item.extract::<PySolveResult>() {
                    Ok(Ok(sr.inner))
                } else if let Ok(fail) = item.extract::<PySolveFailure>() {
                    Ok(Err(fail.inner))
                } else {
                    Err(pyo3::exceptions::PyTypeError::new_err(
                        "solve_results items must be SolveResult or SolveFailure objects",
                    ))
                }
            })
            .collect::<PyResult<Vec<SolveResult>>>()?
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "solve_results must be a SolveResult or list of SolveResult / SolveFailure objects",
        ));
    };

    // List form: centroids must be a list of the same length (the single
    // form returned above — keying on `len() == 1` here used to reject a
    // one-element list of solves paired with a one-element list of centroids).
    let cent_vec: Vec<Vec<Centroid>> = if let Ok(list) = centroids.cast::<pyo3::types::PyList>() {
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
        return list
            .iter()
            .map(|item| {
                let c: PyCentroid = item.extract()?;
                Ok(c.inner)
            })
            .collect();
    }
    // numpy array path: any numeric 2-D dtype.
    if centroids.hasattr("dtype").unwrap_or(false) {
        let arr = extract_f64_array::<Ix2>(centroids)?.ok_or_else(|| {
            pyo3::exceptions::PyTypeError::new_err(
                "centroids array must be a 2-D numeric numpy array (Nx2 or Nx3)",
            )
        })?;
        let a = arr.as_array();
        let ncols = a.shape()[1];
        if ncols < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "centroids array must have at least 2 columns (x, y)",
            ));
        }
        return Ok((0..a.shape()[0])
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
            .collect());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "centroids must be a list of Centroid objects or an Nx2/Nx3 numpy array",
    ))
}

/// View `obj` as a native-float64 numpy array of rank `D`: zero-copy when it
/// already is one (the common case), otherwise an `astype("float64")` copy of
/// any other numeric dtype (float32, ints, big-endian FITS data, …).
/// `Ok(None)` when `obj` is not an ndarray, or is one of another rank.
pub(crate) fn extract_f64_array<'py, D: Dimension>(
    obj: &Bound<'py, PyAny>,
) -> PyResult<Option<PyReadonlyArray<'py, f64, D>>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray<'py, f64, D>>() {
        return Ok(Some(arr));
    }
    if !obj.hasattr("dtype").unwrap_or(false) {
        return Ok(None);
    }
    Ok(obj.call_method1("astype", ("float64",))?.extract().ok())
}

/// Map a `tetra3::Error` to the most appropriate Python exception type, so
/// callers get `ValueError` for bad input, `IOError` for file problems, etc.,
/// instead of an undifferentiated `RuntimeError`.
pub(crate) fn map_tetra3_err(e: tetra3::Error) -> PyErr {
    use pyo3::exceptions::{PyIOError, PyValueError};
    match e {
        tetra3::Error::InvalidInput(m) => PyValueError::new_err(m),
        tetra3::Error::InvalidCatalog(m) => PyValueError::new_err(format!("invalid catalog: {m}")),
        tetra3::Error::Io(e) => PyIOError::new_err(e.to_string()),
        tetra3::Error::Postcard(e) => PyRuntimeError::new_err(format!("postcard error: {e}")),
    }
}

/// Serialize a value with postcard, mapping any error to a Python `RuntimeError`.
///
/// Shared by the pickle (`__reduce__`) implementations so the postcard call and
/// its error conversion are written once rather than per type.
pub(crate) fn to_postcard_bytes<T: Serialize>(value: &T) -> PyResult<Vec<u8>> {
    postcard::to_allocvec(value).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// The shared `__reduce__` body for every pickled wrapper type: postcard-encode
/// the inner value and pair it with the type's `_from_pickle_bytes`
/// reconstructor. Each `#[pymethods]` block delegates here so the pickle
/// plumbing exists once. (A macro emitting whole `#[pymethods]` blocks would
/// need pyo3's `multiple-pymethods` feature and its `inventory` dependency,
/// which isn't worth it for this.)
pub(crate) fn pickle_reduce<T: Serialize>(
    slf: &Bound<'_, impl pyo3::PyClass>,
    inner: &T,
) -> PyResult<(Py<PyAny>, (Vec<u8>,))> {
    let bytes = to_postcard_bytes(inner)?;
    let from_bytes = slf.as_any().get_type().getattr("_from_pickle_bytes")?;
    Ok((from_bytes.unbind(), (bytes,)))
}

/// Deserialize a value from postcard bytes, mapping any error to a Python
/// `RuntimeError`. Counterpart to [`to_postcard_bytes`].
pub(crate) fn from_postcard_bytes<T: DeserializeOwned>(data: &[u8]) -> PyResult<T> {
    postcard::from_bytes::<T>(data).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Resolve image dimensions from either `image_shape=(height, width)` (numpy
/// convention) or separate `image_width`/`image_height`. Exactly one form must
/// be supplied and both dimensions must be non-zero. Returns `(width, height)`.
pub(crate) fn resolve_image_dims(
    image_shape: Option<(u32, u32)>,
    image_width: Option<u32>,
    image_height: Option<u32>,
) -> PyResult<(u32, u32)> {
    let (w, h) = match (image_shape, image_width, image_height) {
        (Some((h, w)), None, None) => (w, h),
        (None, Some(w), Some(h)) => (w, h),
        (Some(_), Some(_), _) | (Some(_), _, Some(_)) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Specify either image_shape or image_width/image_height, not both",
            ))
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Must specify image dimensions via image_shape=(height, width) or image_width + image_height",
            ))
        }
    };
    if w == 0 || h == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "image dimensions must be non-zero, got {w}x{h}"
        )));
    }
    Ok((w, h))
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
        (Some(_), Some(_)) => Err(pyo3::exceptions::PyValueError::new_err(
            both_msg.to_string(),
        )),
        (None, None) => Err(pyo3::exceptions::PyValueError::new_err(
            neither_msg.to_string(),
        )),
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
        (Some(_), Some(_)) => Err(pyo3::exceptions::PyValueError::new_err(
            both_msg.to_string(),
        )),
        (None, None) => Ok(None),
    }
}
