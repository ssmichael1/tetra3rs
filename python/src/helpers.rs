use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

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
fn parse_centroids_single(centroids: &Bound<'_, pyo3::PyAny>) -> PyResult<Vec<Centroid>> {
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
