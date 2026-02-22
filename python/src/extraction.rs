use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use tetra3::centroid_extraction::CentroidExtractionConfig;

use crate::centroid::PyCentroid;

/// Convert a 2D numpy array of any supported dtype to Vec<f32>.
///
/// Supported dtypes: float64, float32, uint8, uint16, int16.
fn image_to_f32(image: &Bound<'_, pyo3::PyAny>) -> PyResult<(Vec<f32>, u32, u32)> {
    // Get the dtype string to dispatch
    let dtype = image.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;

    match (kind.as_str(), itemsize) {
        ("f", 8) => {
            let arr: PyReadonlyArray2<f64> = image.extract()?;
            let a = arr.as_array();
            let h = a.shape()[0] as u32;
            let w = a.shape()[1] as u32;
            Ok((a.iter().map(|&v| v as f32).collect(), w, h))
        }
        ("f", 4) => {
            let arr: PyReadonlyArray2<f32> = image.extract()?;
            let a = arr.as_array();
            let h = a.shape()[0] as u32;
            let w = a.shape()[1] as u32;
            Ok((a.iter().copied().collect(), w, h))
        }
        ("u", 1) => {
            let arr: PyReadonlyArray2<u8> = image.extract()?;
            let a = arr.as_array();
            let h = a.shape()[0] as u32;
            let w = a.shape()[1] as u32;
            Ok((a.iter().map(|&v| v as f32).collect(), w, h))
        }
        ("u", 2) => {
            let arr: PyReadonlyArray2<u16> = image.extract()?;
            let a = arr.as_array();
            let h = a.shape()[0] as u32;
            let w = a.shape()[1] as u32;
            Ok((a.iter().map(|&v| v as f32).collect(), w, h))
        }
        ("i", 2) => {
            let arr: PyReadonlyArray2<i16> = image.extract()?;
            let a = arr.as_array();
            let h = a.shape()[0] as u32;
            let w = a.shape()[1] as u32;
            Ok((a.iter().map(|&v| v as f32).collect(), w, h))
        }
        _ => {
            let dtype_str: String = dtype.str()?.extract()?;
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported image dtype '{}'. Expected float64, float32, uint16, int16, or uint8.",
                dtype_str,
            )))
        }
    }
}

/// Result of centroid extraction from an image.
#[pyclass(name = "ExtractionResult", frozen)]
pub(crate) struct PyExtractionResult {
    centroids: Vec<PyCentroid>,
    image_width: u32,
    image_height: u32,
    background_mean: f64,
    background_sigma: f64,
    threshold: f64,
    num_blobs_raw: usize,
}

#[pymethods]
impl PyExtractionResult {
    /// List of detected centroids, sorted by brightness (brightest first).
    #[getter]
    fn centroids(&self) -> Vec<PyCentroid> {
        self.centroids.clone()
    }

    /// Width of the input image in pixels.
    #[getter]
    fn image_width(&self) -> u32 {
        self.image_width
    }

    /// Height of the input image in pixels.
    #[getter]
    fn image_height(&self) -> u32 {
        self.image_height
    }

    /// Estimated background mean.
    #[getter]
    fn background_mean(&self) -> f64 {
        self.background_mean
    }

    /// Estimated background standard deviation.
    #[getter]
    fn background_sigma(&self) -> f64 {
        self.background_sigma
    }

    /// Detection threshold used.
    #[getter]
    fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Number of raw blobs before filtering.
    #[getter]
    fn num_blobs_raw(&self) -> usize {
        self.num_blobs_raw
    }

    fn __repr__(&self) -> String {
        format!(
            "ExtractionResult(centroids={}, image={}x{}, bg_mean={:.1}, bg_sigma={:.1}, threshold={:.1}, raw_blobs={})",
            self.centroids.len(),
            self.image_width,
            self.image_height,
            self.background_mean,
            self.background_sigma,
            self.threshold,
            self.num_blobs_raw,
        )
    }
}

/// Extract star centroids from a 2D image array.
///
/// Detects stars by sigma-clipping background estimation, thresholding, connected-
/// component labeling, and intensity-weighted centroiding. Each blob's background
/// is refined using the median of nearby non-blob pixels (annulus), and a 2D
/// quadratic fit to the 3×3 peak neighborhood provides sub-pixel precision.
///
/// Args:
///     image: 2D numpy array (height x width) of pixel values.
///         Supported dtypes: float64, float32, uint16, int16, uint8.
///     sigma_threshold: Detection threshold in sigma above background. Default 5.0.
///     min_pixels: Minimum blob size. Default 3.
///     max_pixels: Maximum blob size. Default 10000.
///     max_centroids: Maximum number of centroids to return. None = all.
///     local_bg_block_size: Block size for local background estimation. None = global only.
///     max_elongation: Maximum blob elongation ratio. None = disabled.
///
/// Returns:
///     ExtractionResult with centroids and image statistics.
#[pyfunction]
#[pyo3(signature = (
    image,
    sigma_threshold = 5.0,
    min_pixels = 3,
    max_pixels = 10000,
    max_centroids = None,
    local_bg_block_size = Some(64),
    max_elongation = Some(3.0),
))]
pub(crate) fn extract_centroids(
    image: &Bound<'_, pyo3::PyAny>,
    sigma_threshold: f32,
    min_pixels: usize,
    max_pixels: usize,
    max_centroids: Option<usize>,
    local_bg_block_size: Option<u32>,
    max_elongation: Option<f32>,
) -> PyResult<PyExtractionResult> {
    let (pixels, width, height) = image_to_f32(image)?;

    let config = CentroidExtractionConfig {
        sigma_threshold,
        min_pixels,
        max_pixels,
        max_centroids,
        sigma_clip_iterations: 5,
        sigma_clip_factor: 3.0,
        use_8_connectivity: true,
        local_bg_block_size,
        max_elongation,
    };

    let result =
        tetra3::centroid_extraction::extract_centroids_from_raw(&pixels, width, height, &config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_centroids: Vec<PyCentroid> = result
        .centroids
        .into_iter()
        .map(|c| PyCentroid { inner: c })
        .collect();

    Ok(PyExtractionResult {
        centroids: py_centroids,
        image_width: width,
        image_height: height,
        background_mean: result.background_mean as f64,
        background_sigma: result.background_sigma as f64,
        threshold: result.threshold as f64,
        num_blobs_raw: result.num_blobs_raw,
    })
}
