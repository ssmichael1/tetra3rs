use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use tetra3::centroid_extraction::{CentroidExtractionConfig, CentroidExtractionResult};

use crate::centroid::PyCentroid;

/// The extraction result payload. Serde-derived, so it doubles as the pickle
/// wire format.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct ExtractionData {
    centroids: Vec<tetra3::Centroid>,
    image_width: u32,
    image_height: u32,
    background_mean: f64,
    background_sigma: f64,
    threshold: f64,
    num_blobs_raw: u64,
    psf_sigma_estimate: Option<f64>,
}

impl From<CentroidExtractionResult> for ExtractionData {
    fn from(result: CentroidExtractionResult) -> Self {
        Self {
            centroids: result.centroids,
            image_width: result.image_width,
            image_height: result.image_height,
            background_mean: result.background_mean as f64,
            background_sigma: result.background_sigma as f64,
            threshold: result.threshold as f64,
            num_blobs_raw: result.num_blobs_raw as u64,
            psf_sigma_estimate: result.psf_sigma_estimate.map(f64::from),
        }
    }
}

/// A numpy element type accepted as an image pixel.
trait Pixel: numpy::Element + Copy {
    fn to_f32(self) -> f32;

    /// Convert a contiguous run of pixels. Default is the element-wise cast;
    /// `f32` overrides it with a plain memcpy.
    fn slice_to_f32(s: &[Self]) -> Vec<f32> {
        s.iter().map(|&v| v.to_f32()).collect()
    }
}

macro_rules! impl_pixel_as_f32 {
    ($($t:ty),*) => {$(
        impl Pixel for $t {
            #[inline]
            fn to_f32(self) -> f32 {
                self as f32
            }
        }
    )*};
}
impl_pixel_as_f32!(f64, u8, u16, i16);

impl Pixel for f32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
    fn slice_to_f32(s: &[f32]) -> Vec<f32> {
        s.to_vec()
    }
}

/// Convert a 2D numpy array of any supported dtype to Vec<f32>.
///
/// Supported dtypes: float64, float32, uint8, uint16, int16. Non-native byte
/// order (e.g. the big-endian `>f4` that `np.frombuffer` produces from FITS
/// data) is converted to native order first.
fn image_to_f32(image: &Bound<'_, pyo3::PyAny>) -> PyResult<(Vec<f32>, u32, u32)> {
    use pyo3::exceptions::PyTypeError;

    // Require a numpy array. A plain list (or anything without `dtype`) should
    // get a clear TypeError rather than a confusing AttributeError.
    let Ok(dtype) = image.getattr("dtype") else {
        return Err(PyTypeError::new_err(
            "image must be a 2D numpy array (got an object with no 'dtype')",
        ));
    };
    let ndim: usize = image.getattr("ndim")?.extract()?;
    if ndim != 2 {
        return Err(PyTypeError::new_err(format!(
            "image must be a 2D numpy array, got {ndim} dimension(s)"
        )));
    }

    // Normalize non-native byte order (FITS arrays are big-endian). `astype`
    // with a native-order dtype yields a correctly-valued native copy; recurse
    // once on it (it then passes the is-native check).
    let is_native: bool = dtype.getattr("isnative")?.extract()?;
    if !is_native {
        let native_dtype = dtype.call_method1("newbyteorder", ("=",))?;
        let converted = image.call_method1("astype", (native_dtype,))?;
        return image_to_f32(&converted);
    }

    let kind: String = dtype.getattr("kind")?.extract()?;
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;

    /// Extract one supported dtype and convert element-wise to f32.
    ///
    /// A C-contiguous array converts straight from its flat slice (~5x cheaper
    /// than ndarray's generic strided iterator on a 2048² frame); any other
    /// layout (Fortran order, negative/step strides) takes the strided path.
    /// Both produce identical values.
    fn arr_to_f32<T: Pixel>(image: &Bound<'_, pyo3::PyAny>) -> PyResult<(Vec<f32>, u32, u32)> {
        let arr: PyReadonlyArray2<T> = image.extract()?;
        let a = arr.as_array();
        let h = a.shape()[0] as u32;
        let w = a.shape()[1] as u32;
        let pixels = match a.as_slice() {
            Some(s) => T::slice_to_f32(s),
            None => a.iter().map(|&v| v.to_f32()).collect(),
        };
        Ok((pixels, w, h))
    }

    match (kind.as_str(), itemsize) {
        ("f", 8) => arr_to_f32::<f64>(image),
        ("f", 4) => arr_to_f32::<f32>(image),
        ("u", 1) => arr_to_f32::<u8>(image),
        ("u", 2) => arr_to_f32::<u16>(image),
        ("i", 2) => arr_to_f32::<i16>(image),
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
#[pyclass(name = "ExtractionResult", module = "tetra3rs", frozen)]
pub(crate) struct PyExtractionResult {
    inner: ExtractionData,
}

#[pymethods]
impl PyExtractionResult {
    /// List of detected centroids, sorted by brightness (brightest first).
    #[getter]
    fn centroids(&self) -> Vec<PyCentroid> {
        self.inner
            .centroids
            .iter()
            .map(|c| PyCentroid { inner: c.clone() })
            .collect()
    }

    /// Width of the input image in pixels.
    #[getter]
    fn image_width(&self) -> u32 {
        self.inner.image_width
    }

    /// Height of the input image in pixels.
    #[getter]
    fn image_height(&self) -> u32 {
        self.inner.image_height
    }

    /// Background level (the median block level), in image units.
    #[getter]
    fn background_mean(&self) -> f64 {
        self.inner.background_mean
    }

    /// Background noise sigma per raw pixel (the median block sigma). It is
    /// measured on the detection image and converted assuming uncorrelated
    /// pixel noise, so on resampled or compressed frames it is an equivalent
    /// white-noise level rather than the literal pixel scatter.
    #[getter]
    fn background_sigma(&self) -> f64 {
        self.inner.background_sigma
    }

    /// Detection threshold at the median block, on the (smoothed) detection
    /// image: ``background_mean + sigma_threshold * sigma_detection``.
    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.threshold
    }

    /// Number of connected regions above threshold, before any filter.
    #[getter]
    fn num_blobs_raw(&self) -> usize {
        self.inner.num_blobs_raw as usize
    }

    /// PSF width (Gaussian sigma, pixels) measured from this frame: the
    /// median over the well-exposed, unsaturated stars. None with fewer than
    /// five such stars. Compare with the ``psf_sigma`` you passed; the
    /// estimate saturates near twice that value, so if it comes back much
    /// larger, raise ``psf_sigma`` and look again.
    #[getter]
    fn psf_sigma_estimate(&self) -> Option<f64> {
        self.inner.psf_sigma_estimate
    }

    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<(Py<PyAny>, (Vec<u8>,))> {
        crate::helpers::pickle_reduce(slf, &slf.borrow().inner)
    }

    #[staticmethod]
    fn _from_pickle_bytes(data: &[u8]) -> PyResult<Self> {
        let inner = crate::helpers::from_postcard_bytes::<ExtractionData>(data)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "ExtractionResult(centroids={}, image={}x{}, bg_mean={:.1}, bg_sigma={:.1}, threshold={:.1}, raw_blobs={})",
            self.inner.centroids.len(),
            self.inner.image_width,
            self.inner.image_height,
            self.inner.background_mean,
            self.inner.background_sigma,
            self.inner.threshold,
            self.inner.num_blobs_raw,
        )
    }
}

/// Extract star centroids from a 2D image array.
///
/// Detects on a smoothed copy of the frame and measures on the raw pixels:
/// the frame is (optionally) binned 2x2 and convolved with a small Gaussian
/// (a matched filter for a PSF of width ``psf_sigma``); the background level
/// and a per-block noise sigma are measured on that detection image; pixels
/// above ``level + sigma_threshold * sigma`` are grouped into regions; every
/// local maximum inside a region is a candidate star; and each candidate
/// gets an iterative Gaussian-windowed centroid on the raw pixels, giving
/// position, flux (``brightness``), position covariance (``cov``) and shape
/// (``shape``). Saturated and very broad blobs fall back to the region's
/// center of mass.
///
/// Args:
///     image: 2D numpy array (height x width) of pixel values.
///         Supported dtypes: float64, float32, uint16, int16, uint8.
///         Non-finite values are treated as background.
///     sigma_threshold: Detection threshold in sigma of the detection
///         image's noise — a star is detected when its matched-filter
///         signal-to-noise ratio exceeds this. Default 5.0.
///     psf_sigma: Expected PSF width, the Gaussian sigma in pixels
///         (FWHM / 2.355). Sets the detection kernel and the measurement
///         window; the one parameter worth matching to the optics. The
///         optimum is broad (within a factor ~1.5 of the true width costs
///         under ~30% in position error) and ``psf_sigma_estimate`` on the
///         result reports the width actually measured. Must be in (0, 10].
///         Default 1.5.
///     bg_block_size: Block size in pixels of the background / noise grid.
///         Roughly 1-3% of the image width. Default 64.
///     max_centroids: Maximum number of centroids to return, brightest
///         first. None = all.
///     saturation_level: Raw pixel value at or above which the sensor is
///         saturated. A region whose brightest peak is saturated is measured
///         by its center of mass and yields one centroid. Blobs far broader
///         than ``psf_sigma`` take that path automatically, so this is a
///         refinement for mildly saturated stars. None = not set.
///     border_margin: Drop centroids within this many pixels of an image
///         edge. Stars whose measurement window does not fit in the frame
///         (closer than about ``3 * psf_sigma`` to an edge) are always
///         dropped. Default 0.
///     max_pixels: Regions larger than this many pixels are dropped whole
///         (satellite trails, horizon glow). Default 10000.
///     max_elongation: Maximum elongation (major/minor axis ratio) —
///         rejects streaks, unresolved blends, and saturated stars whose
///         charge has bled far along a column. On a CCD that blooms, raise
///         it (5-6) if the solver lacks bright stars. None = disabled.
///         Default 3.0.
///     reject_hot_pixels: Reject detections much narrower than the PSF (hot
///         pixels, cosmic-ray hits). Turn off for severely undersampled data
///         (PSF FWHM below ~1.2 px). Default True.
///     detection_binning: 1 detects at full resolution, 2 on the frame
///         binned 2x2 (about twice as fast overall; costs roughly a tenth of
///         the detection signal-to-noise for a PSF sigma >= 1.25 px and
///         nothing in accuracy — positions are always measured on unbinned
///         pixels). None picks 2 when ``psf_sigma >= 1.25`` and 1 otherwise.
///         Default None.
///     threads: Worker threads for this call (keyword-only). None (default)
///         uses every available core — or ``RAYON_NUM_THREADS`` if set —
///         except for frames under ~512×512, which run on one thread
///         because waking the workers costs more than it saves. An integer
///         uses exactly that many; 1 is single-threaded. The result is the
///         same for every setting. Safe to use in forked child processes.
///
/// Returns:
///     ExtractionResult with centroids and image statistics.
#[pyfunction]
#[pyo3(signature = (
    image,
    sigma_threshold = 5.0,
    *,
    psf_sigma = 1.5,
    bg_block_size = 64,
    max_centroids = None,
    saturation_level = None,
    border_margin = 0,
    max_pixels = 10000,
    max_elongation = Some(3.0),
    reject_hot_pixels = true,
    detection_binning = None,
    threads = None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn extract_centroids(
    image: &Bound<'_, pyo3::PyAny>,
    sigma_threshold: f32,
    psf_sigma: f32,
    bg_block_size: u32,
    max_centroids: Option<usize>,
    saturation_level: Option<f32>,
    border_margin: u32,
    max_pixels: usize,
    max_elongation: Option<f32>,
    reject_hot_pixels: bool,
    detection_binning: Option<u32>,
    threads: Option<usize>,
) -> PyResult<PyExtractionResult> {
    let config = CentroidExtractionConfig {
        psf_sigma,
        sigma_threshold,
        bg_block_size,
        max_centroids,
        saturation_level,
        border_margin,
        max_pixels,
        max_elongation,
        reject_hot_pixels,
        detection_binning,
    };
    let (pixels, width, height) = image_to_f32(image)?;

    // The image was copied into a pure-Rust buffer above; the extraction
    // runs with the GIL released (so other Python threads keep running), on
    // the module's own thread pool (see `threads`).
    let result = crate::threads::run(image.py(), threads, pixels.len(), || {
        tetra3::centroid_extraction::extract_centroids_from_raw(&pixels, width, height, &config)
    })?
    .map_err(crate::helpers::map_tetra3_err)?;

    Ok(PyExtractionResult {
        inner: result.into(),
    })
}

/// The :func:`extract_centroids` pipeline with its working buffers kept
/// between calls.
///
/// :func:`extract_centroids` allocates its full-image buffers fresh on
/// every call, and the first touch of each page costs more than the
/// allocation. An extractor reuses them, resizing only when the frame size
/// changes, so a frame loop pays that once. Results are bit-identical to
/// :func:`extract_centroids`.
///
/// The buffers are freed with the object. Use one extractor per thread.
/// Pickling keeps no buffers — an unpickled extractor starts empty.
///
/// Example::
///
///     extractor = tetra3rs.CentroidExtractor()
///     for frame in frames:
///         result = extractor.extract(frame, psf_sigma=1.5)
#[pyclass(name = "CentroidExtractor", module = "tetra3rs")]
pub(crate) struct PyCentroidExtractor {
    inner: tetra3::centroid_extraction::CentroidExtractor,
}

#[pymethods]
impl PyCentroidExtractor {
    #[new]
    fn new() -> Self {
        Self {
            inner: tetra3::centroid_extraction::CentroidExtractor::new(),
        }
    }

    /// Extract star centroids from a 2D image array, reusing this
    /// extractor's buffers.
    ///
    /// Takes exactly the arguments of :func:`extract_centroids` (including
    /// the keyword-only ``threads``), with the same defaults, and returns
    /// the same ``ExtractionResult``.
    #[pyo3(signature = (
        image,
        sigma_threshold = 5.0,
        *,
        psf_sigma = 1.5,
        bg_block_size = 64,
        max_centroids = None,
        saturation_level = None,
        border_margin = 0,
        max_pixels = 10000,
        max_elongation = Some(3.0),
        reject_hot_pixels = true,
        detection_binning = None,
        threads = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn extract(
        &mut self,
        image: &Bound<'_, pyo3::PyAny>,
        sigma_threshold: f32,
        psf_sigma: f32,
        bg_block_size: u32,
        max_centroids: Option<usize>,
        saturation_level: Option<f32>,
        border_margin: u32,
        max_pixels: usize,
        max_elongation: Option<f32>,
        reject_hot_pixels: bool,
        detection_binning: Option<u32>,
        threads: Option<usize>,
    ) -> PyResult<PyExtractionResult> {
        let config = CentroidExtractionConfig {
            psf_sigma,
            sigma_threshold,
            bg_block_size,
            max_centroids,
            saturation_level,
            border_margin,
            max_pixels,
            max_elongation,
            reject_hot_pixels,
            detection_binning,
        };
        let (pixels, width, height) = image_to_f32(image)?;
        let inner = &mut self.inner;
        // Input copied to a pure-Rust buffer; the extraction runs with the
        // GIL released (the `&mut self` borrow keeps Python from
        // re-entering), on the module's own thread pool (see `threads`).
        let result = crate::threads::run(image.py(), threads, pixels.len(), || {
            inner.extract_from_raw(&pixels, width, height, &config)
        })?
        .map_err(crate::helpers::map_tetra3_err)?;
        Ok(PyExtractionResult {
            inner: result.into(),
        })
    }

    /// Pickles as a fresh extractor: the buffers are scratch, not state.
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<(Py<PyAny>, ())> {
        Ok((slf.get_type().into_any().unbind(), ()))
    }

    fn __repr__(&self) -> &'static str {
        "CentroidExtractor()"
    }
}
