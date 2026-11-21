//! Extract star centroids from an image.
//!
//! The pipeline detects on a smoothed copy of the frame and measures on the
//! raw pixels:
//!
//! 1. **Detection image.** The frame is (optionally) binned 2×2 and
//!    convolved with a small separable Gaussian — together a matched filter
//!    for a PSF of width [`psf_sigma`](CentroidExtractionConfig::psf_sigma).
//! 2. **Background and noise.** A block-median level and a per-block noise
//!    σ are measured *on the detection image*, so the threshold is right
//!    whatever the pixel noise's correlation, and follows noise that varies
//!    across the frame.
//! 3. **Regions.** Pixels above `level + sigma_threshold · σ` are grouped
//!    into connected regions. A region only says which peaks belong
//!    together.
//! 4. **Peaks.** Each local maximum of the detection image inside a region
//!    is a candidate star, so close pairs separate; a fainter peak must
//!    stand clear of its neighbours' wings to count.
//! 5. **Measurement.** Each candidate gets an iterative Gaussian-windowed
//!    centroid on the raw pixels — the matched filter's maximum evaluated
//!    in the continuum — giving position, flux, position covariance and
//!    shape. Saturated and very broad blobs fall back to the region's
//!    center of mass.
//!
//! Requires the `image` feature.
//!
//! Entry points:
//! - [`extract_centroids_from_raw`] for row-major grayscale `f32` pixels
//!   (FITS, camera SDK output, …).
//! - [`extract_centroids_from_image`] for an already-decoded
//!   [`image::DynamicImage`]; the caller decodes the file.
//! - [`CentroidExtractor`] runs the same pipeline but keeps its full-image
//!   working buffers between calls, for a frame loop.
//!
//! With the `parallel` feature the full-image passes and the per-region
//! measurement run multi-threaded via rayon; results are bit-identical to
//! the sequential build.
//!
//! # Example
//!
//! ```no_run
//! use tetra3::centroid_extraction::{CentroidExtractionConfig, extract_centroids_from_image};
//!
//! let img = image::open("my_star_image.png").unwrap();
//! let config = CentroidExtractionConfig::default();
//! let result = extract_centroids_from_image(&img, &config).unwrap();
//! println!("Found {} stars", result.centroids.len());
//! ```

use crate::centroid::Centroid;
use crate::error::{Error, Result};
use crate::prelude::*;
use image::GenericImageView;

mod background;
mod detect;
mod measure;
mod runs;

/// Configuration for centroid extraction.
#[derive(Debug, Clone)]
pub struct CentroidExtractionConfig {
    /// Expected width of the point-spread function: the Gaussian σ in
    /// pixels (FWHM / 2.355). Sets the detection kernel and the measurement
    /// window, so it is the one parameter worth matching to the optics.
    /// (The window is never narrower than 1 px, whatever this says: a
    /// narrower one cannot be sampled on the pixel grid without the
    /// position wobbling with the star's sub-pixel phase.)
    ///
    /// The optimum is broad — a value within a factor of ~1.5 of the true
    /// width keeps the position error within about 30% of the best
    /// achievable — and
    /// [`CentroidExtractionResult::psf_sigma_estimate`] reports the width
    /// actually measured, to tune it from a real frame. Must be in
    /// `(0, 10]`.
    ///
    /// Default: 1.5
    pub psf_sigma: f32,

    /// Detection threshold, in σ of the noise of the detection image. A
    /// star is detected when its matched-filter signal-to-noise ratio
    /// exceeds this.
    ///
    /// Default: 5.0
    pub sigma_threshold: f32,

    /// Block size, in pixels, of the background and noise grid. Each
    /// `n × n` block contributes one background level and one noise σ; the
    /// level is bilinearly interpolated between block centers, which
    /// removes gradients from vignetting, stray light or the Milky Way.
    /// Smaller blocks follow finer structure but are noisier and risk
    /// absorbing bright stars. Roughly 1–3% of the image width is a good
    /// range.
    ///
    /// Default: 64
    pub bg_block_size: u32,

    /// Maximum number of centroids to return, brightest first. `None`
    /// returns every detection.
    ///
    /// Default: None
    pub max_centroids: Option<usize>,

    /// Raw pixel value at or above which the sensor is saturated. A region
    /// whose brightest peak is saturated is measured by its center of mass
    /// (a clipped profile has no meaningful matched-filter maximum) and
    /// yields a single centroid. Blobs far broader than `psf_sigma` take
    /// the same path automatically, so this is a refinement for mildly
    /// saturated stars, not a requirement.
    ///
    /// Default: None
    pub saturation_level: Option<f32>,

    /// Drop centroids within this many pixels of an image edge. Stars whose
    /// measurement window does not fit in the frame (closer than about
    /// `3 · psf_sigma` to an edge) are always dropped, because a truncated
    /// window biases the position toward the interior.
    ///
    /// Default: 0
    pub border_margin: u32,

    /// Regions larger than this many pixels are dropped whole — satellite
    /// trails, horizon glow, stray-light blobs. Results are brightest
    /// first, so without a cap such an object would be the *top* centroid
    /// handed to the solver.
    ///
    /// Default: 10000
    pub max_pixels: usize,

    /// Maximum elongation (major/minor axis ratio of the measured shape).
    /// More elongated detections are rejected as non-stellar: streaks,
    /// trails, unresolved blends, and saturated stars whose charge has bled
    /// far along a column (their center of mass can be pixels off). On a
    /// CCD that blooms, the brightest stars are the ones this removes; if
    /// the solver then lacks bright stars, raise it (5–6 keeps mild bleeds).
    /// `None` disables the gate.
    ///
    /// Default: `Some(3.0)`
    pub max_elongation: Option<f32>,

    /// Reject detections much narrower than the PSF (measured width under
    /// about 0.4 px, or `0.4 · psf_sigma` if that is larger) — hot pixels
    /// and cosmic-ray hits. Turn this off for severely undersampled data
    /// (PSF FWHM below ~1.2 px), where a real star is geometrically
    /// indistinguishable from a hot pixel.
    ///
    /// Default: true
    pub reject_hot_pixels: bool,

    /// Binning of the detection image: `Some(1)` detects at full
    /// resolution, `Some(2)` on the frame binned 2×2. `None` picks 2 when
    /// `psf_sigma ≥ 1.25` px and 1 otherwise.
    ///
    /// Binning makes every full-image pass four times cheaper. For a PSF
    /// that wide it costs roughly a tenth of the detection signal-to-noise
    /// (stars at the very threshold) and nothing in accuracy, because
    /// positions are always measured on the unbinned pixels. Force
    /// `Some(1)` for the deepest detection.
    ///
    /// Default: None (automatic)
    pub detection_binning: Option<u32>,
}

impl Default for CentroidExtractionConfig {
    fn default() -> Self {
        Self {
            psf_sigma: 1.5,
            sigma_threshold: 5.0,
            bg_block_size: 64,
            max_centroids: None,
            saturation_level: None,
            border_margin: 0,
            max_pixels: 10000,
            max_elongation: Some(3.0),
            reject_hot_pixels: true,
            detection_binning: None,
        }
    }
}

/// Result of centroid extraction: the centroids and diagnostics.
#[derive(Debug, Clone)]
pub struct CentroidExtractionResult {
    /// Extracted centroids, brightest first, in pixel coordinates with
    /// (0, 0) at the image center; +X right, +Y down. Each carries its flux
    /// (`mass`), position covariance (`cov`) and shape (`shape`).
    pub centroids: Vec<Centroid>,

    /// Image width in pixels.
    pub image_width: u32,

    /// Image height in pixels.
    pub image_height: u32,

    /// Background level (the median block level), in image intensity units.
    pub background_mean: f32,

    /// Background noise σ per raw pixel (the median block σ). It is
    /// measured on the detection image and converted assuming uncorrelated
    /// pixel noise, so on resampled or compressed frames it is an
    /// equivalent white-noise level rather than the literal pixel scatter.
    pub background_sigma: f32,

    /// Detection threshold at the median block, on the detection image:
    /// `background_mean + sigma_threshold · σ_detection`.
    pub threshold: f32,

    /// Number of connected regions above threshold, before any filter.
    pub num_blobs_raw: usize,

    /// PSF width (Gaussian σ, pixels) measured from this frame: the median
    /// over the well-exposed, unsaturated stars, corrected for the
    /// measurement window. It is the width as sampled, so it includes the
    /// pixel's own footprint (optics delivering σ = 0.75 px read ≈ 0.80). `None` with fewer than five such stars. Compare
    /// with [`CentroidExtractionConfig::psf_sigma`]; the estimate
    /// saturates near twice the configured value, so if it comes back much
    /// larger, raise `psf_sigma` and look again.
    pub psf_sigma_estimate: Option<f32>,
}

/// Extract star centroids from an already-decoded [`image::DynamicImage`].
///
/// Color images are converted to luminance. Centroids are returned in pixel
/// coordinates with the origin at the image center, suitable for
/// [`SolverDatabase::solve_from_centroids`](crate::SolverDatabase::solve_from_centroids).
///
/// To load from a file, decode it with `image::open(path)?` (which requires
/// the appropriate `image` crate format features in your own `Cargo.toml`)
/// and pass the resulting `DynamicImage` here.
pub fn extract_centroids_from_image(
    img: &image::DynamicImage,
    config: &CentroidExtractionConfig,
) -> Result<CentroidExtractionResult> {
    CentroidExtractor::new().extract_from_image(img, config)
}

/// Extract star centroids from raw grayscale pixel data — pixels from a
/// camera SDK, a FITS file parsed externally, or any other source.
///
/// # Arguments
///
/// * `pixels` - Row-major grayscale pixel values (length `width * height`).
///   Non-finite values are treated as background.
/// * `width`, `height` - Image size in pixels
/// * `config` - Extraction configuration
pub fn extract_centroids_from_raw(
    pixels: &[f32],
    width: u32,
    height: u32,
    config: &CentroidExtractionConfig,
) -> Result<CentroidExtractionResult> {
    CentroidExtractor::new().extract_from_raw(pixels, width, height, config)
}

/// The extraction pipeline with its working buffers kept between calls.
///
/// [`extract_centroids_from_image`] and [`extract_centroids_from_raw`]
/// allocate their full-image buffers fresh on every call, and the first
/// touch of each page costs more than the allocation itself. An extractor
/// reuses them, resizing only when the frame size changes, so a frame loop
/// pays that once. Results are bit-identical to the free functions, which
/// are exactly `CentroidExtractor::new().extract_*(…)`.
///
/// The buffers are released when the extractor is dropped. An extractor is
/// not shared between threads (`&mut self`); use one per thread.
///
/// ```no_run
/// use tetra3::centroid_extraction::{CentroidExtractionConfig, CentroidExtractor};
///
/// let config = CentroidExtractionConfig::default();
/// let mut extractor = CentroidExtractor::new();
/// # let frames: Vec<(Vec<f32>, u32, u32)> = Vec::new();
/// for (pixels, width, height) in &frames {
///     let result = extractor.extract_from_raw(pixels, *width, *height, &config).unwrap();
///     println!("{} stars", result.centroids.len());
/// }
/// ```
pub struct CentroidExtractor {
    /// Grayscale conversion of the last `DynamicImage` (image path only).
    gray: Vec<f32>,
    scratch: detect::Scratch,
}

impl Default for CentroidExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl CentroidExtractor {
    /// An extractor with no buffers allocated yet; the first call sizes them.
    pub fn new() -> Self {
        Self {
            gray: Vec::new(),
            scratch: detect::Scratch::new(),
        }
    }

    /// [`extract_centroids_from_image`] on this extractor's buffers.
    pub fn extract_from_image(
        &mut self,
        img: &image::DynamicImage,
        config: &CentroidExtractionConfig,
    ) -> Result<CentroidExtractionResult> {
        let (width, height) = img.dimensions();
        to_grayscale_f32_into(img, &mut self.gray);
        detect::extract(&self.gray, width, height, config, &mut self.scratch)
    }

    /// [`extract_centroids_from_raw`] on this extractor's buffers.
    pub fn extract_from_raw(
        &mut self,
        pixels: &[f32],
        width: u32,
        height: u32,
        config: &CentroidExtractionConfig,
    ) -> Result<CentroidExtractionResult> {
        check_pixel_len(pixels.len(), width, height)?;
        detect::extract(pixels, width, height, config, &mut self.scratch)
    }
}

// ─── Internal helpers ──────────────────────────────────────────────────────

/// Upper-midpoint order statistic `v[len/2]` — the cheap "median" of the
/// block grid. O(n) selection (partitions the slice in place); `0.0` for an
/// empty slice.
fn midpoint_f32(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let m = values.len() / 2;
    let (_, nth, _) = values.select_nth_unstable_by(m, |a, b| a.total_cmp(b));
    *nth
}

/// Median of the values (partitioned in place, O(n) selection): even lengths
/// average the two central order statistics — `values[n/2]` (the selected
/// element) and `values[n/2 − 1]` (the max of the lower partition that
/// `select_nth` leaves to its left). `0.0` for an empty slice.
fn median_f32(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len();
    let (lower, nth, _) = values.select_nth_unstable_by(n / 2, |a, b| a.total_cmp(b));
    if n.is_multiple_of(2) {
        let prev = lower.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (prev + *nth) / 2.0
    } else {
        *nth
    }
}

/// Sort centroids brightest-first (descending mass; missing mass sorts as 0)
/// and truncate to the configured maximum.
fn sort_and_truncate_by_mass(centroids: &mut Vec<Centroid>, max_centroids: Option<usize>) {
    // Sort (mass, original index) keys rather than the centroids themselves:
    // the key `(mass desc, index asc)` is a strict total order, so an
    // unstable sort of 8-byte keys reproduces exactly the order a stable
    // descending-mass sort would give, at roughly half the cost on dense
    // frames with tens of thousands of detections.
    let mut keys: Vec<(f32, u32)> = centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (c.mass.unwrap_or(0.0), i as u32))
        .collect();
    keys.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    if let Some(max) = max_centroids {
        keys.truncate(max);
    }
    *centroids = keys
        .iter()
        .map(|&(_, i)| centroids[i as usize].clone())
        .collect();
}

/// Validate that a raw pixel buffer matches the claimed dimensions.
fn check_pixel_len(len: usize, width: u32, height: u32) -> Result<()> {
    // Checked multiply: on a 32-bit target `width * height` can wrap in
    // `usize` (e.g. 65536×65536 → 0), letting an undersized buffer through to
    // panic on indexing later. No real buffer can reach a wrapping size.
    let expected = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| {
            Error::InvalidInput(format!("image dimensions {width}x{height} overflow usize"))
        })?;
    if len != expected {
        return Err(Error::InvalidInput(format!(
            "Pixel data length ({len}) does not match width*height ({width}x{height}={expected})",
        )));
    }
    Ok(())
}

/// Parallelism dispatch for the extraction hot paths.
///
/// Each helper has two cfg-gated twins: a [Rayon](https://docs.rs/rayon)
/// work-stealing version under the `parallel` feature and a plain sequential
/// version otherwise. The feature flag lives only here, so the two paths cannot
/// drift apart and the call sites read identically in both configurations.
///
/// All helpers are deterministic: the element-wise maps write disjoint outputs
/// and `map_indices` / `for_each_chunk_mut` assign each index or chunk to a
/// fixed output slot, so results are independent of thread count and the
/// non-`parallel` build is bit-identical to the original sequential code.
///
/// Used by the full-image passes (detection image, bit mask), the block
/// statistics (one task per block row) and the per-region measurement (one
/// task per region, order preserved by index).
pub(super) mod par {
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    /// Map `f` over `0..n` into a `Vec`, preserving index order.
    #[cfg(feature = "parallel")]
    pub fn map_indices<T, F>(n: usize, f: F) -> Vec<T>
    where
        T: Send,
        F: Fn(usize) -> T + Sync + Send,
    {
        (0..n).into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    pub fn map_indices<T, F>(n: usize, f: F) -> Vec<T>
    where
        F: Fn(usize) -> T,
    {
        (0..n).map(f).collect()
    }

    /// Map `f(&mut state, i)` over `0..n` into a `Vec`, preserving index
    /// order. `state` is created by `init` once per worker (once in total
    /// without the feature) and reused across the indices that worker
    /// handles — scratch buffers for a per-item body. `f` must not let its
    /// result depend on what `state` held from earlier indices.
    #[cfg(feature = "parallel")]
    pub fn map_indices_init<S, T, I, F>(n: usize, init: I, f: F) -> Vec<T>
    where
        T: Send,
        I: Fn() -> S + Sync + Send,
        F: Fn(&mut S, usize) -> T + Sync + Send,
    {
        (0..n).into_par_iter().map_init(init, f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    pub fn map_indices_init<S, T, I, F>(n: usize, init: I, mut f: F) -> Vec<T>
    where
        I: FnOnce() -> S,
        F: FnMut(&mut S, usize) -> T,
    {
        let mut state = init();
        (0..n).map(|i| f(&mut state, i)).collect()
    }

    /// Apply `f(i, chunk)` to each disjoint `chunk_len`-sized chunk of `buf`
    /// (one or more image rows per chunk; the last chunk may be shorter).
    #[cfg(feature = "parallel")]
    pub fn for_each_chunk_mut<T, F>(buf: &mut [T], chunk_len: usize, f: F)
    where
        T: Send,
        F: Fn(usize, &mut [T]) + Sync + Send,
    {
        buf.par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(i, c)| f(i, c));
    }
    #[cfg(not(feature = "parallel"))]
    pub fn for_each_chunk_mut<T, F>(buf: &mut [T], chunk_len: usize, mut f: F)
    where
        F: FnMut(usize, &mut [T]),
    {
        for (i, c) in buf.chunks_mut(chunk_len).enumerate() {
            f(i, c);
        }
    }
}

/// Convert a DynamicImage to a Vec<f32> of grayscale values.
fn to_grayscale_f32_into(img: &image::DynamicImage, out: &mut Vec<f32>) {
    use image::DynamicImage;
    out.clear();
    match img {
        // 8-bit grayscale: read the buffer directly (the `to_luma8()`
        // fallback below would clone it first). Alpha is dropped, exactly
        // as the Luma8 conversion does.
        DynamicImage::ImageLuma8(g) => out.extend(g.as_raw().iter().map(|&v| v as f32)),
        DynamicImage::ImageLumaA8(g) => out.extend(g.pixels().map(|p| p.0[0] as f32)),
        // 16-bit images: cast to f32 (values keep their native [0, 65535] range)
        DynamicImage::ImageLuma16(g) => out.extend(g.as_raw().iter().map(|&v| v as f32)),
        DynamicImage::ImageLumaA16(g) => out.extend(g.pixels().map(|p| p.0[0] as f32)),
        DynamicImage::ImageRgb16(rgb) => out.extend(rgb.pixels().map(|p| {
            let [r, g, b] = p.0;
            0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32
        })),
        DynamicImage::ImageRgba16(rgba) => out.extend(rgba.pixels().map(|p| {
            let [r, g, b, _] = p.0;
            0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32
        })),
        // For 32-bit float images
        DynamicImage::ImageRgb32F(rgb) => out.extend(rgb.pixels().map(|p| {
            let [r, g, b] = p.0;
            0.2126 * r + 0.7152 * g + 0.0722 * b
        })),
        DynamicImage::ImageRgba32F(rgba) => out.extend(rgba.pixels().map(|p| {
            let [r, g, b, _] = p.0;
            0.2126 * r + 0.7152 * g + 0.0722 * b
        })),
        // 8-bit and other formats: convert via luma8
        _ => {
            let gray = img.to_luma8();
            out.extend(gray.as_raw().iter().map(|&v| v as f32));
        }
    }
}

#[cfg(test)]
mod tests;
