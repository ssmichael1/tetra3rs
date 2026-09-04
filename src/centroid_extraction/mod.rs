//! Extract star centroids from an astronomical image.
//!
//! This module provides functions to detect and locate stars in pixel data by:
//! 1. Converting the image to grayscale floating-point values
//! 2. Estimating and subtracting the background (sigma-clipped median)
//! 3. Thresholding to identify bright pixels
//! 4. Labeling connected components (blobs)
//! 5. Computing intensity-weighted centroids for each blob, with:
//!    - Per-blob local background from an annulus of non-blob pixels
//!    - Quadratic peak refinement (2D fit to 3×3 around peak pixel)
//!
//! Requires the `image` feature to be enabled.
//!
//! Entry points:
//! - [`extract_centroids_from_image`] for an already-decoded
//!   [`image::DynamicImage`]. The caller is responsible for decoding the
//!   file (using whichever `image` feature flags suit their needs).
//! - [`extract_centroids_from_raw`] for raw grayscale `f32` pixel data —
//!   useful for FITS, camera SDK output, or any other non-standard source.
//! - [`extract_centroids_fast`] is a single-pass "adequate star tracker"
//!   alternative: it reads each pixel once (coarse-grid background +
//!   run-length connected-component moments) for markedly lower latency, at
//!   the cost of faint-star sensitivity and sub-pixel accuracy. The two
//!   functions above stay the default and the right choice for calibration.
//! - [`CentroidExtractor`] runs the same pipeline as the two default
//!   functions but keeps its full-image working buffers between calls, for
//!   a frame loop where every frame has the same size.
//!
//! With the `parallel` feature, the dominant local-background stage and the
//! full-image element-wise maps of the connected-component path, and the
//! background grid + detection bit mask of the fast path, run multi-threaded
//! via rayon; results are bit-identical to the sequential build. (The fast
//! path's run sweep over the mask is sequential — it is proportional to the
//! number of runs, not pixels.)
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
use image::GenericImageView;

mod ccl;
mod fast;
mod runs;

pub use fast::{extract_centroids_fast, FastCentroidConfig};

/// Deblending policy for blobs containing more than one distinct intensity
/// peak (a blended star pair yields a single centroid at the flux-weighted
/// midpoint — a wrong position the pattern hash will happily consume).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeblendMode {
    /// Keep blended blobs as a single merged centroid (historical behavior).
    #[default]
    Off,
    /// Reject blobs with more than one distinct peak — the safe choice for
    /// plate solving, where a missing star costs far less than a wrong
    /// position. A peak counts as distinct when it is a strict local maximum
    /// over its 8-neighborhood, rises above 30% of the blob's peak (over the
    /// local background), and lies more than 2 px from any brighter accepted
    /// peak. Saturated blobs (per `saturation_level`) are exempt: plateau
    /// noise fakes multiple maxima on a genuinely single star.
    Reject,
}

/// Configuration for centroid extraction from an image.
#[derive(Debug, Clone)]
pub struct CentroidExtractionConfig {
    /// Number of sigma above background to use as the detection threshold.
    /// Stars brighter than `background + sigma_threshold * noise` are detected.
    /// Default: 5.0
    pub sigma_threshold: f32,

    /// Minimum number of pixels in a blob to be considered a star.
    /// Helps filter out hot pixels and noise.
    /// Default: 3
    pub min_pixels: usize,

    /// Maximum number of pixels in a blob to be considered a star.
    /// Helps filter out very large extended objects.
    /// Set high enough to include saturated bright stars with large halos.
    /// Default: 10000
    pub max_pixels: usize,

    /// Maximum number of centroids to return, sorted by brightness (mass).
    /// If `None`, all detected centroids are returned.
    /// Default: None
    pub max_centroids: Option<usize>,

    /// Number of iterations for sigma-clipped background estimation.
    /// Default: 5
    pub sigma_clip_iterations: usize,

    /// Sigma clipping factor for background estimation.
    /// Pixels more than this many sigma from the mean are excluded.
    /// Default: 3.0
    pub sigma_clip_factor: f32,

    /// Block size (in pixels) for local background estimation.
    ///
    /// When set to `Some(n)`, the image is divided into `n×n` blocks and
    /// the median value in each block is computed. A smooth background
    /// model is created by bilinear interpolation between block centers
    /// and subtracted before star detection. This removes large-scale
    /// gradients from nebulosity, Milky Way emission, or vignetting.
    ///
    /// A good starting value is 32-128 pixels, or roughly 1-3% of the
    /// image width. Smaller blocks follow finer structure but risk
    /// subtracting real stars.
    ///
    /// When `None`, only global background subtraction is used (original
    /// behavior).
    ///
    /// Default: Some(64)
    pub local_bg_block_size: Option<u32>,

    /// Maximum allowed elongation ratio (major/minor axis) for a detected
    /// blob. Blobs more elongated than this are rejected as non-stellar
    /// (e.g. cosmic rays, satellite trails, diffraction spikes).
    ///
    /// A value of 2.0 means the blob can be at most 2× longer than wide.
    /// Set to a large value (e.g. 100) or `None` to disable.
    ///
    /// Default: `Some(3.0)`
    pub max_elongation: Option<f32>,

    /// Apply a Gaussian matched filter to the bg-subtracted image before
    /// thresholding. When `Some(sigma)`, the image is convolved with a
    /// separable 1-D Gaussian (σ in pixels, kernel truncated at 3σ). The
    /// filtered image is used **only** to form the detection mask —
    /// centroid positions and intensities are still measured on the
    /// unfiltered bg-subtracted image, so photometry is unaffected.
    ///
    /// A matched filter boosts point-source SNR before thresholding —
    /// ~2× peak SNR (≈0.75 mag more depth at the same false-positive rate)
    /// for a σ≈1.5 px PSF. The gain is largest for faint stars in noisy or
    /// dense images, and the optimum is broad: σ within a factor of ~2 of
    /// the true PSF width recovers nearly all of it.
    ///
    /// The detection threshold is automatically scaled by the kernel's
    /// noise-suppression factor, so `sigma_threshold` means "sigmas of the
    /// noise actually present in the thresholded image" whether the filter
    /// is on or off — no retuning needed when toggling it.
    ///
    /// Default: Some(1.5). Set `None` to threshold the unfiltered image
    /// (marginally faster; appropriate when downstream limits like
    /// `max_centroids` make faint-star depth irrelevant).
    pub matched_filter_sigma: Option<f32>,

    /// Maximum DAOFIND-style sharpness: `(peak − mean(8 neighbors)) / peak`,
    /// measured on the background-subtracted image at the blob's peak. Values
    /// near 1 mean the flux is concentrated in a single pixel — a hot pixel
    /// or cosmic-ray hit rather than a star. A critically sampled PSF scores
    /// ~0.5; a strongly undersampled one can reach ~0.85. The default 0.9
    /// passes any system whose PSF spans multiple pixels (the design norm —
    /// star trackers defocus deliberately, because a sub-pixel PSF forfeits
    /// sub-pixel centroiding). Set `None` for severely undersampled data
    /// (PSF FWHM below ~1.5 px), where real stars are geometrically
    /// indistinguishable from hot pixels.
    ///
    /// Default: Some(0.9)
    pub max_sharpness: Option<f32>,

    /// Pixel value at or above which the sensor is considered saturated.
    /// A blob whose peak reaches this level skips quadratic peak refinement
    /// (a flat-topped or bloomed profile has no meaningful sub-pixel
    /// maximum), keeping the center-of-mass position instead.
    ///
    /// Default: None (disabled)
    pub saturation_level: Option<f32>,

    /// What to do with blobs containing more than one distinct intensity
    /// peak (blended star pairs). See [`DeblendMode`].
    ///
    /// Default: [`DeblendMode::Off`]
    pub deblend: DeblendMode,

    /// Drop blobs whose bounding box comes within this many pixels of an
    /// image edge. A star cut off by the frame boundary has a truncated PSF,
    /// which biases its center-of-mass toward the interior — a plausible but
    /// wrong position (only the 3×3 parabola was border-gated before).
    /// A couple of PSF widths (e.g. 3-5 px) is a sensible setting.
    ///
    /// Default: 0 (disabled)
    pub border_margin: u32,
}

impl Default for CentroidExtractionConfig {
    fn default() -> Self {
        Self {
            sigma_threshold: 5.0,
            min_pixels: 3,
            max_pixels: 10000,
            max_centroids: None,
            sigma_clip_iterations: 5,
            sigma_clip_factor: 3.0,
            local_bg_block_size: Some(64),
            max_elongation: Some(3.0),
            matched_filter_sigma: Some(1.5),
            max_sharpness: Some(0.9),
            saturation_level: None,
            deblend: DeblendMode::Off,
            border_margin: 0,
        }
    }
}

/// Result of centroid extraction, containing the centroids and diagnostic info.
#[derive(Debug, Clone)]
pub struct CentroidExtractionResult {
    /// Extracted centroids in pixel coordinates, with (0, 0) at the image center.
    /// +X is right (increasing column), +Y is down (increasing row).
    pub centroids: Vec<Centroid>,

    /// Image width in pixels.
    pub image_width: u32,

    /// Image height in pixels.
    pub image_height: u32,

    /// Estimated background level (in image intensity units).
    pub background_mean: f32,

    /// Estimated background noise standard deviation.
    pub background_sigma: f32,

    /// Detection threshold used (background_mean + sigma_threshold * background_sigma).
    pub threshold: f32,

    /// Number of blobs found before the size/elongation filters are applied
    /// (connected components on the CCL path; detected regions before the
    /// `min_pixels` filter on the fast path).
    pub num_blobs_raw: usize,
}

/// Extract star centroids from an already-decoded [`image::DynamicImage`].
///
/// Performs background subtraction, blob detection, and centroid computation
/// on an in-memory image. Centroids are returned in pixel coordinates with the
/// origin at the image center, suitable for use with
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

/// Extract star centroids from raw grayscale pixel data.
///
/// This is useful when you have pixel data that isn't in a standard image format,
/// e.g. from a camera SDK or FITS file parsed externally.
///
/// # Arguments
///
/// * `pixels` - Row-major grayscale pixel values (length must equal `width * height`)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `config` - Extraction configuration parameters
pub fn extract_centroids_from_raw(
    pixels: &[f32],
    width: u32,
    height: u32,
    config: &CentroidExtractionConfig,
) -> Result<CentroidExtractionResult> {
    CentroidExtractor::new().extract_from_raw(pixels, width, height, config)
}

/// The default extraction pipeline with its working buffers kept between
/// calls.
///
/// [`extract_centroids_from_image`] and [`extract_centroids_from_raw`]
/// allocate their full-image buffers — the grayscale conversion, the clamped
/// and unclamped residual images, the matched filter's output and the
/// detection bit mask, ~48 MB at 2048² — fresh on every call, and the first
/// touch of each page costs more than the allocation itself (~0.3 ms serial,
/// ~0.5 ms with the `parallel` feature at 2048²). An extractor reuses them,
/// resizing only when the frame size changes, so a frame loop pays that once.
/// Results are bit-identical to the free functions, which are exactly
/// `CentroidExtractor::new().extract_*(…)`.
///
/// The buffers are sized for the largest frame seen and released when the
/// extractor is dropped. An extractor is not shared between threads
/// (`&mut self`); use one per thread.
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
    scratch: ccl::Scratch,
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
            scratch: ccl::Scratch::new(),
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
        ccl::extract_from_gray(&self.gray, width, height, config, &mut self.scratch)
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
        ccl::extract_from_gray(pixels, width, height, config, &mut self.scratch)
    }
}

// ─── Internal helpers ──────────────────────────────────────────────────────

/// Upper-midpoint order statistic `v[len/2]` — the extraction pipeline's cheap
/// "median" convention for background grids. O(n) selection (partitions the
/// slice in place); `0.0` for an empty slice.
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
/// and truncate to the configured maximum. Shared tail of both extraction
/// paths.
fn sort_and_truncate_by_mass(centroids: &mut Vec<Centroid>, max_centroids: Option<usize>) {
    // Sort (mass, original index) keys rather than the centroids themselves:
    // the key `(mass desc, index asc)` is a strict total order, so an
    // unstable sort of 8-byte keys reproduces exactly the order a stable
    // descending-mass sort would give — at roughly half the cost on dense
    // frames with tens of thousands of detections. Both extraction paths
    // emit `mass` as `Some(m as f32)` with `m > 0.0` in f64 (or `None`,
    // ranked as 0.0): never NaN and never -0.0, so `total_cmp` agrees with
    // `partial_cmp` on every key present.
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

/// Parallelism dispatch for the centroid-extraction hot paths.
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
/// Both paths parallelize their background grid (one task per block row)
/// and their detection bit mask (16-row chunks). The CCL path additionally
/// runs its residual pass by rows, its run sweep in 64-row bands, and its
/// per-region annulus / moment / deblend stage as one task per region
/// (`map_indices_init`, order preserved by index); the fast path's run
/// sweep stays sequential.
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

    /// Apply `f(i, chunk_a, chunk_b)` to corresponding disjoint
    /// `chunk_len`-sized chunks of two buffers (one image row each).
    #[cfg(feature = "parallel")]
    pub fn for_each_chunk_pair_mut<T, U, F>(a: &mut [T], b: &mut [U], chunk_len: usize, f: F)
    where
        T: Send,
        U: Send,
        F: Fn(usize, &mut [T], &mut [U]) + Sync + Send,
    {
        a.par_chunks_mut(chunk_len)
            .zip(b.par_chunks_mut(chunk_len))
            .enumerate()
            .for_each(|(i, (ca, cb))| f(i, ca, cb));
    }
    #[cfg(not(feature = "parallel"))]
    pub fn for_each_chunk_pair_mut<T, U, F>(a: &mut [T], b: &mut [U], chunk_len: usize, mut f: F)
    where
        F: FnMut(usize, &mut [T], &mut [U]),
    {
        for (i, (ca, cb)) in a
            .chunks_mut(chunk_len)
            .zip(b.chunks_mut(chunk_len))
            .enumerate()
        {
            f(i, ca, cb);
        }
    }
}

/// Coarse block-median background grid shared by both extraction paths.
///
/// The image is divided into `block × block` tiles; each tile's median comes
/// from a phase-staggered stride subsample (a diagonal lattice: every
/// column-residue class is sampled equally, so column-periodic structure —
/// CMOS fixed-pattern noise, Bayer residue — does not alias; the block
/// median's standard error ≈ 1.25σ/√n stays far below detection thresholds).
/// Only non-finite samples are excluded: zeros and negatives are legitimate
/// background on dark-subtracted frames.
///
/// Interpolation is bilinear between block centers and **linearly
/// extrapolates** beyond the outermost centers — clamping (the historical
/// behavior) left any gradient un-modeled across the outer `block/2` border
/// band, which lights up as border false positives at tight thresholds.
pub(super) struct BackgroundGrid {
    grid: Vec<f32>,
    nx: usize,
    ny: usize,
    block: usize,
    stride: usize,
    /// Row-independent half of the interpolation for every image column.
    cols: ColPlan,
}

/// Per-column interpolation parameters of a [`BackgroundGrid`] for a `w`-wide
/// image — the row-independent half of the bilinear blend, computed once with
/// the same [`col_params`] arithmetic the per-pixel accessors use and grouped
/// into segments of constant `(bx0, bx1)`, so blending a whole row is a
/// straight multiply-add loop with no per-pixel divide / floor / clamp.
/// See [`BackgroundGrid::blend_columns`].
struct ColPlan {
    /// Column blend weight `fx` and its complement `1 - fx`.
    fx: Vec<f32>,
    omfx: Vec<f32>,
    /// `(c_start, c_end_exclusive, bx0, bx1)` — maximal column ranges that
    /// share the same pair of grid columns.
    segs: Vec<(usize, usize, usize, usize)>,
}

impl ColPlan {
    fn new(nx: usize, block: usize, w: usize) -> Self {
        let mut fx = Vec::with_capacity(w);
        let mut omfx = Vec::with_capacity(w);
        let mut segs: Vec<(usize, usize, usize, usize)> = Vec::new();
        for c in 0..w {
            let (bx0, bx1, f) = col_params(nx, block, c);
            fx.push(f);
            omfx.push(1.0 - f);
            match segs.last_mut() {
                Some(seg) if seg.2 == bx0 && seg.3 == bx1 => seg.1 = c + 1,
                _ => segs.push((c, c + 1, bx0, bx1)),
            }
        }
        Self { fx, omfx, segs }
    }
}

/// Column-constant part of the interpolation for image column `x`: the two
/// grid columns to blend and the (unclamped — extrapolating) blend weight.
#[inline]
fn col_params(nx: usize, block: usize, x: usize) -> (usize, usize, f32) {
    if nx == 1 {
        return (0, 0, 0.0);
    }
    let bf = (x as f32 - block as f32 / 2.0) / block as f32;
    let bx0 = (bf.floor() as isize).clamp(0, nx as isize - 2) as usize;
    (bx0, bx0 + 1, bf - bx0 as f32)
}

impl BackgroundGrid {
    /// Build the grid. Also returns the global noise σ estimated during the
    /// same pass as the RMS of below-median subsample residuals about their
    /// block median (the half-normal estimator, robust to stars which only
    /// push the distribution upward). The fast path uses this σ directly;
    /// the CCL path ignores it and re-estimates against the bilinear surface
    /// (see `subsample_residuals` + `estimate_background`).
    pub(super) fn build(
        pixels: &[f32],
        w: usize,
        h: usize,
        block: usize,
        stride: usize,
    ) -> (Self, f32) {
        let nx = w.div_ceil(block);
        let ny = h.div_ceil(block);

        // (median, Σresidual², n_below) per block, one task per block row.
        // Each block row walks its sampled image rows once, left to right
        // across all its blocks (forward streaming rather than a strided
        // gather per block); each block still receives its samples in the
        // same y-then-x order, so medians and residual sums are unchanged.
        let per_block_row: Vec<Vec<(f32, f64, usize)>> = par::map_indices(ny, |by| {
            let y0 = by * block;
            let y1 = (y0 + block).min(h);
            let cap = (block / stride + 1).pow(2);
            let mut vals: Vec<Vec<f32>> = (0..nx).map(|_| Vec::with_capacity(cap)).collect();
            let mut y = y0;
            let mut phase = 0usize;
            while y < y1 {
                let row = &pixels[y * w..(y + 1) * w];
                for (bx, block_vals) in vals.iter_mut().enumerate() {
                    let x0 = bx * block;
                    let x1 = (x0 + block).min(w);
                    let mut x = x0 + phase;
                    while x < x1 {
                        let v = row[x];
                        if v.is_finite() {
                            block_vals.push(v);
                        }
                        x += stride;
                    }
                }
                phase = (phase + 1) % stride;
                y += stride;
            }
            vals.iter_mut()
                .map(|block_vals| {
                    let median = midpoint_f32(block_vals);
                    let mut sq = 0.0_f64;
                    let mut n = 0usize;
                    for &v in block_vals.iter() {
                        if v <= median {
                            let d = (v - median) as f64;
                            sq += d * d;
                            n += 1;
                        }
                    }
                    (median, sq, n)
                })
                .collect()
        });

        let per_block = per_block_row.iter().flatten();
        let grid: Vec<f32> = per_block.clone().map(|&(m, _, _)| m).collect();
        let (sq_sum, n_sum) =
            per_block.fold((0.0_f64, 0usize), |(s, n), &(_, sq, k)| (s + sq, n + k));
        let sigma = if n_sum > 0 {
            (sq_sum / n_sum as f64).sqrt() as f32
        } else {
            0.0
        };

        (
            Self {
                grid,
                nx,
                ny,
                block,
                stride,
                cols: ColPlan::new(nx, block, w),
            },
            sigma,
        )
    }

    pub(super) fn stride(&self) -> usize {
        self.stride
    }

    /// Number of grid columns — the length [`Self::blend_row`] expects.
    pub(super) fn grid_width(&self) -> usize {
        self.nx
    }

    /// Representative background level: the midpoint of the block medians.
    pub(super) fn level(&self) -> f32 {
        midpoint_f32(&mut self.grid.clone())
    }

    /// Row-constant part of the interpolation for image row `y`: the two
    /// grid rows to blend and the (unclamped — extrapolating) blend weight.
    #[inline]
    pub(super) fn row_params(&self, y: usize) -> (usize, usize, f32) {
        if self.ny == 1 {
            return (0, 0, 0.0);
        }
        let bf = (y as f32 - self.block as f32 / 2.0) / self.block as f32;
        let by0 = (bf.floor() as isize).clamp(0, self.ny as isize - 2) as usize;
        (by0, by0 + 1, bf - by0 as f32)
    }

    /// Background value at `(x, row)` given `row_params(row)`.
    #[inline]
    pub(super) fn value_at(&self, x: usize, (by0, by1, fy): (usize, usize, f32)) -> f32 {
        let (bx0, bx1, fx) = self.col_params(x);
        let g0 = self.grid[by0 * self.nx + bx0] * (1.0 - fy) + self.grid[by1 * self.nx + bx0] * fy;
        let g1 = self.grid[by0 * self.nx + bx1] * (1.0 - fy) + self.grid[by1 * self.nx + bx1] * fy;
        g0 * (1.0 - fx) + g1 * fx
    }

    /// Blend one grid row for `row_params(row)` into `out` (length `nx`) —
    /// hoists the row-constant half of the interpolation out of per-row
    /// passes; combine with [`Self::blend_columns`].
    #[inline]
    pub(super) fn blend_row(&self, (by0, by1, fy): (usize, usize, f32), out: &mut [f32]) {
        for (bx, g) in out.iter_mut().enumerate() {
            *g = self.grid[by0 * self.nx + bx] * (1.0 - fy) + self.grid[by1 * self.nx + bx] * fy;
        }
    }

    /// Column half of the interpolation for a whole row: with `row_blend`
    /// from [`Self::blend_row`], for every image column `c`
    ///
    /// ```text
    /// out[c] = map(row_blend[bx0] * (1 - fx) + row_blend[bx1] * fx)
    /// ```
    ///
    /// with `(bx0, bx1, fx) = col_params(c)` — the same expression and
    /// operation order as [`Self::value_at`] (which is exactly `blend_row`
    /// followed by this column blend), so `map = |v| v` reproduces
    /// `value_at(c, rp)` bit for bit, but the per-column divide / floor /
    /// clamp is hoisted into the column plan built once by [`Self::build`].
    /// `out.len()` must equal the image width given to `build`. `map` is
    /// applied to each blended value (`|v| v + k·σ` for a detection
    /// threshold, see [`Self::threshold_row`]).
    #[inline]
    pub(super) fn blend_columns(
        &self,
        row_blend: &[f32],
        out: &mut [f32],
        map: impl Fn(f32) -> f32,
    ) {
        debug_assert_eq!(out.len(), self.cols.fx.len());
        for &(c0, c1, bx0, bx1) in &self.cols.segs {
            let (g0, g1) = (row_blend[bx0], row_blend[bx1]);
            let fx = &self.cols.fx[c0..c1];
            let omfx = &self.cols.omfx[c0..c1];
            for ((o, &f), &omf) in out[c0..c1].iter_mut().zip(fx).zip(omfx) {
                *o = map(g0 * omf + g1 * f);
            }
        }
    }

    /// Detection threshold of every column of the row blended into
    /// `row_blend`: `out[c] = value_at(c, rp) + k_sigma`, bit for bit (see
    /// [`Self::blend_columns`]).
    #[inline]
    pub(super) fn threshold_row(&self, row_blend: &[f32], k_sigma: f32, out: &mut [f32]) {
        self.blend_columns(row_blend, out, |v| v + k_sigma);
    }

    #[inline]
    fn col_params(&self, x: usize) -> (usize, usize, f32) {
        col_params(self.nx, self.block, x)
    }
}

/// Elongation ratio (major/minor axis) of a blob from its intensity-weighted
/// central second moments: `√(λ_max/λ_min)` of the 2×2 covariance
/// `[[cxx, cxy], [cxy, cyy]]`. `λ_min` is floored so degenerate (collinear)
/// blobs come out very elongated rather than dividing by zero — the correct
/// verdict for a 1-pixel-wide streak. Shared by both extraction paths.
fn elongation_from_cov(cxx: f64, cyy: f64, cxy: f64) -> f32 {
    let trace = cxx + cyy;
    let det = cxx * cyy - cxy * cxy;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let lambda_max = (trace + disc) / 2.0;
    let lambda_min = (trace - disc).max(1e-12) / 2.0;
    (lambda_max / lambda_min).sqrt() as f32
}

/// 3×3 parabola sub-pixel refinement at the integer peak `(pc, pr)`, gated the
/// same way in both extraction paths: the blob must have ≥ 5 pixels, the peak
/// must not touch the border of the `(w, h)` image, and the fitted position
/// must agree with the center-of-mass estimate `(com_x, com_y)` within 0.5 px
/// (for asymmetric or blended blobs the CoM is more reliable). Returns the
/// refined position, or `None` to keep the CoM.
///
/// When all nine background-subtracted samples are positive, the parabola is
/// fit to **log intensity**: a Gaussian PSF is exactly quadratic in
/// `ln(v)` (`ln(A·e^{−r²/2σ²}) = ln A − r²/2σ²`), which removes most of the
/// linear fit's S-curve bias (~0.05–0.1 px at quarter-pixel peak phases —
/// the classic star-tracker refinement). Blobs with a non-positive sample in
/// the window (faint stars whose wings dip below the local background) keep
/// the linear fit, preserving the previous behavior there.
fn accepted_peak_refine(
    npix: usize,
    (pc, pr): (usize, usize),
    (w, h): (usize, usize),
    (com_x, com_y): (f64, f64),
    v: impl Fn(isize, isize) -> f64,
) -> Option<(f64, f64)> {
    if npix < 5 || pc < 1 || pr < 1 || pc + 1 >= w || pr + 1 >= h {
        return None;
    }
    let mut vals = [[0.0_f64; 3]; 3];
    let mut all_positive = true;
    for dy in -1..=1_isize {
        for dx in -1..=1_isize {
            let val = v(dy, dx);
            vals[(dy + 1) as usize][(dx + 1) as usize] = val;
            all_positive &= val > 0.0;
        }
    }
    if all_positive {
        for row in vals.iter_mut() {
            for val in row.iter_mut() {
                *val = val.ln();
            }
        }
    }
    let (x_off, y_off) =
        quadratic_peak_offset(|dy, dx| vals[(dy + 1) as usize][(dx + 1) as usize])?;
    let qx = pc as f64 + x_off;
    let qy = pr as f64 + y_off;
    let dist_sq = (qx - com_x) * (qx - com_x) + (qy - com_y) * (qy - com_y);
    if dist_sq < 0.25 {
        Some((qx, qy))
    } else {
        None
    }
}

/// Shared tail of the per-region pipeline in both extraction paths, run after
/// the moments, elongation gate, and (CCL only) deblending: the hot-pixel
/// sharpness gate, the quadratic peak refinement (skipped for saturated
/// peaks — a flat top has no meaningful sub-pixel maximum), and assembly of
/// the [`Centroid`] in image-center-origin coordinates. `v(dy, dx)` samples
/// background-subtracted values relative to the integer peak `(pc, pr)`;
/// `(com_x, com_y)` is the center of mass in pixel coordinates,
/// `(cxx, cyy, cxy)` the intensity-weighted central second moments, and
/// `(cx, cy)` the image-center origin. Returns `None` when the blob is
/// rejected as a hot pixel / cosmic ray.
#[allow(clippy::too_many_arguments)]
fn finish_region(
    npix: usize,
    (pc, pr): (usize, usize),
    (w, h): (usize, usize),
    (com_x, com_y): (f64, f64),
    (cxx, cyy, cxy): (f64, f64, f64),
    mass: f64,
    saturated: bool,
    max_sharpness: Option<f32>,
    (cx, cy): (f32, f32),
    v: impl Fn(isize, isize) -> f64,
) -> Option<Centroid> {
    if let Some(max_sharp) = max_sharpness {
        if let Some(s) = peak_sharpness((pc, pr), (w, h), &v) {
            if s > max_sharp as f64 {
                return None;
            }
        }
    }
    let (mut fx, mut fy) = (com_x, com_y);
    if !saturated {
        if let Some((qx, qy)) = accepted_peak_refine(npix, (pc, pr), (w, h), (fx, fy), &v) {
            fx = qx;
            fy = qy;
        }
    }
    Some(Centroid {
        x: fx as f32 - cx,
        y: fy as f32 - cy,
        mass: Some(mass as f32),
        cov: Some(crate::Matrix2::new([
            [cxx as f32, cxy as f32],
            [cxy as f32, cyy as f32],
        ])),
    })
}

/// DAOFIND-style sharpness of a blob peak: `(peak − mean(8 neighbors)) / peak`
/// on background-subtracted values (`v(dy, dx)` samples relative to the peak,
/// the same accessor convention as [`accepted_peak_refine`]). Out-of-bounds
/// neighbors are skipped. Values near 1 mean the flux is concentrated in a
/// single pixel — a hot pixel or cosmic-ray hit; a real PSF puts substantial
/// flux into the neighbors (critically sampled ~0.5, strongly undersampled up
/// to ~0.85). Returns `None` when the peak is non-positive or has no
/// in-bounds neighbors (sharpness undefined — callers should not reject).
fn peak_sharpness(
    (pc, pr): (usize, usize),
    (w, h): (usize, usize),
    v: impl Fn(isize, isize) -> f64,
) -> Option<f64> {
    let peak = v(0, 0);
    if peak <= 0.0 {
        return None;
    }
    let mut sum = 0.0_f64;
    let mut n = 0u32;
    for dy in -1..=1_isize {
        for dx in -1..=1_isize {
            if dy == 0 && dx == 0 {
                continue;
            }
            let rr = pr as isize + dy;
            let cc = pc as isize + dx;
            if rr < 0 || cc < 0 || rr >= h as isize || cc >= w as isize {
                continue;
            }
            sum += v(dy, dx);
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }
    Some((peak - sum / n as f64) / peak)
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

/// Sub-pixel peak offset from a 2-D quadratic fit to a 3×3 neighborhood.
///
/// `v(dy, dx)` samples the (background-subtracted) surface at the peak pixel
/// plus integer offset `(dy, dx)`, `dx`/`dy` ∈ {−1, 0, 1}. Fits a bivariate
/// quadratic and returns the vertex offset `(x_off, y_off)` from the peak
/// pixel, or `None` when the fit is degenerate (near-flat Hessian) or
/// extrapolates beyond half a pixel (an unreliable peak — the caller should
/// fall back to the integer peak / center-of-mass). Shared by the
/// connected-component path ([`compute_blob_centroids`]) and the fast
/// DoG path ([`extract_centroids_fast`]).
fn quadratic_peak_offset(v: impl Fn(isize, isize) -> f64) -> Option<(f64, f64)> {
    let b = (v(0, 1) - v(0, -1)) / 2.0;
    let c_coeff = (v(1, 0) - v(-1, 0)) / 2.0;
    let d = (v(0, 1) + v(0, -1) - 2.0 * v(0, 0)) / 2.0;
    let f = (v(1, 0) + v(-1, 0) - 2.0 * v(0, 0)) / 2.0;
    let e = (v(1, 1) - v(1, -1) - v(-1, 1) + v(-1, -1)) / 4.0;

    let denom = 4.0 * d * f - e * e;
    if denom.abs() <= 1e-10 {
        return None;
    }
    let x_off = (e * c_coeff - 2.0 * f * b) / denom;
    let y_off = (e * b - 2.0 * d * c_coeff) / denom;
    if x_off.abs() <= 0.5 && y_off.abs() <= 0.5 {
        Some((x_off, y_off))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::ccl::estimate_background;
    use super::*;

    #[test]
    fn test_to_grayscale_f32_luma8_direct_arms_match_to_luma8() {
        let (w, h) = (37u32, 11u32);
        let raw: Vec<u8> = (0..w * h).map(|i| (i * 7919 % 256) as u8).collect();
        let luma =
            image::DynamicImage::ImageLuma8(image::GrayImage::from_raw(w, h, raw.clone()).unwrap());
        let expect: Vec<f32> = luma.to_luma8().as_raw().iter().map(|&v| v as f32).collect();
        assert_eq!(gray_of(&luma), expect);
        // LumaA8: alpha is dropped, exactly as `to_luma8()` does.
        let raw_a: Vec<u8> = raw
            .iter()
            .enumerate()
            .flat_map(|(i, &v)| [v, (i % 251) as u8])
            .collect();
        let luma_a =
            image::DynamicImage::ImageLumaA8(image::GrayAlphaImage::from_raw(w, h, raw_a).unwrap());
        let expect_a: Vec<f32> = luma_a
            .to_luma8()
            .as_raw()
            .iter()
            .map(|&v| v as f32)
            .collect();
        assert_eq!(gray_of(&luma_a), expect_a);
        assert_eq!(expect, expect_a);
    }

    #[test]
    fn test_ccl_rejects_degenerate_geometry() {
        let cfg = CentroidExtractionConfig::default();
        // Zero-size and 1-wide images used to panic (chunk size 0 / width-1
        // underflow) rather than return an error.
        assert!(extract_centroids_from_raw(&[], 0, 0, &cfg).is_err());
        assert!(extract_centroids_from_raw(&[1.0], 1, 1, &cfg).is_err());
    }

    /// A reused extractor must give exactly the free functions' output —
    /// including after a *larger* frame left stale data in its buffers and
    /// with the matched filter both on and off (different buffers in play).
    fn gray_of(img: &image::DynamicImage) -> Vec<f32> {
        let mut v = Vec::new();
        to_grayscale_f32_into(img, &mut v);
        v
    }

    #[test]
    fn test_extractor_reuse_is_bit_identical() {
        fn scene(w: u32, h: u32, seed: u32) -> Vec<f32> {
            let mut pixels = vec![50.0_f32; (w * h) as usize];
            let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state
            };
            for p in pixels.iter_mut() {
                *p += (next() % 7) as f32; // a little quantized noise
            }
            for k in 0..12 {
                let sx = 8.0 + (next() % (w - 16)) as f32;
                let sy = 8.0 + (next() % (h - 16)) as f32;
                let amp = 300.0 + 100.0 * (k % 5) as f32;
                for row in 0..h {
                    for col in 0..w {
                        let dx = col as f32 - sx;
                        let dy = row as f32 - sy;
                        pixels[(row * w + col) as usize] +=
                            amp * (-(dx * dx + dy * dy) / (2.0 * 1.5 * 1.5)).exp();
                    }
                }
            }
            pixels
        }
        let same = |a: &CentroidExtractionResult, b: &CentroidExtractionResult| {
            assert_eq!(a.centroids.len(), b.centroids.len());
            for (x, y) in a.centroids.iter().zip(&b.centroids) {
                assert_eq!(x.x.to_bits(), y.x.to_bits());
                assert_eq!(x.y.to_bits(), y.y.to_bits());
                assert_eq!(x.mass.map(f32::to_bits), y.mass.map(f32::to_bits));
                assert_eq!(x.cov, y.cov);
            }
            assert_eq!(a.background_mean.to_bits(), b.background_mean.to_bits());
            assert_eq!(a.background_sigma.to_bits(), b.background_sigma.to_bits());
            assert_eq!(a.threshold.to_bits(), b.threshold.to_bits());
            assert_eq!(a.num_blobs_raw, b.num_blobs_raw);
        };
        let filtered = CentroidExtractionConfig::default();
        let unfiltered = CentroidExtractionConfig {
            matched_filter_sigma: None,
            ..Default::default()
        };
        let global_bg = CentroidExtractionConfig {
            local_bg_block_size: None,
            ..Default::default()
        };
        // (w, h, config) sequence: big → small → big, filter on/off, local/global bg.
        let frames: [(u32, u32, &CentroidExtractionConfig); 6] = [
            (160, 120, &filtered),
            (96, 80, &filtered),
            (96, 80, &unfiltered),
            (160, 120, &global_bg),
            (130, 70, &filtered),
            (96, 80, &filtered),
        ];
        let mut extractor = CentroidExtractor::new();
        for (i, &(w, h, cfg)) in frames.iter().enumerate() {
            let pixels = scene(w, h, i as u32 + 1);
            let fresh = extract_centroids_from_raw(&pixels, w, h, cfg).unwrap();
            assert!(
                fresh.centroids.len() >= 3,
                "frame {i}: {} stars",
                fresh.centroids.len()
            );
            let reused = extractor.extract_from_raw(&pixels, w, h, cfg).unwrap();
            same(&fresh, &reused);
        }
        // Image path shares the grayscale buffer across sizes too.
        for &(w, h) in &[(160u32, 120u32), (96, 80), (160, 120)] {
            let pixels = scene(w, h, 7);
            let img =
                image::DynamicImage::ImageLuma16(image::ImageBuffer::from_fn(w, h, |x, y| {
                    image::Luma([pixels[(y * w + x) as usize] as u16])
                }));
            let fresh = extract_centroids_from_image(&img, &filtered).unwrap();
            let reused = extractor.extract_from_image(&img, &filtered).unwrap();
            same(&fresh, &reused);
        }
    }

    #[test]
    fn test_ccl_rejects_bad_config() {
        let pixels = vec![0.0_f32; 16 * 16];
        let zero_block = CentroidExtractionConfig {
            local_bg_block_size: Some(0),
            ..Default::default()
        };
        assert!(extract_centroids_from_raw(&pixels, 16, 16, &zero_block).is_err());
        let nan_thresh = CentroidExtractionConfig {
            sigma_threshold: f32::NAN,
            ..Default::default()
        };
        assert!(extract_centroids_from_raw(&pixels, 16, 16, &nan_thresh).is_err());
    }

    #[test]
    fn test_background_estimation() {
        // Uniform image: background should be ~100, sigma ~0
        let pixels = vec![100.0_f32; 100 * 100];
        let config = CentroidExtractionConfig::default();
        let (mean, sigma) = estimate_background(&pixels, 100, 100, &config);
        assert!((mean - 100.0).abs() < 1.0);
        assert!(sigma < 1.0);
    }

    #[test]
    fn test_extract_from_raw_single_star() {
        let width = 64u32;
        let height = 64u32;
        let mut pixels = vec![10.0_f32; (width * height) as usize];

        // Place a bright Gaussian-ish star near center
        let star_x = 32.0_f32;
        let star_y = 32.0_f32;
        let sigma_px = 2.0_f32;
        for row in 0..height {
            for col in 0..width {
                let dx = col as f32 - star_x;
                let dy = row as f32 - star_y;
                let r2 = dx * dx + dy * dy;
                pixels[(row * width + col) as usize] +=
                    1000.0 * (-r2 / (2.0 * sigma_px * sigma_px)).exp();
            }
        }

        let config = CentroidExtractionConfig {
            sigma_threshold: 3.0,
            min_pixels: 2,
            ..Default::default()
        };

        let result = extract_centroids_from_raw(&pixels, width, height, &config).unwrap();
        assert_eq!(result.centroids.len(), 1);

        // The centroid should be near the center of the image (0, 0 in pixel coords)
        let c = &result.centroids[0];
        assert!(c.x.abs() < 1.0, "Expected x near 0, got {}", c.x);
        assert!(c.y.abs() < 1.0, "Expected y near 0, got {}", c.y);
        assert!(c.mass.unwrap() > 0.0);
    }

    #[test]
    fn test_fast_path_rejects_trails_and_giant_regions() {
        // A giant bright disc (> max_pixels) and a thin streak must not
        // outrank the real star in the fast path's brightest-first output.
        // bg_grid is set to the frame size so the coarse background cannot
        // absorb the disc (at default grid sizes, structure larger than a
        // block is background-subtracted away before the filters see it).
        let (width, height) = (256u32, 256u32);
        let mut pixels = render_stars(
            width,
            height,
            100.0,
            0.0,
            4.0,
            1.5,
            &[(190.0, 190.0, 800.0)],
        );
        // Flat disc, radius 60 → ~11.3k px, over the default max_pixels.
        for row in 0..height as usize {
            for col in 0..width as usize {
                let (dx, dy) = (col as f32 - 80.0, row as f32 - 80.0);
                if dx * dx + dy * dy < 60.0 * 60.0 {
                    pixels[row * 256 + col] += 500.0;
                }
            }
        }
        // Thin bright streak (a trail segment): 60 px long, 1 px tall,
        // clear of both the disc and the star.
        for col in 20..80 {
            pixels[230 * 256 + col] += 500.0;
        }

        let base = FastCentroidConfig {
            sigma_threshold: 5.0,
            bg_grid: 256,
            ..Default::default()
        };
        // Default max_pixels rejects the disc; the streak needs elongation.
        let res = extract_centroids_fast(&pixels, width, height, &base).unwrap();
        assert_eq!(res.centroids.len(), 2, "star + streak expected");
        assert!(
            res.centroids
                .iter()
                .all(|c| (c.x - (80.0 - 127.5)).abs() > 10.0),
            "disc should be rejected by max_pixels"
        );

        let gated = FastCentroidConfig {
            max_elongation: Some(3.0),
            min_pixels: 5,
            ..base
        };
        let res = extract_centroids_fast(&pixels, width, height, &gated).unwrap();
        assert_eq!(res.centroids.len(), 1, "only the real star should survive");
        assert!(
            (res.centroids[0].x - (190.0 - 127.5)).abs() < 1.0
                && (res.centroids[0].y - (190.0 - 127.5)).abs() < 1.0
        );
        assert!(res.centroids[0].cov.is_some(), "fast path now reports cov");
    }

    #[test]
    fn test_log_parabola_subpixel_accuracy() {
        // A Gaussian PSF is exactly quadratic in log intensity, so the
        // refined position of a bright, point-sampled Gaussian star must be
        // accurate at every sub-pixel phase — including the quarter-pixel
        // phases where the linear-intensity parabola's S-curve bias peaks
        // (~0.03-0.06 px at this PSF width, which would fail this bound).
        let (width, height) = (64u32, 64u32);
        for &(px, py) in &[
            (30.0_f32, 31.0_f32),
            (30.25, 31.25),
            (30.5, 31.4),
            (29.75, 30.6),
        ] {
            let pixels = render_stars(width, height, 100.0, 0.0, 2.0, 1.3, &[(px, py, 5000.0)]);
            let cfg = CentroidExtractionConfig {
                sigma_threshold: 5.0,
                local_bg_block_size: None,
                matched_filter_sigma: None,
                ..Default::default()
            };
            let res = extract_centroids_from_raw(&pixels, width, height, &cfg).unwrap();
            assert_eq!(res.centroids.len(), 1, "phase ({px}, {py})");
            let c = &res.centroids[0];
            let (ex, ey) = (c.x - (px - 31.5), c.y - (py - 31.5));
            assert!(
                ex.abs() < 0.02 && ey.abs() < 0.02,
                "phase ({px}, {py}): error ({ex:.4}, {ey:.4}) px"
            );
        }
    }

    #[test]
    fn test_border_margin() {
        // A star half-off the frame edge centroids to a biased interior
        // position; border_margin drops it while keeping the interior star.
        let (width, height) = (64u32, 64u32);
        let pixels = render_stars(
            width,
            height,
            100.0,
            0.0,
            2.0,
            1.5,
            &[(1.0, 30.0, 1000.0), (40.0, 30.0, 1000.0)],
        );
        let base = CentroidExtractionConfig {
            sigma_threshold: 5.0,
            local_bg_block_size: None,
            ..Default::default()
        };
        let all = extract_centroids_from_raw(&pixels, width, height, &base).unwrap();
        assert_eq!(all.centroids.len(), 2, "margin off: both detected");

        let gated = CentroidExtractionConfig {
            border_margin: 4,
            ..base
        };
        let res = extract_centroids_from_raw(&pixels, width, height, &gated).unwrap();
        assert_eq!(res.centroids.len(), 1, "edge-truncated star dropped");
        assert!((res.centroids[0].x - (40.0 - 31.5)).abs() < 0.5);

        // Fast path honors the same knob.
        let fast = FastCentroidConfig {
            sigma_threshold: 5.0,
            border_margin: 4,
            ..Default::default()
        };
        let res = extract_centroids_fast(&pixels, width, height, &fast).unwrap();
        assert_eq!(res.centroids.len(), 1, "fast path drops the edge star");
    }

    #[test]
    fn test_deblend_reject() {
        // A blended pair (4 px apart, comparable brightness) merges into one
        // blob whose centroid lands between the stars. Off keeps the merged
        // centroid (historical behavior); Reject drops the blob while
        // keeping the isolated star. A saturated flat-top star is exempt
        // even though plateau noise fakes multiple maxima.
        let (width, height) = (96u32, 96u32);
        let mut pixels = render_stars(
            width,
            height,
            100.0,
            0.0,
            4.0,
            1.3,
            &[
                (30.0, 30.0, 2000.0),
                (34.0, 30.0, 1500.0),
                (70.0, 70.0, 2000.0),
            ],
        );
        let base = CentroidExtractionConfig {
            sigma_threshold: 5.0,
            local_bg_block_size: None,
            ..Default::default()
        };
        let merged = extract_centroids_from_raw(&pixels, width, height, &base).unwrap();
        assert_eq!(merged.centroids.len(), 2, "pair merges into one blob");

        let reject = CentroidExtractionConfig {
            deblend: DeblendMode::Reject,
            ..base.clone()
        };
        let res = extract_centroids_from_raw(&pixels, width, height, &reject).unwrap();
        assert_eq!(res.centroids.len(), 1, "blended blob rejected");
        assert!(
            (res.centroids[0].x - (70.0 - 47.5)).abs() < 0.5,
            "isolated star survives"
        );

        // Saturated exemption: clip the pair's peaks flat and mark the level.
        for v in pixels.iter_mut() {
            *v = v.min(600.0);
        }
        let sat = CentroidExtractionConfig {
            deblend: DeblendMode::Reject,
            saturation_level: Some(600.0),
            ..base
        };
        let res = extract_centroids_from_raw(&pixels, width, height, &sat).unwrap();
        assert_eq!(
            res.centroids.len(),
            2,
            "saturated blobs exempt from deblend rejection"
        );
    }

    #[test]
    fn test_deblend_reject_saturation_local_bg() {
        // Same saturated-exemption case as `test_deblend_reject`, but on the
        // default local-background path. Saturation must be judged on the RAW
        // sensor value (== the clip level), NOT the background-subtracted
        // residual `peak_val` (clip − background < clip): with the residual
        // comparison the exemption never fires, plateau noise fakes multiple
        // maxima, and the blended pair is wrongly rejected. This is the path
        // `test_deblend_reject` (local_bg_block_size = None) cannot cover.
        let (width, height) = (96u32, 96u32);
        let bg = 100.0_f32;
        let clip = 600.0_f32;
        let mut pixels = render_stars(
            width,
            height,
            bg,
            0.0,
            4.0,
            1.3,
            &[
                (30.0, 30.0, 2000.0),
                (34.0, 30.0, 1500.0),
                (70.0, 70.0, 2000.0),
            ],
        );
        for v in pixels.iter_mut() {
            *v = v.min(clip);
        }
        // Residual peak ≈ clip − bg = 500 < 600, so a residual-based comparison
        // would classify these clipped stars as unsaturated.
        assert!(clip - bg < clip);

        let sat = CentroidExtractionConfig {
            deblend: DeblendMode::Reject,
            saturation_level: Some(clip),
            local_bg_block_size: Some(16),
            sigma_threshold: 5.0,
            ..Default::default()
        };
        let res = extract_centroids_from_raw(&pixels, width, height, &sat).unwrap();
        assert_eq!(
            res.centroids.len(),
            2,
            "saturated blobs exempt from deblend rejection on the local-bg path"
        );
    }

    #[test]
    fn test_centroid_accuracy_ensemble() {
        // Characterization: ensemble centroid RMSE vs truth for noisy stars
        // at deterministic pseudo-random sub-pixel phases, at two PSF widths
        // bracketing typical trackers (σ 0.9 ≈ TESS-like undersampled,
        // σ 1.5 ≈ deliberately defocused). Run with --nocapture to see the
        // measured RMSE. Guards sub-pixel accuracy regressions and answers
        // "is the centroider the limiting error term?" for improvements like
        // a windowed CoM: the TESS multi-sector calibration residual is
        // ~0.077 px, so an ensemble RMSE well below that means the floor is
        // elsewhere (catalog, proper motion, optics model).
        let (width, height) = (96u32, 96u32);
        for &(sigma_px, amp, bound) in &[
            (0.9_f32, 3000.0_f32, 0.03_f32),
            (1.5, 3000.0, 0.03),
            (1.5, 300.0, 0.12),
        ] {
            let mut se = 0.0_f64;
            let mut n = 0usize;
            for trial in 0..40u64 {
                // splitmix64-derived sub-pixel phase
                let mut z = trial ^ 0x9e37_79b9_7f4a_7c15;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                let px = 47.0 + ((z >> 40) as f32 / 16_777_216.0 - 0.5);
                let py = 47.0 + ((z >> 16 & 0xFF_FFFF) as f32 / 16_777_216.0 - 0.5);
                let pixels =
                    render_stars(width, height, 100.0, 0.0, 20.0, sigma_px, &[(px, py, amp)]);
                let cfg = CentroidExtractionConfig {
                    sigma_threshold: 5.0,
                    local_bg_block_size: None,
                    ..Default::default()
                };
                let res = extract_centroids_from_raw(&pixels, width, height, &cfg).unwrap();
                assert_eq!(
                    res.centroids.len(),
                    1,
                    "σ={sigma_px} amp={amp} trial={trial}"
                );
                let c = &res.centroids[0];
                let (ex, ey) = ((c.x - (px - 47.5)) as f64, (c.y - (py - 47.5)) as f64);
                se += ex * ex + ey * ey;
                n += 1;
            }
            let rmse = (se / (2 * n) as f64).sqrt();
            println!("centroid ensemble RMSE: psf σ={sigma_px} amp={amp} → {rmse:.4} px");
            assert!(
                rmse < bound as f64,
                "σ={sigma_px} amp={amp}: RMSE {rmse:.4} px exceeds {bound}"
            );
        }
    }

    #[test]
    fn test_background_extrapolates_at_borders() {
        // Steep gradient on a frame only 2 background blocks wide — the
        // geometry where a border *clamp* leaves ~7.5 ADU of un-modeled ramp
        // across the outer 32-px band (well above the compensated filtered
        // threshold) and lights it up with false detections. Linear
        // extrapolation beyond the outer block centers must model it away.
        let (width, height) = (128u32, 128u32);
        let pixels = render_stars(width, height, 100.0, 30.0, 20.0, 1.5, &[]);
        let cfg = CentroidExtractionConfig {
            sigma_threshold: 5.0,
            ..Default::default()
        };
        let res = extract_centroids_from_raw(&pixels, width, height, &cfg).unwrap();
        assert_eq!(
            res.centroids.len(),
            0,
            "gradient border band produced detections"
        );
    }

    #[test]
    fn test_matched_filter_depth_gain() {
        // A star too faint for the unfiltered 5σ cut is recovered when the
        // matched filter is on — at the SAME sigma_threshold, because the
        // detection threshold is scaled by the kernel's noise-suppression
        // factor automatically.
        let (width, height) = (64u32, 64u32);
        let pixels = render_stars(width, height, 100.0, 0.0, 20.0, 1.5, &[(30.0, 30.0, 20.0)]);
        let base = CentroidExtractionConfig {
            sigma_threshold: 5.0,
            local_bg_block_size: None,
            matched_filter_sigma: None,
            ..Default::default()
        };
        let unfiltered = extract_centroids_from_raw(&pixels, width, height, &base).unwrap();
        assert_eq!(
            unfiltered.centroids.len(),
            0,
            "star should sit below the unfiltered cut"
        );

        let filtered_cfg = CentroidExtractionConfig {
            matched_filter_sigma: Some(1.5),
            ..base
        };
        let filtered = extract_centroids_from_raw(&pixels, width, height, &filtered_cfg).unwrap();
        assert_eq!(
            filtered.centroids.len(),
            1,
            "matched filter should recover the faint star"
        );
        assert!((filtered.centroids[0].x - (30.0 - 31.5)).abs() < 1.0);
        assert!((filtered.centroids[0].y - (30.0 - 31.5)).abs() < 1.0);
    }

    #[test]
    fn test_matched_filter_no_noise_false_positives() {
        // Pure noise + gradient with the (default-on) filter and local
        // background: the compensated threshold must keep false positives at
        // zero. Regression guard: convolving the *clamped* residual rectified
        // negative noise into a positive DC offset comparable to the
        // compensated threshold, which would light up the whole frame.
        // 4x4+ background blocks: the bilinear background clamps at the
        // outermost block centers, so a steep gradient on a 2-block-wide
        // frame leaves an un-modeled ramp near the borders that exceeds any
        // tight threshold — a (pre-existing) edge-extrapolation limitation,
        // not what this test measures.
        let (width, height) = (256u32, 256u32);
        let pixels = render_stars(width, height, 100.0, 10.0, 20.0, 1.5, &[]);
        let cfg = CentroidExtractionConfig {
            sigma_threshold: 5.0,
            ..Default::default()
        };
        let res = extract_centroids_from_raw(&pixels, width, height, &cfg).unwrap();
        assert_eq!(
            res.centroids.len(),
            0,
            "noise-only frame produced detections"
        );
    }

    #[test]
    fn test_peak_sharpness_values() {
        // Lone hot pixel: all 8 neighbors zero → sharpness exactly 1.
        let hot = |dy: isize, dx: isize| if dy == 0 && dx == 0 { 100.0 } else { 0.0 };
        assert_eq!(peak_sharpness((1, 1), (3, 3), hot), Some(1.0));
        // Flat plateau: neighbors equal the peak → sharpness 0.
        let flat = |_: isize, _: isize| 50.0;
        assert_eq!(peak_sharpness((1, 1), (3, 3), flat), Some(0.0));
        // Corner peak: only the 3 in-bounds neighbors are averaged.
        let corner = |dy: isize, dx: isize| if dy == 0 && dx == 0 { 90.0 } else { 30.0 };
        assert_eq!(
            peak_sharpness((0, 0), (3, 3), corner),
            Some((90.0 - 30.0) / 90.0)
        );
        // Non-positive peak: undefined.
        assert_eq!(peak_sharpness((1, 1), (3, 3), |_, _| -1.0), None);
    }

    #[test]
    fn test_sharpness_gate_rejects_hot_pixel() {
        // A real star plus a single hot pixel. The matched filter smears the
        // hot pixel into a blob that passes `min_pixels`, but its sharpness
        // on the *unfiltered* image (~1.0) trips the gate; the star (~0.5)
        // survives. With the gate disabled, both are detected.
        let (width, height) = (64u32, 64u32);
        let mut pixels = render_stars(width, height, 10.0, 0.0, 2.0, 1.5, &[(20.0, 20.0, 800.0)]);
        pixels[44 * 64 + 44] += 1200.0;

        let base = CentroidExtractionConfig {
            sigma_threshold: 4.0,
            min_pixels: 3,
            matched_filter_sigma: Some(1.5),
            local_bg_block_size: None,
            max_sharpness: Some(0.9),
            ..Default::default()
        };
        let gated = extract_centroids_from_raw(&pixels, width, height, &base).unwrap();
        assert_eq!(
            gated.centroids.len(),
            1,
            "hot pixel should be rejected by the sharpness gate"
        );
        assert!((gated.centroids[0].x - (20.0 - 31.5)).abs() < 1.0);

        let ungated = CentroidExtractionConfig {
            max_sharpness: None,
            ..base
        };
        let all = extract_centroids_from_raw(&pixels, width, height, &ungated).unwrap();
        assert_eq!(
            all.centroids.len(),
            2,
            "gate disabled: hot pixel should be detected"
        );
    }

    #[test]
    fn test_fast_path_sharpness_gate() {
        // Single hot pixel with min_pixels = 1: only the sharpness gate can
        // reject it on the fast path.
        let (width, height) = (64u32, 64u32);
        let mut pixels = render_stars(width, height, 10.0, 0.0, 2.0, 1.5, &[(20.0, 20.0, 800.0)]);
        pixels[44 * 64 + 44] += 1200.0;

        let base = FastCentroidConfig {
            sigma_threshold: 4.0,
            min_pixels: 1,
            max_sharpness: Some(0.9),
            ..Default::default()
        };
        let gated = extract_centroids_fast(&pixels, width, height, &base).unwrap();
        assert_eq!(gated.centroids.len(), 1, "hot pixel should be rejected");

        let ungated = FastCentroidConfig {
            max_sharpness: None,
            ..base
        };
        let all = extract_centroids_fast(&pixels, width, height, &ungated).unwrap();
        assert_eq!(all.centroids.len(), 2, "gate disabled: hot pixel detected");
    }

    #[test]
    fn test_saturation_guard_keeps_com() {
        // A clipped (flat-top) star: with `saturation_level` set the parabola
        // refinement is skipped and the CoM position is kept. The symmetric
        // clipped PSF still centroids onto the true position.
        let (width, height) = (64u32, 64u32);
        let raw = render_stars(width, height, 10.0, 0.0, 1.0, 2.0, &[(30.0, 33.0, 20000.0)]);
        let clipped: Vec<f32> = raw.iter().map(|&v| v.min(1000.0)).collect();

        let config = CentroidExtractionConfig {
            sigma_threshold: 4.0,
            saturation_level: Some(1000.0),
            local_bg_block_size: None,
            ..Default::default()
        };
        let res = extract_centroids_from_raw(&clipped, width, height, &config).unwrap();
        assert_eq!(res.centroids.len(), 1);
        let c = &res.centroids[0];
        assert!(
            (c.x - (30.0 - 31.5)).abs() < 0.3 && (c.y - (33.0 - 31.5)).abs() < 0.3,
            "saturated star CoM off: ({}, {})",
            c.x,
            c.y
        );
    }

    /// A `+inf` or `NaN` pixel must not become a centroid on either path: it
    /// used to flow through `residual.max(0.0)` in the CCL path and emerge as
    /// a `(NaN, NaN)` centroid with infinite mass, ranked brightest.
    #[test]
    fn test_non_finite_pixels_do_not_become_centroids() {
        let (w, h) = (256u32, 256u32);
        let stars = [
            (60.0, 70.0, 3000.0),
            (180.0, 50.0, 2500.0),
            (120.0, 200.0, 2000.0),
            (210.0, 190.0, 1500.0),
        ];
        let clean = render_stars(w, h, 100.0, 0.0, 2.0, 1.5, &stars);
        let mut dirty = clean.clone();
        dirty[10 * w as usize + 10] = f32::INFINITY;
        dirty[60 * w as usize + 100] = f32::NAN;
        dirty[150 * w as usize + 30] = f32::NEG_INFINITY;

        let check =
            |name: &str, clean: &CentroidExtractionResult, dirty: &CentroidExtractionResult| {
                assert_eq!(
                    clean.centroids.len(),
                    stars.len(),
                    "{name}: baseline star count"
                );
                assert_eq!(
                    dirty.centroids.len(),
                    clean.centroids.len(),
                    "{name}: non-finite pixels changed the star count"
                );
                for c in &dirty.centroids {
                    assert!(
                        c.x.is_finite() && c.y.is_finite() && c.mass.is_some_and(f32::is_finite),
                        "{name}: non-finite centroid {c:?}"
                    );
                }
            };

        let ccl_cfg = CentroidExtractionConfig::default();
        check(
            "ccl",
            &extract_centroids_from_raw(&clean, w, h, &ccl_cfg).unwrap(),
            &extract_centroids_from_raw(&dirty, w, h, &ccl_cfg).unwrap(),
        );
        let global_cfg = CentroidExtractionConfig {
            local_bg_block_size: None,
            ..Default::default()
        };
        check(
            "ccl-global-bg",
            &extract_centroids_from_raw(&clean, w, h, &global_cfg).unwrap(),
            &extract_centroids_from_raw(&dirty, w, h, &global_cfg).unwrap(),
        );
        let fast_cfg = FastCentroidConfig::default();
        check(
            "fast",
            &extract_centroids_fast(&clean, w, h, &fast_cfg).unwrap(),
            &extract_centroids_fast(&dirty, w, h, &fast_cfg).unwrap(),
        );
    }

    /// Helper: render Gaussian stars on a background with an optional gradient
    /// and deterministic (seedless) per-pixel noise of amplitude `noise`.
    fn render_stars(
        width: u32,
        height: u32,
        bg: f32,
        gradient: f32,
        noise: f32,
        sigma_px: f32,
        stars: &[(f32, f32, f32)],
    ) -> Vec<f32> {
        let (w, h) = (width as usize, height as usize);
        let mut pixels = vec![0.0_f32; w * h];
        for row in 0..h {
            for col in 0..w {
                // Large-scale gradient the coarse-grid background must reject,
                // plus deterministic hash noise (splitmix64 finalizer). A
                // proper hash matters: the multiplicative Weyl sequence this
                // helper once used is an arithmetic progression mod 1, whose
                // subsequences under any strided sampling are grossly
                // non-uniform — unlike real sensor noise.
                let mut z = (row * w + col) as u64 ^ 0x9e37_79b9_7f4a_7c15;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                z ^= z >> 31;
                let dither = (z >> 40) as f32 / 16_777_216.0 - 0.5;
                pixels[row * w + col] = bg + gradient * (col as f32 / w as f32) + noise * dither;
            }
        }
        for &(sx, sy, brightness) in stars {
            for row in 0..h {
                for col in 0..w {
                    let dx = col as f32 - sx;
                    let dy = row as f32 - sy;
                    let r2 = dx * dx + dy * dy;
                    pixels[row * w + col] += brightness * (-r2 / (2.0 * sigma_px * sigma_px)).exp();
                }
            }
        }
        pixels
    }

    #[test]
    fn test_fast_extract_recovers_stars_over_gradient() {
        let (width, height) = (128u32, 128u32);
        let sigma_px = 1.6_f32;
        // Sub-pixel true positions; a strong left-to-right gradient the
        // coarse-grid background must track, plus realistic noise.
        let stars = [
            (30.3, 30.0, 900.0),
            (90.0, 50.7, 1300.0),
            (60.5, 100.2, 600.0),
        ];
        let pixels = render_stars(width, height, 50.0, 400.0, 8.0, sigma_px, &stars);

        let config = FastCentroidConfig {
            sigma_threshold: 5.0,
            bg_grid: 32,
            ..Default::default()
        };
        let result = extract_centroids_fast(&pixels, width, height, &config).unwrap();
        assert_eq!(
            result.centroids.len(),
            3,
            "expected 3 stars, got {}",
            result.centroids.len()
        );
        // Brightest-first ordering.
        assert!(result.centroids[0].mass.unwrap() >= result.centroids[1].mass.unwrap());

        // Each true star must have a detection within ~0.6 px. The single-pass
        // path is a ~0.5-px-class centroider by design (threshold-clipped CoM +
        // parabola refine) — plenty for solving, not for tight astrometry.
        let cx = (width - 1) as f32 / 2.0;
        let cy = (height - 1) as f32 / 2.0;
        for &(sx, sy, _) in &stars {
            let (tx, ty) = (sx - cx, sy - cy);
            let best = result
                .centroids
                .iter()
                .map(|c| ((c.x - tx).powi(2) + (c.y - ty).powi(2)).sqrt())
                .fold(f32::INFINITY, f32::min);
            assert!(
                best < 0.6,
                "star ({sx}, {sy}) nearest detection {best:.3} px away"
            );
        }
    }

    #[test]
    fn test_fast_extract_merges_touching_pixels_and_caps() {
        let (width, height) = (128u32, 128u32);
        // Two stars 1 px apart form one connected region (correct for a blended
        // pair); a far star is its own region → 2 total.
        let stars = [
            (64.0, 64.0, 1000.0),
            (65.0, 64.0, 950.0),
            (20.0, 20.0, 800.0),
        ];
        let pixels = render_stars(width, height, 30.0, 0.0, 6.0, 1.5, &stars);

        let config = FastCentroidConfig {
            sigma_threshold: 5.0,
            max_centroids: Some(5),
            ..Default::default()
        };
        let result = extract_centroids_fast(&pixels, width, height, &config).unwrap();
        assert_eq!(
            result.centroids.len(),
            2,
            "blended pair should merge to 1 + 1 separate = 2, got {}",
            result.centroids.len()
        );
    }

    #[test]
    fn test_fast_extract_rejects_bad_params() {
        let pixels = vec![0.0_f32; 64 * 64];
        let bad_sigma = FastCentroidConfig {
            sigma_threshold: 0.0,
            ..Default::default()
        };
        assert!(extract_centroids_fast(&pixels, 64, 64, &bad_sigma).is_err());
        let bad_grid = FastCentroidConfig {
            bg_grid: 0,
            ..Default::default()
        };
        assert!(extract_centroids_fast(&pixels, 64, 64, &bad_grid).is_err());
        // Length mismatch.
        assert!(extract_centroids_fast(&pixels, 64, 63, &FastCentroidConfig::default()).is_err());
    }

    #[test]
    fn test_extract_from_raw_multiple_stars() {
        let width = 128u32;
        let height = 128u32;
        let mut pixels = vec![10.0_f32; (width * height) as usize];

        // Place 3 stars at different positions
        let stars = [
            (30.0, 30.0, 800.0),
            (90.0, 50.0, 1200.0),
            (60.0, 100.0, 500.0),
        ];
        let sigma_px = 2.0_f32;

        for &(sx, sy, brightness) in &stars {
            for row in 0..height {
                for col in 0..width {
                    let dx = col as f32 - sx;
                    let dy = row as f32 - sy;
                    let r2 = dx * dx + dy * dy;
                    pixels[(row * width + col) as usize] +=
                        brightness * (-r2 / (2.0 * sigma_px * sigma_px)).exp();
                }
            }
        }

        let config = CentroidExtractionConfig {
            sigma_threshold: 3.0,
            min_pixels: 2,
            ..Default::default()
        };

        let result = extract_centroids_from_raw(&pixels, width, height, &config).unwrap();
        assert_eq!(
            result.centroids.len(),
            3,
            "Expected 3 stars, got {}",
            result.centroids.len()
        );

        // Centroids should be sorted by brightness (descending)
        assert!(result.centroids[0].mass.unwrap() >= result.centroids[1].mass.unwrap());
        assert!(result.centroids[1].mass.unwrap() >= result.centroids[2].mass.unwrap());
    }

    #[test]
    fn test_max_centroids_limit() {
        let width = 128u32;
        let height = 128u32;
        let mut pixels = vec![10.0_f32; (width * height) as usize];

        let stars = [
            (30.0, 30.0, 800.0),
            (90.0, 50.0, 1200.0),
            (60.0, 100.0, 500.0),
        ];
        let sigma_px = 2.0_f32;

        for &(sx, sy, brightness) in &stars {
            for row in 0..height {
                for col in 0..width {
                    let dx = col as f32 - sx;
                    let dy = row as f32 - sy;
                    let r2 = dx * dx + dy * dy;
                    pixels[(row * width + col) as usize] +=
                        brightness * (-r2 / (2.0 * sigma_px * sigma_px)).exp();
                }
            }
        }

        let config = CentroidExtractionConfig {
            sigma_threshold: 3.0,
            min_pixels: 2,
            max_centroids: Some(2),
            ..Default::default()
        };

        let result = extract_centroids_from_raw(&pixels, width, height, &config).unwrap();
        assert_eq!(result.centroids.len(), 2);
    }

    #[test]
    fn test_quadratic_refinement() {
        // Place a Gaussian star at a known sub-pixel offset on uniform background
        let width = 64u32;
        let height = 64u32;
        let bg = 100.0_f32;
        let true_x = 32.3_f32;
        let true_y = 32.7_f32;
        let sigma_px = 2.0_f32;
        let peak_brightness = 2000.0_f32;

        let mut pixels = vec![bg; (width * height) as usize];
        for row in 0..height {
            for col in 0..width {
                let dx = col as f32 - true_x;
                let dy = row as f32 - true_y;
                let r2 = dx * dx + dy * dy;
                pixels[(row * width + col) as usize] +=
                    peak_brightness * (-r2 / (2.0 * sigma_px * sigma_px)).exp();
            }
        }

        let config = CentroidExtractionConfig {
            sigma_threshold: 3.0,
            min_pixels: 3,
            ..Default::default()
        };

        let result = extract_centroids_from_raw(&pixels, width, height, &config).unwrap();
        assert_eq!(
            result.centroids.len(),
            1,
            "Expected 1 star, got {}",
            result.centroids.len()
        );

        // Centroid is in centered coords (origin at image center)
        let c = &result.centroids[0];
        let cx = (width - 1) as f32 / 2.0;
        let cy = (height - 1) as f32 / 2.0;
        let abs_x = c.x + cx;
        let abs_y = c.y + cy;

        let err_x = (abs_x - true_x).abs();
        let err_y = (abs_y - true_y).abs();
        assert!(
            err_x < 0.15,
            "X error too large: centroid={abs_x:.4}, true={true_x}, err={err_x:.4}"
        );
        assert!(
            err_y < 0.15,
            "Y error too large: centroid={abs_y:.4}, true={true_y}, err={err_y:.4}"
        );
    }

    #[test]
    fn test_quadratic_refinement_with_gradient_background() {
        // Place a star on a gradient background to test local background correction
        let width = 128u32;
        let height = 128u32;
        let true_x = 64.4_f32;
        let true_y = 64.6_f32;
        let sigma_px = 2.0_f32;
        let peak_brightness = 2000.0_f32;

        let mut pixels = vec![0.0_f32; (width * height) as usize];
        // Add a gradient background: increases from left to right (50 to 150)
        for row in 0..height {
            for col in 0..width {
                let bg = 50.0 + 100.0 * (col as f32 / width as f32);
                pixels[(row * width + col) as usize] = bg;
            }
        }
        // Add Gaussian star
        for row in 0..height {
            for col in 0..width {
                let dx = col as f32 - true_x;
                let dy = row as f32 - true_y;
                let r2 = dx * dx + dy * dy;
                pixels[(row * width + col) as usize] +=
                    peak_brightness * (-r2 / (2.0 * sigma_px * sigma_px)).exp();
            }
        }

        let config = CentroidExtractionConfig {
            sigma_threshold: 5.0,
            min_pixels: 3,
            ..Default::default()
        };

        let result = extract_centroids_from_raw(&pixels, width, height, &config).unwrap();
        assert!(
            !result.centroids.is_empty(),
            "Should detect at least one star on gradient background"
        );

        // Find the centroid closest to our true position
        let cx = (width - 1) as f32 / 2.0;
        let cy = (height - 1) as f32 / 2.0;
        let best = result
            .centroids
            .iter()
            .min_by(|a, b| {
                let da = (a.x + cx - true_x).powi(2) + (a.y + cy - true_y).powi(2);
                let db = (b.x + cx - true_x).powi(2) + (b.y + cy - true_y).powi(2);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();

        let abs_x = best.x + cx;
        let abs_y = best.y + cy;
        let err_x = (abs_x - true_x).abs();
        let err_y = (abs_y - true_y).abs();
        assert!(
            err_x < 0.3,
            "X error too large on gradient bg: centroid={abs_x:.4}, true={true_x}, err={err_x:.4}"
        );
        assert!(
            err_y < 0.3,
            "Y error too large on gradient bg: centroid={abs_y:.4}, true={true_y}, err={err_y:.4}"
        );
    }
}
