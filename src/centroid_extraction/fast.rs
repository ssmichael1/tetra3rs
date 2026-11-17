//! The fast preset of the extraction pipeline: [`extract_centroids_fast`]
//! and its small [`FastCentroidConfig`]. It runs the same code as
//! [`extract_centroids_from_raw`](super::extract_centroids_from_raw) with the
//! matched filter and the per-blob annulus background switched off and a
//! coarser background subsample.

use super::{
    ccl, check_pixel_len, CentroidExtractionConfig, CentroidExtractionResult, DeblendMode,
};
use crate::error::{Error, Result};

// ─── Fast single-pass star-tracker path ─────────────────────────────────────

/// Configuration for [`extract_centroids_fast`].
///
/// Deliberately small — the knobs a tracker-style detector needs. The
/// matched filter, the per-blob annulus background and deblending of the
/// full [`CentroidExtractionConfig`] do not appear, because this preset
/// switches them off for speed.
#[derive(Debug, Clone)]
pub struct FastCentroidConfig {
    /// Detection threshold in noise sigmas above the local background. A pixel
    /// is "lit" when `value > bg(x, y) + sigma_threshold · σ`. Default: 5.0
    pub sigma_threshold: f32,

    /// Background-grid block size in pixels. The background is the median
    /// of a subsample of each `bg_grid`-sized block, bilinearly interpolated
    /// between block centers, so gradients (vignetting, Milky Way) are
    /// followed. Larger blocks are cheaper but follow gradients more coarsely.
    /// Default: 64
    pub bg_grid: u32,

    /// Minimum pixels in a lit region to count as a star — rejects single hot
    /// pixels and cosmic-ray specks. Default: 2
    pub min_pixels: usize,

    /// Maximum number of centroids to return, brightest first. `None` returns
    /// all detections. For plate solving / tracking a few dozen is plenty.
    /// Default: None
    pub max_centroids: Option<usize>,

    /// Maximum DAOFIND-style sharpness: `(peak − mean(8 neighbors)) / peak`,
    /// measured on the background-subtracted image at the region's peak.
    /// Values near 1 mean single-pixel flux — a hot pixel or cosmic-ray hit.
    /// A critically sampled PSF scores ~0.5; a strongly undersampled one up
    /// to ~0.85. The default 0.9 passes any system whose PSF spans multiple
    /// pixels; set `None` for severely undersampled data (PSF FWHM below
    /// ~1.5 px), where real stars are indistinguishable from hot pixels.
    /// Default: Some(0.9)
    pub max_sharpness: Option<f32>,

    /// Pixel value at or above which the sensor is considered saturated.
    /// A region whose peak reaches this level skips the 3×3 parabola
    /// refinement (a flat-topped profile has no meaningful sub-pixel
    /// maximum), keeping the center-of-mass position.
    /// Default: None (disabled)
    pub saturation_level: Option<f32>,

    /// Maximum pixels in a region. Results are brightest-first, so without a
    /// cap a satellite trail, aircraft streak, or horizon glow becomes the
    /// *top* centroid handed to the solver. Real stars never approach the
    /// default; only raise it for deliberately defocused optics.
    /// Default: 10000
    pub max_pixels: usize,

    /// Maximum elongation ratio (major/minor axis from intensity-weighted
    /// second moments) — rejects streaks and trails too small for
    /// `max_pixels`. Off by default: moment-based elongation is noisy for
    /// regions of only a few pixels (this preset's `min_pixels` default is 2),
    /// so enable it (e.g. 3.0-5.0) when trails are expected and `min_pixels`
    /// is raised enough (≳5) for the moments to be meaningful.
    /// Default: None (disabled)
    pub max_elongation: Option<f32>,

    /// Drop regions whose bounding box comes within this many pixels of an
    /// image edge (truncated PSFs bias the center-of-mass inward).
    /// Default: 0 (disabled)
    pub border_margin: u32,
}

impl Default for FastCentroidConfig {
    fn default() -> Self {
        Self {
            sigma_threshold: 5.0,
            bg_grid: 64,
            min_pixels: 2,
            max_centroids: None,
            max_sharpness: Some(0.9),
            saturation_level: None,
            max_pixels: 10000,
            max_elongation: None,
            border_margin: 0,
        }
    }
}

/// Fast centroid extraction — the "adequate star tracker" preset.
///
/// The same pipeline as [`extract_centroids_from_raw`](super::extract_centroids_from_raw)
/// (block-median background surface, noise estimate against that surface,
/// thresholding into a bit mask, run-length connected regions,
/// intensity-weighted moments, 3×3 peak refinement) configured for latency:
///
/// - no matched filter, so faint-star sensitivity is lower — sized for the
///   stars a tracker locks onto, not deep detection;
/// - no per-blob annulus background: each blob is measured against the
///   block-grid surface alone, which is what dominates the cost on
///   region-dense frames;
/// - the background grid samples every `bg_grid / 8`-th pixel instead of
///   every `/16`-th, and the noise estimate is capped at ~131k samples.
///
/// On sparse 2048² frames this is about 2-3× faster than the default
/// configuration of the full function. With the `parallel` feature the
/// background, mask, sweep and region stages run multi-threaded; results
/// are bit-identical either way.
///
/// `sigma_threshold` means the same thing as in the full pipeline: sigmas
/// of the noise measured against the interpolated background.
///
/// Returns the same [`CentroidExtractionResult`] (centroids in
/// image-center-origin coordinates, brightest first), so it is a drop-in for
/// [`SolverDatabase::solve_from_centroids`](crate::SolverDatabase::solve_from_centroids).
/// A frame loop should use
/// [`CentroidExtractor::extract_fast_from_raw`](super::CentroidExtractor::extract_fast_from_raw),
/// which keeps the working buffers between calls.
pub fn extract_centroids_fast(
    pixels: &[f32],
    width: u32,
    height: u32,
    config: &FastCentroidConfig,
) -> Result<CentroidExtractionResult> {
    extract_fast(pixels, width, height, config, &mut ccl::Scratch::new())
}

/// [`extract_centroids_fast`] on caller-owned working buffers (see
/// [`CentroidExtractor::extract_fast_from_raw`](super::CentroidExtractor::extract_fast_from_raw)).
pub(super) fn extract_fast(
    pixels: &[f32],
    width: u32,
    height: u32,
    config: &FastCentroidConfig,
    scratch: &mut ccl::Scratch,
) -> Result<CentroidExtractionResult> {
    check_pixel_len(pixels.len(), width, height)?;
    if config.bg_grid == 0 {
        return Err(Error::InvalidInput("bg_grid must be >= 1".into()));
    }
    let defaults = CentroidExtractionConfig::default();
    let full = CentroidExtractionConfig {
        sigma_threshold: config.sigma_threshold,
        min_pixels: config.min_pixels,
        max_pixels: config.max_pixels,
        max_centroids: config.max_centroids,
        local_bg_block_size: Some(config.bg_grid),
        max_elongation: config.max_elongation,
        matched_filter_sigma: None,
        max_sharpness: config.max_sharpness,
        saturation_level: config.saturation_level,
        deblend: DeblendMode::Off,
        border_margin: config.border_margin,
        sigma_clip_iterations: defaults.sigma_clip_iterations,
        sigma_clip_factor: defaults.sigma_clip_factor,
    };
    ccl::extract_from_gray(pixels, width, height, &full, ccl::Tuning::FAST, scratch)
}
