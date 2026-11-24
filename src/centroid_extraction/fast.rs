//! Fast single-pass star-tracker extraction path: coarse subsampled
//! background grid + one thresholding pass into a packed bit mask + a
//! run-length sweep over the mask, with moments computed from the run lists.
//! Split out of the crate-facing module; entry via [`extract_centroids_fast`].

use super::{
    check_pixel_len, elongation_from_cov, finish_region, par, runs, sort_and_truncate_by_mass,
    BackgroundGrid, CentroidExtractionResult,
};
use crate::centroid::Centroid;
use crate::error::{Error, Result};
use crate::prelude::*;

// ─── Fast single-pass star-tracker path ─────────────────────────────────────

/// Configuration for [`extract_centroids_fast`].
///
/// Deliberately small — the four knobs a single-pass detector needs. None of
/// the connected-component path's quality filters (block-interpolated
/// background, elongation, per-blob annulus background, sub-pixel agreement
/// gates) appear, because this path trades them away for speed.
#[derive(Debug, Clone)]
pub struct FastCentroidConfig {
    /// Detection threshold in noise sigmas above the local background. A pixel
    /// is "lit" when `value > bg(x, y) + sigma_threshold · σ`. Default: 5.0
    pub sigma_threshold: f32,

    /// Coarse background-grid block size in pixels. The background is estimated
    /// once on a `bg_grid`-spaced grid (from a subsampled pre-pass, ~1/64 the
    /// pixels) and bilinearly interpolated during the main sweep, so gradients
    /// (vignetting, Milky Way) are handled without a full-image background
    /// stage. Larger blocks are cheaper but follow gradients more coarsely.
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
    /// regions of only a few pixels (this path's `min_pixels` default is 2),
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

/// Fast single-pass centroid extraction — the "adequate star tracker" path.
///
/// An alternative to [`extract_centroids_from_raw`](super::extract_centroids_from_raw) that reads each pixel
/// **once**: a cheap subsampled pre-pass builds a coarse background grid, then
/// a single pass thresholds every row against the bilinearly-interpolated
/// background into a packed bit mask (1 bit per pixel), a run-length +
/// union-find sweep over the mask groups lit pixels into connected regions
/// (work proportional to runs, not pixels), and intensity-weighted moments
/// over the run lists give a center-of-mass per region. No convolution, no
/// full-image background buffer, no second pass over the image — so it runs
/// close to memory bandwidth and markedly faster than the connected-component
/// path (which stays the default and the right choice for calibration /
/// faint-star work). With the `parallel` feature the background grid and the
/// bit mask are built multi-threaded; results are bit-identical either way.
///
/// # Trade-offs
///
/// - No matched filter, so faint-star sensitivity is lower — sized for the
///   brightest stars a tracker locks onto, not deep detection.
/// - Center-of-mass is threshold-clipped: sub-pixel accuracy is ~0.1 px for
///   bright stars, degrading for faint ones. A 3×3 parabola refine on the peak
///   (`quadratic_peak_offset`) sharpens it when the region is large enough
///   and the fit agrees with the CoM, matching the CCL path's gate.
/// - A global noise σ is used (adequate when read/shot noise is roughly
///   uniform even where the background level is not).
///
/// Returns the same [`CentroidExtractionResult`] as the CCL path (centroids in
/// image-center-origin coordinates, brightest first), so it is a drop-in for
/// [`SolverDatabase::solve_from_centroids`](crate::SolverDatabase::solve_from_centroids).
/// `background_mean` is the median of the coarse grid, `background_sigma` the
/// global noise σ, `threshold` a representative `bg_mean + k·σ`, and
/// `num_blobs_raw` the region count before the `min_pixels` filter.
pub fn extract_centroids_fast(
    pixels: &[f32],
    width: u32,
    height: u32,
    config: &FastCentroidConfig,
) -> Result<CentroidExtractionResult> {
    let w = width as usize;
    let h = height as usize;
    check_pixel_len(pixels.len(), width, height)?;
    if !(config.sigma_threshold.is_finite() && config.sigma_threshold > 0.0) {
        return Err(Error::InvalidInput(format!(
            "sigma_threshold must be finite and positive, got {}",
            config.sigma_threshold
        )));
    }
    if config.bg_grid == 0 {
        return Err(Error::InvalidInput("bg_grid must be >= 1".into()));
    }
    if w < 2 || h < 2 {
        return Err(Error::InvalidInput("image must be at least 2x2".into()));
    }

    // ── Pre-pass: coarse background grid + global noise σ (subsampled) ──
    let block = config.bg_grid as usize;
    let (bg, sigma) = BackgroundGrid::build(pixels, w, h, block, (block / 8).max(1));
    let nx = w.div_ceil(block);
    let k = config.sigma_threshold;
    let k_sigma = k * sigma;

    // ── Threshold pass: packed detection bit mask ──
    // Per row: blend the two bracketing grid rows (row-constant half of the
    // interpolation), expand to a per-column threshold through the grid's
    // precomputed column plan, and compare the pixels into one bit each.
    // The whole-image mask (h × ⌈w/64⌉ words; 512 KB at 2048²) is filled in
    // independent 16-row chunks — multi-threaded under `parallel` — and
    // then consumed by one sequential run sweep, so region ids are
    // identical to a pixel-by-pixel raster sweep. Moments are computed
    // afterwards from the run lists: lit pixels are ≪1% of the image, so
    // the second touch of them is nearly free.
    const ROWS_PER_CHUNK: usize = 16;
    let words_per_row = w.div_ceil(64);
    let mut mask = vec![0u64; words_per_row * h];
    par::for_each_chunk_mut(&mut mask, words_per_row * ROWS_PER_CHUNK, |ci, chunk| {
        let mut grid_row = vec![0.0_f32; nx];
        let mut thr = vec![0.0_f32; w];
        for (i, words) in chunk.chunks_exact_mut(words_per_row).enumerate() {
            let r = ci * ROWS_PER_CHUNK + i;
            bg.blend_row(bg.row_params(r), &mut grid_row);
            bg.threshold_row(&grid_row, k_sigma, &mut thr);
            runs::pack_lit_row(&pixels[r * w..(r + 1) * w], &thr, words);
        }
    });

    // ── Run-length sweep over the mask via the shared core ──
    let regions = runs::sweep_runs_mask(w, h, words_per_row, &mask);
    let (offsets, order) = regions.group_by_region();

    // ── Emit one centroid per region ──
    // Origin at the geometric image center (W-1)/2, (H-1)/2 (see the CCL path).
    let cx = (width - 1) as f32 / 2.0;
    let cy = (height - 1) as f32 / 2.0;
    let mut centroids: Vec<Centroid> = Vec::new();
    let num_blobs_raw = regions.n_regions;
    'region: for kreg in 0..regions.n_regions {
        let region_runs = &order[offsets[kreg] as usize..offsets[kreg + 1] as usize];
        let extent = regions.extent(region_runs);
        let npix = extent.npix;
        if npix < config.min_pixels || npix > config.max_pixels {
            continue;
        }
        if !extent.clear_of_border(config.border_margin as usize, w, h) {
            continue;
        }

        // Intensity-weighted moments over the run list, background
        // re-interpolated per run (identical arithmetic to the sweep's).
        let mut sum_w = 0.0_f64;
        let mut sum_wx = 0.0_f64;
        let mut sum_wy = 0.0_f64;
        let mut sum_wxx = 0.0_f64;
        let mut sum_wyy = 0.0_f64;
        let mut sum_wxy = 0.0_f64;
        let mut peak_val = f32::NEG_INFINITY;
        let mut peak_x = 0usize;
        let mut peak_y = 0usize;
        for &i in region_runs {
            let run = regions.runs[i as usize];
            let r = run.row as usize;
            let rp = bg.row_params(r);
            let row_off = r * w;
            for c in run.c0 as usize..=run.c1 as usize {
                let p = pixels[row_off + c];
                let weight = (p - bg.value_at(c, rp)).max(0.0) as f64;
                let (cf, rf) = (c as f64, r as f64);
                sum_w += weight;
                sum_wx += weight * cf;
                sum_wy += weight * rf;
                sum_wxx += weight * cf * cf;
                sum_wyy += weight * rf * rf;
                sum_wxy += weight * cf * rf;
                if p > peak_val {
                    peak_val = p;
                    peak_x = c;
                    peak_y = r;
                }
            }
        }
        if sum_w <= 0.0 {
            continue;
        }

        let fx = sum_wx / sum_w;
        let fy = sum_wy / sum_w;

        // Intensity-weighted central second moments — the same statistic the
        // CCL path reports as `cov` and judges elongation on.
        let cxx = sum_wxx / sum_w - fx * fx;
        let cyy = sum_wyy / sum_w - fy * fy;
        let cxy = sum_wxy / sum_w - fx * fy;
        if let Some(max_elong) = config.max_elongation {
            if elongation_from_cov(cxx, cyy, cxy) > max_elong {
                continue 'region;
            }
        }

        let (pc, pr) = (peak_x, peak_y);
        let peak_bg = bg.value_at(pc, bg.row_params(pr)) as f64;
        let v = |dy: isize, dx: isize| -> f64 {
            let rr = (pr as isize + dy) as usize;
            let cc = (pc as isize + dx) as usize;
            pixels[rr * w + cc] as f64 - peak_bg
        };

        // Sharpness gate, peak refinement, and assembly are shared with the
        // CCL path (see `finish_region`).
        let saturated = config.saturation_level.is_some_and(|s| peak_val >= s);
        if let Some(c) = finish_region(
            npix,
            (pc, pr),
            (w, h),
            (fx, fy),
            (cxx, cyy, cxy),
            sum_w,
            saturated,
            config.max_sharpness,
            (cx, cy),
            v,
        ) {
            centroids.push(c);
        }
    }

    sort_and_truncate_by_mass(&mut centroids, config.max_centroids);

    let bg_mean = bg.level();

    Ok(CentroidExtractionResult {
        centroids,
        image_width: width,
        image_height: height,
        background_mean: bg_mean,
        background_sigma: sigma,
        threshold: bg_mean + k * sigma,
        num_blobs_raw,
    })
}
