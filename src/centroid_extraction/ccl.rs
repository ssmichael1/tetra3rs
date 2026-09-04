//! Default connected-component-labeling extraction path: local background
//! subtraction, sigma-clipped global noise stats, optional matched filter,
//! threshold → CCL → two-pass per-blob moments with annulus background, and
//! sub-pixel peak refinement. Split out of the crate-facing module; entry via
//! [`extract_from_gray`].

use std::borrow::Cow;

use numeris::imageproc::{gaussian_blur_into, BorderMode};
use numeris::DynMatrix;

use super::{
    elongation_from_cov, finish_region, median_f32, par, runs, sort_and_truncate_by_mass,
    BackgroundGrid, CentroidExtractionConfig, CentroidExtractionResult, DeblendMode,
};
use crate::centroid::Centroid;
use crate::error::{Error, Result};

/// Full-image working buffers of the pipeline, owned by
/// [`CentroidExtractor`](super::CentroidExtractor) so consecutive frames of
/// the same size reuse them instead of paying `calloc` + first-touch page
/// faults on ~48 MB per 2048² frame. Every element of every buffer is
/// rewritten before it is read, so stale contents never leak into a result
/// and a fresh `Scratch` gives the same output as a reused one.
pub(super) struct Scratch {
    /// Clamped (≥ 0) background-subtracted measurement image.
    clamped: Vec<f32>,
    /// Unclamped residuals — the matched filter's input (only used when the
    /// filter is on). Lent to a `DynMatrix` for the blur and taken back.
    unclamped: Vec<f32>,
    /// The matched filter's output; `gaussian_blur_into` resizes it itself.
    filtered: DynMatrix<f32>,
    /// 1-bit-per-pixel detection mask.
    mask: Vec<u64>,
}

impl Scratch {
    pub(super) fn new() -> Self {
        Self {
            clamped: Vec::new(),
            unclamped: Vec::new(),
            filtered: DynMatrix::zeros(0, 0),
            mask: Vec::new(),
        }
    }
}

/// Set `v`'s length to `n` without zeroing what is already there (the
/// caller overwrites every element).
fn set_len_uninit<T: Copy + Default>(v: &mut Vec<T>, n: usize) {
    if v.len() != n {
        v.resize(n, T::default());
    }
}

/// Shared extraction pipeline for both image and raw-pixel entry points.
pub(super) fn extract_from_gray(
    gray_input: &[f32],
    width: u32,
    height: u32,
    config: &CentroidExtractionConfig,
    scratch: &mut Scratch,
) -> Result<CentroidExtractionResult> {
    let w = width as usize;
    let h = height as usize;
    let Scratch {
        clamped,
        unclamped,
        filtered,
        mask,
    } = scratch;

    // ── Step 0: validate geometry and config ──
    // The pipeline below indexes `width - 1` and chunks the image into rows of
    // width `w`, both of which panic on a degenerate image; and a zero
    // `local_bg_block_size` divides by zero in the background estimator. Reject
    // these up front (the fast path guards the same cases).
    if w < 2 || h < 2 {
        return Err(Error::InvalidInput(format!(
            "image must be at least 2x2, got {width}x{height}"
        )));
    }
    if config.local_bg_block_size == Some(0) {
        return Err(Error::InvalidInput(
            "local_bg_block_size must be >= 1 (or None)".into(),
        ));
    }
    if !config.sigma_threshold.is_finite() {
        return Err(Error::InvalidInput(format!(
            "sigma_threshold must be finite, got {}",
            config.sigma_threshold
        )));
    }

    // The Gaussian kernel spans 2·⌈3σ⌉+1 taps, so an absurd σ is a
    // multi-gigabyte allocation / effective hang, not a blur. 64 px covers
    // any plausible star PSF with two orders of magnitude to spare.
    const MAX_MATCHED_FILTER_SIGMA: f32 = 64.0;
    if let Some(s) = config.matched_filter_sigma {
        if s.is_finite() && s > MAX_MATCHED_FILTER_SIGMA {
            return Err(Error::InvalidInput(format!(
                "matched_filter_sigma must be <= {MAX_MATCHED_FILTER_SIGMA} px, got {s}"
            )));
        }
    }
    let filter_sigma = config
        .matched_filter_sigma
        .filter(|s| s.is_finite() && *s > 0.0);

    // ── Steps 1-2: background model, residuals, and noise stats ──
    // With `local_bg_block_size` set, a block-median grid is built (from a
    // staggered subsample) and everything downstream works from residuals
    // against its bilinear surface. The residuals are produced by ONE fused
    // pass over the image that interpolates the surface on the fly and
    // writes the clamped measurement image and — only when the matched
    // filter is on — the unclamped filter input directly into the blur's
    // matrix. (Blurring the clamped image would rectify negative noise into
    // a positive DC offset; measuring on the unclamped one would let
    // negative pixels cancel star flux.) The materialized full-image
    // background buffer and its separate subtract passes are gone.
    //
    // Noise statistics use the same estimator either way; on the local-bg
    // path it runs on bilinear residuals at the block subsample lattice
    // (identical reference surface, ~stride² fewer samples) instead of a
    // full-image residual buffer.
    let gray: Cow<[f32]>;
    let bg_mean: f32;
    let bg_sigma: f32;
    let mut filter_input: Option<DynMatrix<f32>> = None;

    if let Some(block_size) = config.local_bg_block_size {
        let bs = block_size as usize;
        // Stride bs/16 (vs the fast path's bs/8) keeps extra sampling margin
        // on this calibration-quality path; the build-time σ is ignored in
        // favor of the estimator below, run against the bilinear surface.
        let (bg, _) = BackgroundGrid::build(gray_input, w, h, bs, (bs / 16).max(1));

        let residuals = subsample_residuals(gray_input, w, h, &bg);
        (bg_mean, bg_sigma) = estimate_background(&residuals, width, height, config);

        // Fused residual pass (rows in parallel under the `parallel` feature;
        // each row writes disjoint output, results independent of threads).
        // The bilinear surface is evaluated per row as `blend_row` +
        // `blend_columns` — the same expression and operation order as
        // `value_at`, with the per-pixel divide / floor / clamp hoisted into
        // the grid's column plan — so the residuals are bit-identical to the
        // per-pixel form.
        let nx = bg.grid_width();
        set_len_uninit(clamped, w * h);
        if filter_sigma.is_some() {
            set_len_uninit(unclamped, w * h);
            par::for_each_chunk_pair_mut(clamped, unclamped, w, |y, cr, ur| {
                let mut row_blend = vec![0.0_f32; nx];
                bg.blend_row(bg.row_params(y), &mut row_blend);
                // Surface into `ur`, then residuals in place.
                bg.blend_columns(&row_blend, ur, |v| v);
                let src = &gray_input[y * w..(y + 1) * w];
                for ((c, u), &p) in cr.iter_mut().zip(ur.iter_mut()).zip(src) {
                    // Non-finite pixels (dead/hot columns, NaN padding) are
                    // treated as background: `inf.max(0.0)` is `inf`, and one
                    // such pixel otherwise becomes a NaN centroid ranked first.
                    let r = if p.is_finite() { p - *u } else { 0.0 };
                    *c = r.max(0.0);
                    *u = r;
                }
            });
            filter_input = Some(DynMatrix::from_vec(w, h, std::mem::take(unclamped)));
        } else {
            par::for_each_chunk_mut(clamped, w, |y, cr| {
                let mut row_blend = vec![0.0_f32; nx];
                bg.blend_row(bg.row_params(y), &mut row_blend);
                bg.blend_columns(&row_blend, cr, |v| v);
                let src = &gray_input[y * w..(y + 1) * w];
                for (c, &p) in cr.iter_mut().zip(src) {
                    *c = if p.is_finite() {
                        (p - *c).max(0.0)
                    } else {
                        0.0
                    };
                }
            });
        }
        gray = Cow::Borrowed(clamped);
    } else {
        (bg_mean, bg_sigma) = estimate_background(gray_input, width, height, config);
        // Same non-finite policy as the local-background branch; the copy is
        // only made when the image actually contains such pixels.
        gray = if gray_input.iter().all(|p| p.is_finite()) {
            Cow::Borrowed(gray_input)
        } else {
            Cow::Owned(
                gray_input
                    .iter()
                    .map(|&p| if p.is_finite() { p } else { bg_mean })
                    .collect(),
            )
        };
        if filter_sigma.is_some() {
            set_len_uninit(unclamped, w * h);
            unclamped.copy_from_slice(&gray);
            filter_input = Some(DynMatrix::from_vec(w, h, std::mem::take(unclamped)));
        }
    }
    let gray: &[f32] = &gray;

    // ── Step 3: optional matched filter for thresholding only ──
    // The unclamped residual is convolved with a Gaussian and threshold/CCL
    // run on the filtered copy; centroids are still measured on the
    // unfiltered `gray`, so intensities and CoM positions are unaffected.
    // The detection threshold is scaled by the kernel's white-noise
    // suppression factor so `sigma_threshold` keeps meaning "sigmas of the
    // noise actually present in the thresholded image", filter on or off.
    // `gaussian_blur_into` (numeris ≥ 0.5.19) runs the separable passes in
    // column bands with a per-band halo scratch instead of materializing a
    // full-image intermediate — bit-identical to `gaussian_blur`, one 16 MB
    // buffer less at 2048². Under the `parallel` feature the bands run
    // multi-threaded.
    let (thresh_src, mask_threshold): (&[f32], f32) = match (filter_sigma, filter_input) {
        (Some(sigma), Some(mat)) => {
            gaussian_blur_into(&mat, sigma, BorderMode::Replicate, filtered);
            // Hand the filter input back to the scratch for the next frame.
            *unclamped = mat.into_vec();
            let suppression = gaussian_noise_suppression(sigma);
            (
                filtered.as_slice(),
                bg_mean + config.sigma_threshold * bg_sigma * suppression,
            )
        }
        _ => (gray, bg_mean + config.sigma_threshold * bg_sigma),
    };

    // ── Step 4: threshold into a bit mask, sweep into runs and regions ──
    // The thresholded image is packed one row at a time into a 1-bit-per-
    // pixel mask (`thresh_src[i] > mask_threshold`), and the run-length
    // union-find core reads runs straight off the words, so the sweep costs
    // runs rather than pixels. Downstream stages iterate each region's run
    // list; the annulus's "not in any blob" test is a bit test on the same
    // mask (equivalent: every lit pixel was in some region). 8-connectivity
    // is inherent to the run merging. Nothing downstream reads the filtered
    // image.
    let words_per_row = threshold_to_mask(thresh_src, w, h, mask_threshold, mask);
    let mask: &[u64] = mask;
    // Under `parallel` the rows are labeled in 64-row bands (one task each)
    // and stitched — the same `RunRegions` as the sequential sweep.
    #[cfg(feature = "parallel")]
    let regions = runs::sweep_runs_mask_banded(w, h, words_per_row, mask, 64);
    #[cfg(not(feature = "parallel"))]
    let regions = runs::sweep_runs_mask(w, h, words_per_row, mask);

    // ── Step 5: compute centroids ──
    // Origin at the geometric image center, (W-1)/2 and (H-1)/2 (pixel centers
    // are at integer indices, so for even dimensions this is the intersection
    // of the four central pixels — matching the FITS / astropy / OpenCV
    // convention). +X right, +Y down.
    let cx = (width - 1) as f32 / 2.0;
    let cy = (height - 1) as f32 / 2.0;
    let mut centroids = compute_blob_centroids(
        gray,
        gray_input,
        (mask, words_per_row),
        &regions,
        (w, h),
        (cx, cy),
        config,
    );
    // "Raw" blob count = connected regions before the size/elongation/mass
    // filters, matching the field's documented meaning and the fast path's
    // pre-`min_pixels` region count.
    let num_blobs_raw = regions.n_regions;

    sort_and_truncate_by_mass(&mut centroids, config.max_centroids);

    Ok(CentroidExtractionResult {
        centroids,
        image_width: width,
        image_height: height,
        background_mean: bg_mean,
        background_sigma: bg_sigma,
        threshold: mask_threshold,
        num_blobs_raw,
    })
}

/// Pack `src > thr` into the 1-bit-per-pixel `mask`, `words_per_row =
/// ⌈w/64⌉` `u64`s per image row (returned; padding bits are zero). Rows are
/// packed in independent 16-row chunks — multi-threaded under the `parallel`
/// feature — with [`runs::pack_above_row`], which writes every word of its
/// row, so the mask is identical either way and needs no pre-zeroing.
fn threshold_to_mask(src: &[f32], w: usize, h: usize, thr: f32, mask: &mut Vec<u64>) -> usize {
    const ROWS_PER_CHUNK: usize = 16;
    let words_per_row = w.div_ceil(64);
    set_len_uninit(mask, words_per_row * h);
    par::for_each_chunk_mut(mask, words_per_row * ROWS_PER_CHUNK, |ci, chunk| {
        for (i, words) in chunk.chunks_exact_mut(words_per_row).enumerate() {
            let r = ci * ROWS_PER_CHUNK + i;
            runs::pack_above_row(&src[r * w..(r + 1) * w], thr, words);
        }
    });
    words_per_row
}

/// Background-subtracted residuals at the block subsample lattice (the same
/// staggered lattice [`BackgroundGrid::build`] medians over), against the
/// bilinear surface — the identical reference the full-image residual pass
/// used, so feeding these to [`estimate_background`] preserves its semantics
/// while touching ~stride² fewer samples.
fn subsample_residuals(pixels: &[f32], w: usize, h: usize, bg: &BackgroundGrid) -> Vec<f32> {
    let stride = bg.stride();
    let mut out: Vec<f32> = Vec::with_capacity((w / stride + 1) * (h / stride + 1));
    let mut y = 0usize;
    let mut phase = 0usize;
    while y < h {
        let rp = bg.row_params(y);
        let row = y * w;
        let mut x = phase;
        while x < w {
            let v = pixels[row + x];
            if v.is_finite() {
                out.push(v - bg.value_at(x, rp));
            }
            x += stride;
        }
        phase = (phase + 1) % stride;
        y += stride;
    }
    out
}

/// White-noise standard-deviation suppression factor of the separable 2-D
/// Gaussian blur used by the matched filter.
///
/// For a normalized 1-D kernel `k`, convolving white noise multiplies its
/// standard deviation by `√(Σk²)` per axis, so the separable 2-D factor is
/// `Σk²`. The kernel is replicated exactly as numeris builds it (radius
/// `ceil(3σ)`, `exp(−x²/2σ²)` weights, normalized); for σ ≳ 1 this
/// approaches the continuous limit `1/(2√π·σ)`.
fn gaussian_noise_suppression(sigma: f32) -> f32 {
    let radius = (3.0 * sigma).ceil() as i64;
    let inv_two_sigma_sq = 1.0 / (2.0 * sigma as f64 * sigma as f64);
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for i in -radius..=radius {
        let w = (-((i * i) as f64) * inv_two_sigma_sq).exp();
        sum += w;
        sum_sq += w * w;
    }
    (sum_sq / (sum * sum)) as f32
}

/// Estimate background level and noise.
///
/// Uses the median as the background level and estimates noise from the
/// lower half of the pixel distribution (below the median). This is robust
/// to contamination from stars and nebulosity, which only bias upward.
///
/// The noise estimate sigma-clips the below-median tail to reject remaining
/// outliers, then mirrors the lower-half RMS **about the median** to get the
/// full Gaussian sigma (`E[(v−m)² | v ≤ m] = σ²`).
pub(super) fn estimate_background(
    gray: &[f32],
    _width: u32,
    _height: u32,
    config: &CentroidExtractionConfig,
) -> (f32, f32) {
    let mut values: Vec<f32> = gray.iter().copied().filter(|v| v.is_finite()).collect();
    if values.is_empty() {
        return (0.0, 0.0);
    }

    // Median as robust background level (O(n) selection; see `median_f32`).
    let median = median_f32(&mut values);

    // Estimate noise from pixels at or below the median (uncontaminated by
    // stars, which only push the distribution upward). For Gaussian noise the
    // second moment of the lower half about the *median* equals the full
    // variance — E[(v−m)² | v ≤ m] = σ² — so the lower-half RMS about the
    // median mirrors directly into the full Gaussian sigma. This matches the
    // fast path's `coarse_background`. (Historically this computed the RMS
    // about the lower half's own mean, which for a half-normal is only
    // ≈0.60σ — silently turning a nominal 5σ threshold into a ~3σ one.)
    let mut low_half: Vec<f32> = values.iter().copied().filter(|&v| v <= median).collect();

    // Sigma-clip the lower tail to reject remaining outliers (dead or
    // negative pixels), re-estimating about the median each pass.
    let mut sigma = 0.0_f32;
    for _ in 0..config.sigma_clip_iterations {
        if low_half.is_empty() {
            break;
        }
        let var_sum: f64 = low_half
            .iter()
            .map(|&v| ((v - median) as f64).powi(2))
            .sum();
        sigma = (var_sum / low_half.len() as f64).sqrt() as f32;
        if sigma < 1e-10 {
            break;
        }
        let lo = median - config.sigma_clip_factor * sigma;
        let before = low_half.len();
        low_half.retain(|&v| v >= lo);
        if low_half.len() == before {
            break; // converged
        }
    }

    (median, sigma)
}

/// Compute intensity-weighted centroids for each connected region.
///
/// Consumes the run-length regions from [`runs::sweep_runs_mask`]; each stage
/// iterates the region's run list (row-major, so accumulation order matches
/// the historical bbox-scan order exactly). For each blob that passes size
/// and elongation filters:
/// 1. A local background is estimated from the median of non-blob pixels in a
///    5-pixel annulus around the blob's bounding box.
/// 2. Intensity-weighted moments are accumulated with the local background
///    subtracted, yielding a center-of-mass (CoM) position. Peak pixel is
///    tracked in the same pass.
/// 3. A 2D quadratic is fit to the 3×3 neighborhood around the peak pixel to
///    interpolate the sub-pixel intensity maximum. The quadratic position is
///    used only when it agrees with the CoM (within 0.5 px); otherwise the CoM
///    is kept as-is.
///
/// Centroids are returned in image-center-origin coordinates (`(cx, cy)` is
/// the origin in pixel coordinates), with mass and the intensity-weighted
/// 2×2 covariance `[[cxx, cxy], [cxy, cyy]]` in pixels².
///
/// When `max_elongation` is set in config, blobs with elongation ratio
/// (major/minor axis) exceeding the threshold are rejected as non-stellar.
/// The elongation test uses the **intensity-weighted** second moments —
/// the very same moments reported as `cov` (geometric moments admit a
/// slightly different set of marginal blobs — saturated stars with large
/// halos, etc. — which destabilizes downstream calibration on dense fields
/// like TESS).
///
/// On dense fields (thousands of blobs) this stage is a third or more of
/// extraction wall-clock, so under the `parallel` feature regions are
/// processed as independent tasks: every region reads only shared inputs
/// and writes its own `Option<Centroid>` slot, and the results are collected
/// in region order, so the output is identical to the sequential loop.
fn compute_blob_centroids(
    gray: &[f32],
    raw: &[f32],
    (mask, words_per_row): (&[u64], usize),
    regions: &runs::RunRegions,
    (w, h): (usize, usize),
    (cx, cy): (f32, f32),
    config: &CentroidExtractionConfig,
) -> Vec<Centroid> {
    let (offsets, order) = regions.group_by_region();
    let ctx = BlobContext {
        gray,
        raw,
        mask,
        words_per_row,
        regions,
        offsets: &offsets,
        order: &order,
        w,
        h,
        cx,
        cy,
        config,
    };
    let per_region = par::map_indices_init(regions.n_regions, BlobScratch::default, |s, k| {
        ctx.region_centroid(k, s)
    });
    per_region.into_iter().flatten().collect()
}

/// Read-only inputs shared by every region of one extraction.
struct BlobContext<'a> {
    /// Measurement image (clamped residuals, or the raw image without local
    /// background).
    gray: &'a [f32],
    /// Raw sensor image (saturation is judged on it).
    raw: &'a [f32],
    /// Detection bit mask and its words per row (see `threshold_to_mask`).
    mask: &'a [u64],
    words_per_row: usize,
    regions: &'a runs::RunRegions,
    /// `group_by_region` output: region `k`'s runs are
    /// `order[offsets[k]..offsets[k + 1]]`.
    offsets: &'a [u32],
    order: &'a [u32],
    w: usize,
    h: usize,
    /// Pixel coordinates of the output origin.
    cx: f32,
    cy: f32,
    config: &'a CentroidExtractionConfig,
}

/// Buffers reused across the regions one worker handles (dense fields have
/// thousands; a fresh allocation per region would dominate).
#[derive(Default)]
struct BlobScratch {
    annulus_vals: Vec<f32>,
    maxima: Vec<(f32, usize, usize)>,
    kept: Vec<(usize, usize)>,
}

impl BlobContext<'_> {
    /// The centroid of region `k`, or `None` when a filter rejects it.
    fn region_centroid(&self, k: usize, scratch: &mut BlobScratch) -> Option<Centroid> {
        let Self {
            gray,
            raw,
            mask,
            words_per_row,
            regions,
            w,
            h,
            config,
            ..
        } = *self;
        let region_runs = &self.order[self.offsets[k] as usize..self.offsets[k + 1] as usize];
        let extent = regions.extent(region_runs);
        let pixel_count = extent.npix;
        if pixel_count < config.min_pixels || pixel_count > config.max_pixels {
            return None;
        }
        if !extent.clear_of_border(config.border_margin as usize, w, h) {
            return None;
        }
        let (min_row, max_row, min_col, max_col) = (
            extent.min_row,
            extent.max_row,
            extent.min_col,
            extent.max_col,
        );

        // Reference pixel = bbox top-left, to keep moments numerically stable.
        let ref_col = min_col;
        let ref_row = min_row;

        // --- Per-blob local background from annulus ---
        // Expand bounding box by margin, collect pixels that are not part of
        // *any* region (not lit in the detection mask — every lit pixel was
        // grouped into some region).
        const ANNULUS_MARGIN: usize = 5;
        let r0 = min_row.saturating_sub(ANNULUS_MARGIN);
        let r1 = (max_row + ANNULUS_MARGIN + 1).min(h);
        let c0 = min_col.saturating_sub(ANNULUS_MARGIN);
        let c1 = (max_col + ANNULUS_MARGIN + 1).min(w);
        let annulus_vals = &mut scratch.annulus_vals;
        gather_unlit(
            gray,
            (mask, words_per_row),
            w,
            (r0, r1),
            (c0, c1),
            annulus_vals,
        );

        // Median of annulus (residual local background in bg-subtracted image).
        let local_bg = median_f32(annulus_vals) as f64;

        // --- Single moment pass: intensity-weighted moments with the
        // annulus-local background, tracking the peak in the same sweep ---
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_xx = 0.0_f64;
        let mut sum_yy = 0.0_f64;
        let mut sum_xy = 0.0_f64;
        let mut sum_i = 0.0_f64;
        let mut peak_val = f32::NEG_INFINITY;
        let mut peak_col: usize = ref_col;
        let mut peak_row: usize = ref_row;

        for &i in region_runs {
            let run = regions.runs[i as usize];
            let r = run.row as usize;
            let row_off = r * w;
            for c in run.c0 as usize..=run.c1 as usize {
                let raw = gray[row_off + c];
                if raw > peak_val {
                    peak_val = raw;
                    peak_col = c;
                    peak_row = r;
                }
                let intensity = (raw as f64 - local_bg).max(0.0);
                let dx = c as f64 - ref_col as f64;
                let dy = r as f64 - ref_row as f64;
                sum_x += dx * intensity;
                sum_y += dy * intensity;
                sum_xx += dx * dx * intensity;
                sum_yy += dy * dy * intensity;
                sum_xy += dx * dy * intensity;
                sum_i += intensity;
            }
        }

        if sum_i <= 0.0 {
            return None;
        }

        let dx_bar = sum_x / sum_i;
        let dy_bar = sum_y / sum_i;
        let xbar = ref_col as f64 + dx_bar;
        let ybar = ref_row as f64 + dy_bar;
        let cxx = sum_xx / sum_i - dx_bar * dx_bar;
        let cyy = sum_yy / sum_i - dy_bar * dy_bar;
        let cxy = sum_xy / sum_i - dx_bar * dy_bar;

        // Elongation filter — judged on the same intensity-weighted moments
        // reported as `cov`.
        if let Some(max_elong) = config.max_elongation {
            if elongation_from_cov(cxx, cyy, cxy) > max_elong {
                return None;
            }
        }

        // Saturation is judged on the RAW sensor value at the peak, not the
        // background-subtracted residual `peak_val`: `saturation_level` is
        // documented as a raw ADU level, so subtracting the (positive)
        // background would push a clipped star's residual below the clip
        // level and the exemption would never fire. On the no-local-bg path
        // `gray == raw`, so this matches `peak_val` there. Mirrors the fast
        // path, which tracks its peak on the raw image directly.
        let raw_peak = raw[peak_row * w + peak_col];
        let saturated = config.saturation_level.is_some_and(|s| raw_peak >= s);

        // --- Minimal deblending (see DeblendMode) ---
        // A blended pair centroids to the flux-weighted midpoint — a wrong
        // position the pattern hash will consume. Reject mode drops blobs
        // with more than one distinct peak: strict local maxima over the
        // 8-neighborhood, above 30% of the blob peak (over local
        // background), more than 2 px from any brighter accepted peak.
        // Saturated blobs are exempt (plateau noise fakes maxima on a
        // genuinely single star).
        if config.deblend == DeblendMode::Reject && !saturated {
            let thresh = local_bg + 0.3 * (peak_val as f64 - local_bg);
            let maxima = &mut scratch.maxima;
            maxima.clear();
            for &i in region_runs {
                let run = regions.runs[i as usize];
                let r = run.row as usize;
                let row_off = r * w;
                for c in run.c0 as usize..=run.c1 as usize {
                    let v = gray[row_off + c];
                    if (v as f64) <= thresh {
                        continue;
                    }
                    let mut is_max = true;
                    'nb: for dr in -1..=1_isize {
                        for dc in -1..=1_isize {
                            if dr == 0 && dc == 0 {
                                continue;
                            }
                            let rr = r as isize + dr;
                            let cc = c as isize + dc;
                            if rr < 0 || cc < 0 || rr >= h as isize || cc >= w as isize {
                                continue;
                            }
                            if gray[rr as usize * w + cc as usize] >= v {
                                is_max = false;
                                break 'nb;
                            }
                        }
                    }
                    if is_max {
                        maxima.push((v, c, r));
                    }
                }
            }
            maxima.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let kept = &mut scratch.kept;
            kept.clear();
            for &(_, c, r) in maxima.iter() {
                let distinct = kept.iter().all(|&(kc, kr)| {
                    let dx = c as f64 - kc as f64;
                    let dy = r as f64 - kr as f64;
                    dx * dx + dy * dy > 4.0
                });
                if distinct {
                    kept.push((c, r));
                    if kept.len() > 1 {
                        return None;
                    }
                }
            }
        }

        let (pc, pr) = (peak_col, peak_row);
        // 3x3 grid of background-subtracted values around the peak
        let v = |dy: isize, dx: isize| -> f64 {
            let r = (pr as isize + dy) as usize;
            let c = (pc as isize + dx) as usize;
            gray[r * w + c] as f64 - local_bg
        };

        // Sharpness gate, peak refinement, and assembly are shared with the
        // fast path (see `finish_region`).
        finish_region(
            pixel_count,
            (pc, pr),
            (w, h),
            (xbar, ybar),
            (cxx, cyy, cxy),
            sum_i,
            saturated,
            config.max_sharpness,
            (self.cx, self.cy),
            v,
        )
    }
}

/// Gather into `out` (cleared first), in raster order, the `gray` values of
/// the window rows `r0..r1` × columns `c0..c1` whose detection-mask bit is
/// clear. Branch-free: every window value is written to the next slot and
/// the slot index advances only for unlit pixels (`out` is sized to the
/// window, then truncated) — a mostly-unlit annulus makes the branchy
/// `if !lit { push }` form mispredict on every star pixel.
fn gather_unlit(
    gray: &[f32],
    (mask, words_per_row): (&[u64], usize),
    w: usize,
    (r0, r1): (usize, usize),
    (c0, c1): (usize, usize),
    out: &mut Vec<f32>,
) {
    out.clear();
    out.resize((r1 - r0) * (c1 - c0), 0.0);
    let mut n = 0usize;
    for r in r0..r1 {
        let row = &gray[r * w + c0..r * w + c1];
        let bits = &mask[r * words_per_row..(r + 1) * words_per_row];
        for (i, &v) in row.iter().enumerate() {
            let c = c0 + i;
            let lit = (bits[c / 64] >> (c % 64)) & 1;
            out[n] = v;
            n += (lit == 0) as usize;
        }
    }
    out.truncate(n);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gather_unlit_matches_branchy_gather() {
        let (w, h) = (150usize, 40usize);
        let words_per_row = w.div_ceil(64);
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let gray: Vec<f32> = (0..w * h).map(|_| (next() % 1000) as f32 * 0.25).collect();
        for density in [0u64, 4, 32, 60, 64] {
            let mut mask = vec![0u64; words_per_row * h];
            for r in 0..h {
                for c in 0..w {
                    if next() % 64 < density {
                        mask[r * words_per_row + c / 64] |= 1 << (c % 64);
                    }
                }
                // Garbage padding bits must not matter (never indexed).
                mask[r * words_per_row + words_per_row - 1] |= u64::MAX << (w % 64);
            }
            let windows = [
                (0usize, h, 0usize, w),
                (0, 1, 0, 1),
                (3, 17, 60, 70),
                (10, 11, 0, 150),
                (5, 30, 63, 65),
                (20, 40, 127, 150),
            ];
            for &(r0, r1, c0, c1) in &windows {
                let mut expect = Vec::new();
                for r in r0..r1 {
                    for c in c0..c1 {
                        if (mask[r * words_per_row + c / 64] >> (c % 64)) & 1 == 0 {
                            expect.push(gray[r * w + c]);
                        }
                    }
                }
                let mut got = vec![1.0; 7]; // stale contents are discarded
                gather_unlit(
                    &gray,
                    (&mask, words_per_row),
                    w,
                    (r0, r1),
                    (c0, c1),
                    &mut got,
                );
                assert_eq!(
                    got,
                    expect,
                    "density {density} window {:?}",
                    (r0, r1, c0, c1)
                );
            }
        }
    }
}
