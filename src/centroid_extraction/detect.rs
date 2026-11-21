//! The extraction pipeline: detection image → threshold mask → regions →
//! peaks → windowed measurement. Entry via [`extract`].

use numeris::imageproc::{convolve2d_separable_into, gaussian_kernel_1d, BorderMode};
use numeris::DynMatrix;

use super::background::{quantization_step, BlockGrid, BlockStats};
use super::measure::{elongation, measure, unwindowed_shape, Measurement, Window};
use super::{
    median_f32, par, runs, sort_and_truncate_by_mass, CentroidExtractionConfig,
    CentroidExtractionResult,
};
use crate::centroid::Centroid;
use crate::error::{Error, Result};
use crate::prelude::*;

/// Full-image working buffers, owned by
/// [`CentroidExtractor`](super::CentroidExtractor) so consecutive frames of
/// the same size reuse them instead of paying allocation and first-touch
/// page faults. Every element is rewritten before it is read, so a fresh
/// `Scratch` gives the same output as a reused one.
pub(super) struct Scratch {
    /// Detection image before smoothing (binned, sanitized); lent to a
    /// `DynMatrix` for the convolution and taken back.
    det: Vec<f32>,
    /// Smoothed detection image.
    smoothed: DynMatrix<f32>,
    /// 1-bit-per-pixel detection mask.
    mask: Vec<u64>,
}

impl Scratch {
    pub(super) fn new() -> Self {
        Self {
            det: Vec::new(),
            smoothed: DynMatrix::zeros(0, 0),
            mask: Vec::new(),
        }
    }
}

/// Rows per task of the full-image passes.
const ROWS_PER_CHUNK: usize = 16;

/// Smoothing-kernel truncation radius in σ. The detection statistic only
/// needs the kernel's core: 2.5σ keeps the completeness of a 3σ kernel
/// with fewer taps.
const KERNEL_TRUNCATE: f32 = 2.5;

/// Below this kernel width (in detection pixels) smoothing is skipped: the
/// kernel would be a single tap.
const MIN_KERNEL_SIGMA: f32 = 0.3;

/// Narrowest measurement window (Gaussian σ, pixels).
///
/// The window is sampled on the pixel grid, and the product of a narrow
/// window with a narrow PSF is too sharp for that grid: the estimate then
/// wobbles with the star's sub-pixel phase (about 0.03 px for a 0.7 px PSF
/// under a 0.7 px window). A window of at least one pixel removes the
/// wobble (0.004 px) at a noise cost of ~30% for such a PSF — which only
/// matters for stars too faint to reach the wobble anyway.
const MIN_WINDOW_SIGMA: f32 = 1.0;

/// Check the configuration and return the detection binning factor.
fn validate(cfg: &CentroidExtractionConfig, w: usize, h: usize) -> Result<usize> {
    let bad = |msg: String| Err(Error::InvalidInput(msg));
    if !(cfg.sigma_threshold.is_finite() && cfg.sigma_threshold > 0.0) {
        return bad(format!(
            "sigma_threshold must be finite and positive, got {}",
            cfg.sigma_threshold
        ));
    }
    if !(cfg.psf_sigma.is_finite() && cfg.psf_sigma > 0.0 && cfg.psf_sigma <= 10.0) {
        return bad(format!(
            "psf_sigma must be in (0, 10] pixels, got {}",
            cfg.psf_sigma
        ));
    }
    if cfg.bg_block_size == 0 {
        return bad("bg_block_size must be >= 1".into());
    }
    // Settings that would silently switch a gate off (NaN compares false)
    // or reject every detection are errors, not behaviour.
    if let Some(e) = cfg.max_elongation {
        if e.is_nan() || e < 1.0 {
            return bad(format!("max_elongation must be >= 1 (or None), got {e}"));
        }
    }
    if cfg.saturation_level.is_some_and(f32::is_nan) {
        return bad("saturation_level must not be NaN".into());
    }
    let b = match cfg.detection_binning {
        // Binning 2×2 costs a PSF this wide almost nothing (the bin is a
        // small part of the smoothing it needs anyway); a narrower PSF
        // would lose depth and resolution.
        None if cfg.psf_sigma >= 1.25 => 2,
        None | Some(1) => 1,
        Some(2) => 2,
        Some(n) => return bad(format!("detection_binning must be 1 or 2, got {n}")),
    };
    if w / b < 3 || h / b < 3 {
        return bad(format!(
            "image must be at least {0}x{0}, got {w}x{h}",
            3 * b
        ));
    }
    Ok(b)
}

/// Fill `det` (`wd × hd`) with the frame binned `b × b` (mean), replacing
/// non-finite values by the previous finite value in their row — one such
/// pixel would otherwise spread over the whole smoothing footprint.
fn build_detection_image(
    pixels: &[f32],
    w: usize,
    b: usize,
    (wd, hd): (usize, usize),
    det: &mut Vec<f32>,
) {
    if det.len() != wd * hd {
        det.resize(wd * hd, 0.0);
    }
    par::for_each_chunk_mut(det, wd * ROWS_PER_CHUNK, |ci, chunk| {
        for (i, out) in chunk.chunks_exact_mut(wd).enumerate() {
            let r = ci * ROWS_PER_CHUNK + i;
            let mut ok = true;
            if b == 1 {
                for (o, &p) in out.iter_mut().zip(&pixels[r * w..r * w + wd]) {
                    *o = p;
                    ok &= p.is_finite();
                }
            } else {
                let top = &pixels[2 * r * w..2 * r * w + 2 * wd];
                let bot = &pixels[(2 * r + 1) * w..(2 * r + 1) * w + 2 * wd];
                let (top, _) = top.as_chunks::<2>();
                let (bot, _) = bot.as_chunks::<2>();
                for ((o, t), u) in out.iter_mut().zip(top).zip(bot) {
                    let v = 0.25 * ((t[0] + t[1]) + (u[0] + u[1]));
                    *o = v;
                    ok &= v.is_finite();
                }
            }
            if !ok {
                let mut last = out.iter().copied().find(|p| p.is_finite()).unwrap_or(0.0);
                for p in out.iter_mut() {
                    if p.is_finite() {
                        last = *p;
                    } else {
                        *p = last;
                    }
                }
            }
        }
    });
}

/// Run the pipeline on a `width × height` row-major frame.
pub(super) fn extract(
    pixels: &[f32],
    width: u32,
    height: u32,
    cfg: &CentroidExtractionConfig,
    scratch: &mut Scratch,
) -> Result<CentroidExtractionResult> {
    let (w, h) = (width as usize, height as usize);
    let b = validate(cfg, w, h)?;
    let (wd, hd) = (w / b, h / b);
    let bf = b as f32;

    // ── 1. Detection image: binned, sanitized, smoothed ──
    // Detection runs on the frame binned b×b: binning is itself the first
    // stage of the smoothing, and every later full-image pass then touches
    // b² fewer pixels. Measurement (step 5) always uses the raw pixels.
    //
    // The kernel width, in detection pixels, is chosen so that box(b)
    // followed by the Gaussian has the PSF's variance — a matched filter:
    // σk² = (σp² − (b² − 1)/12) / b².
    build_detection_image(pixels, w, b, (wd, hd), &mut scratch.det);
    let sigma_k = (cfg.psf_sigma.powi(2) - (bf * bf - 1.0) / 12.0)
        .max(0.0)
        .sqrt()
        / bf;
    // White-noise σ of the detection image per unit raw-pixel σ.
    let mut noise_gain = 1.0 / bf;
    let s: &[f32] = if sigma_k >= MIN_KERNEL_SIGMA {
        let kernel = gaussian_kernel_1d::<f32>(sigma_k, KERNEL_TRUNCATE)
            .map_err(|_| Error::InvalidInput("invalid smoothing kernel".into()))?;
        noise_gain *= kernel.iter().map(|v| v * v).sum::<f32>();
        // numeris matrices are column-major: a (wd × hd) matrix whose
        // columns are image rows is exactly the row-major image.
        let src = DynMatrix::from_vec(wd, hd, core::mem::take(&mut scratch.det));
        convolve2d_separable_into(
            &src,
            &kernel,
            &kernel,
            BorderMode::Replicate,
            &mut scratch.smoothed,
        );
        scratch.det = src.into_vec();
        scratch.smoothed.as_slice()
    } else {
        &scratch.det
    };

    // ── 2. Background and noise, measured on the detection image ──
    // Taking the statistics in the domain that is thresholded makes
    // `sigma_threshold` mean what it says whatever the pixel noise's
    // correlation is (resampled, debayered or compressed frames), and the
    // per-block σ follows noise that varies across the frame (vignetting,
    // stray light, a bright sky gradient).
    let block = (cfg.bg_block_size as usize / b).max(1);
    let sigma_floor = quantization_step(pixels) * 0.288_675_13 * noise_gain;
    let stats = BlockStats::measure(s, wd, hd, block, sigma_floor);
    let k = cfg.sigma_threshold;
    // Never below the rounding error of the interpolated surface: on a
    // noiseless frame the blend of equal block levels can land an ulp under
    // the pixel value, and `p > thr` would light arbitrary patches.
    let margin = stats.level.rounding_margin();
    let thr = stats.level.with_values(
        stats
            .level
            .grid
            .iter()
            .zip(&stats.sigma)
            .map(|(&m, &sg)| m + (k * sg).max(margin))
            .collect(),
    );

    // ── 3. Threshold into a bit mask, sweep into runs and regions ──
    let words_per_row = wd.div_ceil(64);
    let mask = &mut scratch.mask;
    if mask.len() != words_per_row * hd {
        mask.resize(words_per_row * hd, 0);
    }
    par::for_each_chunk_mut(mask, words_per_row * ROWS_PER_CHUNK, |ci, chunk| {
        let mut grid_row = vec![0.0_f32; thr.nx];
        let mut thr_row = vec![0.0_f32; wd];
        for (i, words) in chunk.chunks_exact_mut(words_per_row).enumerate() {
            let r = ci * ROWS_PER_CHUNK + i;
            thr.blend_row(thr.row_params(r), &mut grid_row);
            thr.blend_columns(&grid_row, &mut thr_row);
            runs::pack_lit_row(&s[r * wd..(r + 1) * wd], &thr_row, words);
        }
    });
    #[cfg(feature = "parallel")]
    let regions = runs::sweep_runs_mask_banded(wd, hd, words_per_row, mask, 64);
    #[cfg(not(feature = "parallel"))]
    let regions = runs::sweep_runs_mask(wd, hd, words_per_row, mask);
    let (offsets, order) = regions.group_by_region();

    // ── 4–5. Peaks and measurement, one task per region ──
    let ctx = Ctx {
        raw: pixels,
        s,
        level: &stats.level,
        sigma: &stats.sigma,
        regions: &regions,
        w,
        h,
        wd,
        hd,
        b,
        cfg,
        win: Window::new(cfg.psf_sigma.max(MIN_WINDOW_SIGMA)),
        raw_noise_per_det: 1.0 / noise_gain as f64,
    };
    let per_region: Vec<Vec<Star>> =
        par::map_indices_init(regions.n_regions, Vec::new, |peaks, kreg| {
            let region_runs = &order[offsets[kreg] as usize..offsets[kreg + 1] as usize];
            ctx.region(region_runs, peaks)
        });
    let stars: Vec<Star> = per_region.into_iter().flatten().collect();

    // PSF width actually seen: the median over well-measured stars.
    let mut widths: Vec<f32> = stars.iter().filter_map(|st| st.psf_sigma).collect();
    let psf_sigma_estimate = (widths.len() >= 5).then(|| median_f32(&mut widths));

    let mut centroids: Vec<Centroid> = stars.into_iter().map(|st| st.centroid).collect();
    sort_and_truncate_by_mass(&mut centroids, cfg.max_centroids);

    let background_mean = stats.level.midpoint();
    Ok(CentroidExtractionResult {
        centroids,
        image_width: width,
        image_height: height,
        background_mean,
        background_sigma: stats.sigma_global / noise_gain,
        threshold: background_mean + k * stats.sigma_global,
        num_blobs_raw: regions.n_regions,
        psf_sigma_estimate,
    })
}

/// One accepted detection.
struct Star {
    centroid: Centroid,
    /// PSF width implied by this star's windowed moments, when it is bright
    /// enough and cleanly measured for that to mean something.
    psf_sigma: Option<f32>,
}

/// Read-only inputs shared by every region of one extraction.
struct Ctx<'a> {
    /// Raw frame (`w × h`).
    raw: &'a [f32],
    /// Smoothed detection image (`wd × hd`, binned `b × b`).
    s: &'a [f32],
    /// Background level and per-block noise σ of the detection image.
    level: &'a BlockGrid,
    sigma: &'a [f32],
    regions: &'a runs::RunRegions,
    w: usize,
    h: usize,
    wd: usize,
    hd: usize,
    b: usize,
    cfg: &'a CentroidExtractionConfig,
    win: Window,
    /// Raw-pixel noise σ per unit detection-image σ, for white noise.
    raw_noise_per_det: f64,
}

/// Minimum flux signal-to-noise for a star to contribute to the PSF-width
/// estimate.
const PSF_ESTIMATE_MIN_SNR: f64 = 30.0;

impl Ctx<'_> {
    /// Whether detection pixel `(c, r)` is a local maximum of the smoothed
    /// image over its 8-neighbourhood. Ties go to the pixel that comes
    /// first in raster order, so a flat plateau yields a few candidates
    /// rather than none.
    fn is_peak(&self, c: usize, r: usize) -> bool {
        let (w, h, s) = (self.wd, self.hd, self.s);
        let v = s[r * w + c];
        for rr in r.saturating_sub(1)..(r + 2).min(h) {
            for cc in c.saturating_sub(1)..(c + 2).min(w) {
                if rr == r && cc == c {
                    continue;
                }
                let n = s[rr * w + cc];
                let earlier = rr < r || (rr == r && cc < c);
                if !(n < v || (n == v && !earlier)) {
                    return false;
                }
            }
        }
        true
    }

    /// Raw-pixel position of the center of detection pixel `(c, r)`.
    fn raw_pos(&self, c: usize, r: usize) -> (f64, f64) {
        let off = (self.b - 1) as f64 / 2.0;
        ((c * self.b) as f64 + off, (r * self.b) as f64 + off)
    }

    /// Whether any raw pixel under detection pixel `(c, r)` is saturated.
    fn saturated(&self, c: usize, r: usize) -> bool {
        let Some(sat) = self.cfg.saturation_level else {
            return false;
        };
        let b = self.b;
        (r * b..r * b + b).any(|rr| (c * b..c * b + b).any(|cc| self.raw[rr * self.w + cc] >= sat))
    }

    /// Background level and raw-pixel noise σ at detection pixel `(c, r)`.
    fn local_stats(&self, c: usize, r: usize) -> (f32, f64) {
        let bg = self.level.value_at(c, self.level.row_params(r));
        let sigma = self.sigma[self.level.block_index(c, r)] as f64 * self.raw_noise_per_det;
        (bg, sigma)
    }

    /// The stars of one region.
    ///
    /// A region (connected pixels of the smoothed image above threshold)
    /// only says which peaks belong together; each local maximum inside it
    /// is a candidate star and is measured on its own. The brightest peak
    /// is the region's reason to exist and needs no further test. A fainter
    /// one must prove it is a separate source and not noise or PSF
    /// structure riding on a neighbour's wing.
    fn region(&self, region_runs: &[u32], peaks: &mut Vec<(f32, u32, u32)>) -> Vec<Star> {
        let (w, h, cfg) = (self.w, self.h, self.cfg);
        let extent = self.regions.extent(region_runs);
        if extent.npix * self.b * self.b > cfg.max_pixels {
            return Vec::new();
        }
        peaks.clear();
        for &i in region_runs {
            let run = self.regions.runs[i as usize];
            let r = run.row as usize;
            for c in run.c0 as usize..=run.c1 as usize {
                if self.is_peak(c, r) {
                    peaks.push((self.s[r * self.wd + c], c as u32, r as u32));
                }
            }
        }
        // Brightest first; raster order breaks ties so the result does not
        // depend on how the runs were visited.
        peaks.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then((a.2, a.1).cmp(&(b.2, b.1))));
        let Some(&(_, pc, pr)) = peaks.first() else {
            return Vec::new();
        };
        let s2 = self.win.sigma2;
        // A measurement starts at a detection-pixel center, which is up to
        // half a detection pixel from the true peak of the smoothed image.
        let slack = self.b as f64;

        // ── Brightest peak ──
        // It is the region's reason to exist, so it needs no significance
        // test, and a little more room to settle than the others.
        let (pc, pr) = (pc as usize, pr as usize);
        let (bg, sigma_raw) = self.local_stats(pc, pr);
        let saturated = self.saturated(pc, pr);
        let primary = if saturated {
            None
        } else {
            measure(
                self.raw,
                (w, h),
                bg,
                self.raw_pos(pc, pr),
                &self.win,
                1.0 + slack,
            )
        };
        // Saturated, or far broader than the window can resolve (object σ
        // over twice `psf_sigma`), or not settling: the window is sitting
        // on a plateau and its fixed point means little. The region's
        // center of mass is the right estimate for such a blob, and the
        // region yields that one centroid.
        let on_plateau = primary.is_some_and(|m| m.mean_var() > 0.8 * s2 || !m.converged);
        if saturated || on_plateau {
            return self
                .region_com(region_runs, bg, sigma_raw)
                .into_iter()
                .collect();
        }
        let mut out = Vec::new();
        let mut accepted: Vec<(f64, f64)> = Vec::new();
        if let Some(p) = primary {
            accepted.push((p.x, p.y));
            out.extend(self.finish(&p, sigma_raw));
        }

        // ── Fainter peaks ──
        // Each must: settle within a detection pixel of where it started
        // (rather than slide into a neighbour); land at least 2·psf_sigma
        // from every star already accepted; and, in a local fit of a
        // Gaussian plus a constant pedestal, show an amplitude that is
        // significant against the noise and at least half the pedestal it
        // sits on. The last test is what rejects bumps on a bright star's
        // wing, where the pedestal is the wing.
        let k = cfg.sigma_threshold as f64;
        let min_sep2 = 4.0 * s2;
        for &(_, c, r) in peaks.iter().skip(1) {
            let (c, r) = (c as usize, r as usize);
            let (bg, sigma_raw) = self.local_stats(c, r);
            let Some(m) = measure(self.raw, (w, h), bg, self.raw_pos(c, r), &self.win, slack)
            else {
                continue;
            };
            let clear = accepted
                .iter()
                .all(|&(ax, ay)| (ax - m.x).powi(2) + (ay - m.y).powi(2) >= min_sep2);
            if !m.converged
                || !clear
                || m.amp < k * sigma_raw * m.amp_var.sqrt()
                || m.amp < 0.5 * m.pedestal
            {
                continue;
            }
            accepted.push((m.x, m.y));
            out.extend(self.finish(&m, sigma_raw));
        }
        out
    }

    /// Whether `(x, y)` (raw pixels) is clear of the configured border.
    fn in_frame(&self, x: f64, y: f64) -> bool {
        let bm = self.cfg.border_margin as f64;
        x >= bm && y >= bm && x <= (self.w - 1) as f64 - bm && y <= (self.h - 1) as f64 - bm
    }

    /// Build a [`Centroid`] in image-center-origin coordinates.
    fn centroid(
        &self,
        (x, y): (f64, f64),
        flux: f64,
        var: (f64, f64),
        shape: [f64; 3],
    ) -> Centroid {
        let (cx, cy) = ((self.w - 1) as f64 / 2.0, (self.h - 1) as f64 / 2.0);
        Centroid {
            x: (x - cx) as f32,
            y: (y - cy) as f32,
            mass: Some(flux as f32),
            cov: Some(crate::Matrix2::new([
                [var.0 as f32, 0.0],
                [0.0, var.1 as f32],
            ])),
            shape: Some(crate::Matrix2::new([
                [shape[0] as f32, shape[2] as f32],
                [shape[2] as f32, shape[1] as f32],
            ])),
        }
    }

    /// Apply the shape and border gates to a windowed measurement and turn
    /// it into a star.
    fn finish(&self, m: &Measurement, sigma_raw: f64) -> Option<Star> {
        let s2 = self.win.sigma2;
        if !self.in_frame(m.x, m.y) {
            return None;
        }
        // A matched PSF measures σw²/2 per axis; a single hot pixel about
        // 0.04·σw² at σw = 1.5. Below 0.15·σw² the object is narrower than
        // ~0.4·σw: a hot pixel or cosmic-ray hit.
        if self.cfg.reject_hot_pixels && m.mean_var() < 0.15 * s2 {
            return None;
        }
        let (shape, elong) = unwindowed_shape(m.moments, s2);
        if self
            .cfg
            .max_elongation
            .is_some_and(|max| elong > max as f64)
        {
            return None;
        }
        let var = sigma_raw * sigma_raw;
        let snr = m.flux / (var * m.flux_var).sqrt().max(f64::MIN_POSITIVE);
        // Width estimate from stars the window resolves comfortably.
        let psf_sigma = (snr >= PSF_ESTIMATE_MIN_SNR && m.mean_var() < 0.75 * s2)
            .then(|| (0.5 * (shape[0] + shape[1])).sqrt() as f32);
        Some(Star {
            centroid: self.centroid((m.x, m.y), m.flux, (var * m.var_x, var * m.var_y), shape),
            psf_sigma,
        })
    }

    /// Intensity-weighted center of mass of the region's pixels above the
    /// background — the estimate for saturated and very broad blobs.
    ///
    /// The elongation gate applies here too, saturated or not: a star whose
    /// charge has bled far along a column has a center of mass that can be
    /// pixels from the star, and is better dropped than trusted.
    fn region_com(&self, region_runs: &[u32], bg: f32, sigma_raw: f64) -> Option<Star> {
        let (w, b) = (self.w, self.b);
        let (mut s0, mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        // Noise term of the position variance: Σ (offset from the mean)².
        let (mut n, mut nx, mut ny, mut nxx, mut nyy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        // Reference pixel = the region's first, to keep the sums small.
        let first = self.regions.runs[region_runs[0] as usize];
        let (rc, rr) = (
            (first.c0 as usize * b) as f64,
            (first.row as usize * b) as f64,
        );
        for &i in region_runs {
            let run = self.regions.runs[i as usize];
            for r in run.row as usize * b..(run.row as usize + 1) * b {
                let dy = r as f64 - rr;
                for c in run.c0 as usize * b..(run.c1 as usize + 1) * b {
                    let p = self.raw[r * w + c];
                    let d = if p.is_finite() {
                        (p - bg).max(0.0) as f64
                    } else {
                        0.0
                    };
                    let dx = c as f64 - rc;
                    s0 += d;
                    sx += d * dx;
                    sy += d * dy;
                    sxx += d * dx * dx;
                    syy += d * dy * dy;
                    sxy += d * dx * dy;
                    n += 1.0;
                    nx += dx;
                    ny += dy;
                    nxx += dx * dx;
                    nyy += dy * dy;
                }
            }
        }
        if s0 <= 0.0 {
            return None;
        }
        let (mx, my) = (sx / s0, sy / s0);
        let (x, y) = (rc + mx, rr + my);
        if !self.in_frame(x, y) {
            return None;
        }
        let shape = [sxx / s0 - mx * mx, syy / s0 - my * my, sxy / s0 - mx * my];
        let too_long = |max: f32| elongation(shape) > max as f64;
        if self.cfg.max_elongation.is_some_and(too_long) {
            return None;
        }
        // Var(Σ d·(x − x̄)) / s0² with independent pixel noise.
        let v = sigma_raw * sigma_raw / (s0 * s0);
        let var = (
            v * (nxx - 2.0 * mx * nx + n * mx * mx),
            v * (nyy - 2.0 * my * ny + n * my * my),
        );
        Some(Star {
            centroid: self.centroid((x, y), s0, var, shape),
            psf_sigma: None,
        })
    }
}
