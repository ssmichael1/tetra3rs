//! Unit tests for the extraction pipeline, on synthetic frames with known
//! star positions.

use super::background::{quantization_step, BlockStats};
use super::measure::{measure, unwindowed_shape, Window};
use super::*;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};

const READ_NOISE: f32 = 5.0;
const BG: f32 = 100.0;

/// erf, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
fn erf(x: f64) -> f64 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    s * (1.0 - poly * (-x * x).exp())
}

/// Fraction of a unit-flux 1-D Gaussian that lands in pixel `i`.
fn pixel_frac(i: i64, mu: f64, sigma: f64) -> f64 {
    let s = sigma * core::f64::consts::SQRT_2;
    0.5 * (erf((i as f64 + 0.5 - mu) / s) - erf((i as f64 - 0.5 - mu) / s))
}

/// A synthetic frame under construction: noiseless signal plus truth.
struct Frame {
    w: usize,
    h: usize,
    psf: f64,
    signal: Vec<f32>,
    /// True positions in pixel coordinates (pixel centers at integers).
    truth: Vec<(f64, f64)>,
}

impl Frame {
    fn new(w: usize, h: usize, psf: f64) -> Self {
        Self {
            w,
            h,
            psf,
            signal: vec![0.0; w * h],
            truth: Vec::new(),
        }
    }

    /// Total flux of a star with matched-filter SNR `snr` against the read
    /// noise.
    fn flux(&self, snr: f64) -> f64 {
        snr * READ_NOISE as f64 * 2.0 * core::f64::consts::PI.sqrt() * self.psf
    }

    /// Add a pixel-integrated Gaussian star of total flux `flux`.
    fn star(&mut self, x: f64, y: f64, flux: f64) {
        let r = (6.0 * self.psf).ceil() as i64 + 1;
        let (xi, yi) = (x.round() as i64, y.round() as i64);
        for row in (yi - r).max(0)..=(yi + r).min(self.h as i64 - 1) {
            let fy = pixel_frac(row, y, self.psf);
            for col in (xi - r).max(0)..=(xi + r).min(self.w as i64 - 1) {
                self.signal[row as usize * self.w + col as usize] +=
                    (flux * fy * pixel_frac(col, x, self.psf)) as f32;
            }
        }
        self.truth.push((x, y));
    }

    /// Stars of SNR `snr` on a jittered lattice of pitch `cell`.
    fn lattice(&mut self, cell: usize, snr: f64, rng: &mut StdRng) {
        let flux = self.flux(snr);
        for gy in 0..self.h / cell {
            for gx in 0..self.w / cell {
                let x = (gx * cell) as f64 + cell as f64 / 2.0 + rng.random::<f64>() - 0.5;
                let y = (gy * cell) as f64 + cell as f64 / 2.0 + rng.random::<f64>() - 0.5;
                self.star(x, y, flux);
            }
        }
    }

    /// Render: background `bg(u, v)` (`u`, `v` in 0..1), Gaussian noise of
    /// σ `noise(u, v)`.
    fn render_with(
        &self,
        rng: &mut StdRng,
        bg: impl Fn(f32, f32) -> f32,
        noise: impl Fn(f32, f32) -> f32,
    ) -> Vec<f32> {
        let normal = Normal::new(0.0_f32, 1.0).unwrap();
        let mut out = vec![0.0; self.w * self.h];
        for row in 0..self.h {
            for col in 0..self.w {
                let (u, v) = (col as f32 / self.w as f32, row as f32 / self.h as f32);
                out[row * self.w + col] =
                    bg(u, v) + self.signal[row * self.w + col] + noise(u, v) * normal.sample(rng);
            }
        }
        out
    }

    /// Render on a flat background with the standard read noise.
    fn render(&self, rng: &mut StdRng) -> Vec<f32> {
        self.render_with(rng, |_, _| BG, |_, _| READ_NOISE)
    }
}

fn run(
    pixels: &[f32],
    w: usize,
    h: usize,
    cfg: &CentroidExtractionConfig,
) -> CentroidExtractionResult {
    extract_centroids_from_raw(pixels, w as u32, h as u32, cfg).unwrap()
}

/// Detections in pixel coordinates (top-left origin).
fn positions(res: &CentroidExtractionResult) -> Vec<(f64, f64)> {
    let cx = (res.image_width - 1) as f64 / 2.0;
    let cy = (res.image_height - 1) as f64 / 2.0;
    res.centroids
        .iter()
        .map(|c| (c.x as f64 + cx, c.y as f64 + cy))
        .collect()
}

/// For each truth position the distance to the nearest detection, and the
/// number of detections farther than `radius` from every truth position.
fn match_truth(truth: &[(f64, f64)], dets: &[(f64, f64)], radius: f64) -> (Vec<f64>, usize) {
    let dist = |a: &(f64, f64), b: &(f64, f64)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
    let nearest = truth
        .iter()
        .map(|t| {
            dets.iter()
                .map(|d| dist(t, d))
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let stray = dets
        .iter()
        .filter(|d| truth.iter().all(|t| dist(t, d) > radius))
        .count();
    (nearest, stray)
}

/// Per-axis RMS of radial errors.
fn rms_axis(errs: &[f64]) -> f64 {
    (errs.iter().map(|e| e * e).sum::<f64>() / (2.0 * errs.len() as f64)).sqrt()
}

// ─── Accuracy ───────────────────────────────────────────────────────────────

#[test]
fn position_error_tracks_the_noise_limit() {
    // Read-noise Cramér–Rao bound per axis: √2·σ_psf / SNR.
    for (snr, binning) in [
        (20.0, Some(1)),
        (50.0, None),
        (1000.0, None),
        (50.0, Some(1)),
    ] {
        let mut rng = StdRng::seed_from_u64(7);
        let mut f = Frame::new(512, 512, 1.5);
        f.lattice(32, snr, &mut rng);
        let px = f.render(&mut rng);
        let cfg = CentroidExtractionConfig {
            detection_binning: binning,
            ..Default::default()
        };
        let res = run(&px, f.w, f.h, &cfg);
        let (errs, stray) = match_truth(&f.truth, &positions(&res), 1.5);
        assert!(
            errs.iter().all(|&e| e < 1.5),
            "snr {snr}: a star was missed"
        );
        assert_eq!(stray, 0, "snr {snr}: stray detections");
        let bound = core::f64::consts::SQRT_2 * 1.5 / snr;
        let rms = rms_axis(&errs);
        assert!(
            rms < 1.25 * bound,
            "snr {snr}: rms {rms:.4} px vs bound {bound:.4}"
        );
    }
}

#[test]
fn reported_covariance_matches_the_scatter() {
    let mut rng = StdRng::seed_from_u64(11);
    let mut f = Frame::new(768, 768, 1.5);
    f.lattice(32, 20.0, &mut rng);
    let px = f.render(&mut rng);
    let res = run(&px, f.w, f.h, &CentroidExtractionConfig::default());
    let (errs, _) = match_truth(&f.truth, &positions(&res), 1.5);
    let actual = rms_axis(&errs);
    let predicted = (res
        .centroids
        .iter()
        .map(|c| {
            let m = c.cov.unwrap();
            0.5 * (m[(0, 0)] + m[(1, 1)]) as f64
        })
        .sum::<f64>()
        / res.centroids.len() as f64)
        .sqrt();
    assert!(
        (predicted / actual - 1.0).abs() < 0.2,
        "predicted σ {predicted:.4} vs actual {actual:.4}"
    );
    // Noise and PSF width are reported in raw-pixel terms.
    assert!((res.background_sigma / READ_NOISE - 1.0).abs() < 0.1);
    assert!((res.background_mean - BG).abs() < 0.5);
}

#[test]
fn shape_and_psf_estimate_report_the_psf_width() {
    for psf in [1.2_f64, 1.5, 2.0] {
        let mut rng = StdRng::seed_from_u64(13);
        let mut f = Frame::new(512, 512, psf);
        f.lattice(40, 300.0, &mut rng);
        let px = f.render(&mut rng);
        // Deliberately not matched to the true width.
        let res = run(&px, f.w, f.h, &CentroidExtractionConfig::default());
        // A pixel-integrated Gaussian is broader by the pixel's own
        // variance, 1/12.
        let expect = (psf * psf + 1.0 / 12.0).sqrt();
        let est = res.psf_sigma_estimate.expect("enough bright stars") as f64;
        assert!(
            (est / expect - 1.0).abs() < 0.08,
            "psf {psf}: estimate {est:.3}"
        );
        let s = res.centroids[0].shape.unwrap();
        let width = (0.5 * (s[(0, 0)] + s[(1, 1)]) as f64).sqrt();
        assert!(
            (width / expect - 1.0).abs() < 0.1,
            "psf {psf}: shape {width:.3}"
        );
    }
}

#[test]
fn psf_width_mismatch_degrades_gently() {
    // Config says 1.5; the optics deliver 1.0 or 2.5.
    for psf in [1.0_f64, 2.5] {
        let mut rng = StdRng::seed_from_u64(17);
        let mut f = Frame::new(512, 512, psf);
        f.lattice(32, 100.0, &mut rng);
        let px = f.render(&mut rng);
        let res = run(&px, f.w, f.h, &CentroidExtractionConfig::default());
        let (errs, stray) = match_truth(&f.truth, &positions(&res), 1.5);
        assert!(errs.iter().all(|&e| e < 1.5));
        assert_eq!(stray, 0);
        let bound = core::f64::consts::SQRT_2 * psf / 100.0;
        assert!(
            rms_axis(&errs) < 1.6 * bound,
            "psf {psf}: {:.4}",
            rms_axis(&errs)
        );
    }
}

#[test]
fn binned_and_full_detection_measure_the_same_positions() {
    let mut rng = StdRng::seed_from_u64(19);
    let mut f = Frame::new(400, 400, 1.5);
    f.lattice(40, 200.0, &mut rng);
    let px = f.render(&mut rng);
    let at = |b| {
        let cfg = CentroidExtractionConfig {
            detection_binning: Some(b),
            ..Default::default()
        };
        positions(&run(&px, f.w, f.h, &cfg))
    };
    let (full, binned) = (at(1), at(2));
    assert_eq!(full.len(), f.truth.len());
    let (d, stray) = match_truth(&full, &binned, 0.5);
    assert_eq!(stray, 0);
    // The measurement is on raw pixels either way; only its starting point
    // differs.
    assert!(
        d.iter().all(|&e| e < 5e-3),
        "max {:?}",
        d.iter().cloned().fold(0.0, f64::max)
    );
}

// ─── Detection statistics ───────────────────────────────────────────────────

#[test]
fn pure_noise_gives_almost_no_detections() {
    let mut rng = StdRng::seed_from_u64(23);
    let f = Frame::new(1024, 1024, 1.5);
    let px = f.render(&mut rng);
    for binning in [Some(1), Some(2)] {
        let cfg = CentroidExtractionConfig {
            detection_binning: binning,
            ..Default::default()
        };
        let n = run(&px, f.w, f.h, &cfg).centroids.len();
        assert!(n <= 3, "{n} detections in a noise-only frame");
    }
}

#[test]
fn threshold_holds_on_a_gradient_with_varying_noise() {
    // A steep, curved background and noise that grows 4× across the frame:
    // the threshold must follow both, or the quiet steep corner fills with
    // false detections.
    let mut rng = StdRng::seed_from_u64(29);
    let mut f = Frame::new(1024, 1024, 1.5);
    f.lattice(128, 400.0, &mut rng);
    let bg =
        |u: f32, v: f32| BG + 600.0 * (u + 0.5 * v - 1.2 * ((u - 0.5).powi(2) + (v - 0.5).powi(2)));
    let px = f.render_with(&mut rng, bg, |u, _| READ_NOISE * (1.0 + 3.0 * u));
    let res = run(&px, f.w, f.h, &CentroidExtractionConfig::default());
    let (errs, stray) = match_truth(&f.truth, &positions(&res), 2.0);
    assert!(errs.iter().all(|&e| e < 1.0), "bright stars must survive");
    assert!(stray <= 3, "{stray} false detections");
}

#[test]
fn threshold_is_correct_for_correlated_noise() {
    // Smooth the noise first (as resampling or debayering does): a
    // threshold assuming white noise would be far too low for it.
    let mut rng = StdRng::seed_from_u64(31);
    let (w, h) = (768usize, 768usize);
    let white = Frame::new(w, h, 1.5).render_with(&mut rng, |_, _| 0.0, |_, _| 10.0);
    let mut px = vec![BG; w * h];
    for r in 1..h - 1 {
        for c in 1..w - 1 {
            let mut sum = 0.0;
            for dr in 0..3 {
                for dc in 0..3 {
                    sum += white[(r + dr - 1) * w + c + dc - 1];
                }
            }
            px[r * w + c] += sum / 3.0;
        }
    }
    let n = run(&px, w, h, &CentroidExtractionConfig::default())
        .centroids
        .len();
    assert!(n <= 3, "{n} detections in correlated noise");
}

#[test]
fn flat_and_degenerate_frames_give_nothing() {
    let cfg = CentroidExtractionConfig::default();
    for value in [0.0_f32, 1234.5, f32::NAN, f32::INFINITY] {
        let px = vec![value; 200 * 160];
        let res = run(&px, 200, 160, &cfg);
        assert!(res.centroids.is_empty(), "value {value}");
    }
}

#[test]
fn quantized_frame_with_sub_count_noise_does_not_flood() {
    // An 8-bit style frame: background exactly 3 with an occasional 4 —
    // the measured σ of the smoothed image is zero almost everywhere.
    let mut rng = StdRng::seed_from_u64(37);
    let (w, h) = (512usize, 512usize);
    let mut px: Vec<f32> = (0..w * h)
        .map(|_| if rng.random::<f32>() < 0.02 { 4.0 } else { 3.0 })
        .collect();
    let mut f = Frame::new(w, h, 1.5);
    f.star(200.3, 150.7, 4000.0);
    f.star(350.6, 400.2, 4000.0);
    for (p, s) in px.iter_mut().zip(&f.signal) {
        *p = (*p + s).round();
    }
    let res = run(&px, w, h, &CentroidExtractionConfig::default());
    let (errs, stray) = match_truth(&f.truth, &positions(&res), 2.0);
    assert!(errs.iter().all(|&e| e < 0.2), "{errs:?}");
    assert!(stray <= 5, "{stray} detections on single-count speckle");
}

#[test]
fn quantization_step_sees_integer_data_only() {
    let mut rng = StdRng::seed_from_u64(41);
    let cont = Frame::new(256, 256, 1.5).render(&mut rng);
    assert!(quantization_step(&cont) < 0.05 * READ_NOISE);
    let ints: Vec<f32> = cont.iter().map(|p| p.round()).collect();
    assert_eq!(quantization_step(&ints), 1.0);
    assert_eq!(quantization_step(&[7.0; 100]), 0.0);
    assert_eq!(quantization_step(&[f32::NAN; 100]), 0.0);
}

#[test]
fn block_stats_recover_level_and_noise_on_a_slope() {
    let mut rng = StdRng::seed_from_u64(43);
    let (w, h) = (512usize, 384usize);
    // 0.5 ADU/px along y: 32 ADU across a block, against σ = 2.
    let px = Frame::new(w, h, 1.5).render_with(&mut rng, |_, v| 50.0 + 0.5 * 384.0 * v, |_, _| 2.0);
    let stats = BlockStats::measure(&px, w, h, 64, 0.0);
    assert!(
        (stats.sigma_global / 2.0 - 1.0).abs() < 0.1,
        "σ {}",
        stats.sigma_global
    );
    for (y, x) in [(40usize, 40usize), (200, 300), (350, 480)] {
        let got = stats.level.value_at(x, stats.level.row_params(y));
        let want = 50.0 + 0.5 * y as f32;
        assert!(
            (got - want).abs() < 1.0,
            "level at ({x},{y}): {got} vs {want}"
        );
    }
}

// ─── Hazards ────────────────────────────────────────────────────────────────

#[test]
fn hot_pixels_are_rejected_unless_asked_for() {
    let mut rng = StdRng::seed_from_u64(47);
    let mut f = Frame::new(512, 512, 1.5);
    f.lattice(128, 100.0, &mut rng);
    let mut px = f.render(&mut rng);
    let hot: Vec<(f64, f64)> = (0..40)
        .map(|i| (60.0 + 10.0 * i as f64, 20.0 + 128.0 * (i % 4) as f64))
        .collect();
    for &(x, y) in &hot {
        px[y as usize * f.w + x as usize] += 150.0 * READ_NOISE;
    }
    let on_hot = |cfg: &CentroidExtractionConfig| {
        let dets = positions(&run(&px, f.w, f.h, cfg));
        let (near, _) = match_truth(&hot, &dets, 1.5);
        let (stars, _) = match_truth(&f.truth, &dets, 1.5);
        assert!(stars.iter().all(|&e| e < 0.2), "stars must survive");
        near.iter().filter(|&&d| d < 1.5).count()
    };
    assert_eq!(on_hot(&CentroidExtractionConfig::default()), 0);
    let keep = CentroidExtractionConfig {
        reject_hot_pixels: false,
        ..Default::default()
    };
    assert_eq!(on_hot(&keep), hot.len());
}

#[test]
fn close_pairs_are_separated() {
    let mut rng = StdRng::seed_from_u64(53);
    let mut f = Frame::new(512, 512, 1.5);
    let flux = f.flux(100.0);
    for i in 0..16 {
        let (x, y) = (60.0 + 100.0 * (i % 4) as f64, 60.0 + 100.0 * (i / 4) as f64);
        let ang = 0.4 * i as f64;
        f.star(x, y, flux);
        f.star(x + 8.0 * ang.cos(), y + 8.0 * ang.sin(), 0.6 * flux);
    }
    let px = f.render(&mut rng);
    for binning in [Some(1), Some(2)] {
        let cfg = CentroidExtractionConfig {
            detection_binning: binning,
            ..Default::default()
        };
        let dets = positions(&run(&px, f.w, f.h, &cfg));
        assert_eq!(dets.len(), f.truth.len(), "binning {binning:?}");
        let (errs, _) = match_truth(&f.truth, &dets, 1.0);
        assert!(
            errs.iter().all(|&e| e < 0.2),
            "binning {binning:?}: {errs:?}"
        );
    }
}

#[test]
fn bright_star_wings_do_not_spawn_companions() {
    // Very bright stars with photon noise: the wings are far above
    // threshold and noisy, but each star must come out once.
    let mut rng = StdRng::seed_from_u64(59);
    let mut f = Frame::new(512, 512, 1.5);
    let flux = f.flux(20_000.0);
    for i in 0..9 {
        f.star(
            90.3 + 160.0 * (i % 3) as f64,
            85.6 + 160.0 * (i / 3) as f64,
            flux,
        );
    }
    let normal = Normal::new(0.0_f32, 1.0).unwrap();
    let px: Vec<f32> = f
        .signal
        .iter()
        .map(|&s| BG + s + (READ_NOISE * READ_NOISE + s).sqrt() * normal.sample(&mut rng))
        .collect();
    for binning in [Some(1), Some(2)] {
        let cfg = CentroidExtractionConfig {
            detection_binning: binning,
            ..Default::default()
        };
        let dets = positions(&run(&px, f.w, f.h, &cfg));
        assert_eq!(dets.len(), 9, "binning {binning:?}: {dets:?}");
    }
}

#[test]
fn saturated_stars_are_located_with_or_without_a_level() {
    const SAT: f32 = 4000.0;
    let mut rng = StdRng::seed_from_u64(61);
    let mut f = Frame::new(600, 600, 1.5);
    let peak_frac = pixel_frac(0, 0.0, 1.5).powi(2);
    for (i, factor) in [2.0, 10.0, 50.0, 2.0, 10.0, 50.0, 2.0, 10.0, 50.0]
        .iter()
        .enumerate()
    {
        let (x, y) = (
            100.4 + 200.0 * (i % 3) as f64,
            99.7 + 200.0 * (i / 3) as f64,
        );
        f.star(x, y, factor * (SAT - BG) as f64 / peak_frac);
    }
    let px: Vec<f32> = f.render(&mut rng).iter().map(|p| p.min(SAT)).collect();
    for level in [Some(SAT), None] {
        let cfg = CentroidExtractionConfig {
            saturation_level: level,
            ..Default::default()
        };
        let res = run(&px, f.w, f.h, &cfg);
        // Each saturated star once, as the brightest detections; at most a
        // stray threshold-level noise peak besides.
        let dets = positions(&res);
        assert!(dets.len() <= f.truth.len() + 1, "level {level:?}: {dets:?}");
        let (errs, _) = match_truth(&f.truth, &dets[..f.truth.len()], 1.0);
        assert!(errs.iter().all(|&e| e < 0.1), "level {level:?}: {errs:?}");
    }
}

#[test]
fn trails_and_large_blobs_are_dropped() {
    let mut rng = StdRng::seed_from_u64(67);
    let mut f = Frame::new(512, 512, 1.5);
    f.star(100.0, 400.0, f.flux(200.0));
    let truth = f.truth.clone();
    // A satellite trail: stars strung along a diagonal, one per pixel.
    for i in 0..300 {
        f.star(80.0 + i as f64, 60.0 + 0.6 * i as f64, f.flux(60.0));
    }
    // A broad glow, σ = 25 px.
    let mut px = f.render(&mut rng);
    for r in 0..f.h {
        for c in 0..f.w {
            let d2 = (c as f32 - 400.0).powi(2) + (r as f32 - 380.0).powi(2);
            px[r * f.w + c] += 400.0 * (-d2 / (2.0 * 25.0 * 25.0)).exp();
        }
    }
    let cfg = CentroidExtractionConfig {
        max_pixels: 2000,
        ..Default::default()
    };
    let dets = positions(&run(&px, f.w, f.h, &cfg));
    let (errs, stray) = match_truth(&truth, &dets, 2.0);
    assert!(errs[0] < 0.1, "the real star must survive");
    assert_eq!(stray, 0, "trail or glow leaked through: {dets:?}");
}

#[test]
fn non_finite_pixels_are_tolerated() {
    let mut rng = StdRng::seed_from_u64(71);
    let mut f = Frame::new(400, 400, 1.5);
    f.lattice(80, 100.0, &mut rng);
    let mut px = f.render(&mut rng);
    // A dead column, a NaN block in a corner, and scattered infinities well
    // away from the stars (which sit near cell centers).
    for r in 0..f.h {
        px[r * f.w + 2] = f32::NAN;
    }
    for r in 0..10 {
        for c in 390..400 {
            px[r * f.w + c] = f32::NAN;
        }
    }
    for i in 0..5 {
        px[(80 * i + 5) * f.w + 80 * i + 79] = if i % 2 == 0 {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        };
    }
    for binning in [Some(1), Some(2)] {
        let cfg = CentroidExtractionConfig {
            detection_binning: binning,
            ..Default::default()
        };
        let res = run(&px, f.w, f.h, &cfg);
        for c in &res.centroids {
            assert!(c.x.is_finite() && c.y.is_finite() && c.mass.unwrap().is_finite());
            let cov = c.cov.unwrap();
            assert!(cov[(0, 0)].is_finite() && cov[(1, 1)].is_finite());
        }
        let (errs, _) = match_truth(&f.truth, &positions(&res), 1.0);
        assert!(
            errs.iter().all(|&e| e < 0.2),
            "binning {binning:?}: {errs:?}"
        );
    }
}

// ─── Interface ──────────────────────────────────────────────────────────────

#[test]
fn results_are_brightest_first_and_truncated() {
    let mut rng = StdRng::seed_from_u64(73);
    let mut f = Frame::new(400, 400, 1.5);
    for i in 0..16 {
        let snr = 30.0 * (1 + (i * 7) % 16) as f64;
        f.star(
            50.0 + 100.0 * (i % 4) as f64,
            50.0 + 100.0 * (i / 4) as f64,
            f.flux(snr),
        );
    }
    let px = f.render(&mut rng);
    let all = run(&px, f.w, f.h, &CentroidExtractionConfig::default());
    assert_eq!(all.centroids.len(), 16);
    assert!(all
        .centroids
        .windows(2)
        .all(|p| p[0].mass.unwrap() >= p[1].mass.unwrap()));
    // PSF-weighted flux is in the same units as the star's total flux.
    let brightest = all.centroids[0].mass.unwrap() as f64;
    assert!(
        (brightest / f.flux(480.0) - 1.0).abs() < 0.05,
        "{brightest}"
    );
    let cfg = CentroidExtractionConfig {
        max_centroids: Some(5),
        ..Default::default()
    };
    let top = run(&px, f.w, f.h, &cfg);
    assert_eq!(top.centroids, all.centroids[..5]);
}

#[test]
fn coordinates_are_relative_to_the_image_center() {
    // One star exactly at the geometric center of an even-sized frame, one
    // offset: (W − 1)/2 is the origin, +x right, +y down.
    let mut rng = StdRng::seed_from_u64(79);
    let mut f = Frame::new(200, 160, 1.5);
    f.star(99.5, 79.5, f.flux(2000.0));
    f.star(149.5, 49.5, f.flux(1000.0));
    let res = run(
        &f.render(&mut rng),
        f.w,
        f.h,
        &CentroidExtractionConfig::default(),
    );
    assert_eq!(res.centroids.len(), 2);
    let (a, b) = (&res.centroids[0], &res.centroids[1]);
    assert!(a.x.abs() < 0.02 && a.y.abs() < 0.02, "{a:?}");
    assert!(
        (b.x - 50.0).abs() < 0.02 && (b.y + 30.0).abs() < 0.02,
        "{b:?}"
    );
    assert_eq!((res.image_width, res.image_height), (200, 160));
}

#[test]
fn edge_stars_and_border_margin() {
    let mut rng = StdRng::seed_from_u64(83);
    let mut f = Frame::new(300, 300, 1.5);
    let flux = f.flux(300.0);
    f.star(2.0, 150.0, flux); // window cannot fit: always dropped
    f.star(12.0, 150.0, flux); // inside the frame, inside a 20 px margin
    f.star(150.0, 150.0, flux);
    let px = f.render(&mut rng);
    let n = |margin| {
        let cfg = CentroidExtractionConfig {
            border_margin: margin,
            ..Default::default()
        };
        run(&px, f.w, f.h, &cfg).centroids.len()
    };
    assert_eq!(n(0), 2);
    assert_eq!(n(20), 1);
}

#[test]
fn odd_sizes_and_small_frames_work() {
    let mut rng = StdRng::seed_from_u64(89);
    for (w, h) in [(301usize, 257usize), (65, 64), (33, 47)] {
        let mut f = Frame::new(w, h, 1.5);
        f.star(w as f64 / 2.0 + 0.3, h as f64 / 2.0 - 0.2, f.flux(300.0));
        let px = f.render(&mut rng);
        for binning in [Some(1), Some(2)] {
            let cfg = CentroidExtractionConfig {
                detection_binning: binning,
                ..Default::default()
            };
            let dets = positions(&run(&px, w, h, &cfg));
            let (errs, _) = match_truth(&f.truth, &dets, 1.0);
            assert!(errs[0] < 0.1, "{w}x{h} binning {binning:?}: {dets:?}");
        }
    }
    // Too small for any window: no panic, no detections.
    let cfg = CentroidExtractionConfig {
        detection_binning: Some(1),
        ..Default::default()
    };
    assert!(run(&[1.0; 16], 4, 4, &cfg).centroids.is_empty());
}

#[test]
fn invalid_input_is_reported() {
    let px = vec![0.0_f32; 64 * 64];
    let bad = |cfg: CentroidExtractionConfig| {
        assert!(matches!(
            extract_centroids_from_raw(&px, 64, 64, &cfg),
            Err(Error::InvalidInput(_))
        ));
    };
    let d = CentroidExtractionConfig::default;
    for v in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        bad(CentroidExtractionConfig {
            sigma_threshold: v,
            ..d()
        });
    }
    for v in [0.0, -1.0, f32::NAN, 10.5] {
        bad(CentroidExtractionConfig {
            psf_sigma: v,
            ..d()
        });
    }
    bad(CentroidExtractionConfig {
        bg_block_size: 0,
        ..d()
    });
    bad(CentroidExtractionConfig {
        detection_binning: Some(3),
        ..d()
    });
    bad(CentroidExtractionConfig {
        detection_binning: Some(0),
        ..d()
    });
    for v in [0.5, 0.0, -2.0, f32::NAN] {
        bad(CentroidExtractionConfig {
            max_elongation: Some(v),
            ..d()
        });
    }
    bad(CentroidExtractionConfig {
        saturation_level: Some(f32::NAN),
        ..d()
    });
    // Wrong buffer length, and frames too small to detect on.
    assert!(extract_centroids_from_raw(&px, 64, 63, &d()).is_err());
    assert!(extract_centroids_from_raw(&px[..4], 2, 2, &d()).is_err());
    assert!(extract_centroids_from_raw(&px[..25], 5, 5, &d()).is_err());
}

#[test]
fn oversized_background_block_is_one_block() {
    // A block larger than the frame must still sample the whole frame, not
    // a single pixel.
    let mut rng = StdRng::seed_from_u64(107);
    let mut f = Frame::new(200, 150, 1.5);
    f.lattice(50, 100.0, &mut rng);
    let px = f.render(&mut rng);
    let cfg = CentroidExtractionConfig {
        bg_block_size: 4096,
        ..Default::default()
    };
    let res = run(&px, f.w, f.h, &cfg);
    assert!(
        (res.background_mean - BG).abs() < 1.0,
        "{}",
        res.background_mean
    );
    // One block's worth of samples: a coarse σ, but a real one.
    assert!((res.background_sigma / READ_NOISE - 1.0).abs() < 0.4);
    let (errs, stray) = match_truth(&f.truth, &positions(&res), 1.5);
    assert!(errs.iter().all(|&e| e < 0.3));
    assert_eq!(stray, 0);
}

#[test]
fn extreme_settings_do_not_panic() {
    let mut rng = StdRng::seed_from_u64(97);
    let mut f = Frame::new(128, 96, 1.5);
    f.lattice(32, 200.0, &mut rng);
    let px = f.render(&mut rng);
    for psf_sigma in [0.2, 0.5, 4.0, 10.0] {
        for bg_block_size in [1, 7, 4096] {
            for detection_binning in [Some(1), Some(2)] {
                let cfg = CentroidExtractionConfig {
                    psf_sigma,
                    bg_block_size,
                    detection_binning,
                    max_pixels: 0,
                    ..Default::default()
                };
                let _ = run(&px, f.w, f.h, &cfg);
            }
        }
    }
}

#[test]
fn extractor_reuse_matches_fresh_calls() {
    let mut rng = StdRng::seed_from_u64(101);
    let cfg = CentroidExtractionConfig::default();
    let mut ex = CentroidExtractor::new();
    // Change the frame size between calls so buffers shrink and grow.
    for (w, h) in [(256usize, 192usize), (400, 400), (128, 300), (400, 400)] {
        let mut f = Frame::new(w, h, 1.5);
        f.lattice(40, 80.0, &mut rng);
        let px = f.render(&mut rng);
        let fresh = run(&px, w, h, &cfg);
        let reused = ex.extract_from_raw(&px, w as u32, h as u32, &cfg).unwrap();
        assert_eq!(fresh.centroids, reused.centroids, "{w}x{h}");
        assert_eq!(fresh.background_sigma, reused.background_sigma);
    }
}

#[test]
fn image_entry_point_matches_raw() {
    let mut rng = StdRng::seed_from_u64(103);
    let mut f = Frame::new(256, 200, 1.5);
    f.lattice(50, 300.0, &mut rng);
    let px16: Vec<u16> = f
        .render(&mut rng)
        .iter()
        .map(|p| p.round() as u16)
        .collect();
    let raw: Vec<f32> = px16.iter().map(|&p| p as f32).collect();
    let img = image::DynamicImage::ImageLuma16(
        image::ImageBuffer::from_raw(f.w as u32, f.h as u32, px16).unwrap(),
    );
    let cfg = CentroidExtractionConfig::default();
    let a = extract_centroids_from_image(&img, &cfg).unwrap();
    let b = run(&raw, f.w, f.h, &cfg);
    assert_eq!(a.centroids, b.centroids);
    assert_eq!(a.centroids.len(), f.truth.len());
}

// ─── Measurement ────────────────────────────────────────────────────────────

#[test]
fn windowed_centroid_is_exact_without_noise() {
    let mut f = Frame::new(64, 64, 1.5);
    let win = Window::new(1.5);
    for (i, (dx, dy)) in [(0.0, 0.0), (0.25, -0.4), (0.5, 0.5), (-0.37, 0.13)]
        .iter()
        .enumerate()
    {
        f.signal.fill(0.0);
        f.truth.clear();
        f.star(30.0 + dx, 33.0 + dy, 1000.0);
        let m = measure(&f.signal, (64, 64), 0.0, (30.0, 33.0), &win, 2.0).unwrap();
        assert!(m.converged, "case {i}");
        let err = ((m.x - 30.0 - dx).powi(2) + (m.y - 33.0 - dy).powi(2)).sqrt();
        assert!(err < 2e-3, "case {i}: error {err:.5}");
        assert!(
            (m.flux / 1000.0 - 1.0).abs() < 0.03,
            "case {i}: flux {}",
            m.flux
        );
        // A background error (here 6% of the star's peak, far more than
        // the block statistics ever miss by) barely moves the position:
        // the window is symmetric but for its truncation.
        let off = measure(&f.signal, (64, 64), -4.0, (30.0, 33.0), &win, 2.0).unwrap();
        assert!(
            (off.x - m.x).abs() < 2e-3 && (off.y - m.y).abs() < 2e-3,
            "case {i}"
        );
    }
}

#[test]
fn window_shape_distinguishes_hot_pixels_and_broad_blobs() {
    let win = Window::new(1.5);
    let s2 = win.sigma2;
    let mut hot = vec![0.0_f32; 64 * 64];
    hot[32 * 64 + 32] = 500.0;
    let m = measure(&hot, (64, 64), 0.0, (32.0, 32.0), &win, 2.0).unwrap();
    assert!(m.mean_var() < 0.15 * s2, "hot pixel var {}", m.mean_var());

    let mut f = Frame::new(64, 64, 1.5);
    f.star(32.0, 32.0, 1000.0);
    let m = measure(&f.signal, (64, 64), 0.0, (32.0, 32.0), &win, 2.0).unwrap();
    assert!(
        (m.mean_var() / (0.5 * s2) - 1.0).abs() < 0.1,
        "matched var {}",
        m.mean_var()
    );
    let (shape, elong) = unwindowed_shape(m.moments, s2);
    assert!(
        (shape[0] / (2.25 + 1.0 / 12.0) - 1.0).abs() < 0.05,
        "{shape:?}"
    );
    assert!(elong < 1.02);

    let flat = vec![100.0_f32; 64 * 64];
    let m = measure(&flat, (64, 64), 0.0, (32.0, 32.0), &win, 2.0).unwrap();
    assert!(m.mean_var() > 0.8 * s2, "plateau var {}", m.mean_var());
}

#[test]
fn measure_rejects_windows_off_the_frame_and_empty_flux() {
    let win = Window::new(1.5);
    let img = vec![10.0_f32; 40 * 40];
    assert!(measure(&img, (40, 40), 0.0, (3.0, 20.0), &win, 2.0).is_none());
    assert!(measure(&img, (40, 40), 0.0, (20.0, 37.0), &win, 2.0).is_none());
    // Nothing above the background.
    assert!(measure(&img, (40, 40), 10.0, (20.0, 20.0), &win, 2.0).is_none());
    assert!(measure(&img, (40, 40), 50.0, (20.0, 20.0), &win, 2.0).is_none());
}
