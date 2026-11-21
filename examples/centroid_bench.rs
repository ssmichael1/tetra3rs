//! Ground-truth benchmark for centroid extraction.
//!
//! Renders synthetic frames of pixel-integrated Gaussian stars at known
//! sub-pixel positions and scores the extractor on completeness, position
//! error (against the read-noise Cramér–Rao bound, `√2·σ_psf / SNR` per
//! axis), false detections and run time, plus hazard scenes: a background
//! gradient with spatially varying noise, hot pixels, close pairs and
//! saturated stars. `SNR` is the matched-filter signal-to-noise ratio.
//!
//! ```sh
//! cargo run --release --features image,parallel --example centroid_bench
//! T3_PSF=2.5 T3_ONLY=full cargo run --release --features image --example centroid_bench
//! ```
//!
//! Each configuration (`auto`, `full` = full-resolution detection, `binned`,
//! `auto-sat` = with a saturation level) is run on every scene. Columns:
//! `rms/axis` is the per-axis RMS position error of the stars found within
//! 1.5 px of truth; `robust σ` the same from the median error; `cov σ` what
//! the reported `Centroid::cov` predicts. A `false` detection is one not
//! matched to any true star (a second detection on a star counts).
//!
//! Environment:
//! - `T3_PSF` true PSF σ in px (default 1.5); `T3_CFG_PSF` the `psf_sigma`
//!   given to the extractor (default: the true value)
//! - `T3_SHOT=1` adds star photon noise to the SNR scene (gain 1 e⁻/ADU)
//! - `T3_SIZE` frame side (default 2048); `T3_FRAMES` frames per scene
//!   (default 3); `T3_ONLY=name` runs one configuration
//! - `T3_ELONG` `max_elongation` (default 3.0; 0 disables)
//! - `T3_CHECKSUM=1` prints a checksum of every result (to compare builds);
//!   `T3_FALSEMAP=1` lists each false detection on stderr

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::time::Instant;
use tetra3::centroid_extraction::*;
use tetra3::Centroid;

const READ_NOISE: f32 = 5.0;
const BG_LEVEL: f32 = 100.0;
const SNR_BINS: [f32; 9] = [4.0, 5.0, 6.0, 8.0, 12.0, 20.0, 50.0, 200.0, 1000.0];

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

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

/// Fraction of a unit-flux 1-D Gaussian (center `mu`, width `sigma`) that
/// lands in pixel `i` (which spans `i − 0.5 .. i + 0.5`).
fn pixel_frac(i: i64, mu: f64, sigma: f64) -> f64 {
    let s = sigma * std::f64::consts::SQRT_2;
    0.5 * (erf((i as f64 + 0.5 - mu) / s) - erf((i as f64 - 0.5 - mu) / s))
}

#[derive(Clone, Copy)]
struct Truth {
    x: f64,
    y: f64,
    /// Total flux in ADU.
    flux: f64,
    /// Index into the scene's group labels.
    group: usize,
}

struct Scene {
    name: String,
    w: usize,
    h: usize,
    pixels: Vec<f32>,
    truth: Vec<Truth>,
    groups: Vec<String>,
    /// Positions a detection may legitimately sit on without being a star
    /// (unused) or must not sit on (hot pixels).
    hot: Vec<(usize, usize)>,
}

struct SceneBuilder {
    w: usize,
    h: usize,
    sigma_psf: f64,
    signal: Vec<f32>,
    truth: Vec<Truth>,
    hot: Vec<(usize, usize)>,
}

impl SceneBuilder {
    fn new(w: usize, h: usize, sigma_psf: f64) -> Self {
        Self {
            w,
            h,
            sigma_psf,
            signal: vec![0.0; w * h],
            truth: Vec::new(),
            hot: Vec::new(),
        }
    }

    fn add_star(&mut self, x: f64, y: f64, flux: f64, group: usize) {
        let r = (6.0 * self.sigma_psf).ceil() as i64 + 1;
        let (xi, yi) = (x.round() as i64, y.round() as i64);
        let fx: Vec<f64> = (xi - r..=xi + r)
            .map(|i| pixel_frac(i, x, self.sigma_psf))
            .collect();
        for row in yi - r..=yi + r {
            if row < 0 || row >= self.h as i64 {
                continue;
            }
            let fy = pixel_frac(row, y, self.sigma_psf);
            for (k, col) in (xi - r..=xi + r).enumerate() {
                if col < 0 || col >= self.w as i64 {
                    continue;
                }
                self.signal[row as usize * self.w + col as usize] += (flux * fy * fx[k]) as f32;
            }
        }
        self.truth.push(Truth { x, y, flux, group });
    }

    /// Flux giving a matched-filter SNR of `snr` against read noise.
    fn flux_for_snr(&self, snr: f32) -> f64 {
        snr as f64 * READ_NOISE as f64 * 2.0 * std::f64::consts::PI.sqrt() * self.sigma_psf
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        self,
        name: &str,
        groups: Vec<String>,
        rng: &mut StdRng,
        gradient: f32,
        shot: bool,
        saturation: Option<f32>,
    ) -> Scene {
        let (w, h) = (self.w, self.h);
        let normal = Normal::new(0.0_f32, 1.0).unwrap();
        let mut pixels = vec![0.0_f32; w * h];
        for row in 0..h {
            for col in 0..w {
                let i = row * w + col;
                // Linear tilt plus a radial vignette-like falloff.
                let (u, v) = (col as f32 / w as f32, row as f32 / h as f32);
                let r2 = (u - 0.5).powi(2) + (v - 0.5).powi(2);
                let bg = BG_LEVEL + gradient * (u + 0.5 * v - 1.2 * r2);
                let s = self.signal[i];
                let var = READ_NOISE * READ_NOISE
                    + if shot {
                        s + (bg - BG_LEVEL).max(0.0)
                    } else {
                        0.0
                    };
                let mut p = bg + s + var.sqrt() * normal.sample(rng);
                if let Some(sat) = saturation {
                    p = p.min(sat);
                }
                pixels[i] = p;
            }
        }
        for &(c, r) in &self.hot {
            pixels[r * w + c] += READ_NOISE * (30.0 + 300.0 * rng.random::<f32>());
        }
        Scene {
            name: name.to_string(),
            w,
            h,
            pixels,
            truth: self.truth,
            groups,
            hot: self.hot,
        }
    }
}

/// Isolated stars on a jittered lattice, cycling through the SNR bins.
fn snr_scene(size: usize, sigma_psf: f64, seed: u64, gradient: f32, shot: bool) -> Scene {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut b = SceneBuilder::new(size, size, sigma_psf);
    let cell = 40usize;
    let n = size / cell;
    let mut k = (seed as usize) % SNR_BINS.len();
    for gy in 0..n {
        for gx in 0..n {
            let x = (gx * cell) as f64 + 12.0 + 16.0 * rng.random::<f64>();
            let y = (gy * cell) as f64 + 12.0 + 16.0 * rng.random::<f64>();
            let bin = k % SNR_BINS.len();
            // Decorrelate bin from lattice position.
            k += 1 + (rng.random::<u32>() % 3) as usize;
            b.add_star(x, y, b.flux_for_snr(SNR_BINS[bin]), bin);
        }
    }
    let groups = SNR_BINS.iter().map(|s| format!("snr{s:>5}")).collect();
    let name = if gradient > 0.0 { "gradient" } else { "snr" };
    b.finish(name, groups, &mut rng, gradient, shot, None)
}

/// Stars at SNR 50 with 400 hot pixels scattered between them.
fn hot_scene(size: usize, sigma_psf: f64, seed: u64) -> Scene {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut b = SceneBuilder::new(size, size, sigma_psf);
    let cell = 80usize;
    let n = size / cell;
    for gy in 0..n {
        for gx in 0..n {
            let x = (gx * cell) as f64 + 12.0 + 16.0 * rng.random::<f64>();
            let y = (gy * cell) as f64 + 12.0 + 16.0 * rng.random::<f64>();
            b.add_star(x, y, b.flux_for_snr(50.0), 0);
        }
    }
    for _ in 0..400 {
        let gx = rng.random::<u32>() as usize % n;
        let gy = rng.random::<u32>() as usize % n;
        // Far corner of the cell from the star.
        b.hot.push((
            gx * cell + 50 + rng.random::<u32>() as usize % 20,
            gy * cell + 50 + rng.random::<u32>() as usize % 20,
        ));
    }
    b.finish("hot", vec!["snr 50".into()], &mut rng, 0.0, false, None)
}

const PAIR_SEPS: [f64; 6] = [2.0, 3.0, 4.0, 5.0, 7.0, 10.0];

/// Equal-brightness pairs (SNR 100 each) at a range of separations.
fn pair_scene(size: usize, sigma_psf: f64, seed: u64) -> Scene {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut b = SceneBuilder::new(size, size, sigma_psf);
    let cell = 64usize;
    let n = size / cell;
    let mut k = 0usize;
    for gy in 0..n {
        for gx in 0..n {
            let x = (gx * cell) as f64 + 28.0 + 8.0 * rng.random::<f64>();
            let y = (gy * cell) as f64 + 28.0 + 8.0 * rng.random::<f64>();
            let g = k % PAIR_SEPS.len();
            k += 1;
            let ang = rng.random::<f64>() * std::f64::consts::TAU;
            let (dx, dy) = (
                0.5 * PAIR_SEPS[g] * ang.cos(),
                0.5 * PAIR_SEPS[g] * ang.sin(),
            );
            let f = b.flux_for_snr(100.0);
            b.add_star(x - dx, y - dy, f, g);
            b.add_star(x + dx, y + dy, 0.6 * f, g);
        }
    }
    let groups = PAIR_SEPS.iter().map(|s| format!("sep{s:>4}")).collect();
    b.finish("pairs", groups, &mut rng, 0.0, false, None)
}

const SAT_FACTORS: [f64; 4] = [0.5, 2.0, 10.0, 50.0];

/// Bright stars whose peak is `factor ×` the saturation level (4000 ADU).
fn saturated_scene(size: usize, sigma_psf: f64, seed: u64) -> Scene {
    const SAT: f64 = 4000.0;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut b = SceneBuilder::new(size, size, sigma_psf);
    let cell = 128usize;
    let n = size / cell;
    let mut k = 0usize;
    for gy in 0..n {
        for gx in 0..n {
            let x = (gx * cell) as f64 + 56.0 + 16.0 * rng.random::<f64>();
            let y = (gy * cell) as f64 + 56.0 + 16.0 * rng.random::<f64>();
            let g = k % SAT_FACTORS.len();
            k += 1;
            // Peak of a centered pixel-integrated Gaussian ≈ flux·erf(…)².
            let peak_frac = pixel_frac(0, 0.0, sigma_psf).powi(2);
            b.add_star(
                x,
                y,
                SAT_FACTORS[g] * (SAT - BG_LEVEL as f64) / peak_frac,
                g,
            );
        }
    }
    let groups = SAT_FACTORS
        .iter()
        .map(|s| format!("peak {s:>4}x"))
        .collect();
    b.finish("saturated", groups, &mut rng, 0.0, true, Some(SAT as f32))
}

#[derive(Default, Clone)]
struct GroupScore {
    n_truth: usize,
    n_found: usize,
    /// Σ(dx² + dy²) over found.
    sq: f64,
    /// Radial errors of found.
    errs: Vec<f64>,
    /// Σ predicted per-axis variance (from `cov`) over found.
    pred: f64,
    crlb: f64,
}

#[derive(Default)]
struct Score {
    groups: Vec<GroupScore>,
    false_pos: usize,
    on_hot: usize,
    n_det: usize,
}

const MATCH_RADIUS: f64 = 1.5;

fn score(scene: &Scene, sigma_psf: f64, dets: &[Centroid], acc: &mut Score) {
    let (cx, cy) = ((scene.w - 1) as f64 / 2.0, (scene.h - 1) as f64 / 2.0);
    let masses: Vec<f32> = dets.iter().map(|c| c.mass.unwrap_or(0.0)).collect();
    let pred: Vec<f64> = dets
        .iter()
        .map(|c| {
            c.cov
                .map_or(f64::NAN, |m| 0.5 * (m[(0, 0)] + m[(1, 1)]) as f64)
        })
        .collect();
    let dets_mass = |i: usize| masses[i];
    let dets: Vec<(f64, f64)> = dets
        .iter()
        .map(|c| (c.x as f64 + cx, c.y as f64 + cy))
        .collect();
    // Bucket detections on a coarse grid for nearest-neighbour lookup.
    let cell = 8.0;
    let gw = (scene.w as f64 / cell) as usize + 1;
    let gh = (scene.h as f64 / cell) as usize + 1;
    let mut grid: Vec<Vec<u32>> = vec![Vec::new(); gw * gh];
    for (i, &(x, y)) in dets.iter().enumerate() {
        if x < 0.0 || y < 0.0 || !x.is_finite() || !y.is_finite() {
            continue;
        }
        let (gx, gy) = (
            ((x / cell) as usize).min(gw - 1),
            ((y / cell) as usize).min(gh - 1),
        );
        grid[gy * gw + gx].push(i as u32);
    }
    let near = |x: f64, y: f64, radius: f64| -> Option<(usize, f64)> {
        let (gx, gy) = ((x / cell) as isize, (y / cell) as isize);
        let mut best: Option<(usize, f64)> = None;
        for yy in gy - 1..=gy + 1 {
            for xx in gx - 1..=gx + 1 {
                if xx < 0 || yy < 0 || xx >= gw as isize || yy >= gh as isize {
                    continue;
                }
                for &i in &grid[yy as usize * gw + xx as usize] {
                    let (dx, dy) = (dets[i as usize].0 - x, dets[i as usize].1 - y);
                    let d = (dx * dx + dy * dy).sqrt();
                    if d <= radius && best.is_none_or(|b| d < b.1) {
                        best = Some((i as usize, d));
                    }
                }
            }
        }
        best
    };
    if acc.groups.len() < scene.groups.len() {
        acc.groups.resize(scene.groups.len(), GroupScore::default());
    }
    let mut used = vec![false; dets.len()];
    for t in &scene.truth {
        let g = &mut acc.groups[t.group];
        g.n_truth += 1;
        // Read-noise CRLB per axis: σ_x = √(8π)·σ_psf²·σ_n / F.
        g.crlb = (8.0 * std::f64::consts::PI).sqrt() * sigma_psf * sigma_psf * READ_NOISE as f64
            / t.flux;
        if let Some((i, d)) = near(t.x, t.y, MATCH_RADIUS) {
            g.n_found += 1;
            g.sq += d * d;
            g.errs.push(d);
            g.pred += pred[i];
            used[i] = true;
        }
    }
    acc.n_det += dets.len();
    for (i, &(x, y)) in dets.iter().enumerate() {
        if used[i] {
            continue;
        }
        // Unmatched: a second detection on a star still counts as false
        // (a duplicate the solver would consume as a star).
        acc.false_pos += 1;
        if std::env::var_os("T3_FALSEMAP").is_some() {
            eprintln!(
                "FALSE {} {:7.1} {:7.1} mass {:?}",
                scene.name,
                x,
                y,
                dets_mass(i)
            );
        }
        if scene
            .hot
            .iter()
            .any(|&(c, r)| (x - c as f64).abs() < 1.5 && (y - r as f64).abs() < 1.5)
        {
            acc.on_hot += 1;
        }
    }
}

type Extractor = Box<dyn Fn(&[f32], u32, u32) -> CentroidExtractionResult>;

fn extractors(sigma_psf: f32) -> Vec<(&'static str, Extractor)> {
    let make = |cfg: CentroidExtractionConfig| -> Extractor {
        let ex = std::cell::RefCell::new(CentroidExtractor::new());
        Box::new(move |p, w, h| ex.borrow_mut().extract_from_raw(p, w, h, &cfg).unwrap())
    };
    let base = CentroidExtractionConfig {
        psf_sigma: env_f32("T3_CFG_PSF", sigma_psf),
        max_elongation: Some(env_f32("T3_ELONG", 3.0)).filter(|e| *e > 0.0),
        ..Default::default()
    };
    vec![
        ("auto", make(base.clone())),
        (
            "full",
            make(CentroidExtractionConfig {
                detection_binning: Some(1),
                ..base.clone()
            }),
        ),
        (
            "binned",
            make(CentroidExtractionConfig {
                detection_binning: Some(2),
                ..base.clone()
            }),
        ),
        (
            "auto-sat",
            make(CentroidExtractionConfig {
                saturation_level: Some(4000.0),
                ..base.clone()
            }),
        ),
    ]
}

fn main() {
    let sigma_psf = env_f32("T3_PSF", 1.5);
    let shot = env_f32("T3_SHOT", 0.0) > 0.0;
    let size = env_f32("T3_SIZE", 2048.0) as usize;
    let frames = env_f32("T3_FRAMES", 3.0) as u64;
    let only = std::env::var("T3_ONLY").ok();

    println!(
        "PSF σ = {sigma_psf} px, read noise {READ_NOISE}, shot noise {}, {size}² × {frames} frames, parallel = {}",
        if shot { "on" } else { "off" },
        cfg!(feature = "parallel")
    );

    type SceneFn = Box<dyn Fn(u64) -> Scene>;
    let sp = sigma_psf as f64;
    let scene_fns: Vec<SceneFn> = vec![
        Box::new(move |s| snr_scene(size, sp, s, 0.0, shot)),
        Box::new(move |s| snr_scene(size, sp, s, 600.0, true)),
        Box::new(move |s| hot_scene(size, sp, 100 + s)),
        Box::new(move |s| pair_scene(size, sp, 200 + s)),
        Box::new(move |s| saturated_scene(size, sp, 300 + s)),
    ];
    let scenes: Vec<Vec<Scene>> = scene_fns
        .iter()
        .map(|f| (0..frames).map(f).collect())
        .collect();

    for (name, ex) in extractors(sigma_psf) {
        if only.as_deref().is_some_and(|o| o != name) {
            continue;
        }
        println!("\n══ {name} ══");
        for set in &scenes {
            let mut acc = Score::default();
            let mut best_ms = f64::INFINITY;
            let (mut psf_est, mut bg_sigma) = (None, 0.0);
            let mut checksum = 0u64;
            for scene in set {
                let (w, h) = (scene.w as u32, scene.h as u32);
                let mut res = None;
                for _ in 0..7 {
                    let t = Instant::now();
                    res = Some(ex(&scene.pixels, w, h));
                    best_ms = best_ms.min(t.elapsed().as_secs_f64() * 1e3);
                }
                let res = res.unwrap();
                for c in &res.centroids {
                    for v in [c.x, c.y, c.mass.unwrap_or(0.0)] {
                        checksum = checksum.rotate_left(5) ^ v.to_bits() as u64;
                    }
                }
                psf_est = res.psf_sigma_estimate;
                bg_sigma = res.background_sigma;
                score(scene, sp, &res.centroids, &mut acc);
            }
            let s0 = &set[0];
            println!(
                "── {:<9} {:6.2} ms   detections {:>6}   false {:>5}   on hot {:>3}   psf est {:?}   noise {:.2}",
                s0.name, best_ms, acc.n_det, acc.false_pos, acc.on_hot, psf_est, bg_sigma
            );
            if std::env::var_os("T3_CHECKSUM").is_some() {
                println!("   checksum {checksum:016x}");
            }
            for (g, label) in acc.groups.iter().zip(&s0.groups) {
                let mut errs = g.errs.clone();
                errs.sort_by(|a, b| a.total_cmp(b));
                let med = errs.get(errs.len() / 2).copied().unwrap_or(f64::NAN);
                let bad = errs.iter().filter(|&&e| e > 0.5).count();
                // Per-axis RMS, and a robust per-axis σ from the median
                // radial error (Rayleigh: median = 1.1774 σ).
                let rms = (g.sq / (2.0 * g.n_found.max(1) as f64)).sqrt();
                let robust = med / 1.1774;
                println!(
                    "   {label}  found {:>4}/{:<4} {:5.1}%  rms/axis {:.4}  robust σ {:.4}  ({:4.2}× CRLB {:.4})  >0.5px {:>3}  cov σ {:.4}",
                    g.n_found,
                    g.n_truth,
                    100.0 * g.n_found as f64 / g.n_truth.max(1) as f64,
                    rms,
                    robust,
                    robust / g.crlb,
                    g.crlb,
                    bad,
                    (g.pred / g.n_found.max(1) as f64).sqrt()
                );
            }
        }
    }
}
