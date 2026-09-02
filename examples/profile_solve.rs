//! Leaf-level solver profiler.
//!
//! Builds a 10°-FOV database once, then solves many random orientations and
//! reports where time goes inside the pattern search (image-side edges, catalog-side
//! edges, SVD, verification query/match, wcs_refine) plus operation counts.
//!
//! Run with the `profile` feature for the per-bucket timing breakdown:
//!
//! ```sh
//! cargo run --release --features profile --example profile_solve
//! cargo run --release --features profile --example profile_solve -- 5000   # n trials
//! ```
//!
//! Without the feature it still reports wall-clock solve time (no breakdown).

use std::time::Instant;

use numeris::{Matrix3, Vector3};
use tetra3::{Centroid, SolveConfig, SolverDatabase};

#[path = "../tests/common/mod.rs"]
mod common;
use common::{rotation_from_ra_dec_roll, Rng};

fn generate_centroids(
    db: &SolverDatabase,
    rot: &Matrix3<f32>,
    half_fov: f32,
    pixel_scale: f32,
    rng: &mut Rng,
) -> Vec<Centroid> {
    let mags: Vec<f32> = db.star_catalog.stars().iter().map(|s| s.mag).collect();
    common::project_field(
        &db.star_vectors,
        &mags,
        rot,
        half_fov,
        pixel_scale,
        0.0,
        rng,
    )
}

fn main() {
    let n_trials: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);

    let catalog_path = "data/gaia_merged.bin";
    if !std::path::Path::new(catalog_path).exists() {
        eprintln!("missing {catalog_path} — run from the crate root with the catalog present");
        std::process::exit(1);
    }

    // T3_FOV_DEG=x sets the camera FOV (default 10°); the database's max FOV
    // scales with it (1.2×, as the default 12° does for 10°).
    let fov_deg: f32 = std::env::var("T3_FOV_DEG")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);
    // 10° FOV database (shared with the golden dump and statistical tests).
    let config = tetra3::GenerateDatabaseConfig {
        max_fov_deg: fov_deg * 1.2,
        ..common::profiler_db_config()
    };

    eprintln!("Building database from {catalog_path} …");
    let t_build = Instant::now();
    let db = SolverDatabase::generate_from_gaia(catalog_path, &config).expect("db generation");
    eprintln!(
        "  {} stars, {} patterns, table {} ({:.1}s)",
        db.star_catalog.len(),
        db.props.num_patterns,
        db.pattern_catalog.len(),
        t_build.elapsed().as_secs_f32()
    );

    let fov_rad = fov_deg.to_radians();
    let half_fov = fov_rad / 2.0;
    let image_width = 1024u32;
    let pixel_scale = {
        let f = (image_width as f32 / 2.0) / (fov_rad / 2.0).tan();
        1.0 / f
    };

    // T3_FOV_BIAS=x biases the solver's FOV estimate by a fractional amount
    // (e.g. 0.15 → solver is told the FOV is 15% larger than truth) while the
    // centroids are still generated at the true FOV. Exercises the FOV sweep.
    let fov_bias: f32 = std::env::var("T3_FOV_BIAS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    // T3_PATTERN_STARS=N caps the pattern-forming centroids at the N
    // brightest (SolveConfig::pattern_checking_stars; default when unset).
    let pattern_checking_stars: u32 = std::env::var("T3_PATTERN_STARS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(SolveConfig::DEFAULT_PATTERN_CHECKING_STARS);

    // T3_ABERRATION=1 sets an observer velocity (Earth-like, 30 km/s) so
    // the catalog is aberration-corrected on every solve.
    let observer_velocity_km_s = std::env::var("T3_ABERRATION")
        .ok()
        .map(|_| [10.0, -20.0, 20.0]);

    let solve_config = SolveConfig {
        fov_max_error_rad: Some(2.0_f32.to_radians()),
        match_radius: 0.01,
        match_threshold: 1e-5,
        solve_timeout_ms: Some(10_000),
        match_max_error: None,
        pattern_checking_stars,
        observer_velocity_km_s,
        ..SolveConfig::new(fov_rad * (1.0 + fov_bias), image_width, image_width)
    };

    // Scenario knobs (env vars):
    //   T3_SPURIOUS=K  append K uniform-random false centroids to each field
    //   T3_RANDOM=1    each field is ENTIRELY random centroids (forces no-match:
    //                  full combination enumeration × full FOV sweep)
    //   T3_FOV_BIAS=x  bias the solver's FOV estimate by fraction x (see below)
    //   T3_MAX_CENTROIDS=N  keep only the N brightest true centroids per field
    //   T3_PATTERN_STARS=N  cap pattern-forming centroids at the N brightest
    //   T3_ABERRATION=1     aberration-correct the catalog (observer velocity set)
    //   T3_FOV_DEG=x        camera FOV in degrees (database max FOV = 1.2×; default 10)
    //                  (sparse-field scenario; fields with fewer than 4 are
    //                  regenerated as usual)
    let spurious: usize = std::env::var("T3_SPURIOUS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let random_only = std::env::var("T3_RANDOM").is_ok();
    let max_centroids: Option<usize> = std::env::var("T3_MAX_CENTROIDS")
        .ok()
        .and_then(|s| s.parse().ok());
    let half_w_px = half_fov / pixel_scale; // image half-extent in pixels
    eprintln!(
        "Scenario: random_only={random_only}, spurious_per_field={spurious}, fov_bias={fov_bias}, max_centroids={max_centroids:?}, pattern_stars={pattern_checking_stars}"
    );

    let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);
    let add_spurious = |c: &mut Vec<Centroid>, rng: &mut Rng, k: usize| {
        for _ in 0..k {
            c.push(Centroid {
                x: (rng.unit() * 2.0 - 1.0) * half_w_px,
                y: (rng.unit() * 2.0 - 1.0) * half_w_px,
                mass: Some(rng.unit() * 5.0),
                cov: None,
            });
        }
    };

    // Pre-generate centroid sets so RNG / projection work is outside the timed
    // loop. `truths` holds each field's true boresight so solutions can be
    // checked for correctness (a wrong-attitude accept must count as a false
    // positive, not a solve).
    let mut sets: Vec<Vec<Centroid>> = Vec::with_capacity(n_trials as usize);
    let mut truths: Vec<Vector3<f32>> = Vec::with_capacity(n_trials as usize);
    while (sets.len() as u32) < n_trials {
        let ra = rng.unit() * 2.0 * std::f32::consts::PI;
        let dec = (rng.unit() * 2.0 - 1.0).asin();
        let roll = rng.unit() * 2.0 * std::f32::consts::PI;
        let rot = rotation_from_ra_dec_roll(ra, dec, roll);
        let boresight =
            Vector3::from_array([dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()]);
        let mut c = if random_only {
            Vec::new()
        } else {
            generate_centroids(&db, &rot, half_fov, pixel_scale, &mut rng)
        };
        if let Some(n) = max_centroids {
            c.sort_by(|a, b| {
                b.mass
                    .unwrap_or(f32::MIN)
                    .partial_cmp(&a.mass.unwrap_or(f32::MIN))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            c.truncate(n);
        }
        if random_only {
            add_spurious(&mut c, &mut rng, spurious.max(30));
        } else {
            add_spurious(&mut c, &mut rng, spurious);
        }
        if c.len() >= 4 {
            sets.push(c);
            truths.push(boresight);
        }
    }

    // Warm-up (page-in, branch predictors) — not measured.
    for c in sets.iter().take(20) {
        let _ = db.solve_from_centroids(c, &solve_config);
    }

    #[cfg(feature = "profile")]
    tetra3::solver::profiling::reset();

    let mut n_found = 0u32;
    let mut n_wrong = 0u32;
    let mut total_solve_ns: u128 = 0;
    let t_all = Instant::now();
    for (c, truth) in sets.iter().zip(&truths) {
        let t = Instant::now();
        let r = db.solve_from_centroids(c, &solve_config);
        total_solve_ns += t.elapsed().as_nanos();
        if let Ok(sol) = r {
            n_found += 1;
            // Check the solved boresight against truth: an accepted wrong
            // attitude (> 1° off) is a false positive, not a solve. For
            // T3_RANDOM fields the stored truth is meaningless, but any
            // accept there is a false positive by construction anyway.
            let q = sol.qicrs2cam;
            let bs = q.to_rotation_matrix().transpose() * Vector3::from_array([0.0, 0.0, 1.0]);
            let dot = (bs[0] * truth[0] + bs[1] * truth[1] + bs[2] * truth[2]).clamp(-1.0, 1.0);
            if dot.acos() > 1.0_f32.to_radians() {
                n_wrong += 1;
            }
        }
    }
    let wall = t_all.elapsed();

    println!("\n═══════════════════════════════════════════════════════════════");
    println!(
        "Profiled {} solves ({} found, {} WRONG-ATTITUDE), wall {:.3}s",
        sets.len(),
        n_found,
        n_wrong,
        wall.as_secs_f64()
    );
    println!(
        "Mean solve: {:.1} µs   (sum {:.3}s)",
        total_solve_ns as f64 / sets.len() as f64 / 1000.0,
        total_solve_ns as f64 / 1e9
    );

    #[cfg(not(feature = "profile"))]
    println!("\n(build with --features profile for the per-bucket breakdown)");

    #[cfg(feature = "profile")]
    {
        use tetra3::solver::profiling::buckets as bk;
        let snap = tetra3::solver::profiling::snapshot();

        // Ordered presentation: timed buckets first, then count-only buckets.
        const TIMED: &[&str] = &[
            bk::IMAGE_EDGES,
            bk::CAT_EDGES,
            bk::KEY_ENUM,
            bk::SVD,
            bk::VERIFY_QUERY,
            bk::VERIFY_MATCH,
            bk::WCS_REFINE,
        ];
        const COUNTS: &[&str] = &[
            bk::FOV_PASS,
            bk::COMBOS,
            bk::CANDIDATES,
            bk::RATIO_PASS,
            bk::VERIFY_QUERY_STARS,
        ];

        let get = |name: &str| -> (u128, u64) {
            snap.iter()
                .find(|(k, _, _)| *k == name)
                .map(|(_, ns, n)| (*ns, *n))
                .unwrap_or((0, 0))
        };
        let timed_total: u128 = TIMED.iter().map(|b| get(b).0).sum();

        println!("\n  Leaf timing buckets (instrumented spans):");
        println!(
            "    {:<14} {:>10} {:>7} {:>12} {:>10}",
            "bucket", "total_ms", "%timed", "calls", "ns/call"
        );
        for b in TIMED {
            let (ns, n) = get(b);
            let ms = ns as f64 / 1e6;
            let pct = if timed_total > 0 {
                100.0 * ns as f64 / timed_total as f64
            } else {
                0.0
            };
            let per = if n > 0 { ns as f64 / n as f64 } else { 0.0 };
            println!("    {b:<14} {ms:>10.2} {pct:>6.1}% {n:>12} {per:>9.0}");
        }
        println!(
            "    {:<14} {:>10.2}",
            "TIMED TOTAL",
            timed_total as f64 / 1e6
        );
        println!(
            "    (timed spans = {:.1}% of summed solve time; remainder is loop/hash/glue)",
            100.0 * timed_total as f64 / total_solve_ns as f64
        );

        println!("\n  Operation counts (totals across all solves):");
        for b in COUNTS {
            let (_, n) = get(b);
            println!(
                "    {:<22} {:>14}  ({:.1} / solve)",
                b,
                n,
                n as f64 / sets.len() as f64
            );
        }

        // wcs_refine internals — nested INSIDE the wcs_refine bucket above, so
        // shown as a share of wcs_refine (not the global timed total).
        const WCS_TIMED: &[&str] = &[
            bk::WCS_REASSOC_QUERY,
            bk::WCS_REASSOC_PROJECT,
            bk::WCS_REASSOC_MATCH,
        ];
        const WCS_COUNTS: &[&str] = &[
            bk::WCS_OUTER,
            bk::WCS_INNER,
            bk::WCS_REASSOC_CALL,
            bk::WCS_REASSOC_STARS,
        ];
        let wcs_total = get(bk::WCS_REFINE).0;
        if wcs_total > 0 {
            println!(
                "\n  wcs_refine internals (share of the {:.1} ms wcs_refine total):",
                wcs_total as f64 / 1e6
            );
            for b in WCS_TIMED {
                let (ns, n) = get(b);
                let pct = 100.0 * ns as f64 / wcs_total as f64;
                let per = if n > 0 { ns as f64 / n as f64 } else { 0.0 };
                println!(
                    "    {:<20} {:>10.2} ms {:>6.1}%  ({:>10} calls, {:.0} ns/call)",
                    b,
                    ns as f64 / 1e6,
                    pct,
                    n,
                    per
                );
            }
            let reassoc: u128 = WCS_TIMED.iter().map(|b| get(b).0).sum();
            println!(
                "    {:<20} {:>10.2} ms {:>6.1}%  (Phase-D re-association total; remainder = LS/residual/clip transcendentals)",
                "→ reassoc subtotal",
                reassoc as f64 / 1e6,
                100.0 * reassoc as f64 / wcs_total as f64,
            );
            for b in WCS_COUNTS {
                let (_, n) = get(b);
                println!(
                    "    {:<20} {:>14}  ({:.1} / solve)",
                    b,
                    n,
                    n as f64 / sets.len() as f64
                );
            }
        }

        // The key question for the N×N precompute decision:
        let img = get(bk::IMAGE_EDGES).0 as f64;
        println!(
            "\n  → image-side edge angles = {:.1}% of timed work, {:.1}% of total solve time",
            if timed_total > 0 {
                100.0 * img / timed_total as f64
            } else {
                0.0
            },
            100.0 * img / total_solve_ns as f64,
        );
    }
    println!("═══════════════════════════════════════════════════════════════\n");
}
