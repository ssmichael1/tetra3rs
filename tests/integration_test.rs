//! Integration tests: build a database from Hipparcos, generate synthetic centroids
//! from known pointing directions, and verify the solver recovers the correct attitude.

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use tetra3::{Centroid, GenerateDatabaseConfig, SolveConfig, SolveStatus, SolverDatabase};

/// Build a small test database (wide FOV for speed) and solve a synthetic image.
#[test]
fn test_generate_and_solve_hipparcos() {
    // Initialize tracing for debug output
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();

    // ── Step 1: Generate a small database ──
    let config = GenerateDatabaseConfig {
        max_fov_deg: 20.0,
        min_fov_deg: None,             // single-scale
        star_max_magnitude: Some(6.0), // bright stars only for fast build
        pattern_max_error: 0.005,      // coarser bins for small DB
        lattice_field_oversampling: 30,
        patterns_per_lattice_field: 25,
        verification_stars_per_fov: 50,
        multiscale_step: 1.5,
        epoch_proper_motion_year: Some(2025.0),
        catalog_nside: 8,
    };

    let db = SolverDatabase::generate_from_hipparcos("data/hip2.dat", &config)
        .expect("Failed to generate database");

    println!(
        "Database: {} stars, {} patterns, table size {}",
        db.star_catalog.len(),
        db.props.num_patterns,
        db.pattern_catalog.len()
    );
    assert!(db.props.num_patterns > 0, "Should have generated patterns");

    // ── Step 2: Generate synthetic centroids ──
    // Point the camera toward Orion's belt region: RA ≈ 83°, Dec ≈ -1°
    let target_ra = 83.0_f32.to_radians();
    let target_dec = (-1.0_f32).to_radians();

    // Build a rotation that points the camera boresight (+Z) at this RA/Dec.
    // Camera frame: +X right, +Y down, +Z boresight.
    //
    // The boresight direction in ICRS:
    let boresight_icrs = Vector3::new(
        target_dec.cos() * target_ra.cos(),
        target_dec.cos() * target_ra.sin(),
        target_dec.sin(),
    );
    // Choose "up" direction (celestial north projected onto the image plane)
    let north_icrs = Vector3::new(0.0, 0.0, 1.0);
    // Camera Z = boresight (ICRS direction)
    let cam_z = boresight_icrs.normalize();
    // Camera X = right = perpendicular to boresight and north
    let cam_x = north_icrs.cross(&cam_z).normalize();
    // Camera Y = down = Z × X
    let cam_y = cam_z.cross(&cam_x);

    // Rotation matrix: rows are camera axes expressed in ICRS
    let rot = nalgebra::Matrix3::new(
        cam_x.x, cam_x.y, cam_x.z, cam_y.x, cam_y.y, cam_y.z, cam_z.x, cam_z.y, cam_z.z,
    );
    // This R satisfies: camera_vec = R * icrs_vec
    let true_quat = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rot));

    // Simulate a 15° FOV camera
    let fov_rad = 15.0_f32.to_radians();
    let half_fov = fov_rad / 2.0;

    // Find catalog stars visible in this FOV
    let nearby = db
        .star_catalog
        .query_indices_from_uvec(boresight_icrs, half_fov * 1.2);

    println!("Stars near boresight: {}", nearby.len());

    // Project each visible star to centroid coordinates
    let mut centroids: Vec<Centroid> = Vec::new();
    for &idx in &nearby {
        let sv = &db.star_vectors[idx];
        let icrs_v = Vector3::new(sv[0], sv[1], sv[2]);
        let cam_v = rot * icrs_v;

        if cam_v.z > 0.01 {
            let cx = cam_v.x / cam_v.z; // radians from boresight
            let cy = cam_v.y / cam_v.z;

            // Only keep stars within the FOV
            if cx.abs() < half_fov && cy.abs() < half_fov {
                centroids.push(Centroid {
                    x: cx,
                    y: cy,
                    mass: Some(10.0 - db.star_catalog.stars()[idx].mag), // brighter = higher mass
                    cov: None,
                });
            }
        }
    }

    println!("Synthetic centroids: {}", centroids.len());
    assert!(
        centroids.len() >= 4,
        "Need at least 4 centroids for solving, got {}",
        centroids.len()
    );

    // ── Step 3: Solve ──
    let solve_config = SolveConfig {
        fov_estimate_rad: fov_rad,
        fov_max_error_rad: Some(5.0_f32.to_radians()), // generous tolerance
        match_radius: 0.01,
        match_threshold: 1e-5,
        solve_timeout_ms: Some(30_000), // 30s for test
        match_max_error: None,
    };

    let result = db.solve_from_centroids(&centroids, &solve_config);

    println!("Solve status: {:?}", result.status);
    println!("Solve time: {:.1} ms", result.solve_time_ms);
    if let Some(n) = result.num_matches {
        println!("Matches: {}", n);
    }
    if let Some(rmse) = result.rmse_rad {
        println!("RMSE: {:.1} arcsec", rmse.to_degrees() * 3600.0);
    }
    if let Some(prob) = result.prob {
        println!("Probability: {:.2e}", prob);
    }

    assert_eq!(
        result.status,
        SolveStatus::MatchFound,
        "Solver should find a match"
    );

    // ── Step 4: Verify the recovered quaternion ──
    let solved_quat = result
        .qicrs2cam
        .expect("Should have quaternion on MatchFound");

    // Compare the solved boresight direction with the true one.
    // solved_quat rotates ICRS → camera, so boresight in ICRS = solved_quat.inverse() * [0,0,1]
    let solved_boresight = solved_quat.inverse() * Vector3::new(0.0, 0.0, 1.0);
    let true_boresight = true_quat.inverse() * Vector3::new(0.0, 0.0, 1.0);

    let angle_error = angular_separation(&solved_boresight, &true_boresight);

    println!(
        "Boresight error: {:.4}° ({:.1} arcsec)",
        angle_error.to_degrees(),
        angle_error.to_degrees() * 3600.0
    );

    // Should be within 0.5 degrees for a wide-FOV solve with no noise
    assert!(
        angle_error < 0.5_f32.to_radians(),
        "Boresight error {:.3}° exceeds 0.5° tolerance",
        angle_error.to_degrees()
    );
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Numerically stable angular separation between two unit vectors.
/// Uses atan2(|cross|, dot) which avoids the precision loss of acos near 0/π.
fn angular_separation(a: &Vector3<f32>, b: &Vector3<f32>) -> f32 {
    let cross = a.cross(b);
    cross.norm().atan2(a.dot(b))
}

// ── Helpers for the statistical test ──────────────────────────────────────────

/// Build rotation matrix (ICRS → camera) from boresight RA/Dec and roll angle.
fn rotation_from_ra_dec_roll(ra: f32, dec: f32, roll: f32) -> Matrix3<f32> {
    let boresight = Vector3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin());

    // Camera Z = boresight direction
    let cam_z = boresight.normalize();

    // Reference "up" in ICRS — use celestial north unless boresight is near a pole
    let north = Vector3::new(0.0, 0.0, 1.0);
    let raw_x = north.cross(&cam_z);
    let cam_x_noroll = if raw_x.norm() > 1e-6 {
        raw_x.normalize()
    } else {
        // Near pole: fall back to ICRS X-axis as reference
        let fallback = Vector3::new(1.0, 0.0, 0.0);
        fallback.cross(&cam_z).normalize()
    };
    let cam_y_noroll = cam_z.cross(&cam_x_noroll);

    // Apply roll (rotation around boresight)
    let cam_x = cam_x_noroll * roll.cos() + cam_y_noroll * roll.sin();
    let cam_y = -cam_x_noroll * roll.sin() + cam_y_noroll * roll.cos();

    Matrix3::new(
        cam_x.x, cam_x.y, cam_x.z, cam_y.x, cam_y.y, cam_y.z, cam_z.x, cam_z.y, cam_z.z,
    )
}

/// Generate synthetic centroids for a given rotation and FOV from the database.
/// If `noise_sigma_rad` is non-zero, adds Gaussian noise to each centroid coordinate.
fn generate_centroids_with_noise(
    db: &SolverDatabase,
    rot: &Matrix3<f32>,
    boresight_icrs: &Vector3<f32>,
    half_fov: f32,
    noise_sigma_rad: f32,
    rng: &mut StdRng,
) -> Vec<Centroid> {
    let nearby = db
        .star_catalog
        .query_indices_from_uvec(*boresight_icrs, half_fov * 1.2);

    let noise_dist = Normal::new(0.0f32, noise_sigma_rad).unwrap();

    let mut centroids = Vec::new();
    for &idx in &nearby {
        let sv = &db.star_vectors[idx];
        let icrs_v = Vector3::new(sv[0], sv[1], sv[2]);
        let cam_v = rot * icrs_v;

        if cam_v.z > 0.01 {
            let cx = cam_v.x / cam_v.z;
            let cy = cam_v.y / cam_v.z;

            if cx.abs() < half_fov && cy.abs() < half_fov {
                let nx = if noise_sigma_rad > 0.0 {
                    noise_dist.sample(rng)
                } else {
                    0.0
                };
                let ny = if noise_sigma_rad > 0.0 {
                    noise_dist.sample(rng)
                } else {
                    0.0
                };
                centroids.push(Centroid {
                    x: cx + nx,
                    y: cy + ny,
                    mass: Some(10.0 - db.star_catalog.stars()[idx].mag),
                    cov: None,
                });
            }
        }
    }
    centroids
}

/// Generate synthetic centroids without noise (convenience wrapper).
fn generate_centroids(
    db: &SolverDatabase,
    rot: &Matrix3<f32>,
    boresight_icrs: &Vector3<f32>,
    half_fov: f32,
) -> Vec<Centroid> {
    // Use a dummy RNG — noise_sigma_rad=0 means it's never sampled
    let mut dummy_rng = StdRng::seed_from_u64(0);
    generate_centroids_with_noise(db, rot, boresight_icrs, half_fov, 0.0, &mut dummy_rng)
}

/// Solve 1000 random orientations with a 10° FOV camera and report statistics.
#[test]
fn test_statistical_1000_random_orientations() {
    let _ = tracing_subscriber::fmt().with_env_filter("warn").try_init();

    // ── Build database for 10° FOV ──
    let config = GenerateDatabaseConfig {
        max_fov_deg: 12.0,
        min_fov_deg: None,
        star_max_magnitude: Some(7.0),
        pattern_max_error: 0.003,
        lattice_field_oversampling: 50,
        patterns_per_lattice_field: 50,
        verification_stars_per_fov: 40,
        multiscale_step: 1.5,
        epoch_proper_motion_year: Some(2025.0),
        catalog_nside: 8,
    };

    let db = SolverDatabase::generate_from_hipparcos("data/hip2.dat", &config)
        .expect("Failed to generate database");

    println!("\n══════════════════════════════════════════════════════════════");
    println!(
        "Database: {} stars, {} patterns, table size {}",
        db.star_catalog.len(),
        db.props.num_patterns,
        db.pattern_catalog.len()
    );

    // ── Solve config ──
    let fov_rad = 10.0_f32.to_radians();
    let half_fov = fov_rad / 2.0;

    let solve_config = SolveConfig {
        fov_estimate_rad: fov_rad,
        fov_max_error_rad: Some(2.0_f32.to_radians()),
        match_radius: 0.01,
        match_threshold: 1e-5,
        solve_timeout_ms: Some(10_000),
        match_max_error: None,
    };

    // Threshold for classifying a solve as "correct" vs "misidentified"
    let correct_threshold_arcsec = 180.0; // 3 arcmin — generous for lost-in-space
    let wrong_threshold_arcsec = 3600.0; // 1° — clearly wrong star field

    // ── Sample 1000 random orientations ──
    let n_trials: u32 = 1000;
    let mut rng = StdRng::seed_from_u64(42);

    let mut n_correct = 0u32;
    let mut n_imprecise = 0u32; // matched but error > 3 arcmin
    let mut n_wrong = 0u32; // matched but error > 1° (wrong field)
    let mut n_too_few = 0u32;
    let mut n_no_match = 0u32;
    let mut n_timeout = 0u32;

    // Stats for all solved orientations
    let mut all_errors_arcsec = Vec::new();
    let mut all_rmse_arcsec = Vec::new();
    let mut all_match_counts = Vec::new();

    // Track all solve times (including failures)
    let mut all_solve_times_ms = Vec::new();

    for trial in 0..n_trials {
        // Uniform random point on sphere
        let ra: f32 = rng.random::<f32>() * 2.0 * std::f32::consts::PI;
        let dec: f32 = (rng.random::<f32>() * 2.0 - 1.0).asin(); // uniform in sin(dec)
        let roll: f32 = rng.random::<f32>() * 2.0 * std::f32::consts::PI;

        let rot = rotation_from_ra_dec_roll(ra, dec, roll);
        let boresight_icrs = Vector3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin());

        let centroids = generate_centroids(&db, &rot, &boresight_icrs, half_fov);

        if centroids.len() < 4 {
            n_too_few += 1;
            continue;
        }

        let result = db.solve_from_centroids(&centroids, &solve_config);

        match result.status {
            SolveStatus::MatchFound => {
                all_solve_times_ms.push(result.solve_time_ms);

                // Compute boresight error
                let true_quat =
                    UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rot));
                let solved_quat = result.qicrs2cam.unwrap();
                let solved_boresight = solved_quat.inverse() * Vector3::new(0.0, 0.0, 1.0);
                let true_boresight = true_quat.inverse() * Vector3::new(0.0, 0.0, 1.0);
                let err_rad = angular_separation(&solved_boresight, &true_boresight);
                let err_arcsec = err_rad.to_degrees() * 3600.0;

                all_errors_arcsec.push(err_arcsec);
                if let Some(n) = result.num_matches {
                    all_match_counts.push(n);
                }
                if let Some(rmse) = result.rmse_rad {
                    all_rmse_arcsec.push(rmse.to_degrees() * 3600.0);
                }

                if err_arcsec < correct_threshold_arcsec {
                    n_correct += 1;
                } else if err_arcsec < wrong_threshold_arcsec {
                    n_imprecise += 1;
                    println!(
                        "  Trial {:4}: IMPRECISE err={:.1}\" matches={} RA={:.1}° Dec={:.1}° ({} centroids)",
                        trial, err_arcsec, result.num_matches.unwrap_or(0),
                        ra.to_degrees(), dec.to_degrees(), centroids.len(),
                    );
                } else {
                    n_wrong += 1;
                    println!(
                        "  Trial {:4}: WRONG err={:.1}\" matches={} RA={:.1}° Dec={:.1}° ({} centroids)",
                        trial, err_arcsec, result.num_matches.unwrap_or(0),
                        ra.to_degrees(), dec.to_degrees(), centroids.len(),
                    );
                }
            }
            SolveStatus::NoMatch => {
                n_no_match += 1;
                all_solve_times_ms.push(result.solve_time_ms);
            }
            SolveStatus::Timeout => {
                n_timeout += 1;
                all_solve_times_ms.push(result.solve_time_ms);
            }
            SolveStatus::TooFew => n_too_few += 1,
        }

        // Progress reporting
        if (trial + 1) % 200 == 0 {
            println!(
                "  Progress: {}/{} trials, {} correct, {} imprecise, {} wrong, {} failed",
                trial + 1,
                n_trials,
                n_correct,
                n_imprecise,
                n_wrong,
                n_no_match + n_timeout,
            );
        }
    }

    // ── Report statistics ──
    let n_attempted = n_trials - n_too_few;
    let n_solved = n_correct + n_imprecise + n_wrong;

    println!("\n══════════════════════════════════════════════════════════════");
    println!("RESULTS: 10° FOV, mag ≤ 7.0, {} trials", n_trials);
    println!("══════════════════════════════════════════════════════════════");
    println!(
        "  Correct (<3'):  {:4} ({:.1}%)",
        n_correct,
        100.0 * n_correct as f64 / n_attempted as f64
    );
    println!(
        "  Imprecise:      {:4} ({:.1}%)  (3'–1° error)",
        n_imprecise,
        100.0 * n_imprecise as f64 / n_attempted as f64
    );
    println!(
        "  Wrong (>1°):    {:4} ({:.1}%)",
        n_wrong,
        100.0 * n_wrong as f64 / n_attempted as f64
    );
    println!(
        "  No match:       {:4} ({:.1}%)",
        n_no_match,
        100.0 * n_no_match as f64 / n_attempted as f64
    );
    println!("  Timeout:        {:4}", n_timeout);
    println!("  Too few stars:  {:4}", n_too_few);
    println!(
        "  Solve rate:     {:.1}% ({}/{})",
        100.0 * n_solved as f64 / n_attempted as f64,
        n_solved,
        n_attempted
    );

    if !all_errors_arcsec.is_empty() {
        let mut sorted = all_errors_arcsec.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let mean: f32 = sorted.iter().sum::<f32>() / n as f32;
        let median = sorted[n / 2];
        let p95 = sorted[(n as f64 * 0.95) as usize];
        let p99 = sorted[(n as f64 * 0.99) as usize];
        let max = *sorted.last().unwrap();

        println!("\n  Boresight error — all solves (arcsec):");
        println!("    Mean:   {:8.2}", mean);
        println!("    Median: {:8.2}", median);
        println!("    P95:    {:8.2}", p95);
        println!("    P99:    {:8.2}", p99);
        println!("    Max:    {:8.2}", max);
    }

    if !all_rmse_arcsec.is_empty() {
        let mut sorted = all_rmse_arcsec.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let mean: f32 = sorted.iter().sum::<f32>() / n as f32;
        let p95 = sorted[(n as f64 * 0.95) as usize];
        let max = *sorted.last().unwrap();

        println!("\n  Fit RMSE — all solves (arcsec):");
        println!("    Mean:   {:8.2}", mean);
        println!("    P95:    {:8.2}", p95);
        println!("    Max:    {:8.2}", max);
    }

    if !all_solve_times_ms.is_empty() {
        all_solve_times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = all_solve_times_ms.len();
        let mean: f32 = all_solve_times_ms.iter().sum::<f32>() / n as f32;
        let p95 = all_solve_times_ms[(n as f64 * 0.95) as usize];
        let max = *all_solve_times_ms.last().unwrap();

        println!("\n  Solve time — all attempts (ms):");
        println!("    Mean:   {:8.4}", mean);
        println!("    P95:    {:8.4}", p95);
        println!("    Max:    {:8.4}", max);
    }

    if !all_match_counts.is_empty() {
        let mean: f32 =
            all_match_counts.iter().map(|&n| n as f32).sum::<f32>() / all_match_counts.len() as f32;
        let min = all_match_counts.iter().cloned().min().unwrap();

        println!("\n  Star matches per solve:");
        println!("    Mean:   {:8.1}", mean);
        println!("    Min:    {:8}", min);
    }

    println!("══════════════════════════════════════════════════════════════\n");

    // ── Assertions ──
    let correct_rate = n_correct as f64 / n_attempted as f64;
    assert!(
        correct_rate > 0.95,
        "Correct solve rate {:.1}% is below 95% (correct {}, attempted {})",
        correct_rate * 100.0,
        n_correct,
        n_attempted,
    );

    let wrong_rate = n_wrong as f64 / n_attempted as f64;
    assert!(
        wrong_rate < 0.01,
        "Wrong identification rate {:.1}% exceeds 1% ({} wrong of {} attempted)",
        wrong_rate * 100.0,
        n_wrong,
        n_attempted,
    );
}

/// Solve 1000 random orientations with a 10° FOV camera and 4"/axis centroid noise.
#[test]
fn test_statistical_1000_noisy_centroids() {
    let _ = tracing_subscriber::fmt().with_env_filter("warn").try_init();

    let noise_sigma_arcsec = 4.0;
    let noise_sigma_rad = (noise_sigma_arcsec / 3600.0_f32).to_radians();

    // ── Build database for 10° FOV ──
    let config = GenerateDatabaseConfig {
        max_fov_deg: 12.0,
        min_fov_deg: None,
        star_max_magnitude: Some(7.0),
        pattern_max_error: 0.003,
        lattice_field_oversampling: 50,
        patterns_per_lattice_field: 50,
        verification_stars_per_fov: 40,
        multiscale_step: 1.5,
        epoch_proper_motion_year: Some(2025.0),
        catalog_nside: 8,
    };

    let db = SolverDatabase::generate_from_hipparcos("data/hip2.dat", &config)
        .expect("Failed to generate database");

    println!("\n══════════════════════════════════════════════════════════════");
    println!(
        "Database: {} stars, {} patterns, table size {}",
        db.star_catalog.len(),
        db.props.num_patterns,
        db.pattern_catalog.len()
    );
    println!(
        "Centroid noise: σ = {:.1}\" per axis ({:.2e} rad)",
        noise_sigma_arcsec, noise_sigma_rad
    );

    // ── Solve config ──
    let fov_rad = 10.0_f32.to_radians();
    let half_fov = fov_rad / 2.0;

    let solve_config = SolveConfig {
        fov_estimate_rad: fov_rad,
        fov_max_error_rad: Some(2.0_f32.to_radians()),
        match_radius: 0.01,
        match_threshold: 1e-5,
        solve_timeout_ms: Some(10_000),
        match_max_error: None,
    };

    let correct_threshold_arcsec = 180.0;
    let wrong_threshold_arcsec = 3600.0;

    // ── Sample 1000 random orientations ──
    let n_trials: u32 = 1000;
    let mut rng = StdRng::seed_from_u64(123); // different seed from noiseless test

    let mut n_correct = 0u32;
    let mut n_imprecise = 0u32;
    let mut n_wrong = 0u32;
    let mut n_too_few = 0u32;
    let mut n_no_match = 0u32;
    let mut n_timeout = 0u32;

    let mut all_errors_arcsec = Vec::new();
    let mut all_rmse_arcsec = Vec::new();
    let mut all_match_counts = Vec::new();
    let mut all_solve_times_ms = Vec::new();

    for trial in 0..n_trials {
        let ra: f32 = rng.random::<f32>() * 2.0 * std::f32::consts::PI;
        let dec: f32 = (rng.random::<f32>() * 2.0 - 1.0).asin();
        let roll: f32 = rng.random::<f32>() * 2.0 * std::f32::consts::PI;

        let rot = rotation_from_ra_dec_roll(ra, dec, roll);
        let boresight_icrs = Vector3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin());

        let centroids = generate_centroids_with_noise(
            &db,
            &rot,
            &boresight_icrs,
            half_fov,
            noise_sigma_rad,
            &mut rng,
        );

        if centroids.len() < 4 {
            n_too_few += 1;
            continue;
        }

        let result = db.solve_from_centroids(&centroids, &solve_config);

        match result.status {
            SolveStatus::MatchFound => {
                all_solve_times_ms.push(result.solve_time_ms);

                let true_quat =
                    UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rot));
                let solved_quat = result.qicrs2cam.unwrap();
                let solved_boresight = solved_quat.inverse() * Vector3::new(0.0, 0.0, 1.0);
                let true_boresight = true_quat.inverse() * Vector3::new(0.0, 0.0, 1.0);
                let err_rad = angular_separation(&solved_boresight, &true_boresight);
                let err_arcsec = err_rad.to_degrees() * 3600.0;

                all_errors_arcsec.push(err_arcsec);
                if let Some(n) = result.num_matches {
                    all_match_counts.push(n);
                }
                if let Some(rmse) = result.rmse_rad {
                    all_rmse_arcsec.push(rmse.to_degrees() * 3600.0);
                }

                if err_arcsec < correct_threshold_arcsec {
                    n_correct += 1;
                } else if err_arcsec < wrong_threshold_arcsec {
                    n_imprecise += 1;
                    println!(
                        "  Trial {:4}: IMPRECISE err={:.1}\" matches={} RA={:.1}° Dec={:.1}° ({} centroids)",
                        trial, err_arcsec, result.num_matches.unwrap_or(0),
                        ra.to_degrees(), dec.to_degrees(), centroids.len(),
                    );
                } else {
                    n_wrong += 1;
                    println!(
                        "  Trial {:4}: WRONG err={:.1}\" matches={} RA={:.1}° Dec={:.1}° ({} centroids)",
                        trial, err_arcsec, result.num_matches.unwrap_or(0),
                        ra.to_degrees(), dec.to_degrees(), centroids.len(),
                    );
                }
            }
            SolveStatus::NoMatch => {
                n_no_match += 1;
                all_solve_times_ms.push(result.solve_time_ms);
            }
            SolveStatus::Timeout => {
                n_timeout += 1;
                all_solve_times_ms.push(result.solve_time_ms);
            }
            SolveStatus::TooFew => n_too_few += 1,
        }

        if (trial + 1) % 200 == 0 {
            println!(
                "  Progress: {}/{} trials, {} correct, {} imprecise, {} wrong, {} failed",
                trial + 1,
                n_trials,
                n_correct,
                n_imprecise,
                n_wrong,
                n_no_match + n_timeout,
            );
        }
    }

    // ── Report statistics ──
    let n_attempted = n_trials - n_too_few;
    let n_solved = n_correct + n_imprecise + n_wrong;

    println!("\n══════════════════════════════════════════════════════════════");
    println!(
        "RESULTS: 10° FOV, mag ≤ 7.0, σ = {}\" noise, {} trials",
        noise_sigma_arcsec, n_trials
    );
    println!("══════════════════════════════════════════════════════════════");
    println!(
        "  Correct (<3'):  {:4} ({:.1}%)",
        n_correct,
        100.0 * n_correct as f64 / n_attempted as f64
    );
    println!(
        "  Imprecise:      {:4} ({:.1}%)  (3'–1° error)",
        n_imprecise,
        100.0 * n_imprecise as f64 / n_attempted as f64
    );
    println!(
        "  Wrong (>1°):    {:4} ({:.1}%)",
        n_wrong,
        100.0 * n_wrong as f64 / n_attempted as f64
    );
    println!(
        "  No match:       {:4} ({:.1}%)",
        n_no_match,
        100.0 * n_no_match as f64 / n_attempted as f64
    );
    println!("  Timeout:        {:4}", n_timeout);
    println!("  Too few stars:  {:4}", n_too_few);
    println!(
        "  Solve rate:     {:.1}% ({}/{})",
        100.0 * n_solved as f64 / n_attempted as f64,
        n_solved,
        n_attempted
    );

    if !all_errors_arcsec.is_empty() {
        let mut sorted = all_errors_arcsec.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let mean: f32 = sorted.iter().sum::<f32>() / n as f32;
        let median = sorted[n / 2];
        let p95 = sorted[(n as f64 * 0.95) as usize];
        let p99 = sorted[(n as f64 * 0.99) as usize];
        let max = *sorted.last().unwrap();

        println!("\n  Boresight error — all solves (arcsec):");
        println!("    Mean:   {:8.2}", mean);
        println!("    Median: {:8.2}", median);
        println!("    P95:    {:8.2}", p95);
        println!("    P99:    {:8.2}", p99);
        println!("    Max:    {:8.2}", max);
    }

    if !all_rmse_arcsec.is_empty() {
        let mut sorted = all_rmse_arcsec.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let mean: f32 = sorted.iter().sum::<f32>() / n as f32;
        let p95 = sorted[(n as f64 * 0.95) as usize];
        let max = *sorted.last().unwrap();

        println!("\n  Fit RMSE — all solves (arcsec):");
        println!("    Mean:   {:8.2}", mean);
        println!("    P95:    {:8.2}", p95);
        println!("    Max:    {:8.2}", max);
    }

    if !all_solve_times_ms.is_empty() {
        all_solve_times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = all_solve_times_ms.len();
        let mean: f32 = all_solve_times_ms.iter().sum::<f32>() / n as f32;
        let p95 = all_solve_times_ms[(n as f64 * 0.95) as usize];
        let max = *all_solve_times_ms.last().unwrap();

        println!("\n  Solve time — all attempts (ms):");
        println!("    Mean:   {:8.4}", mean);
        println!("    P95:    {:8.4}", p95);
        println!("    Max:    {:8.4}", max);
    }

    if !all_match_counts.is_empty() {
        let mean: f32 =
            all_match_counts.iter().map(|&n| n as f32).sum::<f32>() / all_match_counts.len() as f32;
        let min = all_match_counts.iter().cloned().min().unwrap();

        println!("\n  Star matches per solve:");
        println!("    Mean:   {:8.1}", mean);
        println!("    Min:    {:8}", min);
    }

    println!("══════════════════════════════════════════════════════════════\n");

    // ── Assertions (relaxed for noisy data) ──
    let correct_rate = n_correct as f64 / n_attempted as f64;
    assert!(
        correct_rate > 0.90,
        "Correct solve rate {:.1}% is below 90% with {}\" noise (correct {}, attempted {})",
        correct_rate * 100.0,
        noise_sigma_arcsec,
        n_correct,
        n_attempted,
    );

    let wrong_rate = n_wrong as f64 / n_attempted as f64;
    assert!(
        wrong_rate < 0.02,
        "Wrong identification rate {:.1}% exceeds 2% with {}\" noise ({} wrong of {} attempted)",
        wrong_rate * 100.0,
        noise_sigma_arcsec,
        n_wrong,
        n_attempted,
    );
}
