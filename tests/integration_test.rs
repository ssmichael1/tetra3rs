//! Integration tests: build a database from Gaia, generate synthetic centroids
//! from known pointing directions, and verify the solver recovers the correct attitude.

mod test_data;

use numeris::{Matrix3, Quaternion, Vector3};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use tetra3::{Centroid, GenerateDatabaseConfig, SolveConfig, SolveStatus, SolverDatabase};

/// Path to the Gaia merged catalog (downloaded from GCS if missing).
fn gaia_catalog_path() -> String {
    test_data::ensure_test_file("data/gaia_merged.bin")
}

/// Build a small test database (wide FOV for speed) and solve a synthetic image.
#[test]
fn test_generate_and_solve() {
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

    let catalog_path = gaia_catalog_path();
    let db = SolverDatabase::generate_from_gaia(&catalog_path, &config)
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
    let boresight_icrs = Vector3::from_array([
        target_dec.cos() * target_ra.cos(),
        target_dec.cos() * target_ra.sin(),
        target_dec.sin(),
    ]);
    // Choose "up" direction (celestial north projected onto the image plane)
    let north_icrs = Vector3::from_array([0.0, 0.0, 1.0]);
    // Camera Z = boresight (ICRS direction)
    let cam_z = boresight_icrs.normalize();
    // Camera X = right = perpendicular to boresight and north
    let cam_x = north_icrs.cross(&cam_z).normalize();
    // Camera Y = down = Z × X
    let cam_y = cam_z.cross(&cam_x);

    // Rotation matrix: rows are camera axes expressed in ICRS
    let rot = Matrix3::new([
        [cam_x[0], cam_x[1], cam_x[2]],
        [cam_y[0], cam_y[1], cam_y[2]],
        [cam_z[0], cam_z[1], cam_z[2]],
    ]);
    // This R satisfies: camera_vec = R * icrs_vec
    let true_quat = Quaternion::from_rotation_matrix(&rot);

    // Simulate a 15° FOV camera with 1024x1024 sensor
    let fov_rad = 15.0_f32.to_radians();
    let half_fov = fov_rad / 2.0;
    let image_width = 1024u32;
    let image_height = 1024u32;
    // True pinhole pixel scale (1/f); matches the solver's internal convention.
    let pixel_scale = {
        let f = (image_width as f32 / 2.0) / (fov_rad / 2.0).tan();
        1.0 / f
    };

    // Find catalog stars visible in this FOV
    let nearby = db
        .star_catalog
        .query_indices_from_uvec(boresight_icrs, half_fov * 1.2);

    println!("Stars near boresight: {}", nearby.len());

    // Project each visible star to centroid pixel coordinates
    let mut centroids: Vec<Centroid> = Vec::new();
    for &idx in &nearby {
        let sv = &db.star_vectors[idx];
        let icrs_v = Vector3::from_array([sv[0], sv[1], sv[2]]);
        let cam_v = rot * icrs_v;

        if cam_v[2] > 0.01 {
            let cx_rad = cam_v[0] / cam_v[2]; // radians from boresight
            let cy_rad = cam_v[1] / cam_v[2];

            // Only keep stars within the FOV
            if cx_rad.abs() < half_fov && cy_rad.abs() < half_fov {
                centroids.push(Centroid {
                    x: cx_rad / pixel_scale, // convert to pixels from image center
                    y: cy_rad / pixel_scale,
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
        image_width,
        image_height,
        fov_max_error_rad: Some(5.0_f32.to_radians()), // generous tolerance
        match_radius: 0.01,
        match_threshold: 1e-5,
        solve_timeout_ms: Some(30_000), // 30s for test
        match_max_error: None,
        refine_iterations: 2,
        ..Default::default()
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
    let solved_boresight = solved_quat.inverse() * Vector3::from_array([0.0, 0.0, 1.0]);
    let true_boresight = true_quat.inverse() * Vector3::from_array([0.0, 0.0, 1.0]);

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
    let boresight = Vector3::from_array([dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()]);

    // Camera Z = boresight direction
    let cam_z = boresight.normalize();

    // Reference "up" in ICRS — use celestial north unless boresight is near a pole
    let north = Vector3::from_array([0.0, 0.0, 1.0]);
    let raw_x = north.cross(&cam_z);
    let cam_x_noroll = if raw_x.norm() > 1e-6 {
        raw_x.normalize()
    } else {
        // Near pole: fall back to ICRS X-axis as reference
        let fallback = Vector3::from_array([1.0, 0.0, 0.0]);
        fallback.cross(&cam_z).normalize()
    };
    let cam_y_noroll = cam_z.cross(&cam_x_noroll);

    // Apply roll (rotation around boresight)
    let cam_x = cam_x_noroll * roll.cos() + cam_y_noroll * roll.sin();
    let cam_y = -cam_x_noroll * roll.sin() + cam_y_noroll * roll.cos();

    Matrix3::new([
        [cam_x[0], cam_x[1], cam_x[2]],
        [cam_y[0], cam_y[1], cam_y[2]],
        [cam_z[0], cam_z[1], cam_z[2]],
    ])
}

/// Generate synthetic centroids (in pixel coordinates) for a given rotation and FOV.
/// If `noise_sigma_px` is non-zero, adds Gaussian noise (in pixels) to each centroid coordinate.
fn generate_centroids_with_noise(
    db: &SolverDatabase,
    rot: &Matrix3<f32>,
    boresight_icrs: &Vector3<f32>,
    half_fov: f32,
    pixel_scale: f32,
    noise_sigma_px: f32,
    rng: &mut StdRng,
) -> Vec<Centroid> {
    let nearby = db
        .star_catalog
        .query_indices_from_uvec(*boresight_icrs, half_fov * 1.2);

    let noise_dist = Normal::new(0.0f32, noise_sigma_px.max(1e-30)).unwrap();

    let mut centroids = Vec::new();
    for &idx in &nearby {
        let sv = &db.star_vectors[idx];
        let icrs_v = Vector3::from_array([sv[0], sv[1], sv[2]]);
        let cam_v = rot * icrs_v;

        if cam_v[2] > 0.01 {
            let cx_rad = cam_v[0] / cam_v[2];
            let cy_rad = cam_v[1] / cam_v[2];

            if cx_rad.abs() < half_fov && cy_rad.abs() < half_fov {
                let nx = if noise_sigma_px > 0.0 {
                    noise_dist.sample(rng)
                } else {
                    0.0
                };
                let ny = if noise_sigma_px > 0.0 {
                    noise_dist.sample(rng)
                } else {
                    0.0
                };
                centroids.push(Centroid {
                    x: cx_rad / pixel_scale + nx,
                    y: cy_rad / pixel_scale + ny,
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
    pixel_scale: f32,
) -> Vec<Centroid> {
    let mut dummy_rng = StdRng::seed_from_u64(0);
    generate_centroids_with_noise(
        db,
        rot,
        boresight_icrs,
        half_fov,
        pixel_scale,
        0.0,
        &mut dummy_rng,
    )
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
        patterns_per_lattice_field: 100,
        verification_stars_per_fov: 40,
        multiscale_step: 1.5,
        epoch_proper_motion_year: Some(2025.0),
        catalog_nside: 8,
    };

    let db = SolverDatabase::generate_from_gaia(&gaia_catalog_path(), &config)
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
    let image_width = 1024u32;
    let image_height = 1024u32;
    // True pinhole pixel scale (1/f); matches the solver's internal convention.
    let pixel_scale = {
        let f = (image_width as f32 / 2.0) / (fov_rad / 2.0).tan();
        1.0 / f
    };

    let solve_config = SolveConfig {
        fov_estimate_rad: fov_rad,
        image_width,
        image_height,
        fov_max_error_rad: Some(2.0_f32.to_radians()),
        match_radius: 0.01,
        match_threshold: 1e-5,
        solve_timeout_ms: Some(10_000),
        match_max_error: None,
        refine_iterations: 2,
        ..Default::default()
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
        let boresight_icrs = Vector3::from_array([dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()]);

        let centroids = generate_centroids(&db, &rot, &boresight_icrs, half_fov, pixel_scale);

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
                    Quaternion::from_rotation_matrix(&rot);
                let solved_quat = result.qicrs2cam.unwrap();
                let solved_boresight = solved_quat.inverse() * Vector3::from_array([0.0, 0.0, 1.0]);
                let true_boresight = true_quat.inverse() * Vector3::from_array([0.0, 0.0, 1.0]);
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

#[test]
fn test_save_and_load_database() {
    let _ = tracing_subscriber::fmt().with_env_filter("warn").try_init();

    let config = GenerateDatabaseConfig {
        max_fov_deg: 12.0,
        min_fov_deg: None,
        star_max_magnitude: Some(7.0),
        pattern_max_error: 0.005,
        lattice_field_oversampling: 100,
        patterns_per_lattice_field: 60,
        verification_stars_per_fov: 50,
        multiscale_step: 1.5,
        epoch_proper_motion_year: Some(2025.0),
        catalog_nside: 16,
    };

    let db = SolverDatabase::generate_from_gaia(&gaia_catalog_path(), &config)
        .expect("Failed to generate database");

    // Save to a temporary file
    let tmp_path = "temp_db.bin";
    db.save_to_file(tmp_path).expect("Failed to save database");

    // Load it back
    let loaded_db = SolverDatabase::load_from_file(tmp_path).expect("Failed to load database");

    // Verify properties match
    assert_eq!(db.star_catalog.len(), loaded_db.star_catalog.len());
    assert_eq!(db.props.num_patterns, loaded_db.props.num_patterns);
    assert_eq!(db.pattern_catalog.len(), loaded_db.pattern_catalog.len());

    // Clean up temporary file
    std::fs::remove_file(tmp_path).expect("Failed to delete temporary file");
}

/// Solve 1000 random orientations with a 10° FOV camera and 4"/axis centroid noise.
#[test]
fn test_statistical_1000_noisy_centroids() {
    let _ = tracing_subscriber::fmt().with_env_filter("warn").try_init();

    let noise_sigma_arcsec = 4.0;

    // ── Build database for 10° FOV ──
    let config = GenerateDatabaseConfig {
        max_fov_deg: 12.0,
        min_fov_deg: None,
        star_max_magnitude: Some(7.0),
        pattern_max_error: 0.003,
        lattice_field_oversampling: 50,
        patterns_per_lattice_field: 100,
        verification_stars_per_fov: 60,
        multiscale_step: 1.5,
        epoch_proper_motion_year: Some(2025.0),
        catalog_nside: 16,
    };

    let db = SolverDatabase::generate_from_gaia(&gaia_catalog_path(), &config)
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
    let image_width = 1024u32;
    let image_height = 1024u32;
    // True pinhole pixel scale (1/f); matches the solver's internal convention.
    let pixel_scale = {
        let f = (image_width as f32 / 2.0) / (fov_rad / 2.0).tan();
        1.0 / f
    };
    let noise_sigma_px = (noise_sigma_arcsec / 3600.0_f32).to_radians() / pixel_scale;

    println!(
        "Centroid noise: σ = {:.1}\" per axis ({:.2} px)",
        noise_sigma_arcsec, noise_sigma_px
    );

    let solve_config = SolveConfig {
        fov_estimate_rad: fov_rad,
        image_width,
        image_height,
        fov_max_error_rad: Some(2.0_f32.to_radians()),
        match_radius: 0.01,
        match_threshold: 1e-5,
        solve_timeout_ms: Some(10_000),
        match_max_error: None,
        refine_iterations: 2,
        ..Default::default()
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
    let mut all_roll_errors_arcsec = Vec::new();
    let mut all_rmse_arcsec = Vec::new();
    let mut all_match_counts = Vec::new();
    let mut all_solve_times_ms = Vec::new();

    for trial in 0..n_trials {
        let ra: f32 = rng.random::<f32>() * 2.0 * std::f32::consts::PI;
        let dec: f32 = (rng.random::<f32>() * 2.0 - 1.0).asin();
        let roll: f32 = rng.random::<f32>() * 2.0 * std::f32::consts::PI;

        let rot = rotation_from_ra_dec_roll(ra, dec, roll);
        let boresight_icrs = Vector3::from_array([dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()]);

        let centroids = generate_centroids_with_noise(
            &db,
            &rot,
            &boresight_icrs,
            half_fov,
            pixel_scale,
            noise_sigma_px,
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
                    Quaternion::from_rotation_matrix(&rot);
                let solved_quat = result.qicrs2cam.unwrap();
                let solved_boresight = solved_quat.inverse() * Vector3::from_array([0.0, 0.0, 1.0]);
                let true_boresight = true_quat.inverse() * Vector3::from_array([0.0, 0.0, 1.0]);
                let err_rad = angular_separation(&solved_boresight, &true_boresight);
                let err_arcsec = err_rad.to_degrees() * 3600.0;

                // Roll error: angle between the camera x-axes (projected
                // perpendicular to the true boresight) of the true vs solved rotations.
                let cam_x = Vector3::from_array([1.0_f32, 0.0, 0.0]);
                let true_x_icrs = true_quat.inverse() * cam_x;
                let solved_x_icrs = solved_quat.inverse() * cam_x;
                let proj_true = true_x_icrs - true_boresight * true_x_icrs.dot(&true_boresight);
                let proj_solved =
                    solved_x_icrs - true_boresight * solved_x_icrs.dot(&true_boresight);
                let roll_err_rad = proj_true
                    .normalize()
                    .dot(&proj_solved.normalize())
                    .clamp(-1.0, 1.0)
                    .acos();
                let roll_err_arcsec = roll_err_rad.to_degrees() * 3600.0;
                all_roll_errors_arcsec.push(roll_err_arcsec);

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

    if !all_roll_errors_arcsec.is_empty() {
        let mut sorted = all_roll_errors_arcsec.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let mean: f32 = sorted.iter().sum::<f32>() / n as f32;
        let median = sorted[n / 2];
        let p95 = sorted[(n as f64 * 0.95) as usize];
        let p99 = sorted[(n as f64 * 0.99) as usize];
        let max = *sorted.last().unwrap();

        println!("\n  Roll error — all solves (arcsec):");
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

/// Tracking-mode test: solve with LIS, then perturb the attitude and re-solve
/// using the perturbed attitude as a hint. Verify the hinted solve succeeds
/// (and ideally faster).
#[test]
fn test_tracking_with_attitude_hint() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "warn".into()))
        .try_init();

    // Small DB matching test_generate_and_solve so it builds quickly.
    let config = GenerateDatabaseConfig {
        max_fov_deg: 20.0,
        min_fov_deg: None,
        star_max_magnitude: Some(6.0),
        pattern_max_error: 0.005,
        lattice_field_oversampling: 30,
        patterns_per_lattice_field: 25,
        verification_stars_per_fov: 50,
        multiscale_step: 1.5,
        epoch_proper_motion_year: Some(2025.0),
        catalog_nside: 8,
    };

    let db = SolverDatabase::generate_from_gaia(&gaia_catalog_path(), &config)
        .expect("Failed to generate database");

    let fov_rad = 15.0_f32.to_radians();
    let half_fov = fov_rad / 2.0;
    let image_width = 1024u32;
    let image_height = 1024u32;
    let pixel_scale = fov_rad / image_width as f32;

    let mut rng = StdRng::seed_from_u64(7);

    // Number of fields to test. Each: solve LIS, then re-solve with hint.
    let n_trials = 20u32;
    let mut n_lis_ok = 0u32;
    let mut n_track_ok = 0u32;
    let mut n_track_recovers_perturbed = 0u32;
    let mut lis_time_ms = Vec::new();
    let mut track_time_ms = Vec::new();

    let perturb_arcmin = 30.0_f32; // hint within 0.5° of truth
    let perturb_rad = perturb_arcmin / 60.0 * std::f32::consts::PI / 180.0;

    for trial in 0..n_trials {
        let ra: f32 = rng.random::<f32>() * 2.0 * std::f32::consts::PI;
        let dec: f32 = (rng.random::<f32>() * 2.0 - 1.0).asin();
        let roll: f32 = rng.random::<f32>() * 2.0 * std::f32::consts::PI;

        let rot = rotation_from_ra_dec_roll(ra, dec, roll);
        let boresight_icrs =
            Vector3::from_array([dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()]);
        let centroids = generate_centroids(&db, &rot, &boresight_icrs, half_fov, pixel_scale);
        if centroids.len() < 4 {
            continue;
        }

        // ── Step 1: LIS solve (no hint) ──
        let lis_config = SolveConfig {
            fov_estimate_rad: fov_rad,
            image_width,
            image_height,
            fov_max_error_rad: Some(2.0_f32.to_radians()),
            solve_timeout_ms: Some(10_000),
            ..Default::default()
        };
        let lis_result = db.solve_from_centroids(&centroids, &lis_config);
        if lis_result.status != SolveStatus::MatchFound {
            continue;
        }
        n_lis_ok += 1;
        lis_time_ms.push(lis_result.solve_time_ms);
        let lis_quat = lis_result.qicrs2cam.expect("MatchFound implies quaternion");

        // ── Step 2: perturb the attitude by `perturb_rad` around a random axis ──
        let axis_x: f32 = rng.random::<f32>() - 0.5;
        let axis_y: f32 = rng.random::<f32>() - 0.5;
        let axis_z: f32 = rng.random::<f32>() - 0.5;
        let axis = Vector3::from_array([axis_x, axis_y, axis_z]).normalize();
        let half = perturb_rad / 2.0;
        let s = half.sin();
        let perturbation = Quaternion::new(half.cos(), s * axis[0], s * axis[1], s * axis[2]);
        let hinted_quat = perturbation * lis_quat;

        // ── Step 3: re-solve with the perturbed attitude as a hint ──
        // Reuse the camera model from the LIS result (refined focal length).
        let track_config = SolveConfig {
            fov_estimate_rad: fov_rad,
            image_width,
            image_height,
            attitude_hint: Some(hinted_quat),
            hint_uncertainty_rad: 1.0_f32.to_radians(),
            strict_hint: true, // disable LIS fallback so we measure tracking alone
            solve_timeout_ms: Some(2_000),
            camera_model: lis_result
                .camera_model
                .clone()
                .expect("MatchFound implies camera_model"),
            ..Default::default()
        };
        let track_result = db.solve_from_centroids(&centroids, &track_config);
        if track_result.status == SolveStatus::MatchFound {
            n_track_ok += 1;
            track_time_ms.push(track_result.solve_time_ms);

            // Verify the tracked solution agrees with the LIS solution.
            // On noiseless synthetic data the two paths converge to the same
            // fixed point of wcs_refine, so agreement is at f32 floating-point
            // noise (effectively zero). 1″ is very loose and only catches
            // gross regressions — tighten if we ever want to detect subtler
            // divergence between the two paths.
            let tq = track_result.qicrs2cam.unwrap();
            let lis_bs = lis_quat.inverse() * Vector3::from_array([0.0, 0.0, 1.0]);
            let track_bs = tq.inverse() * Vector3::from_array([0.0, 0.0, 1.0]);
            let agreement = angular_separation(&lis_bs, &track_bs);
            const AGREEMENT_THRESHOLD_ARCSEC: f32 = 1.0;
            if agreement < (AGREEMENT_THRESHOLD_ARCSEC / 3600.0).to_radians() {
                n_track_recovers_perturbed += 1;
            } else {
                println!(
                    "  Trial {:2}: tracked but disagrees with LIS by {:.2}\"",
                    trial,
                    agreement.to_degrees() * 3600.0
                );
            }
        } else {
            println!(
                "  Trial {:2}: tracking FAILED (status={:?}, perturb={:.1}')",
                trial, track_result.status, perturb_arcmin
            );
        }
    }

    let mean = |v: &[f32]| -> f32 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f32>() / v.len() as f32
        }
    };

    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Tracking-mode test ({} trials, {:.1}' hint perturbation)", n_trials, perturb_arcmin);
    println!("    LIS solves successful:        {:3}/{}", n_lis_ok, n_trials);
    println!("    Tracking solves successful:   {:3}/{}", n_track_ok, n_lis_ok);
    println!("    Tracking agrees with LIS:     {:3}/{}", n_track_recovers_perturbed, n_track_ok);
    println!("    Mean LIS time:      {:7.2} ms", mean(&lis_time_ms));
    println!("    Mean tracking time: {:7.2} ms", mean(&track_time_ms));
    println!("══════════════════════════════════════════════════════════════\n");

    assert!(n_lis_ok >= 15, "LIS only solved {}/{} — DB may be too sparse", n_lis_ok, n_trials);
    assert!(
        n_track_ok as f64 / n_lis_ok as f64 > 0.90,
        "Tracking only succeeded for {}/{} of LIS-solved frames",
        n_track_ok,
        n_lis_ok
    );
    assert!(
        n_track_recovers_perturbed as f64 / n_track_ok.max(1) as f64 > 0.95,
        "Tracking matched but disagreed with LIS in {}/{} cases",
        n_track_ok - n_track_recovers_perturbed,
        n_track_ok
    );
}

/// Regression test for issue #13: multiscale databases that produce a pattern
/// table larger than rkyv's 32-bit offset limit (~2 GB) should save and load
/// successfully under the sharded `PatternCatalog` layout.
///
/// This test is expensive: it generates a multiscale database covering several
/// FOV octaves, which typically produces tens of millions of unique patterns
/// and requires multi-GB RAM. Marked `#[ignore]` so it only runs when
/// explicitly requested:
///
/// ```sh
/// cargo test --release --test integration_test test_multiscale_sharded_database -- --ignored --nocapture
/// ```
#[test]
#[ignore = "slow: generates a multi-GB pattern catalog; run with --ignored"]
fn test_multiscale_sharded_database() {
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();

    // FOV range chosen to exceed the old 2 GB single-Vec limit — covers 0.5° to 5°.
    let config = GenerateDatabaseConfig {
        max_fov_deg: 5.0,
        min_fov_deg: Some(0.5),
        star_max_magnitude: Some(9.0),
        pattern_max_error: 0.002,
        lattice_field_oversampling: 100,
        patterns_per_lattice_field: 50,
        verification_stars_per_fov: 150,
        multiscale_step: 1.5,
        epoch_proper_motion_year: Some(2025.0),
        catalog_nside: 16,
    };

    let catalog_path = test_data::ensure_test_file("data/gaia_merged.bin");
    println!("Generating multiscale database 0.5°–5°…");
    let db = SolverDatabase::generate_from_gaia(&catalog_path, &config)
        .expect("multiscale database generation");

    let total_slots = db.pattern_catalog.len();
    let n_shards = db.pattern_catalog.shards.len();
    println!(
        "  {} pattern slots across {} shard(s) ({} patterns stored)",
        total_slots, n_shards, db.props.num_patterns
    );
    assert!(
        n_shards >= 2,
        "expected ≥2 shards for this FOV range to exercise the sharding path \
         (got {} slots in {} shard)",
        total_slots,
        n_shards
    );

    let tmp_path = std::env::temp_dir().join("tetra3rs_multiscale_test.rkyv");
    println!("Saving to {}…", tmp_path.display());
    db.save_to_file(tmp_path.to_str().unwrap())
        .expect("save_to_file");

    println!("Loading…");
    let loaded = SolverDatabase::load_from_file(tmp_path.to_str().unwrap())
        .expect("load_from_file");
    assert_eq!(loaded.pattern_catalog.len(), total_slots);
    assert_eq!(loaded.pattern_catalog.shards.len(), n_shards);
    assert_eq!(loaded.props.num_patterns, db.props.num_patterns);

    std::fs::remove_file(tmp_path).ok();
}
