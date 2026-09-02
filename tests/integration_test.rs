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
        fov_max_error_rad: Some(5.0_f32.to_radians()), // generous tolerance
        match_radius: 0.01,
        match_threshold: 1e-5,
        solve_timeout_ms: Some(30_000), // 30s for test
        match_max_error: None,
        ..SolveConfig::new(fov_rad, image_width, image_height)
    };

    let result = db.solve_from_centroids(&centroids, &solve_config);

    let solution = result.expect("Solver should find a match");

    // Attitude covariance: finite, symmetric, positive-diagonal, and the
    // boresight uncertainty sits between rmse/(10·√n) and rmse — the matched
    // stars average the per-star scatter down, but only by ~√n.
    let cov = solution.attitude_cov_rad2;
    for i in 0..3 {
        assert!(
            cov[i][i].is_finite() && cov[i][i] > 0.0,
            "cov diag {i}: {}",
            cov[i][i]
        );
        for j in 0..3 {
            assert!((cov[i][j] - cov[j][i]).abs() <= 1e-12 * cov[i][i].abs().max(cov[j][j].abs()));
        }
    }
    let sigma = solution.attitude_sigma_rad();
    let pointing = sigma[1].hypot(sigma[2]);
    let rmse = solution.rmse_rad as f64;
    let n = solution.num_matches as f64;
    assert!(
        pointing < rmse,
        "pointing σ {pointing:.3e} vs rmse {rmse:.3e}"
    );
    assert!(
        pointing > rmse / (10.0 * n.sqrt()),
        "pointing σ {pointing:.3e} implausibly small"
    );
    println!("Solve time: {:.1} ms", solution.solve_time_ms);
    println!("Matches: {}", solution.num_matches);
    println!(
        "RMSE: {:.1} arcsec",
        solution.rmse_rad.to_degrees() * 3600.0
    );
    println!("Probability: {:.2e}", solution.prob);

    // ── Step 4: Verify the recovered quaternion ──
    let solved_quat = solution.qicrs2cam;

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

/// Regression test: `Solution.matched_centroid_indices` must index the
/// *caller's* input slice, even when the solver drops non-finite centroids up
/// front. Dropping compacts the working list, shifting every index at or
/// beyond the drop point; without a remap, `matched_centroid_indices` would be
/// off by the number of dropped centroids ahead of each match, silently
/// pairing the wrong observed positions with catalog stars (e.g. corrupting a
/// `calibrate_camera` distortion fit).
#[test]
fn test_matched_indices_survive_dropped_centroid() {
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();

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

    // Same Orion's-belt pointing as `test_generate_and_solve`.
    let target_ra = 83.0_f32.to_radians();
    let target_dec = (-1.0_f32).to_radians();
    let boresight_icrs = Vector3::from_array([
        target_dec.cos() * target_ra.cos(),
        target_dec.cos() * target_ra.sin(),
        target_dec.sin(),
    ]);
    let cam_z = boresight_icrs.normalize();
    let cam_x = Vector3::from_array([0.0, 0.0, 1.0])
        .cross(&cam_z)
        .normalize();
    let cam_y = cam_z.cross(&cam_x);
    let rot = Matrix3::new([
        [cam_x[0], cam_x[1], cam_x[2]],
        [cam_y[0], cam_y[1], cam_y[2]],
        [cam_z[0], cam_z[1], cam_z[2]],
    ]);

    let fov_rad = 15.0_f32.to_radians();
    let half_fov = fov_rad / 2.0;
    let (image_width, image_height) = (1024u32, 1024u32);
    let pixel_scale = 1.0 / ((image_width as f32 / 2.0) / (fov_rad / 2.0).tan());

    // Project visible stars to centroids, remembering each centroid's true
    // catalog id so we can check the reported index → id pairing.
    let mut centroids: Vec<Centroid> = Vec::new();
    let mut source_ids: Vec<i64> = Vec::new();
    for &idx in &db
        .star_catalog
        .query_indices_from_uvec(boresight_icrs, half_fov * 1.2)
    {
        let sv = &db.star_vectors[idx];
        let cam_v = rot * Vector3::from_array([sv[0], sv[1], sv[2]]);
        if cam_v[2] > 0.01 {
            let (cx_rad, cy_rad) = (cam_v[0] / cam_v[2], cam_v[1] / cam_v[2]);
            if cx_rad.abs() < half_fov && cy_rad.abs() < half_fov {
                centroids.push(Centroid {
                    x: cx_rad / pixel_scale,
                    y: cy_rad / pixel_scale,
                    mass: Some(10.0 - db.star_catalog.stars()[idx].mag),
                    cov: None,
                });
                source_ids.push(db.star_catalog_ids[idx]);
            }
        }
    }
    assert!(
        centroids.len() >= 5,
        "need >= 5 centroids, got {}",
        centroids.len()
    );

    // Insert a NaN centroid partway through so every real centroid after it is
    // shifted by one in the working (post-drop) frame. It sits among the
    // bright stars (mass high) so the pre-drop brightness order would place it
    // inside the tested set had it not been dropped.
    let insert_at = 2;
    centroids.insert(
        insert_at,
        Centroid {
            x: f32::NAN,
            y: 12.0,
            mass: Some(100.0),
            cov: None,
        },
    );
    source_ids.insert(insert_at, i64::MIN); // sentinel: must never be matched

    let solve_config = SolveConfig {
        fov_max_error_rad: Some(5.0_f32.to_radians()),
        match_radius: 0.01,
        match_threshold: 1e-5,
        solve_timeout_ms: Some(30_000),
        match_max_error: None,
        ..SolveConfig::new(fov_rad, image_width, image_height)
    };

    let solution = db
        .solve_from_centroids(&centroids, &solve_config)
        .expect("solve should succeed despite the dropped NaN centroid");

    assert_eq!(
        solution.matched_centroid_indices.len(),
        solution.matched_catalog_ids.len(),
    );
    assert!(
        !solution.matched_centroid_indices.is_empty(),
        "expected matches"
    );

    for (&ci, &cat_id) in solution
        .matched_centroid_indices
        .iter()
        .zip(&solution.matched_catalog_ids)
    {
        // Index must land in the caller's slice and never on the NaN entry.
        assert!(ci < centroids.len(), "index {ci} out of caller range");
        assert!(
            centroids[ci].x.is_finite() && centroids[ci].y.is_finite(),
            "matched index {ci} points at the dropped non-finite centroid",
        );
        // The reported index must identify the centroid actually generated
        // from the matched catalog star — the core off-by-one check.
        assert_eq!(
            source_ids[ci], cat_id,
            "index {ci} maps to catalog id {} but the solution paired it with {cat_id}",
            source_ids[ci],
        );
    }
}

/// Small single-scale test database used by the non-finite-input tests below.
fn small_test_db() -> SolverDatabase {
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
    SolverDatabase::generate_from_gaia(&gaia_catalog_path(), &config)
        .expect("Failed to generate database")
}

/// Noiseless synthetic centroids for the Orion's-belt pointing used by
/// `test_generate_and_solve` (15° FOV, 1024×1024), brightest first by mass.
fn orion_synthetic_centroids(db: &SolverDatabase, fov_rad: f32, image_width: u32) -> Vec<Centroid> {
    let target_ra = 83.0_f32.to_radians();
    let target_dec = (-1.0_f32).to_radians();
    let boresight_icrs = Vector3::from_array([
        target_dec.cos() * target_ra.cos(),
        target_dec.cos() * target_ra.sin(),
        target_dec.sin(),
    ]);
    let cam_z = boresight_icrs.normalize();
    let cam_x = Vector3::from_array([0.0, 0.0, 1.0])
        .cross(&cam_z)
        .normalize();
    let cam_y = cam_z.cross(&cam_x);
    let rot = Matrix3::new([
        [cam_x[0], cam_x[1], cam_x[2]],
        [cam_y[0], cam_y[1], cam_y[2]],
        [cam_z[0], cam_z[1], cam_z[2]],
    ]);
    let half_fov = fov_rad / 2.0;
    let pixel_scale = 1.0 / ((image_width as f32 / 2.0) / (fov_rad / 2.0).tan());

    let mut centroids = Vec::new();
    for &idx in &db
        .star_catalog
        .query_indices_from_uvec(boresight_icrs, half_fov * 1.2)
    {
        let sv = &db.star_vectors[idx];
        let cam_v = rot * Vector3::from_array([sv[0], sv[1], sv[2]]);
        if cam_v[2] > 0.01 {
            let (cx_rad, cy_rad) = (cam_v[0] / cam_v[2], cam_v[1] / cam_v[2]);
            if cx_rad.abs() < half_fov && cy_rad.abs() < half_fov {
                centroids.push(Centroid {
                    x: cx_rad / pixel_scale,
                    y: cy_rad / pixel_scale,
                    mass: Some(10.0 - db.star_catalog.stars()[idx].mag),
                    cov: None,
                });
            }
        }
    }
    centroids
}

/// `calibrate_camera` must tolerate a non-finite centroid at a *matched*
/// index (e.g. the caller passes a differently-filtered array than the one
/// solved). It used to feed NaN through the WCS normal equations and the
/// pooled polynomial fit and return `Ok` with an all-NaN model.
#[test]
fn test_calibrate_tolerates_nan_centroid() {
    let db = small_test_db();
    let fov_rad = 15.0_f32.to_radians();
    let (w, h) = (1024u32, 1024u32);
    let centroids = orion_synthetic_centroids(&db, fov_rad, w);

    let solve_config = SolveConfig {
        fov_max_error_rad: Some(5.0_f32.to_radians()),
        solve_timeout_ms: Some(30_000),
        ..SolveConfig::new(fov_rad, w, h)
    };
    let solution = db
        .solve_from_centroids(&centroids, &solve_config)
        .expect("solve should succeed");
    assert!(solution.matched_centroid_indices.len() >= 6);

    let cal_config = tetra3::CalibrateConfig {
        model: tetra3::DistortionModelType::Polynomial { order: 2 },
        ..Default::default()
    };
    let sr: tetra3::SolveResult = Ok(solution.clone());

    let clean = tetra3::calibrate_camera(&[&sr], &[&centroids], &db, w, h, &cal_config)
        .expect("clean calibration");

    let mut poisoned = centroids.clone();
    poisoned[solution.matched_centroid_indices[0]].x = f32::NAN;
    let dirty = tetra3::calibrate_camera(&[&sr], &[&poisoned], &db, w, h, &cal_config)
        .expect("calibration with one NaN centroid must still succeed");

    dirty
        .camera_model
        .validate()
        .expect("fitted model must validate");
    assert!(dirty.rmse_after_px.is_finite());
    assert!(dirty.n_inliers > 0);
    // Exactly the poisoned match is lost; the fit itself is unchanged.
    assert!(
        dirty.n_inliers + 1 >= clean.n_inliers,
        "{} vs {}",
        dirty.n_inliers,
        clean.n_inliers
    );
    assert!(
        (dirty.rmse_after_px - clean.rmse_after_px).abs() < 0.05,
        "rmse {} vs {}",
        dirty.rmse_after_px,
        clean.rmse_after_px
    );
    assert!((dirty.camera_model.focal_length_px - clean.camera_model.focal_length_px).abs() < 1e-6);
}

/// A `Some(NaN)` mass must be treated exactly like `None` (unknown), not fed
/// into the brightness sort's comparator.
#[test]
fn test_nan_mass_treated_as_unknown() {
    let db = small_test_db();
    let fov_rad = 15.0_f32.to_radians();
    let (w, h) = (1024u32, 1024u32);
    let centroids = orion_synthetic_centroids(&db, fov_rad, w);

    let with_none: Vec<Centroid> = centroids
        .iter()
        .enumerate()
        .map(|(i, c)| Centroid {
            mass: if i % 3 == 0 { None } else { c.mass },
            ..*c
        })
        .collect();
    let with_nan: Vec<Centroid> = centroids
        .iter()
        .enumerate()
        .map(|(i, c)| Centroid {
            mass: if i % 3 == 0 { Some(f32::NAN) } else { c.mass },
            ..*c
        })
        .collect();

    let solve_config = SolveConfig {
        fov_max_error_rad: Some(5.0_f32.to_radians()),
        solve_timeout_ms: Some(30_000),
        ..SolveConfig::new(fov_rad, w, h)
    };
    let a = db
        .solve_from_centroids(&with_none, &solve_config)
        .expect("solve with None masses");
    let b = db
        .solve_from_centroids(&with_nan, &solve_config)
        .expect("solve with NaN masses");

    assert_eq!(a.matched_centroid_indices, b.matched_centroid_indices);
    assert_eq!(a.matched_catalog_ids, b.matched_catalog_ids);
    let qa = a.qicrs2cam.to_rotation_matrix();
    let qb = b.qicrs2cam.to_rotation_matrix();
    for r in 0..3 {
        for c in 0..3 {
            assert!((qa[(r, c)] - qb[(r, c)]).abs() < 1e-6);
        }
    }
}

/// Solve a mirrored (parity-flipped) synthetic field end to end.
///
/// Regression test: the finalize path used to rebuild the rotation with a
/// parity branch that produced a reflection (det −1), so every
/// `parity_flip = true` solve returned a meaningless quaternion and huge
/// residuals. The rest of the suite never exercised this (the skyview tests
/// pre-correct parity and the synthetic fields are proper), so the full
/// pipeline is covered here: mirrored centroids → parity detection → WCS
/// refinement → quaternion / residuals / pixel_to_world.
#[test]
fn test_parity_flipped_solve() {
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();

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

    // A non-trivial roll matters: the broken branch also depended on θ.
    let ra = 83.0_f32.to_radians();
    let dec = (-1.0_f32).to_radians();
    let roll = 40.0_f32.to_radians();
    let rot = rotation_from_ra_dec_roll(ra, dec, roll);
    let boresight_icrs =
        Vector3::from_array([dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()]);

    let fov_rad = 15.0_f32.to_radians();
    let image_width = 1024u32;
    let image_height = 1024u32;
    let pixel_scale = {
        let f = (image_width as f32 / 2.0) / (fov_rad / 2.0).tan();
        1.0 / f
    };

    // Generate a proper field, then mirror the x-axis (e.g. a FITS image with
    // det(CD) < 0 read without parity correction).
    let mut centroids = generate_centroids(&db, &rot, &boresight_icrs, fov_rad / 2.0, pixel_scale);
    assert!(centroids.len() >= 4, "need ≥4 centroids");
    for c in &mut centroids {
        c.x = -c.x;
    }

    let solve_config = SolveConfig {
        fov_max_error_rad: Some(5.0_f32.to_radians()),
        match_radius: 0.01,
        match_threshold: 1e-5,
        solve_timeout_ms: Some(30_000),
        match_max_error: None,
        ..SolveConfig::new(fov_rad, image_width, image_height)
    };

    let solution = db
        .solve_from_centroids(&centroids, &solve_config)
        .expect("mirrored field should solve");

    assert!(
        solution.parity_flip,
        "mirrored field must be detected as parity-flipped"
    );

    // Full-attitude check: negating x undoes the mirror, so qicrs2cam must be
    // the rotation of the original (pre-mirror) camera frame.
    let solved_rot = solution.qicrs2cam.to_rotation_matrix();
    let rel = solved_rot * rot.transpose();
    let trace = rel[(0, 0)] + rel[(1, 1)] + rel[(2, 2)];
    let attitude_err = (((trace - 1.0) / 2.0).clamp(-1.0, 1.0) as f64).acos();
    println!(
        "Parity solve: attitude error {:.1}\", rmse {:.1}\", {} matches",
        attitude_err.to_degrees() * 3600.0,
        solution.rmse_rad.to_degrees() * 3600.0,
        solution.num_matches
    );
    assert!(
        attitude_err < (120.0 / 3600.0_f64).to_radians(),
        "attitude error {:.1}\" exceeds 120\"",
        attitude_err.to_degrees() * 3600.0
    );

    // The finalize residuals are recomputed from the quaternion-bearing
    // rotation — they blow up to degrees if the conventions diverge.
    assert!(
        solution.rmse_rad.to_degrees() * 3600.0 < 60.0,
        "rmse {:.1}\" exceeds 60\"",
        solution.rmse_rad.to_degrees() * 3600.0
    );

    // pixel_to_world takes *observed* (mirrored) pixels; the camera model
    // applies the parity internally. Check every matched star round-trips.
    for (k, &cent_idx) in solution.matched_centroid_indices.iter().enumerate() {
        let cat_id = solution.matched_catalog_ids[k];
        let star_idx = db
            .star_catalog_ids
            .iter()
            .position(|&id| id == cat_id)
            .expect("matched catalog id present");
        let sv = &db.star_vectors[star_idx];
        let star_v = Vector3::from_array([sv[0], sv[1], sv[2]]);

        let (ra_deg, dec_deg) =
            solution.pixel_to_world(centroids[cent_idx].x as f64, centroids[cent_idx].y as f64);
        let (ra_r, dec_r) = (ra_deg.to_radians() as f32, dec_deg.to_radians() as f32);
        let pred_v = Vector3::from_array([
            dec_r.cos() * ra_r.cos(),
            dec_r.cos() * ra_r.sin(),
            dec_r.sin(),
        ]);
        let sep = angular_separation(&pred_v, &star_v);
        assert!(
            sep.to_degrees() * 3600.0 < 60.0,
            "pixel_to_world off by {:.1}\" for catalog id {}",
            sep.to_degrees() * 3600.0,
            cat_id
        );
    }
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
        fov_max_error_rad: Some(2.0_f32.to_radians()),
        match_radius: 0.01,
        match_threshold: 1e-5,
        solve_timeout_ms: Some(10_000),
        match_max_error: None,
        ..SolveConfig::new(fov_rad, image_width, image_height)
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
        let boresight_icrs =
            Vector3::from_array([dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()]);

        let centroids = generate_centroids(&db, &rot, &boresight_icrs, half_fov, pixel_scale);

        if centroids.len() < 4 {
            n_too_few += 1;
            continue;
        }

        let result = db.solve_from_centroids(&centroids, &solve_config);

        match result {
            Ok(solution) => {
                all_solve_times_ms.push(solution.solve_time_ms);

                // Compute boresight error
                let true_quat = Quaternion::from_rotation_matrix(&rot);
                let solved_quat = solution.qicrs2cam;
                let solved_boresight = solved_quat.inverse() * Vector3::from_array([0.0, 0.0, 1.0]);
                let true_boresight = true_quat.inverse() * Vector3::from_array([0.0, 0.0, 1.0]);
                let err_rad = angular_separation(&solved_boresight, &true_boresight);
                let err_arcsec = err_rad.to_degrees() * 3600.0;

                all_errors_arcsec.push(err_arcsec);
                all_match_counts.push(solution.num_matches);
                all_rmse_arcsec.push(solution.rmse_rad.to_degrees() * 3600.0);

                if err_arcsec < correct_threshold_arcsec {
                    n_correct += 1;
                } else if err_arcsec < wrong_threshold_arcsec {
                    n_imprecise += 1;
                    println!(
                        "  Trial {:4}: IMPRECISE err={:.1}\" matches={} RA={:.1}° Dec={:.1}° ({} centroids)",
                        trial, err_arcsec, solution.num_matches,
                        ra.to_degrees(), dec.to_degrees(), centroids.len(),
                    );
                } else {
                    n_wrong += 1;
                    println!(
                        "  Trial {:4}: WRONG err={:.1}\" matches={} RA={:.1}° Dec={:.1}° ({} centroids)",
                        trial, err_arcsec, solution.num_matches,
                        ra.to_degrees(), dec.to_degrees(), centroids.len(),
                    );
                }
            }
            Err(fail) => match fail.status {
                SolveStatus::NoMatch => {
                    n_no_match += 1;
                    all_solve_times_ms.push(fail.solve_time_ms);
                }
                SolveStatus::Timeout => {
                    n_timeout += 1;
                    all_solve_times_ms.push(fail.solve_time_ms);
                }
                SolveStatus::TooFew => n_too_few += 1,
                SolveStatus::InvalidConfig => {
                    panic!("statistical trials use a valid config; got InvalidConfig")
                }
            },
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

    // The file carries the format header, and a legacy (pre-header) bare
    // postcard payload still loads through the same entry point.
    let bytes = std::fs::read(tmp_path).expect("read saved database");
    assert_eq!(
        &bytes[..4],
        b"T3DB",
        "saved database must start with the magic"
    );
    let legacy = postcard::to_allocvec(&db).expect("bare postcard payload");
    let from_legacy = SolverDatabase::from_bytes(&legacy).expect("legacy payload must load");
    assert_eq!(from_legacy.props.num_patterns, db.props.num_patterns);
    let from_header = SolverDatabase::from_bytes(&bytes).expect("header payload must load");
    assert_eq!(from_header.pattern_catalog.len(), db.pattern_catalog.len());

    // Clean up temporary file
    std::fs::remove_file(tmp_path).expect("Failed to delete temporary file");

    // ── Corruption is rejected at load, not deferred to a mid-solve panic ──
    // Each tampered database decodes cleanly through postcard; without
    // validate() the inconsistency only surfaces as an out-of-bounds index
    // (or unbounded key enumeration) during solve_from_centroids.
    let n_stars = db.star_catalog.len() as u32;

    let mut tampered = db.clone();
    tampered.star_vectors.truncate(db.star_vectors.len() / 2);
    assert!(
        tampered.validate().is_err(),
        "truncated star_vectors must fail validation"
    );

    let mut tampered = db.clone();
    tampered.star_catalog_ids.push(0);
    assert!(
        tampered.validate().is_err(),
        "over-long star_catalog_ids must fail validation"
    );

    let mut tampered = db.clone();
    let slot = tampered
        .pattern_catalog
        .entries
        .iter()
        .position(|e| !e.is_empty())
        .expect("generated database has at least one pattern");
    tampered.pattern_catalog.entries[slot].star_indices = [n_stars, 0, 0, 0];
    assert!(
        tampered.validate().is_err(),
        "pattern entry indexing past the star table must fail validation"
    );

    let mut tampered = db.clone();
    tampered.props.pattern_bins = u32::MAX; // would explode key enumeration
    assert!(
        tampered.validate().is_err(),
        "pattern_bins inconsistent with pattern_max_error must fail validation"
    );

    // An invalid solve config fails fast with InvalidConfig instead of
    // burning the search to a guaranteed NoMatch (or warn-and-proceed).
    let dummy_centroids: Vec<Centroid> = (0..6)
        .map(|i| Centroid {
            x: 40.0 * i as f32 - 100.0,
            y: 25.0 * i as f32 - 60.0,
            mass: Some(100.0),
            cov: None,
        })
        .collect();
    let fail = db
        .solve_from_centroids(&dummy_centroids, &SolveConfig::default())
        .expect_err("placeholder camera model must not solve");
    assert_eq!(fail.status, SolveStatus::InvalidConfig);
    let mut bad_cfg = SolveConfig::new(20.0_f32.to_radians(), 1024, 768);
    bad_cfg.match_radius = f32::NAN;
    let fail = db
        .solve_from_centroids(&dummy_centroids, &bad_cfg)
        .expect_err("NaN match_radius must not solve");
    assert_eq!(fail.status, SolveStatus::InvalidConfig);

    // And end-to-end: a bit-flipped file either fails to decode (postcard) or
    // fails validation — it must never load successfully with corrupt indices.
    let bad_path = "temp_db_corrupt.bin";
    tampered.star_vectors.truncate(3);
    let bytes = tampered.to_bytes().expect("serialize tampered db");
    std::fs::write(bad_path, &bytes).expect("write corrupt db");
    assert!(
        SolverDatabase::load_from_file(bad_path).is_err(),
        "corrupt database file must be rejected at load"
    );
    std::fs::remove_file(bad_path).ok();
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
        fov_max_error_rad: Some(2.0_f32.to_radians()),
        match_radius: 0.01,
        match_threshold: 1e-5,
        solve_timeout_ms: Some(10_000),
        match_max_error: None,
        ..SolveConfig::new(fov_rad, image_width, image_height)
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
        let boresight_icrs =
            Vector3::from_array([dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()]);

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

        match result {
            Ok(solution) => {
                all_solve_times_ms.push(solution.solve_time_ms);

                let true_quat = Quaternion::from_rotation_matrix(&rot);
                let solved_quat = solution.qicrs2cam;
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
                all_match_counts.push(solution.num_matches);
                all_rmse_arcsec.push(solution.rmse_rad.to_degrees() * 3600.0);

                if err_arcsec < correct_threshold_arcsec {
                    n_correct += 1;
                } else if err_arcsec < wrong_threshold_arcsec {
                    n_imprecise += 1;
                    println!(
                        "  Trial {:4}: IMPRECISE err={:.1}\" matches={} RA={:.1}° Dec={:.1}° ({} centroids)",
                        trial, err_arcsec, solution.num_matches,
                        ra.to_degrees(), dec.to_degrees(), centroids.len(),
                    );
                } else {
                    n_wrong += 1;
                    println!(
                        "  Trial {:4}: WRONG err={:.1}\" matches={} RA={:.1}° Dec={:.1}° ({} centroids)",
                        trial, err_arcsec, solution.num_matches,
                        ra.to_degrees(), dec.to_degrees(), centroids.len(),
                    );
                }
            }
            Err(fail) => match fail.status {
                SolveStatus::NoMatch => {
                    n_no_match += 1;
                    all_solve_times_ms.push(fail.solve_time_ms);
                }
                SolveStatus::Timeout => {
                    n_timeout += 1;
                    all_solve_times_ms.push(fail.solve_time_ms);
                }
                SolveStatus::TooFew => n_too_few += 1,
                SolveStatus::InvalidConfig => {
                    panic!("statistical trials use a valid config; got InvalidConfig")
                }
            },
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
            fov_max_error_rad: Some(2.0_f32.to_radians()),
            solve_timeout_ms: Some(10_000),
            ..SolveConfig::new(fov_rad, image_width, image_height)
        };
        let Ok(lis_solution) = db.solve_from_centroids(&centroids, &lis_config) else {
            continue;
        };
        n_lis_ok += 1;
        lis_time_ms.push(lis_solution.solve_time_ms);
        let lis_quat = lis_solution.qicrs2cam;

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
            attitude_hint: Some(hinted_quat),
            hint_uncertainty_rad: 1.0_f32.to_radians(),
            strict_hint: true, // disable LIS fallback so we measure tracking alone
            solve_timeout_ms: Some(2_000),
            ..SolveConfig::with_camera_model(lis_solution.camera_model.clone())
        };
        match db.solve_from_centroids(&centroids, &track_config) {
            Ok(track_solution) => {
                n_track_ok += 1;
                track_time_ms.push(track_solution.solve_time_ms);

                // Verify the tracked solution agrees with the LIS solution.
                // On noiseless synthetic data the two paths converge to the same
                // fixed point of wcs_refine, so agreement is at f32 floating-point
                // noise (effectively zero). 1″ is very loose and only catches
                // gross regressions — tighten if we ever want to detect subtler
                // divergence between the two paths.
                let tq = track_solution.qicrs2cam;
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
            }
            Err(fail) => {
                println!(
                    "  Trial {:2}: tracking FAILED (status={:?}, perturb={:.1}')",
                    trial, fail.status, perturb_arcmin
                );
            }
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
    println!(
        "  Tracking-mode test ({} trials, {:.1}' hint perturbation)",
        n_trials, perturb_arcmin
    );
    println!(
        "    LIS solves successful:        {:3}/{}",
        n_lis_ok, n_trials
    );
    println!(
        "    Tracking solves successful:   {:3}/{}",
        n_track_ok, n_lis_ok
    );
    println!(
        "    Tracking agrees with LIS:     {:3}/{}",
        n_track_recovers_perturbed, n_track_ok
    );
    println!("    Mean LIS time:      {:7.2} ms", mean(&lis_time_ms));
    println!("    Mean tracking time: {:7.2} ms", mean(&track_time_ms));
    println!("══════════════════════════════════════════════════════════════\n");

    assert!(
        n_lis_ok >= 15,
        "LIS only solved {}/{} — DB may be too sparse",
        n_lis_ok,
        n_trials
    );
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

/// Regression test for issue #13: multiscale databases that produce very
/// large pattern tables should save and load successfully.
///
/// Originally exercised the sharded `PatternCatalog` workaround for rkyv's
/// 32-bit offset limit. After the postcard migration the offset limit is
/// gone, so this test simply verifies a multi-octave database round-trips.
///
/// This test is expensive: it generates a multiscale database covering
/// several FOV octaves, which typically produces tens of millions of unique
/// patterns and requires multi-GB RAM. Marked `#[ignore]` so it only runs
/// when explicitly requested:
///
/// ```sh
/// cargo test --release --test integration_test test_multiscale_database -- --ignored --nocapture
/// ```
#[test]
#[ignore = "slow: generates a multi-GB pattern catalog; run with --ignored"]
fn test_multiscale_database() {
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();

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
    println!(
        "  {} pattern slots ({} patterns stored)",
        total_slots, db.props.num_patterns
    );

    let tmp_path = std::env::temp_dir().join("tetra3rs_multiscale_test.bin");
    println!("Saving to {}…", tmp_path.display());
    db.save_to_file(tmp_path.to_str().unwrap())
        .expect("save_to_file");

    println!("Loading…");
    let loaded =
        SolverDatabase::load_from_file(tmp_path.to_str().unwrap()).expect("load_from_file");
    assert_eq!(loaded.pattern_catalog.len(), total_slots);
    assert_eq!(loaded.props.num_patterns, db.props.num_patterns);

    std::fs::remove_file(tmp_path).ok();
}

/// `max_patterns_checked` bounds the lost-in-space search by work rather
/// than wall time: a field of random centroids never matches, and a tiny
/// budget ends the search with `Timeout` deterministically — no clock
/// involved — while `Some(0)` is rejected up front as `InvalidConfig`.
#[test]
fn test_pattern_budget_reports_timeout() {
    fn centroid(x: f32, y: f32) -> Centroid {
        Centroid {
            x,
            y,
            mass: None,
            cov: None,
        }
    }

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

    let mut rng = StdRng::seed_from_u64(42);
    let image_width = 1024u32;
    let centroids: Vec<Centroid> = (0..16)
        .map(|_| {
            centroid(
                (rng.random::<f32>() - 0.5) * image_width as f32,
                (rng.random::<f32>() - 0.5) * image_width as f32,
            )
        })
        .collect();

    // Generous wall-clock so the *pattern* budget is what trips.
    let base = || SolveConfig {
        solve_timeout_ms: Some(60_000),
        fov_max_error_rad: Some(2.0_f32.to_radians()),
        ..SolveConfig::new(15.0_f32.to_radians(), image_width, image_width)
    };

    let budgeted = SolveConfig {
        max_patterns_checked: Some(3),
        ..base()
    };
    let err = db
        .solve_from_centroids(&centroids, &budgeted)
        .expect_err("random centroids must not solve");
    assert_eq!(err.status, SolveStatus::Timeout);
    assert!(
        err.solve_time_ms < 1000.0,
        "3-pattern budget should end the search almost immediately, took {} ms",
        err.solve_time_ms
    );

    // Unbounded budget on the same field exhausts the combinations instead.
    let unbounded = SolveConfig {
        max_patterns_checked: None,
        ..base()
    };
    let err = db
        .solve_from_centroids(&centroids, &unbounded)
        .expect_err("random centroids must not solve");
    assert_eq!(err.status, SolveStatus::NoMatch);

    let zero = SolveConfig {
        max_patterns_checked: Some(0),
        ..base()
    };
    let err = db
        .solve_from_centroids(&centroids, &zero)
        .expect_err("zero budget is an invalid config");
    assert_eq!(err.status, SolveStatus::InvalidConfig);
}

/// Calibration must correct stellar aberration with the velocity each solve
/// recorded: the differential aberration across a frame (≈ 1e-4 of the
/// field) is otherwise fitted as lens distortion or left in the residuals.
/// Synthetic sky: catalog positions aberrated for a 30 km/s observer,
/// projected through an ideal pinhole (no distortion), solved with the
/// velocity given, then calibrated (a) with the recorded velocity and (b)
/// with it stripped from the solutions.
#[test]
fn test_calibration_corrects_aberration_per_image() {
    let db = small_test_db();
    let fov_rad = 15.0_f32.to_radians();
    let (w, h) = (2048u32, 2048u32);
    let half_fov = fov_rad / 2.0;
    let pixel_scale = 1.0 / ((w as f32 / 2.0) / half_fov.tan());
    let velocity = [0.0f64, 30.0, 0.0]; // km/s, ICRS
    let c_km_s = 299_792.458f64;
    let beta = [
        velocity[0] / c_km_s,
        velocity[1] / c_km_s,
        velocity[2] / c_km_s,
    ];

    // Two pointings on the sky where the velocity is nearly transverse
    // (largest differential aberration): boresights near ±X.
    let pointings = [
        (0.0f32, 5.0f32.to_radians(), 0.3f32),
        (0.2f32, -8.0f32.to_radians(), 1.1f32),
    ];
    let mut solve_results: Vec<tetra3::SolveResult> = Vec::new();
    let mut stripped: Vec<tetra3::SolveResult> = Vec::new();
    let mut centroid_sets: Vec<Vec<Centroid>> = Vec::new();
    for &(ra, dec, roll) in &pointings {
        let rot = rotation_from_ra_dec_roll(ra, dec, roll);
        let mut centroids = Vec::new();
        for (i, sv) in db.star_vectors.iter().enumerate() {
            // Apparent direction: s' = (s + β) / |s + β|.
            let a = [
                sv[0] as f64 + beta[0],
                sv[1] as f64 + beta[1],
                sv[2] as f64 + beta[2],
            ];
            let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
            let v = rot
                * Vector3::from_array([(a[0] / n) as f32, (a[1] / n) as f32, (a[2] / n) as f32]);
            if v[2] > 0.01 {
                let (cx, cy) = (v[0] / v[2], v[1] / v[2]);
                if cx.abs() < half_fov && cy.abs() < half_fov {
                    centroids.push(Centroid {
                        x: cx / pixel_scale,
                        y: cy / pixel_scale,
                        mass: Some(10.0 - db.star_catalog.stars()[i].mag),
                        cov: None,
                    });
                }
            }
        }
        let config = SolveConfig {
            fov_max_error_rad: Some(2.0_f32.to_radians()),
            solve_timeout_ms: Some(30_000),
            observer_velocity_km_s: Some(velocity),
            ..SolveConfig::new(fov_rad, w, h)
        };
        let solution = db
            .solve_from_centroids(&centroids, &config)
            .expect("aberrated field should solve");
        assert_eq!(solution.observer_velocity_km_s, Some(velocity));
        let mut without = solution.clone();
        without.observer_velocity_km_s = None;
        solve_results.push(Ok(solution));
        stripped.push(Ok(without));
        centroid_sets.push(centroids);
    }

    let cal_config = tetra3::CalibrateConfig {
        model: tetra3::DistortionModelType::Polynomial { order: 3 },
        ..Default::default()
    };
    let cents: Vec<&[Centroid]> = centroid_sets.iter().map(|c| c.as_slice()).collect();
    for n_images in [1usize, 2] {
        let with: Vec<&tetra3::SolveResult> = solve_results.iter().take(n_images).collect();
        let without: Vec<&tetra3::SolveResult> = stripped.iter().take(n_images).collect();
        let corrected = tetra3::calibrate_camera(&with, &cents[..n_images], &db, w, h, &cal_config)
            .expect("calibration with recorded velocity");
        let uncorrected =
            tetra3::calibrate_camera(&without, &cents[..n_images], &db, w, h, &cal_config)
                .expect("calibration without velocity");
        // Bias of the fitted model relative to an ideal pinhole at its own
        // focal length: the polynomial's order-0/1 terms absorb a uniform
        // shift and a linear stretch, so the aberration shows up here, not
        // in the residual.
        let model_bias = |cam: &tetra3::CameraModel| -> f64 {
            let f = cam.focal_length_px;
            let hw = w as f64 / 2.0;
            let hh = h as f64 / 2.0;
            [(0.0, 0.0), (hw, hh), (-hw, hh), (hw, -hh), (-hw, -hh)]
                .iter()
                .map(|&(x, y)| {
                    let (px, py) = cam.tanplane_to_pixel(x / f, y / f);
                    ((px - x).powi(2) + (py - y).powi(2)).sqrt()
                })
                .fold(0.0, f64::max)
        };
        let bias_with = model_bias(&corrected.camera_model);
        let bias_without = model_bias(&uncorrected.camera_model);
        println!(
            "{n_images} image(s): model bias with velocity {bias_with:.4} px, without {bias_without:.4} px \
             (rmse after {:.4} / {:.4})",
            corrected.rmse_after_px, uncorrected.rmse_after_px
        );
        // With the recorded velocity the sky is a perfect pinhole.
        assert!(bias_with < 0.03, "corrected model bias {bias_with} px");
        // Without it: the single-image path fits the uniform shift (~20″,
        // 0.75 px here) into the optical center; the multi-image path
        // re-fits each attitude and is left with the differential part
        // (~1e-4 of the field, a few hundredths of a pixel at the corners).
        let min_bias = if n_images == 1 { 0.3 } else { 0.02 };
        assert!(
            bias_without > min_bias && bias_without > 20.0 * bias_with,
            "stripping the velocity should bias the model: {bias_without} px vs {bias_with} px"
        );
    }
}
