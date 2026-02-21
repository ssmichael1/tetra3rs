//! Integration test: extract star centroids from SkyView survey FITS images (10° FOV),
//! solve for attitude, and compare against the WCS solution in the FITS header.
//!
//! These images use simple TAN (gnomonic) projection with CDELT (no SIP distortion),
//! data in HDU 0, and are 2048×2048 pixels at ~17.6"/px.

mod test_data;

use nalgebra::{Rotation3, UnitQuaternion, Vector3};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use tetra3::{
    CentroidExtractionConfig, GenerateDatabaseConfig, SolveConfig, SolveStatus, SolverDatabase,
};

// ═══════════════════════════════════════════════════════════════════════════
// Minimal FITS reader
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
enum FitsValue {
    Float(f64),
    Int(i64),
    #[allow(dead_code)]
    Str(String),
}

struct FitsHdu {
    headers: HashMap<String, FitsValue>,
    data_offset: u64,
    data_len: u64,
}

fn parse_header_card(card: &[u8; 80]) -> Option<(String, FitsValue)> {
    let card_str = String::from_utf8_lossy(card);
    let keyword = card_str[..8].trim().to_string();
    if keyword.is_empty() || keyword == "COMMENT" || keyword == "HISTORY" || keyword == "END" {
        return None;
    }
    if card_str.len() < 10 || &card_str[8..10] != "= " {
        return None;
    }
    let value_str = card_str[10..].trim();
    let value = if value_str.starts_with('\'') {
        if let Some(end) = value_str[1..].find('\'') {
            FitsValue::Str(value_str[1..1 + end].trim().to_string())
        } else {
            FitsValue::Str(value_str[1..].trim().to_string())
        }
    } else {
        let num_part = if let Some(slash) = value_str.find('/') {
            value_str[..slash].trim()
        } else {
            value_str
        };
        if let Ok(i) = num_part.parse::<i64>() {
            FitsValue::Int(i)
        } else if let Ok(f) = num_part.parse::<f64>() {
            FitsValue::Float(f)
        } else {
            FitsValue::Str(num_part.to_string())
        }
    };
    Some((keyword, value))
}

fn read_fits_hdus(path: &str) -> Vec<FitsHdu> {
    let mut file = File::open(path).expect("Failed to open FITS file");
    let mut hdus = Vec::new();
    let mut offset: u64 = 0;

    loop {
        let mut headers = HashMap::new();
        let mut found_end = false;

        loop {
            let mut block = [0u8; 2880];
            if file.read_exact(&mut block).is_err() {
                return hdus;
            }
            offset += 2880;
            for i in 0..36 {
                let card: &[u8; 80] = block[i * 80..(i + 1) * 80].try_into().unwrap();
                let card_str = String::from_utf8_lossy(card);
                if card_str.starts_with("END") {
                    found_end = true;
                    break;
                }
                if let Some((k, v)) = parse_header_card(card) {
                    headers.insert(k, v);
                }
            }
            if found_end {
                break;
            }
        }

        let naxis = match headers.get("NAXIS") {
            Some(FitsValue::Int(n)) => *n as usize,
            _ => 0,
        };
        let bitpix = match headers.get("BITPIX") {
            Some(FitsValue::Int(n)) => *n,
            _ => 8,
        };

        let mut data_len: u64 = if naxis > 0 {
            let bytes_per_pixel = (bitpix.unsigned_abs() as u64) / 8;
            let mut npixels: u64 = 1;
            for i in 1..=naxis {
                let key = format!("NAXIS{}", i);
                if let Some(FitsValue::Int(n)) = headers.get(&key) {
                    npixels *= *n as u64;
                }
            }
            npixels * bytes_per_pixel
        } else {
            0
        };

        if let Some(FitsValue::Int(pcount)) = headers.get("PCOUNT") {
            data_len += *pcount as u64;
        }

        let data_offset = offset;
        let padded = ((data_len + 2879) / 2880) * 2880;
        offset += padded;
        file.seek(SeekFrom::Start(offset)).ok();

        hdus.push(FitsHdu {
            headers,
            data_offset,
            data_len,
        });
    }
}

fn get_f64(hdu: &FitsHdu, key: &str) -> Option<f64> {
    match hdu.headers.get(key) {
        Some(FitsValue::Float(f)) => Some(*f),
        Some(FitsValue::Int(i)) => Some(*i as f64),
        _ => None,
    }
}

fn read_f32_image(path: &str, hdu: &FitsHdu) -> Vec<f32> {
    let mut file = File::open(path).expect("Failed to open FITS file");
    file.seek(SeekFrom::Start(hdu.data_offset)).unwrap();
    let npixels = hdu.data_len as usize / 4;
    let mut buf = vec![0u8; hdu.data_len as usize];
    file.read_exact(&mut buf).unwrap();
    let mut pixels = Vec::with_capacity(npixels);
    for i in 0..npixels {
        let bytes: [u8; 4] = buf[i * 4..(i + 1) * 4].try_into().unwrap();
        pixels.push(f32::from_be_bytes(bytes));
    }
    pixels
}

// ═══════════════════════════════════════════════════════════════════════════
// WCS helpers
// ═══════════════════════════════════════════════════════════════════════════

fn radec_to_uvec(ra_deg: f64, dec_deg: f64) -> Vector3<f32> {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    Vector3::new(
        (dec.cos() * ra.cos()) as f32,
        (dec.cos() * ra.sin()) as f32,
        dec.sin() as f32,
    )
}

fn uvec_to_radec(v: &Vector3<f32>) -> (f64, f64) {
    let dec = (v.z as f64).asin();
    let ra = (v.y as f64).atan2(v.x as f64);
    let ra_deg = ra.to_degrees();
    let ra_deg = if ra_deg < 0.0 { ra_deg + 360.0 } else { ra_deg };
    (ra_deg, dec.to_degrees())
}

fn angular_separation(a: &Vector3<f32>, b: &Vector3<f32>) -> f32 {
    let cross = a.cross(b).norm();
    let dot = a.dot(b);
    cross.atan2(dot)
}

/// Build attitude quaternion from CDELT-based WCS (TAN projection, no rotation).
///
/// CDELT1 < 0 (RA increases left/east), CDELT2 > 0 (Dec increases up/north).
/// No CROTA2, no CD matrix — axes are aligned with RA/Dec.
fn wcs_to_quaternion_cdelt(hdu: &FitsHdu) -> UnitQuaternion<f32> {
    let crval1 = get_f64(hdu, "CRVAL1").unwrap();
    let crval2 = get_f64(hdu, "CRVAL2").unwrap();

    let ra0 = crval1.to_radians();
    let dec0 = crval2.to_radians();

    let boresight = Vector3::new(
        (dec0.cos() * ra0.cos()) as f32,
        (dec0.cos() * ra0.sin()) as f32,
        dec0.sin() as f32,
    );

    let sin_ra = ra0.sin() as f32;
    let cos_ra = ra0.cos() as f32;
    let sin_dec = dec0.sin() as f32;
    let cos_dec = dec0.cos() as f32;

    // Right-handed tangent plane basis at boresight:
    //   +X = East = direction of increasing RA = (-sin_ra, cos_ra, 0)
    //   +Y = North = direction of increasing Dec
    //   +Z = Boresight (into the sky)
    //   Verify: East × North = Boresight ✓ (right-handed)
    //
    // After parity correction (negating centroid x for CDELT1 < 0), the
    // extracted centroids use +X = East, +Y = North, +Z = boresight.
    // This quaternion must match that convention.
    let e_east = Vector3::new(-sin_ra, cos_ra, 0.0).normalize();
    let e_north = Vector3::new(-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec).normalize();

    let cam_x_icrs = e_east;
    let cam_y_icrs = e_north;
    let cam_z_icrs = boresight.normalize();

    let rot = nalgebra::Matrix3::new(
        cam_x_icrs.x,
        cam_x_icrs.y,
        cam_x_icrs.z,
        cam_y_icrs.x,
        cam_y_icrs.y,
        cam_y_icrs.z,
        cam_z_icrs.x,
        cam_z_icrs.y,
        cam_z_icrs.z,
    );

    UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rot))
}

// ═══════════════════════════════════════════════════════════════════════════
// The test
// ═══════════════════════════════════════════════════════════════════════════

struct SkyViewTestCase {
    filename: &'static str,
    description: &'static str,
}

const SKYVIEW_TEST_CASES: &[SkyViewTestCase] = &[
    SkyViewTestCase {
        filename: "orion_region_10deg.fits",
        description: "Orion region (many bright stars)",
    },
    SkyViewTestCase {
        filename: "pleiades_region_10deg.fits",
        description: "Pleiades region",
    },
    SkyViewTestCase {
        filename: "cygnus_region_10deg.fits",
        description: "Cygnus region (Milky Way)",
    },
    SkyViewTestCase {
        filename: "andromeda_m31_10deg.fits",
        description: "Andromeda / M31 region",
    },
    SkyViewTestCase {
        filename: "north_pole_10deg.fits",
        description: "Near north celestial pole",
    },
    SkyViewTestCase {
        filename: "sagittarius_10deg.fits",
        description: "Sagittarius (galactic center)",
    },
    SkyViewTestCase {
        filename: "cassiopeia_10deg.fits",
        description: "Cassiopeia region",
    },
    SkyViewTestCase {
        filename: "scorpius_antares_10deg.fits",
        description: "Scorpius / Antares region",
    },
    SkyViewTestCase {
        filename: "south_pole_10deg.fits",
        description: "Near south celestial pole",
    },
    SkyViewTestCase {
        filename: "virgo_cluster_10deg.fits",
        description: "Virgo cluster region",
    },
];

/// Build database suitable for ~10° FOV SkyView images.
fn build_skyview_database() -> SolverDatabase {
    let config = GenerateDatabaseConfig {
        max_fov_deg: 15.0,
        min_fov_deg: None,
        star_max_magnitude: Some(7.0),
        pattern_max_error: 0.005,
        lattice_field_oversampling: 100,
        patterns_per_lattice_field: 150,
        verification_stars_per_fov: 50,
        multiscale_step: 1.5,
        epoch_proper_motion_year: Some(2000.0), // SkyView images use J2000 epoch
        catalog_nside: 8,
    };

    let catalog_path = test_data::ensure_test_file("data/hip2.dat");
    SolverDatabase::generate_from_hipparcos(&catalog_path, &config)
        .expect("Failed to generate database from Hipparcos catalog")
}

#[test]
fn test_skyview_fits_solve() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("debug")
        .try_init();

    // Ensure all test files are downloaded
    test_data::ensure_test_file("data/hip2.dat");
    for tc in SKYVIEW_TEST_CASES {
        test_data::ensure_test_file(&format!("data/skyview_10deg_test_images/{}", tc.filename));
    }

    let db = build_skyview_database();
    println!("\n══════════════════════════════════════════════════════════════");
    println!(
        "Database: {} stars, {} patterns",
        db.star_catalog.len(),
        db.props.num_patterns,
    );

    let mut passed = 0;
    let mut failed = 0;

    for tc in SKYVIEW_TEST_CASES {
        let fits_path = format!("data/skyview_10deg_test_images/{}", tc.filename);
        println!("\n══════════════════════════════════════════════════════════════");
        println!("Testing: {} ({})", tc.filename, tc.description);

        // ── Read the FITS file (data is in HDU 0) ──
        let hdus = read_fits_hdus(&fits_path);
        assert!(!hdus.is_empty(), "Expected at least 1 HDU in FITS file");

        let image_hdu = &hdus[0];
        let naxis1 = match image_hdu.headers.get("NAXIS1") {
            Some(FitsValue::Int(n)) => *n as u32,
            _ => panic!("Missing NAXIS1"),
        };
        let naxis2 = match image_hdu.headers.get("NAXIS2") {
            Some(FitsValue::Int(n)) => *n as u32,
            _ => panic!("Missing NAXIS2"),
        };
        println!("  Image size: {} x {}", naxis1, naxis2);

        let pixels = read_f32_image(&fits_path, image_hdu);
        assert_eq!(pixels.len(), (naxis1 as usize) * (naxis2 as usize));

        // Handle NaN/Inf pixels
        let clean_pixels: Vec<f32> = pixels
            .iter()
            .map(|&v| {
                if v.is_nan() || v.is_infinite() {
                    0.0
                } else {
                    v
                }
            })
            .collect();

        // Get WCS info
        let crval_ra = get_f64(image_hdu, "CRVAL1").unwrap();
        let crval_dec = get_f64(image_hdu, "CRVAL2").unwrap();
        let true_boresight = radec_to_uvec(crval_ra, crval_dec);

        let cdelt1 = get_f64(image_hdu, "CDELT1").unwrap(); // deg/px (negative)
        let cdelt2 = get_f64(image_hdu, "CDELT2").unwrap(); // deg/px (positive)
        let pixel_scale_deg = cdelt2.abs(); // same magnitude for both axes
        let fov_h_deg = pixel_scale_deg * naxis1 as f64;
        let fov_v_deg = pixel_scale_deg * naxis2 as f64;

        println!("  WCS: RA={:.4}°, Dec={:.4}°", crval_ra, crval_dec);
        println!(
            "  FOV: {:.2}° x {:.2}°, pixel scale {:.2}\"/px",
            fov_h_deg,
            fov_v_deg,
            pixel_scale_deg * 3600.0
        );
        println!("  CDELT: ({:.6}, {:.6}) deg/px", cdelt1, cdelt2);

        // ── Extract centroids ──
        // The module now handles local background subtraction internally
        // (block median + bilinear interpolation), so we can use moderate
        // sigma threshold. After local BG subtraction the residual is
        // mostly noise + point sources.
        let extract_config = CentroidExtractionConfig {
            sigma_threshold: 10.0,
            min_pixels: 3,
            max_pixels: 10000,
            max_centroids: Some(200),
            sigma_clip_iterations: 5,
            sigma_clip_factor: 3.0,
            use_8_connectivity: true,
            local_bg_block_size: Some(64),
            max_elongation: Some(3.0),
        };

        let extraction =
            tetra3::extract_centroids_from_raw(&clean_pixels, naxis1, naxis2, &extract_config)
                .expect("Centroid extraction failed");

        println!(
            "  Extracted {} centroids (from {} raw blobs)",
            extraction.centroids.len(),
            extraction.num_blobs_raw
        );
        println!(
            "  Background: mean={:.1}, sigma={:.1}, threshold={:.1}",
            extraction.background_mean, extraction.background_sigma, extraction.threshold
        );

        // ── Fix parity: negate x for CDELT1 < 0 ──
        // The centroid extractor gives +x = increasing column = West (when CDELT1 < 0).
        // This makes (West, North, Boresight) a LEFT-handed frame (det = -1).
        // The solver needs a RIGHT-handed frame, so we negate x to get +x = East.
        let parity_x: f32 = if cdelt1 < 0.0 { -1.0 } else { 1.0 };
        let parity_y: f32 = if cdelt2 < 0.0 { -1.0 } else { 1.0 };
        let corrected_centroids: Vec<tetra3::Centroid> = extraction
            .centroids
            .iter()
            .map(|c| tetra3::Centroid {
                x: c.x * parity_x,
                y: c.y * parity_y,
                mass: c.mass,
                cov: c.cov,
            })
            .collect();

        // Print first 10 centroids (after parity correction)
        println!(
            "  First 10 centroids (parity corrected, x*={}, y*={}):",
            parity_x, parity_y
        );
        let pixel_scale = (fov_h_deg as f32).to_radians() / naxis1 as f32;
        for (i, c) in corrected_centroids.iter().take(10).enumerate() {
            let x_rad = c.x * pixel_scale;
            let y_rad = c.y * pixel_scale;
            println!(
                "    [{:2}] x={:+.1} px ({:+.3}°), y={:+.1} px ({:+.3}°), mass={:.0}",
                i,
                c.x,
                x_rad.to_degrees(),
                c.y,
                y_rad.to_degrees(),
                c.mass.unwrap_or(0.0)
            );
        }

        // ── Validate centroids by projecting catalog stars using known WCS ──
        let wcs_q = wcs_to_quaternion_cdelt(image_hdu);
        let wcs_boresight_uvec = wcs_q.inverse() * Vector3::new(0.0, 0.0, 1.0);
        let (wcs_ra, wcs_dec) = uvec_to_radec(&wcs_boresight_uvec);
        println!(
            "  WCS quaternion boresight: RA={:.4}°, Dec={:.4}°",
            wcs_ra, wcs_dec
        );

        let half_fov =
            ((fov_h_deg * fov_h_deg + fov_v_deg * fov_v_deg).sqrt() as f32 / 2.0).to_radians();
        let nearby_indices = db
            .star_catalog
            .query_indices_from_uvec(wcs_boresight_uvec, half_fov * 1.2);

        let mut catalog_centroids: Vec<(f32, f32, f32)> = Vec::new(); // (x_px, y_px, mag)
        for &idx in &nearby_indices {
            let sv = &db.star_vectors[idx];
            let icrs_v = Vector3::new(sv[0], sv[1], sv[2]);
            let cam_v = wcs_q * icrs_v;
            if cam_v.z > 0.01 {
                let cx_rad = cam_v.x / cam_v.z;
                let cy_rad = cam_v.y / cam_v.z;
                let half_fov_h = (fov_h_deg as f32 / 2.0).to_radians();
                let half_fov_v = (fov_v_deg as f32 / 2.0).to_radians();
                if cx_rad.abs() < half_fov_h && cy_rad.abs() < half_fov_v {
                    // Convert to pixel coordinates from image center
                    catalog_centroids.push((
                        cx_rad / pixel_scale,
                        cy_rad / pixel_scale,
                        db.star_catalog.stars()[idx].mag,
                    ));
                }
            }
        }
        catalog_centroids.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        println!(
            "  Catalog stars in FOV: {} (via WCS projection)",
            catalog_centroids.len()
        );
        println!("  Brightest 10 catalog stars (projected vs corrected centroids):");
        for (i, &(cx, cy, mag)) in catalog_centroids.iter().take(10).enumerate() {
            let mut best_dist = f32::MAX;
            let mut best_j = 0;
            for (j, ec) in corrected_centroids.iter().enumerate() {
                let dx = ec.x - cx;
                let dy = ec.y - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < best_dist {
                    best_dist = dist;
                    best_j = j;
                }
            }
            let dist_rad = best_dist * pixel_scale;
            println!(
                "    [{:2}] mag={:.1} pos=({:+.1} px, {:+.1} px) → closest centroid [{:2}] dist={:.1} px ({:.1}\")",
                i,
                mag,
                cx,
                cy,
                best_j,
                best_dist,
                dist_rad.to_degrees() * 3600.0,
            );
        }

        if extraction.centroids.len() < 4 {
            println!("  SKIP: Too few centroids");
            continue;
        }

        // ── Reference solve: use catalog stars projected through WCS ──
        let ref_centroids: Vec<tetra3::Centroid> = catalog_centroids
            .iter()
            .map(|&(cx, cy, mag)| tetra3::Centroid {
                x: cx,
                y: cy,
                mass: Some(10.0 - mag),
                cov: None,
            })
            .collect();

        let solve_config = SolveConfig {
            fov_estimate_rad: (fov_h_deg as f32).to_radians(),
            image_width: naxis1,
            image_height: naxis2,
            fov_max_error_rad: Some((3.0_f32).to_radians()),
            match_radius: 0.01,
            match_threshold: 1e-5,
            solve_timeout_ms: Some(60_000),
            match_max_error: None,
            refine_iterations: 2,
            ..Default::default()
        };

        let ref_result = db.solve_from_centroids(&ref_centroids, &solve_config);
        println!(
            "  Reference solve (WCS catalog): {:?} ({} centroids, {:.1}ms)",
            ref_result.status,
            ref_centroids.len(),
            ref_result.solve_time_ms
        );
        if ref_result.status == SolveStatus::MatchFound {
            let ref_q = ref_result.qicrs2cam.unwrap();
            let ref_boresight = ref_q.inverse() * Vector3::new(0.0, 0.0, 1.0);
            let ref_err = angular_separation(&ref_boresight, &true_boresight);
            println!(
                "  Reference boresight error: {:.1}\"",
                ref_err.to_degrees() * 3600.0
            );
        }

        // ── Solve from parity-corrected extracted centroids ──
        let result = db.solve_from_centroids(&corrected_centroids, &solve_config);

        println!("  Solve status: {:?}", result.status);
        println!("  Solve time:   {:.1} ms", result.solve_time_ms);

        if result.status == SolveStatus::MatchFound {
            let solved_q = result.qicrs2cam.unwrap();
            let solved_boresight = solved_q.inverse() * Vector3::new(0.0, 0.0, 1.0);
            let (solved_ra, solved_dec) = uvec_to_radec(&solved_boresight);
            let error_rad = angular_separation(&solved_boresight, &true_boresight);
            let error_arcmin = error_rad.to_degrees() * 60.0;

            println!(
                "  Solved:       RA={:.4}°, Dec={:.4}°",
                solved_ra, solved_dec
            );
            println!(
                "  Boresight error: {:.2}' ({:.1}\")",
                error_arcmin,
                error_arcmin * 60.0
            );
            if let Some(n) = result.num_matches {
                println!("  Matched stars: {}", n);
            }
            if let Some(rmse) = result.rmse_rad {
                println!("  RMSE:         {:.1}\"", rmse.to_degrees() * 3600.0);
            }
            if let Some(fov) = result.fov_rad {
                println!("  Solved FOV:   {:.2}°", fov.to_degrees());
            }

            if error_arcmin > 30.0 {
                println!(
                    "  *** FAIL: boresight error {:.1}' exceeds 30' ***",
                    error_arcmin
                );
                failed += 1;
            } else {
                println!("  PASS");
                passed += 1;
            }
        } else {
            println!("  *** FAIL: no match found ***");
            failed += 1;
        }
    }

    println!("\n══════════════════════════════════════════════════════════════");
    println!(
        "RESULTS: {}/{} passed, {}/{} failed",
        passed,
        SKYVIEW_TEST_CASES.len(),
        failed,
        SKYVIEW_TEST_CASES.len()
    );
    println!("══════════════════════════════════════════════════════════════");
    assert_eq!(failed, 0, "{} SkyView solve tests failed", failed);
}
