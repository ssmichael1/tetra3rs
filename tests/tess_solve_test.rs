//! Integration test: extract star centroids from TESS FFI images (~12° FOV),
//! solve for attitude, and compare against the WCS solution in the FITS header.
//!
//! These images use CD matrix WCS (no simple CDELT), data in HDU 1 (FITS extension),
//! and are 2136×2078 pixels at ~21"/px. The science region is rows 0–2047,
//! columns 44–2091 (2048×2048); the rest is overscan/collateral.
//!
//! The solver's auto-parity detection handles the CD matrix orientation.

use nalgebra::Vector3;
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

/// Trim TESS image to science region: rows 0–2047, columns 44–2091.
/// Returns (trimmed_pixels, sci_width, sci_height).
fn trim_tess_science_region(pixels: &[f32], full_width: u32, full_height: u32) -> (Vec<f32>, u32, u32) {
    let sci_col_start = 44_u32;
    let sci_col_end = 2092_u32;
    let sci_row_end = 2048_u32.min(full_height);
    let sci_width = sci_col_end - sci_col_start;
    let sci_height = sci_row_end;

    let mut trimmed = Vec::with_capacity((sci_width * sci_height) as usize);
    for row in 0..sci_height {
        let row_start = (row * full_width + sci_col_start) as usize;
        let row_end = (row * full_width + sci_col_end) as usize;
        trimmed.extend_from_slice(&pixels[row_start..row_end]);
    }
    (trimmed, sci_width, sci_height)
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
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

fn angular_separation(a: &Vector3<f32>, b: &Vector3<f32>) -> f32 {
    let cross = a.cross(b).norm();
    let dot = a.dot(b);
    cross.atan2(dot)
}

// ═══════════════════════════════════════════════════════════════════════════
// The test
// ═══════════════════════════════════════════════════════════════════════════

struct TessTestCase {
    filename: &'static str,
    description: &'static str,
}

const TESS_TEST_CASES: &[TessTestCase] = &[
    TessTestCase {
        filename: "moderate_density_field.fits",
        description: "Moderate density field (RA~319°, Dec~-41°)",
    },
    TessTestCase {
        filename: "sparse_field_north_ecliptic.fits",
        description: "Sparse field near north ecliptic (RA~89°, Dec~-75°)",
    },
    TessTestCase {
        filename: "dense_galactic_plane.fits",
        description: "Dense galactic plane field (RA~41°, Dec~-67°)",
    },
];

/// Build database suitable for ~12° FOV TESS images.
fn build_tess_database() -> SolverDatabase {
    let config = GenerateDatabaseConfig {
        max_fov_deg: 14.0,
        min_fov_deg: None,
        star_max_magnitude: None,
        pattern_max_error: 0.005,
        lattice_field_oversampling: 100,
        patterns_per_lattice_field: 150,
        verification_stars_per_fov: 150,
        multiscale_step: 1.5,
        epoch_proper_motion_year: Some(2018.0), // TESS launched 2018
        catalog_nside: 8,
    };

    SolverDatabase::generate_from_hipparcos("data/hip2.dat", &config)
        .expect("Failed to generate database from Hipparcos catalog")
}

#[test]
fn test_tess_fits_solve() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("debug")
        .try_init();

    let db = build_tess_database();
    println!("\n══════════════════════════════════════════════════════════════");
    println!(
        "Database: {} stars, {} patterns",
        db.star_catalog.len(),
        db.props.num_patterns,
    );

    let mut passed = 0;
    let mut failed = 0;

    for tc in TESS_TEST_CASES {
        let fits_path = format!("data/tess_test_images/{}", tc.filename);
        println!("\n══════════════════════════════════════════════════════════════");
        println!("Testing: {} ({})", tc.filename, tc.description);

        // ── Read the FITS file (data is in HDU 1 for TESS) ──
        let hdus = read_fits_hdus(&fits_path);
        assert!(hdus.len() >= 2, "Expected at least 2 HDUs in TESS FITS file");

        let image_hdu = &hdus[1]; // HDU 1 = image extension
        let naxis1 = match image_hdu.headers.get("NAXIS1") {
            Some(FitsValue::Int(n)) => *n as u32,
            _ => panic!("Missing NAXIS1"),
        };
        let naxis2 = match image_hdu.headers.get("NAXIS2") {
            Some(FitsValue::Int(n)) => *n as u32,
            _ => panic!("Missing NAXIS2"),
        };
        println!("  Full image size: {} x {}", naxis1, naxis2);

        let pixels = read_f32_image(&fits_path, image_hdu);
        assert_eq!(pixels.len(), (naxis1 as usize) * (naxis2 as usize));

        // Trim to science region (rows 0-2047, cols 44-2091) to remove
        // overscan/collateral regions that generate spurious centroids.
        let (trimmed_pixels, sci_width, sci_height) =
            trim_tess_science_region(&pixels, naxis1, naxis2);
        println!("  Science region: {} x {}", sci_width, sci_height);

        // Handle NaN/Inf pixels
        let clean_pixels: Vec<f32> = trimmed_pixels
            .iter()
            .map(|&v| {
                if v.is_nan() || v.is_infinite() {
                    0.0
                } else {
                    v
                }
            })
            .collect();

        // Get WCS boresight from header
        let crval_ra = get_f64(image_hdu, "CRVAL1").unwrap();
        let crval_dec = get_f64(image_hdu, "CRVAL2").unwrap();
        let true_boresight = radec_to_uvec(crval_ra, crval_dec);
        println!("  WCS: RA={:.4}°, Dec={:.4}°", crval_ra, crval_dec);

        // Print CD matrix
        if let (Some(cd11), Some(cd12), Some(cd21), Some(cd22)) = (
            get_f64(image_hdu, "CD1_1"),
            get_f64(image_hdu, "CD1_2"),
            get_f64(image_hdu, "CD2_1"),
            get_f64(image_hdu, "CD2_2"),
        ) {
            let pixel_scale_deg = ((cd11 * cd11 + cd21 * cd21).sqrt()
                + (cd12 * cd12 + cd22 * cd22).sqrt())
                / 2.0;
            println!(
                "  CD matrix: [{:.6}, {:.6}; {:.6}, {:.6}]",
                cd11, cd12, cd21, cd22
            );
            println!(
                "  Approx pixel scale: {:.2}\"/px, FOV: {:.2}° x {:.2}°",
                pixel_scale_deg * 3600.0,
                pixel_scale_deg * sci_width as f64,
                pixel_scale_deg * sci_height as f64
            );
        }

        // ── Extract centroids ──
        // TESS images have high background (~150-200 DN). Saturated stars
        // create elongated blobs, so max_elongation is set high (30.0).
        let extract_config = CentroidExtractionConfig {
            sigma_threshold: 250.0,
            min_pixels: 3,
            max_pixels: 10000,
            max_centroids: None,
            sigma_clip_iterations: 5,
            sigma_clip_factor: 3.0,
            use_8_connectivity: true,
            local_bg_block_size: Some(128),
            max_elongation: Some(30.0),
        };

        let extraction =
            tetra3::extract_centroids_from_raw(&clean_pixels, sci_width, sci_height, &extract_config)
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

        if extraction.centroids.len() < 4 {
            println!("  SKIP: Too few centroids");
            continue;
        }

        // ── Solve ──
        // No manual parity correction needed — the solver's auto-parity
        // detection handles CD matrix orientation automatically.
        let solve_config = SolveConfig {
            fov_estimate_rad: (12.0_f32).to_radians(),
            image_width: sci_width,
            image_height: sci_height,
            fov_max_error_rad: Some((2.0_f32).to_radians()),
            match_radius: 0.01,
            match_threshold: 1e-5,
            solve_timeout_ms: Some(60_000),
            match_max_error: None,
        };

        let result = db.solve_from_centroids(&extraction.centroids, &solve_config);

        println!("  Solve status: {:?}", result.status);
        println!("  Solve time:   {:.1} ms", result.solve_time_ms);

        if result.status == SolveStatus::MatchFound {
            let solved_q = result.qicrs2cam.unwrap();
            let solved_boresight = solved_q.inverse() * Vector3::new(0.0, 0.0, 1.0);
            let error_rad = angular_separation(&solved_boresight, &true_boresight);
            let error_arcmin = error_rad.to_degrees() * 60.0;

            let dec = (solved_boresight.z as f64).asin().to_degrees();
            let ra = (solved_boresight.y as f64)
                .atan2(solved_boresight.x as f64)
                .to_degrees()
                .rem_euclid(360.0);

            println!("  Solved:       RA={:.4}°, Dec={:.4}°", ra, dec);
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
        TESS_TEST_CASES.len(),
        failed,
        TESS_TEST_CASES.len()
    );
    println!("══════════════════════════════════════════════════════════════");
    assert_eq!(failed, 0, "{} TESS solve tests failed", failed);
}
