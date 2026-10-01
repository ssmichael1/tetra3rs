//! Integration test: extract star centroids from TESS FFI images (~12° FOV),
//! solve for attitude, and compare against the WCS solution in the FITS header.
//!
//! These images use CD matrix WCS (no simple CDELT), data in HDU 1 (FITS extension),
//! and are 2136×2078 pixels at ~21"/px. The science region is rows 0–2047,
//! columns 44–2091 (2048×2048); the rest is overscan/collateral.
//!
//! The solver's auto-parity detection handles the CD matrix orientation.

mod test_data;

use numeris::Vector3;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;
use tetra3::{
    extract_centroids_fast, CalibrateConfig, CentroidExtractionConfig, DistortionModelType,
    FastCentroidConfig, GenerateDatabaseConfig, SolveConfig, SolverDatabase,
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
    let value = if let Some(rest) = value_str.strip_prefix('\'') {
        if let Some(end) = rest.find('\'') {
            FitsValue::Str(rest[..end].trim().to_string())
        } else {
            FitsValue::Str(rest.trim().to_string())
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
            let bytes_per_pixel = bitpix.unsigned_abs() / 8;
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
        let padded = data_len.div_ceil(2880) * 2880;
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
fn trim_tess_science_region(
    pixels: &[f32],
    full_width: u32,
    full_height: u32,
) -> (Vec<f32>, u32, u32) {
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
    Vector3::from_array([
        (dec.cos() * ra.cos()) as f32,
        (dec.cos() * ra.sin()) as f32,
        dec.sin() as f32,
    ])
}

fn angular_separation(a: &Vector3<f32>, b: &Vector3<f32>) -> f32 {
    let cross = a.cross(b);
    let cross_norm = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    let dot = a.dot(b);
    cross_norm.atan2(dot)
}

/// Get a SIP polynomial coefficient from the header (e.g. "A_2_1").
/// Returns 0.0 if the key is not present.
fn get_sip_coeff(hdu: &FitsHdu, prefix: &str, p: usize, q: usize) -> f64 {
    let key = format!("{}_{}_{}", prefix, p, q);
    get_f64(hdu, &key).unwrap_or(0.0)
}

/// Apply SIP distortion correction (forward: pixel → corrected pixel).
/// u, v are pixel offsets from CRPIX (0-indexed: x - (CRPIX1 - 1)).
/// Returns (u + f(u,v), v + g(u,v)) where f and g are the SIP polynomials.
fn apply_sip_forward(hdu: &FitsHdu, u: f64, v: f64) -> (f64, f64) {
    let a_order = match hdu.headers.get("A_ORDER") {
        Some(FitsValue::Int(n)) => *n as usize,
        _ => return (u, v), // no SIP
    };
    let b_order = match hdu.headers.get("B_ORDER") {
        Some(FitsValue::Int(n)) => *n as usize,
        _ => return (u, v),
    };

    let mut f = 0.0;
    for p in 0..=a_order {
        for q in 0..=(a_order - p) {
            let c = get_sip_coeff(hdu, "A", p, q);
            if c != 0.0 {
                f += c * u.powi(p as i32) * v.powi(q as i32);
            }
        }
    }

    let mut g = 0.0;
    for p in 0..=b_order {
        for q in 0..=(b_order - p) {
            let c = get_sip_coeff(hdu, "B", p, q);
            if c != 0.0 {
                g += c * u.powi(p as i32) * v.powi(q as i32);
            }
        }
    }

    (u + f, v + g)
}

/// Compute the RA/Dec (degrees) at a given pixel coordinate in the full-frame
/// image using the full WCS chain: pixel → SIP → CD matrix → TAN deprojection.
/// pixel_x, pixel_y are 0-indexed full-frame coordinates.
fn pixel_to_radec(hdu: &FitsHdu, pixel_x: f64, pixel_y: f64) -> (f64, f64) {
    // CRPIX is 1-indexed in FITS
    let crpix1 = get_f64(hdu, "CRPIX1").expect("Missing CRPIX1");
    let crpix2 = get_f64(hdu, "CRPIX2").expect("Missing CRPIX2");
    let crval1 = get_f64(hdu, "CRVAL1").expect("Missing CRVAL1");
    let crval2 = get_f64(hdu, "CRVAL2").expect("Missing CRVAL2");

    // Convert to 1-indexed FITS convention, then offset from CRPIX
    let u = (pixel_x + 1.0) - crpix1;
    let v = (pixel_y + 1.0) - crpix2;

    // Apply SIP distortion (forward: pixel → corrected pixel)
    let (u_sip, v_sip) = apply_sip_forward(hdu, u, v);

    // Apply CD matrix to get intermediate world coordinates (degrees)
    let cd11 = get_f64(hdu, "CD1_1").unwrap_or(0.0);
    let cd12 = get_f64(hdu, "CD1_2").unwrap_or(0.0);
    let cd21 = get_f64(hdu, "CD2_1").unwrap_or(0.0);
    let cd22 = get_f64(hdu, "CD2_2").unwrap_or(0.0);

    let xi = cd11 * u_sip + cd12 * v_sip; // degrees
    let eta = cd21 * u_sip + cd22 * v_sip; // degrees

    // TAN (gnomonic) deprojection
    let xi_rad = xi.to_radians();
    let eta_rad = eta.to_radians();
    let crval1_rad = crval1.to_radians();
    let crval2_rad = crval2.to_radians();

    let denom = crval2_rad.cos() - eta_rad * crval2_rad.sin();
    let ra_rad = crval1_rad + (xi_rad).atan2(denom);
    let dec_rad = (crval2_rad.sin() + eta_rad * crval2_rad.cos())
        .atan2((xi_rad.powi(2) + denom.powi(2)).sqrt());

    (ra_rad.to_degrees().rem_euclid(360.0), dec_rad.to_degrees())
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
        max_fov_deg: 15.0,
        min_fov_deg: Some(11.0),
        star_max_magnitude: None,
        pattern_max_error: 0.005,
        lattice_field_oversampling: 100,
        patterns_per_lattice_field: 150,
        verification_stars_per_fov: 1000,
        multiscale_step: 1.5,
        epoch_proper_motion_year: Some(2018.0), // TESS launched 2018
        catalog_nside: 8,
    };

    let catalog_path = test_data::ensure_test_file("data/gaia_merged.bin");
    SolverDatabase::generate_from_gaia(&catalog_path, &config)
        .expect("Failed to generate database from Gaia catalog")
}

#[test]
fn test_tess_fits_solve() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
        .is_test(true)
        .try_init();

    // Ensure all test files are downloaded
    for tc in TESS_TEST_CASES {
        test_data::ensure_test_file(&format!("data/tess_test_images/{}", tc.filename));
    }

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
        assert!(
            hdus.len() >= 2,
            "Expected at least 2 HDUs in TESS FITS file"
        );

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

        // Compute true boresight from the WCS at the center of the science region.
        // The science region starts at column 44 in the full frame, so the center
        // pixel in full-frame coordinates is (44 + sci_width/2, sci_height/2).
        // Geometric center of the trimmed science region in full-frame
        // 0-indexed coords: region spans columns [44, 44+sci_width-1], rows
        // [0, sci_height-1], so the center is (W-1)/2 — matching tetra3rs's
        // centroid origin (issue #28).
        let center_x = 44.0 + (sci_width - 1) as f64 / 2.0;
        let center_y = (sci_height - 1) as f64 / 2.0;
        let (boresight_ra, boresight_dec) = pixel_to_radec(image_hdu, center_x, center_y);
        let true_boresight = radec_to_uvec(boresight_ra, boresight_dec);

        let crval_ra = get_f64(image_hdu, "CRVAL1").unwrap();
        let crval_dec = get_f64(image_hdu, "CRVAL2").unwrap();
        println!("  WCS CRVAL: RA={:.4}°, Dec={:.4}°", crval_ra, crval_dec);
        println!(
            "  WCS boresight (center pixel): RA={:.4}°, Dec={:.4}°",
            boresight_ra, boresight_dec
        );

        // Print CD matrix
        if let (Some(cd11), Some(cd12), Some(cd21), Some(cd22)) = (
            get_f64(image_hdu, "CD1_1"),
            get_f64(image_hdu, "CD1_2"),
            get_f64(image_hdu, "CD2_1"),
            get_f64(image_hdu, "CD2_2"),
        ) {
            let pixel_scale_deg =
                ((cd11 * cd11 + cd21 * cd21).sqrt() + (cd12 * cd12 + cd22 * cd22).sqrt()) / 2.0;
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
            sigma_threshold: 150.0,
            min_pixels: 3,
            max_pixels: 10000,
            max_centroids: None,
            sigma_clip_iterations: 5,
            sigma_clip_factor: 3.0,
            local_bg_block_size: Some(128),
            max_elongation: Some(30.0),
            matched_filter_sigma: None,
            ..Default::default()
        };

        let extraction = tetra3::extract_centroids_from_raw(
            &clean_pixels,
            sci_width,
            sci_height,
            &extract_config,
        )
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
        // The solver assumes a perfect pinhole (gnomonic) projection. TESS has
        // significant SIP distortion (up to ~65 px at corners), so the solved
        // attitude will have ~1-5' boresight error and elevated RMSE. This is a
        // known limitation — pre-undistorting centroids doesn't help because the
        // solver's internal projection model also assumes uniform pixel scale.
        let solve_config = SolveConfig {
            fov_max_error_rad: Some((2.0_f32).to_radians()),
            match_radius: 0.005,
            match_threshold: 1e-5,
            solve_timeout_ms: Some(60_000),
            match_max_error: None,
            ..SolveConfig::new((12.0_f32).to_radians(), sci_width, sci_height)
        };

        let result = db.solve_from_centroids(&extraction.centroids, &solve_config);

        if let Ok(solution) = &result {
            println!("  Solve time:   {:.1} ms", solution.solve_time_ms);
            let solved_q = solution.qicrs2cam;
            let solved_boresight = solved_q.inverse() * Vector3::from_array([0.0, 0.0, 1.0]);
            let error_rad = angular_separation(&solved_boresight, &true_boresight);
            let error_arcmin = error_rad.to_degrees() * 60.0;

            let dec = (solved_boresight[2] as f64).asin().to_degrees();
            let ra = (solved_boresight[1] as f64)
                .atan2(solved_boresight[0] as f64)
                .to_degrees()
                .rem_euclid(360.0);

            println!("  Solved:       RA={:.4}°, Dec={:.4}°", ra, dec);
            println!(
                "  Boresight error: {:.2}' ({:.1}\")",
                error_arcmin,
                error_arcmin * 60.0
            );
            println!("  Matched stars: {}", solution.num_matches);
            println!(
                "  RMSE:         {:.1}\"",
                solution.rmse_rad.to_degrees() * 3600.0
            );
            println!("  Solved FOV:   {:.2}°", solution.fov_rad.to_degrees());

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

/// Benchmark: fast single-pass centroid path vs. the connected-component path
/// on real TESS frames — speed, centroid-position agreement, end-to-end solve.
///
/// `#[ignore]` (timing/diagnostic, not a pass/fail gate). Run with:
///   cargo test --test tess_solve_test --features image -- --ignored --nocapture bench_fast_vs_ccl
/// Add `--features image,parallel` to time the multi-threaded blurs.
#[test]
#[ignore]
fn bench_fast_vs_ccl_extraction() {
    const REPS: u32 = 5;

    for tc in TESS_TEST_CASES {
        test_data::ensure_test_file(&format!("data/tess_test_images/{}", tc.filename));
    }
    let db = build_tess_database();

    println!("\n══════════════════════════════════════════════════════════════");
    println!("Fast single-pass vs. connected-component centroid extraction (best of {REPS})");
    println!("══════════════════════════════════════════════════════════════");

    for tc in TESS_TEST_CASES {
        let fits_path = format!("data/tess_test_images/{}", tc.filename);
        let hdus = read_fits_hdus(&fits_path);
        let image_hdu = &hdus[1];
        let naxis1 = match image_hdu.headers.get("NAXIS1") {
            Some(FitsValue::Int(n)) => *n as u32,
            _ => panic!("Missing NAXIS1"),
        };
        let naxis2 = match image_hdu.headers.get("NAXIS2") {
            Some(FitsValue::Int(n)) => *n as u32,
            _ => panic!("Missing NAXIS2"),
        };
        let pixels = read_f32_image(&fits_path, image_hdu);
        let (trimmed, sci_width, sci_height) = trim_tess_science_region(&pixels, naxis1, naxis2);
        let clean: Vec<f32> = trimmed
            .iter()
            .map(|&v| if v.is_finite() { v } else { 0.0 })
            .collect();

        // Geometric center of the trimmed science region in full-frame
        // 0-indexed coords: region spans columns [44, 44+sci_width-1], rows
        // [0, sci_height-1], so the center is (W-1)/2 — matching tetra3rs's
        // centroid origin (issue #28).
        let center_x = 44.0 + (sci_width - 1) as f64 / 2.0;
        let center_y = (sci_height - 1) as f64 / 2.0;
        let (b_ra, b_dec) = pixel_to_radec(image_hdu, center_x, center_y);
        let true_boresight = radec_to_uvec(b_ra, b_dec);

        let ccl_config = CentroidExtractionConfig {
            sigma_threshold: 150.0,
            min_pixels: 3,
            max_pixels: 10000,
            local_bg_block_size: Some(128),
            max_elongation: Some(30.0),
            ..Default::default()
        };
        // Single-pass detector: threshold in noise-σ above a coarse-grid
        // background (independent of the ~250-DN raw threshold above).
        let fast_config = FastCentroidConfig {
            sigma_threshold: 8.0,
            bg_grid: 128,
            min_pixels: 3,
            max_centroids: None,
            ..Default::default()
        };

        let best = |f: &dyn Fn() -> usize| -> (f64, usize) {
            let mut best_ms = f64::INFINITY;
            let mut n = 0;
            for _ in 0..REPS {
                let t = Instant::now();
                n = f();
                best_ms = best_ms.min(t.elapsed().as_secs_f64() * 1e3);
            }
            (best_ms, n)
        };

        let ccl = tetra3::extract_centroids_from_raw(&clean, sci_width, sci_height, &ccl_config)
            .expect("ccl extract");
        let fast = extract_centroids_fast(&clean, sci_width, sci_height, &fast_config)
            .expect("fast extract");

        let (ccl_ms, _) = best(&|| {
            tetra3::extract_centroids_from_raw(&clean, sci_width, sci_height, &ccl_config)
                .unwrap()
                .centroids
                .len()
        });
        let (fast_ms, _) = best(&|| {
            extract_centroids_fast(&clean, sci_width, sci_height, &fast_config)
                .unwrap()
                .centroids
                .len()
        });

        // Cross-match: nearest CCL centroid to each fast centroid, ≤ 2 px.
        let mut deltas: Vec<f32> = Vec::new();
        for f in &fast.centroids {
            let mut best_d = f32::INFINITY;
            for c in &ccl.centroids {
                let d = ((f.x - c.x).powi(2) + (f.y - c.y).powi(2)).sqrt();
                best_d = best_d.min(d);
            }
            if best_d <= 2.0 {
                deltas.push(best_d);
            }
        }
        deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (median_d, max_d) = if deltas.is_empty() {
            (f32::NAN, f32::NAN)
        } else {
            (deltas[deltas.len() / 2], *deltas.last().unwrap())
        };

        let solve_config = SolveConfig {
            fov_max_error_rad: Some((2.0_f32).to_radians()),
            match_radius: 0.005,
            match_threshold: 1e-5,
            solve_timeout_ms: Some(60_000),
            match_max_error: None,
            ..SolveConfig::new((12.0_f32).to_radians(), sci_width, sci_height)
        };
        let boresight_err_arcmin = |cents: &[tetra3::Centroid]| -> Option<f32> {
            let sol = db.solve_from_centroids(cents, &solve_config).ok()?;
            let bs = sol.qicrs2cam.inverse() * Vector3::from_array([0.0, 0.0, 1.0]);
            Some(angular_separation(&bs, &true_boresight).to_degrees() * 60.0)
        };
        let ccl_err = boresight_err_arcmin(&ccl.centroids);
        let fast_err = boresight_err_arcmin(&fast.centroids);

        println!("\n{} ({}×{})", tc.filename, sci_width, sci_height);
        println!(
            "  CCL : {:6.1} ms   {:4} centroids   solve {}",
            ccl_ms,
            ccl.centroids.len(),
            ccl_err.map_or("FAIL".into(), |e| format!("{e:.2}' err")),
        );
        println!(
            "  Fast: {:6.1} ms   {:4} centroids   solve {}",
            fast_ms,
            fast.centroids.len(),
            fast_err.map_or("FAIL".into(), |e| format!("{e:.2}' err")),
        );
        println!(
            "  speedup {:.2}×   |   {} matched ≤2px: median Δ {:.3} px, max {:.3} px",
            ccl_ms / fast_ms,
            deltas.len(),
            median_d,
            max_d,
        );
    }
    println!("══════════════════════════════════════════════════════════════");
}

/// Test the polynomial distortion fitting pipeline on TESS images.
///
/// For each TESS image we:
/// 1. Solve with raw (distorted) centroids.
/// 2. Fit a 4th-order polynomial distortion model from the solve result.
/// 3. Re-solve with the fitted distortion model applied.
/// 4. Verify that the distortion-corrected solve has lower RMSE.
/// 5. Verify that the solved RA/Dec of the center pixel matches the FITS WCS
///    solution within 1 arcmin.
#[test]
fn test_tess_distortion_fit_and_center_accuracy() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
        .is_test(true)
        .try_init();

    for tc in TESS_TEST_CASES {
        test_data::ensure_test_file(&format!("data/tess_test_images/{}", tc.filename));
    }

    let db = build_tess_database();

    println!("\n══════════════════════════════════════════════════════════════");
    println!("DISTORTION TEST: fit polynomial model, re-solve, check center");

    let mut failed = 0;

    for tc in TESS_TEST_CASES {
        let fits_path = format!("data/tess_test_images/{}", tc.filename);
        println!("\n──────────────────────────────────────────────────────────────");
        println!("Testing: {} ({})", tc.filename, tc.description);

        // ── Read and prepare image ──
        let hdus = read_fits_hdus(&fits_path);
        assert!(hdus.len() >= 2);
        let image_hdu = &hdus[1];

        let naxis1 = match image_hdu.headers.get("NAXIS1") {
            Some(FitsValue::Int(n)) => *n as u32,
            _ => panic!("Missing NAXIS1"),
        };
        let naxis2 = match image_hdu.headers.get("NAXIS2") {
            Some(FitsValue::Int(n)) => *n as u32,
            _ => panic!("Missing NAXIS2"),
        };

        let pixels = read_f32_image(&fits_path, image_hdu);
        let (trimmed_pixels, sci_width, sci_height) =
            trim_tess_science_region(&pixels, naxis1, naxis2);

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

        // ── Extract centroids ──
        let extract_config = CentroidExtractionConfig {
            sigma_threshold: 150.0,
            min_pixels: 3,
            max_pixels: 10000,
            max_centroids: None,
            sigma_clip_iterations: 5,
            sigma_clip_factor: 3.0,
            local_bg_block_size: Some(128),
            max_elongation: Some(30.0),
            matched_filter_sigma: None,
            ..Default::default()
        };

        let extraction = tetra3::extract_centroids_from_raw(
            &clean_pixels,
            sci_width,
            sci_height,
            &extract_config,
        )
        .expect("Centroid extraction failed");

        println!("  Extracted {} centroids", extraction.centroids.len());

        if extraction.centroids.len() < 4 {
            println!("  SKIP: Too few centroids");
            continue;
        }

        // ── 1. Initial solve (raw centroids) ──
        let solve_cfg = SolveConfig {
            fov_max_error_rad: Some((2.0_f32).to_radians()),
            match_radius: 0.005,
            match_threshold: 1e-5,
            solve_timeout_ms: Some(60_000),
            match_max_error: None,
            ..SolveConfig::new((12.0_f32).to_radians(), sci_width, sci_height)
        };

        let result_raw = db.solve_from_centroids(&extraction.centroids, &solve_cfg);
        let solution_raw = result_raw
            .as_ref()
            .unwrap_or_else(|_| panic!("Raw solve failed for {}", tc.filename));

        let rmse_raw_arcsec = solution_raw.rmse_rad.to_degrees() as f64 * 3600.0;
        println!(
            "  Raw solve:   RMSE={:.1}\", {} matches",
            rmse_raw_arcsec, solution_raw.num_matches,
        );

        // ── 2. Calibrate camera model (polynomial order 4) ──
        let cal_result = tetra3::calibrate_camera(
            &[&result_raw],
            &[&extraction.centroids[..]],
            &db,
            sci_width,
            sci_height,
            &CalibrateConfig {
                model: DistortionModelType::Polynomial { order: 4 },
                ..CalibrateConfig::default()
            },
        )
        .expect("calibration should succeed");
        println!(
            "  Calibration: RMSE {:.3} -> {:.3} px, {} inliers, {} outliers",
            cal_result.rmse_before_px,
            cal_result.rmse_after_px,
            cal_result.n_inliers,
            cal_result.n_outliers,
        );

        // ── 3. Re-solve with camera model ──
        let solve_cfg_dist = SolveConfig {
            camera_model: cal_result.camera_model.clone(),
            ..solve_cfg
        };
        let result_dist = db.solve_from_centroids(&extraction.centroids, &solve_cfg_dist);
        let solution_dist = result_dist
            .as_ref()
            .unwrap_or_else(|_| panic!("Distortion-corrected solve failed for {}", tc.filename));

        let rmse_dist_arcsec = solution_dist.rmse_rad.to_degrees() as f64 * 3600.0;
        println!(
            "  Dist solve:  RMSE={:.1}\", {} matches",
            rmse_dist_arcsec, solution_dist.num_matches,
        );

        // ── 4. Verify center pixel RA/Dec matches FITS WCS ──
        let (solved_ra, solved_dec) = solution_dist.pixel_to_world(0.0, 0.0);

        // Center of science region in full-frame 0-indexed coordinates
        let center_x_ff = 44.0 + (sci_width - 1) as f64 / 2.0;
        let center_y_ff = (sci_height - 1) as f64 / 2.0;
        let (wcs_ra, wcs_dec) = pixel_to_radec(image_hdu, center_x_ff, center_y_ff);

        let sep_arcmin = angular_separation(
            &radec_to_uvec(solved_ra, solved_dec),
            &radec_to_uvec(wcs_ra, wcs_dec),
        )
        .to_degrees()
            * 60.0;

        println!(
            "  Center pixel: solved=({:.4} deg, {:.4} deg), WCS=({:.4} deg, {:.4} deg), sep={:.2}'",
            solved_ra, solved_dec, wcs_ra, wcs_dec, sep_arcmin,
        );

        // ── 5. Assertions ──
        let mut test_passed = true;

        if rmse_dist_arcsec >= rmse_raw_arcsec {
            println!(
                "  *** FAIL: distortion-corrected RMSE ({:.1}\") >= raw RMSE ({:.1}\") ***",
                rmse_dist_arcsec, rmse_raw_arcsec,
            );
            test_passed = false;
        }

        if sep_arcmin >= 1.0 {
            println!(
                "  *** FAIL: center pixel separation {:.2}' exceeds 1' ***",
                sep_arcmin,
            );
            test_passed = false;
        }

        if test_passed {
            println!("  PASS");
        } else {
            failed += 1;
        }
    }

    println!("\n══════════════════════════════════════════════════════════════");
    assert_eq!(failed, 0, "{} TESS distortion tests failed", failed);
}

/// Test multi-image camera calibration using tiered solve passes.
///
/// Uses 10 TESS Camera 1, CCD 1 images from different sectors — same optics,
/// different sky pointings. Matches the tiered solve+calibrate approach from
/// the `tess_multi_image.ipynb` notebook:
///
/// 1. Extract centroids from all images up front.
/// 2. Run 4 tiered passes, each progressively tighter:
///    - Solve all images with the current camera model.
///    - Calibrate a new camera model from all solve results.
///    - Feed the calibrated model into the next pass.
/// 3. After all passes, verify:
///    - All images solved successfully.
///    - RMSE < 15" for all images.
///    - Center pixel RA/Dec within 10" of FITS WCS solution.
#[test]
fn test_tess_multi_image_calibration() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .is_test(true)
        .try_init();

    // Same CCD (Camera 1, CCD 1) across 10 sectors — matching notebook sector list
    let same_ccd_images: &[(&str, &str)] = &[
        ("data/tess_same_ccd/sector01_cam1_ccd1.fits", "Sector 1"),
        ("data/tess_same_ccd/sector02_cam1_ccd1.fits", "Sector 2"),
        ("data/tess_same_ccd/sector03_cam1_ccd1.fits", "Sector 3"),
        ("data/tess_same_ccd/sector04_cam1_ccd1.fits", "Sector 4"),
        ("data/tess_same_ccd/sector05_cam1_ccd1.fits", "Sector 5"),
        ("data/tess_same_ccd/sector06_cam1_ccd1.fits", "Sector 6"),
        ("data/tess_same_ccd/sector13_cam1_ccd1.fits", "Sector 13"),
        ("data/tess_same_ccd/sector14_cam1_ccd1.fits", "Sector 14"),
        ("data/tess_same_ccd/sector15_cam1_ccd1.fits", "Sector 15"),
        ("data/tess_same_ccd/sector17_cam1_ccd1.fits", "Sector 17"),
    ];

    for &(path, _) in same_ccd_images {
        test_data::ensure_test_file(path);
    }

    // ── Database with notebook parameters ──
    let db = {
        let config = GenerateDatabaseConfig {
            max_fov_deg: 14.0,
            min_fov_deg: None,
            star_max_magnitude: None,
            pattern_max_error: 0.005,
            lattice_field_oversampling: 100,
            patterns_per_lattice_field: 500,
            verification_stars_per_fov: 3000,
            multiscale_step: 1.5,
            epoch_proper_motion_year: Some(2018.0),
            catalog_nside: 8,
        };
        let catalog_path = test_data::ensure_test_file("data/gaia_merged.bin");
        SolverDatabase::generate_from_gaia(&catalog_path, &config)
            .expect("Failed to generate database")
    };

    println!("\n══════════════════════════════════════════════════════════════");
    println!("MULTI-IMAGE TIERED CALIBRATION TEST");
    println!(
        "Database: {} stars, {} patterns",
        db.star_catalog.len(),
        db.props.num_patterns,
    );

    // ── Extract centroids from all images up front ──
    let extract_config = CentroidExtractionConfig {
        sigma_threshold: 180.0,
        min_pixels: 3,
        max_pixels: 10000,
        max_centroids: None,
        sigma_clip_iterations: 5,
        sigma_clip_factor: 3.0,
        local_bg_block_size: Some(16),
        max_elongation: Some(6.0),
        matched_filter_sigma: None,
        ..Default::default()
    };

    struct ImageData {
        centroids: Vec<tetra3::Centroid>,
        sci_width: u32,
        sci_height: u32,
        image_hdu_headers: HashMap<String, FitsValue>,
        description: &'static str,
    }

    let mut images: Vec<ImageData> = Vec::new();

    for &(fits_path, description) in same_ccd_images {
        let hdus = read_fits_hdus(fits_path);
        assert!(hdus.len() >= 2, "Expected at least 2 HDUs in {}", fits_path);
        let image_hdu = &hdus[1];

        let naxis1 = match image_hdu.headers.get("NAXIS1") {
            Some(FitsValue::Int(n)) => *n as u32,
            _ => panic!("Missing NAXIS1 in {}", fits_path),
        };
        let naxis2 = match image_hdu.headers.get("NAXIS2") {
            Some(FitsValue::Int(n)) => *n as u32,
            _ => panic!("Missing NAXIS2 in {}", fits_path),
        };

        let pixels = read_f32_image(fits_path, image_hdu);
        let (trimmed_pixels, sci_width, sci_height) =
            trim_tess_science_region(&pixels, naxis1, naxis2);

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

        let extraction = tetra3::extract_centroids_from_raw(
            &clean_pixels,
            sci_width,
            sci_height,
            &extract_config,
        )
        .expect("Centroid extraction failed");

        println!(
            "  {}: {} x {}, {} centroids",
            description,
            sci_width,
            sci_height,
            extraction.centroids.len()
        );

        images.push(ImageData {
            centroids: extraction.centroids,
            sci_width,
            sci_height,
            image_hdu_headers: image_hdu.headers.clone(),
            description,
        });
    }

    let sci_width = images[0].sci_width;
    let sci_height = images[0].sci_height;

    // ── Tiered solve+calibrate passes ──
    // Each pass: solve all images → calibrate → feed model to next pass.
    // Progressively tighter match_radius and higher polynomial order.
    struct PassConfig {
        match_radius: f32,
        calibration_order: u32,
        fov_max_error_deg: f32,
    }

    let pass_configs = [
        PassConfig {
            match_radius: 0.005,
            calibration_order: 3,
            fov_max_error_deg: 0.5,
        },
        PassConfig {
            match_radius: 0.005,
            calibration_order: 4,
            fov_max_error_deg: 0.5,
        },
        PassConfig {
            match_radius: 0.003,
            calibration_order: 5,
            fov_max_error_deg: 0.5,
        },
        PassConfig {
            match_radius: 0.002,
            calibration_order: 6,
            fov_max_error_deg: 0.5,
        },
    ];

    let mut camera_model: Option<tetra3::CameraModel> = None;
    let mut results: Vec<tetra3::SolveResult> = Vec::new();

    for (pass_idx, pcfg) in pass_configs.iter().enumerate() {
        results.clear();

        let fov_estimate_rad = match &camera_model {
            Some(cm) => cm.fov_rad() as f32,
            None => (11.8_f32).to_radians(),
        };

        println!(
            "\n  Pass {}: match_radius={}, order={}, fov={:.2}°",
            pass_idx + 1,
            pcfg.match_radius,
            pcfg.calibration_order,
            fov_estimate_rad.to_degrees(),
        );

        for img in &images {
            let solve_cfg = SolveConfig {
                fov_max_error_rad: Some(pcfg.fov_max_error_deg.to_radians()),
                match_radius: pcfg.match_radius,
                match_threshold: 1e-5,
                solve_timeout_ms: Some(60_000),
                ..SolveConfig::with_camera_model(camera_model.clone().unwrap_or_else(|| {
                    tetra3::CameraModel::from_fov(
                        fov_estimate_rad as f64,
                        img.sci_width,
                        img.sci_height,
                    )
                }))
            };

            let result = db.solve_from_centroids(&img.centroids, &solve_cfg);

            let status_str = match &result {
                Ok(solution) => format!(
                    "OK  RMSE={:.1}\", {} matches",
                    solution.rmse_rad.to_degrees() as f64 * 3600.0,
                    solution.num_matches
                ),
                Err(fail) => format!("{:?}", fail.status),
            };
            println!("    {}: {}", img.description, status_str);

            results.push(result);
        }

        // Calibrate from all results
        let solve_refs: Vec<&tetra3::SolveResult> = results.iter().collect();
        let centroid_refs: Vec<&[tetra3::Centroid]> =
            images.iter().map(|img| img.centroids.as_slice()).collect();

        let cal_result = tetra3::calibrate_camera(
            &solve_refs,
            &centroid_refs,
            &db,
            sci_width,
            sci_height,
            &CalibrateConfig {
                model: DistortionModelType::Polynomial {
                    order: pcfg.calibration_order,
                },
                ..CalibrateConfig::default()
            },
        )
        .expect("calibration should succeed");

        println!(
            "    Calibration: RMSE {:.3} -> {:.3} px, {} inliers, {} outliers",
            cal_result.rmse_before_px,
            cal_result.rmse_after_px,
            cal_result.n_inliers,
            cal_result.n_outliers,
        );

        camera_model = Some(cal_result.camera_model);
    }

    // ── Verify final results ──
    println!("\n  Final results:");
    let mut failed = 0;
    let n_images = images.len();
    let arcsec_per_px = results[0]
        .as_ref()
        .map(|s| s.fov_rad.to_degrees() as f64)
        .unwrap_or(0.0)
        * 3600.0
        / sci_width as f64;

    for (img, result) in images.iter().zip(results.iter()) {
        let Ok(solution) = result else {
            println!("    {}: *** FAIL: no match ***", img.description);
            failed += 1;
            continue;
        };

        let rmse_arcsec = solution.rmse_rad.to_degrees() as f64 * 3600.0;
        let rmse_px = rmse_arcsec / arcsec_per_px;

        // Compare center pixel against FITS WCS
        let hdu_for_wcs = FitsHdu {
            headers: img.image_hdu_headers.clone(),
            data_offset: 0,
            data_len: 0,
        };

        let (solved_ra, solved_dec) = solution.pixel_to_world(0.0, 0.0);

        let center_x_ff = 44.0 + (img.sci_width - 1) as f64 / 2.0;
        let center_y_ff = (img.sci_height - 1) as f64 / 2.0;
        let (wcs_ra, wcs_dec) = pixel_to_radec(&hdu_for_wcs, center_x_ff, center_y_ff);

        let sep_arcsec = angular_separation(
            &radec_to_uvec(solved_ra, solved_dec),
            &radec_to_uvec(wcs_ra, wcs_dec),
        )
        .to_degrees() as f64
            * 3600.0;

        println!(
            "    {}: {} matches, RMSE={:.2}\" ({:.3} px), vs WCS={:.2}\"",
            img.description, solution.num_matches, rmse_arcsec, rmse_px, sep_arcsec,
        );

        if rmse_arcsec > 15.0 {
            println!("      *** FAIL: RMSE {:.1}\" exceeds 15\" ***", rmse_arcsec,);
            failed += 1;
        }
        if sep_arcsec > 10.0 {
            println!(
                "      *** FAIL: WCS separation {:.1}\" exceeds 10\" ***",
                sep_arcsec,
            );
            failed += 1;
        }
    }

    println!("\n══════════════════════════════════════════════════════════════");
    let n_solved = results.iter().filter(|r| r.is_ok()).count();
    println!(
        "RESULT: {}/{} solved, {} failures",
        n_solved, n_images, failed,
    );
    assert_eq!(
        failed, 0,
        "{} multi-image calibration checks failed",
        failed
    );
}
