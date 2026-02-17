//! Integration test: extract star centroids from real TESS FFI FITS images,
//! solve for attitude, and compare against the WCS solution in the FITS header.

use nalgebra::{Rotation3, UnitQuaternion, Vector3};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use tetra3::{
    CentroidExtractionConfig, GenerateDatabaseConfig, SolveConfig, SolveStatus, SolverDatabase,
};

// ═══════════════════════════════════════════════════════════════════════════
// Minimal FITS reader — enough to parse TESS FFI files (f32 IMAGE in HDU 1)
// ═══════════════════════════════════════════════════════════════════════════

/// Parsed FITS header keyword value (simplified).
#[derive(Debug, Clone)]
enum FitsValue {
    Float(f64),
    Int(i64),
    Str(String),
    Bool(bool),
}

/// A minimal FITS HDU.
struct FitsHdu {
    headers: HashMap<String, FitsValue>,
    data_offset: u64, // byte offset where data begins in file
    data_len: u64,    // bytes of data (before padding)
}

/// Parse a single 80-char FITS header card.
fn parse_header_card(card: &[u8; 80]) -> Option<(String, FitsValue)> {
    let card_str = String::from_utf8_lossy(card);

    // Keyword is first 8 chars
    let keyword = card_str[..8].trim().to_string();
    if keyword.is_empty() || keyword == "COMMENT" || keyword == "HISTORY" || keyword == "END" {
        return None;
    }

    // Value indicator "= " at columns 8-9
    if card_str.len() < 10 || &card_str[8..10] != "= " {
        return None;
    }

    let value_str = card_str[10..].trim();

    // Try to parse value
    let value = if value_str.starts_with('\'') {
        // String value: extract between single quotes
        if let Some(end) = value_str[1..].find('\'') {
            FitsValue::Str(value_str[1..1 + end].trim().to_string())
        } else {
            FitsValue::Str(value_str[1..].trim().to_string())
        }
    } else if value_str.starts_with('T') {
        FitsValue::Bool(true)
    } else if value_str.starts_with('F') {
        FitsValue::Bool(false)
    } else {
        // Strip any inline comment after " /"
        let num_part = if let Some(slash) = value_str.find('/') {
            value_str[..slash].trim()
        } else {
            value_str
        };
        // Try integer first, then float
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

/// Read FITS HDUs from a file. Returns a vec of HDUs with their headers and data offsets.
fn read_fits_hdus(path: &str) -> Vec<FitsHdu> {
    let mut file = File::open(path).expect("Failed to open FITS file");
    let mut hdus = Vec::new();
    let mut offset: u64 = 0;

    loop {
        let mut headers = HashMap::new();
        let mut found_end = false;

        // Read header blocks (2880-byte blocks of 80-char cards)
        loop {
            let mut block = [0u8; 2880];
            if file.read_exact(&mut block).is_err() {
                // EOF
                return hdus;
            }
            offset += 2880;

            for i in 0..36 {
                let card: &[u8; 80] = block[i * 80..(i + 1) * 80]
                    .try_into()
                    .expect("slice with incorrect length");
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

        // Compute data size from NAXIS, BITPIX, etc.
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

        // Check for PCOUNT/GCOUNT (extensions)
        if let Some(FitsValue::Int(pcount)) = headers.get("PCOUNT") {
            data_len += *pcount as u64;
        }

        let data_offset = offset;

        // Pad to 2880-byte boundary
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

/// Get a float value from FITS headers.
fn get_f64(hdu: &FitsHdu, key: &str) -> Option<f64> {
    match hdu.headers.get(key) {
        Some(FitsValue::Float(f)) => Some(*f),
        Some(FitsValue::Int(i)) => Some(*i as f64),
        _ => None,
    }
}

/// Read f32 image data from a FITS HDU (big-endian).
fn read_f32_image(path: &str, hdu: &FitsHdu) -> Vec<f32> {
    let mut file = File::open(path).expect("Failed to open FITS file");
    file.seek(SeekFrom::Start(hdu.data_offset))
        .expect("Failed to seek");

    let npixels = hdu.data_len as usize / 4;
    let mut buf = vec![0u8; hdu.data_len as usize];
    file.read_exact(&mut buf)
        .expect("Failed to read image data");

    // FITS stores data big-endian
    let mut pixels = Vec::with_capacity(npixels);
    for i in 0..npixels {
        let bytes: [u8; 4] = buf[i * 4..(i + 1) * 4].try_into().unwrap();
        pixels.push(f32::from_be_bytes(bytes));
    }
    pixels
}

// ═══════════════════════════════════════════════════════════════════════════
// WCS helpers — convert CRVAL to a boresight unit vector
// ═══════════════════════════════════════════════════════════════════════════

/// Convert RA/Dec (degrees) to a unit vector in ICRS.
fn radec_to_uvec(ra_deg: f64, dec_deg: f64) -> Vector3<f32> {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    Vector3::new(
        (dec.cos() * ra.cos()) as f32,
        (dec.cos() * ra.sin()) as f32,
        dec.sin() as f32,
    )
}

/// Convert a unit vector back to (RA, Dec) in degrees.
fn uvec_to_radec(v: &Vector3<f32>) -> (f64, f64) {
    let dec = (v.z as f64).asin();
    let ra = (v.y as f64).atan2(v.x as f64);
    let ra_deg = ra.to_degrees();
    let ra_deg = if ra_deg < 0.0 { ra_deg + 360.0 } else { ra_deg };
    (ra_deg, dec.to_degrees())
}

/// Angular separation between two unit vectors (radians).
fn angular_separation(a: &Vector3<f32>, b: &Vector3<f32>) -> f32 {
    let cross = a.cross(b).norm();
    let dot = a.dot(b);
    cross.atan2(dot)
}

// ═══════════════════════════════════════════════════════════════════════════
// Build the attitude quaternion from the TESS WCS CD matrix
// ═══════════════════════════════════════════════════════════════════════════

/// Build a quaternion for the WCS tangent-plane frame at CRVAL.
///
/// The corrected centroids use the full CD matrix pipeline, so they live
/// in the WCS tangent-plane frame where:
///   +X = increasing xi  = East  (direction of increasing RA)
///   +Y = increasing eta = North (direction of increasing Dec)
///   +Z = boresight (CRVAL direction)
///
/// This quaternion is independent of the CD matrix — it only depends on CRVAL.
fn wcs_to_quaternion(hdu: &FitsHdu) -> UnitQuaternion<f32> {
    let crval1 = get_f64(hdu, "CRVAL1").unwrap(); // RA at reference pixel (deg)
    let crval2 = get_f64(hdu, "CRVAL2").unwrap(); // Dec at reference pixel (deg)

    let ra0 = crval1.to_radians();
    let dec0 = crval2.to_radians();

    let sin_ra = ra0.sin() as f32;
    let cos_ra = ra0.cos() as f32;
    let sin_dec = dec0.sin() as f32;
    let cos_dec = dec0.cos() as f32;

    // Tangent-plane basis vectors in ICRS at the boresight:
    //   +X = East  = direction of increasing RA  = (-sin_ra, cos_ra, 0)
    //   +Y = North = direction of increasing Dec = (-sin_dec*cos_ra, -sin_dec*sin_ra, cos_dec)
    //   +Z = Boresight
    //   Verify: East × North = Boresight ✓ (right-handed)
    let cam_x_icrs = Vector3::new(-sin_ra, cos_ra, 0.0).normalize();
    let cam_y_icrs =
        Vector3::new(-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec).normalize();
    let cam_z_icrs = Vector3::new(
        cos_dec * cos_ra,
        cos_dec * sin_ra,
        sin_dec,
    )
    .normalize();

    // Build rotation matrix: R * icrs = tangent-plane frame
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

/// Test data: each TESS FFI file with its expected boresight (from CRVAL).
struct TessTestCase {
    filename: &'static str,
    description: &'static str,
}

const TESS_TEST_CASES: &[TessTestCase] = &[
    TessTestCase {
        filename: "sparse_field_north_ecliptic.fits",
        description: "Sparse field near north ecliptic pole",
    },
    TessTestCase {
        filename: "moderate_density_field.fits",
        description: "Moderate density star field",
    },
    TessTestCase {
        filename: "dense_galactic_plane.fits",
        description: "Dense field near galactic plane",
    },
];

/// Build a database suitable for TESS FOV (~12° per CCD, 17° diagonal, ~21" pixel scale).
fn build_tess_database() -> SolverDatabase {
    let config = GenerateDatabaseConfig {
        max_fov_deg: 20.0, // TESS CCD diagonal is ~17°, need margin
        min_fov_deg: None,
        star_max_magnitude: Some(9.0), // TESS detects stars to ~mag 10+, but we use 7 for speed
        pattern_max_error: 0.005,
        lattice_field_oversampling: 100,
        patterns_per_lattice_field: 150,
        verification_stars_per_fov: 50,
        multiscale_step: 1.5,
        epoch_proper_motion_year: Some(2018.55), // TESS observation date: 2018-07-25
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

    // Build a database for TESS-like FOV
    let db = build_tess_database();
    println!("\n══════════════════════════════════════════════════════════════");
    println!(
        "Database: {} stars, {} patterns",
        db.star_catalog.len(),
        db.props.num_patterns,
    );

    let mut all_passed = true;

    for tc in TESS_TEST_CASES {
        let fits_path = format!("data/tess_test_images/{}", tc.filename);
        println!("\n══════════════════════════════════════════════════════════════");
        println!("Testing: {} ({})", tc.filename, tc.description);

        // ── Read the FITS file ──
        let hdus = read_fits_hdus(&fits_path);
        assert!(hdus.len() >= 2, "Expected at least 2 HDUs in FITS file");

        let image_hdu = &hdus[1];
        let naxis1 = match image_hdu.headers.get("NAXIS1") {
            Some(FitsValue::Int(n)) => *n as u32,
            _ => panic!("Missing NAXIS1"),
        };
        let naxis2 = match image_hdu.headers.get("NAXIS2") {
            Some(FitsValue::Int(n)) => *n as u32,
            _ => panic!("Missing NAXIS2"),
        };

        println!("  Image size: {} x {}", naxis1, naxis2);

        // Read pixel data
        let pixels = read_f32_image(&fits_path, image_hdu);
        assert_eq!(pixels.len(), (naxis1 as usize) * (naxis2 as usize));

        // Get the WCS boresight from the header (truth)
        let crval_ra = get_f64(image_hdu, "CRVAL1").unwrap();
        let crval_dec = get_f64(image_hdu, "CRVAL2").unwrap();
        let true_boresight = radec_to_uvec(crval_ra, crval_dec);
        println!(
            "  WCS boresight: RA={:.4}°, Dec={:.4}°",
            crval_ra, crval_dec
        );

        // Compute FOV from CD matrix
        let cd11 = get_f64(image_hdu, "CD1_1").unwrap();
        let cd12 = get_f64(image_hdu, "CD1_2").unwrap();
        let cd21 = get_f64(image_hdu, "CD2_1").unwrap();
        let cd22 = get_f64(image_hdu, "CD2_2").unwrap();
        let det_cd = cd11 * cd22 - cd12 * cd21;
        let pixel_scale_deg = det_cd.abs().sqrt();
        let fov_h_deg = pixel_scale_deg * naxis1 as f64;
        let fov_v_deg = pixel_scale_deg * naxis2 as f64;
        let fov_diag_deg = (fov_h_deg * fov_h_deg + fov_v_deg * fov_v_deg).sqrt();
        println!(
            "  FOV: {:.1}° x {:.1}° (diagonal {:.1}°), pixel scale {:.2}\"/px",
            fov_h_deg,
            fov_v_deg,
            fov_diag_deg,
            pixel_scale_deg * 3600.0
        );
        println!(
            "  CD matrix: [{:.6}, {:.6}; {:.6}, {:.6}], det={:.2e}",
            cd11, cd12, cd21, cd22, det_cd
        );

        // ── Crop to science pixels only ──
        // TESS FFIs include overscan columns and virtual/smear rows.
        // Science area: columns 45-2092 (0-indexed: 44..2092), rows 0-2047.
        let sci_col_start = 44_u32;
        let sci_col_end = 2092_u32;
        let sci_row_start = 0_u32;
        let sci_row_end = 2048_u32;

        let sci_width = sci_col_end - sci_col_start;
        let sci_height = sci_row_end - sci_row_start;

        // Extract science region and handle NaN pixels
        let mut sci_pixels = Vec::with_capacity((sci_width * sci_height) as usize);
        for row in sci_row_start..sci_row_end {
            for col in sci_col_start..sci_col_end {
                let idx = (row * naxis1 + col) as usize;
                let v = pixels[idx];
                sci_pixels.push(if v.is_nan() || v.is_infinite() {
                    0.0
                } else {
                    v
                });
            }
        }

        // Recompute FOV for science region only
        let sci_fov_h_deg = pixel_scale_deg * sci_width as f64;
        let sci_fov_v_deg = pixel_scale_deg * sci_height as f64;
        let sci_fov_diag_deg =
            (sci_fov_h_deg * sci_fov_h_deg + sci_fov_v_deg * sci_fov_v_deg).sqrt();
        println!(
            "  Science region: {}x{} px, FOV {:.1}° x {:.1}° (diagonal {:.1}°)",
            sci_width, sci_height, sci_fov_h_deg, sci_fov_v_deg, sci_fov_diag_deg
        );

        // ── Save science region as PNG for visual inspection ──
        {
            // Percentile stretch: sort a sample of pixels to find 1st and 99th percentile
            let mut sorted_vals: Vec<f32> =
                sci_pixels.iter().copied().filter(|v| *v > 0.0).collect();
            sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let lo = sorted_vals[sorted_vals.len() / 100]; // 1st percentile
            let hi = sorted_vals[sorted_vals.len() * 99 / 100]; // 99th percentile
            let range = hi - lo;
            println!("  PNG stretch: lo={:.1}, hi={:.1}", lo, hi);

            let mut img_buf = image::GrayImage::new(sci_width, sci_height);
            for row in 0..sci_height {
                for col in 0..sci_width {
                    let idx = (row * sci_width + col) as usize;
                    let v = sci_pixels[idx];
                    let normed = if range > 0.0 {
                        ((v - lo) / range).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    // Apply sqrt stretch for better visibility of faint stars
                    let byte = (normed.sqrt() * 255.0) as u8;
                    img_buf.put_pixel(col, row, image::Luma([byte]));
                }
            }
            let png_path = format!(
                "data/tess_test_images/{}.png",
                tc.filename.trim_end_matches(".fits")
            );
            img_buf.save(&png_path).expect("Failed to save PNG");
            println!("  Saved: {}", png_path);
        }

        // ── Extract centroids ──
        let extract_config = CentroidExtractionConfig {
            fov_horizontal_deg: sci_fov_h_deg as f32,
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
            tetra3::extract_centroids_from_raw(&sci_pixels, sci_width, sci_height, &extract_config)
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

        // Print first 10 centroids for debugging
        println!("  First 10 centroids (raw pixel-scaled):");
        for (i, c) in extraction.centroids.iter().take(10).enumerate() {
            println!(
                "    [{:2}] x={:+.5} rad ({:+.3}°), y={:+.5} rad ({:+.3}°), mass={:.0}",
                i,
                c.x,
                c.x.to_degrees(),
                c.y,
                c.y.to_degrees(),
                c.mass.unwrap_or(0.0)
            );
        }

        // ── Apply SIP distortion correction ──
        // TESS images use TAN-SIP projection: pixel coords must be corrected
        // for optical distortion before converting to tangent-plane coords.
        //
        // Pipeline:
        //   1. Convert centroid (x,y) back to pixel position in full frame
        //   2. Compute (u,v) = pixel offset from CRPIX
        //   3. Apply SIP forward polynomial: du = Σ A_p_q · u^p · v^q
        //   4. Corrected offset: u' = u + du, v' = v + dv
        //   5. Apply CD matrix: (ξ,η) = CD · (u',v') in degrees
        //   6. Convert to radians for solver centroids

        let crpix1 = get_f64(image_hdu, "CRPIX1").unwrap();
        let crpix2 = get_f64(image_hdu, "CRPIX2").unwrap();

        // Parse SIP polynomial orders
        let a_order = match image_hdu.headers.get("A_ORDER") {
            Some(FitsValue::Int(n)) => *n as usize,
            _ => 0,
        };
        let b_order = match image_hdu.headers.get("B_ORDER") {
            Some(FitsValue::Int(n)) => *n as usize,
            _ => 0,
        };

        // Collect SIP A coefficients
        let mut a_coeffs: HashMap<(usize, usize), f64> = HashMap::new();
        for p in 0..=a_order {
            for q in 0..=(a_order - p) {
                let key = format!("A_{}_{}", p, q);
                if let Some(FitsValue::Float(v)) = image_hdu.headers.get(&key) {
                    a_coeffs.insert((p, q), *v);
                }
            }
        }

        // Collect SIP B coefficients
        let mut b_coeffs: HashMap<(usize, usize), f64> = HashMap::new();
        for p in 0..=b_order {
            for q in 0..=(b_order - p) {
                let key = format!("B_{}_{}", p, q);
                if let Some(FitsValue::Float(v)) = image_hdu.headers.get(&key) {
                    b_coeffs.insert((p, q), *v);
                }
            }
        }

        println!(
            "  SIP correction: A_ORDER={}, B_ORDER={}, {} A coeffs, {} B coeffs",
            a_order,
            b_order,
            a_coeffs.len(),
            b_coeffs.len()
        );

        // Pixel scale used by centroid extraction (for converting back to pixel coords)
        let pixel_size_rad = (sci_fov_h_deg as f64).to_radians() / sci_width as f64;

        // Build corrected centroids using the full WCS pipeline:
        //   1. Convert centroid → pixel position in full frame
        //   2. Compute (u,v) = pixel offset from CRPIX
        //   3. Apply SIP forward polynomial: u' = u + A(u,v), v' = v + B(u,v)
        //   4. Apply CD matrix: (xi, eta) = CD · (u', v') → tangent plane (degrees)
        //   5. Convert to radians for solver centroids
        //
        // This is critical for TESS because the CD matrix columns are not
        // orthogonal (~88.6° instead of 90°), so a uniform pixel scale gives
        // direction-dependent angular errors that exceed the pattern tolerance.
        let mut max_du = 0.0_f64;
        let mut max_dv = 0.0_f64;
        let corrected_centroids: Vec<tetra3::Centroid> = extraction
            .centroids
            .iter()
            .map(|c| {
                // Step 1: Convert from pixel-scaled (radians) back to pixel position in science region
                let col_sci = (c.x as f64) / pixel_size_rad + (sci_width as f64) / 2.0;
                let row_sci = (c.y as f64) / pixel_size_rad + (sci_height as f64) / 2.0;

                // Step 2: Convert to full-frame pixel coords (0-indexed), then to FITS 1-indexed
                let col_full = col_sci + sci_col_start as f64;
                let row_full = row_sci + sci_row_start as f64;

                // Step 3: Compute offset from CRPIX (CRPIX is 1-indexed in FITS)
                let u = col_full + 1.0 - crpix1;
                let v = row_full + 1.0 - crpix2;

                // Step 4: Apply SIP polynomial correction (in pixel space)
                let mut du = 0.0_f64;
                for (&(p, q), &coeff) in &a_coeffs {
                    du += coeff * (u.powi(p as i32)) * (v.powi(q as i32));
                }
                let mut dv = 0.0_f64;
                for (&(p, q), &coeff) in &b_coeffs {
                    dv += coeff * (u.powi(p as i32)) * (v.powi(q as i32));
                }

                if du.abs() > max_du {
                    max_du = du.abs();
                }
                if dv.abs() > max_dv {
                    max_dv = dv.abs();
                }

                let u_corr = u + du;
                let v_corr = v + dv;

                // Step 5: Apply CD matrix → tangent plane coordinates (degrees → radians)
                let xi_rad = (cd11 * u_corr + cd12 * v_corr).to_radians();
                let eta_rad = (cd21 * u_corr + cd22 * v_corr).to_radians();

                tetra3::Centroid {
                    x: xi_rad as f32,
                    y: eta_rad as f32,
                    mass: c.mass,
                    cov: c.cov,
                }
            })
            .collect();

        println!(
            "  SIP max correction: du={:.1} px ({:.1}\"), dv={:.1} px ({:.1}\")",
            max_du,
            max_du * pixel_size_rad.to_degrees() * 3600.0,
            max_dv,
            max_dv * pixel_size_rad.to_degrees() * 3600.0
        );

        // Print first 10 corrected centroids (tangent-plane frame)
        println!("  First 10 corrected centroids (SIP + CD matrix):");
        for (i, c) in corrected_centroids.iter().take(10).enumerate() {
            println!(
                "    [{:2}] x={:+.5} rad ({:+.3}°), y={:+.5} rad ({:+.3}°)",
                i,
                c.x,
                c.x.to_degrees(),
                c.y,
                c.y.to_degrees(),
            );
        }

        // ── Pixel-space diagnostic ──
        // Project catalog stars to pixel coordinates via TAN+CD^-1 (bypass quaternion)
        // and compare directly with centroid pixel positions.
        println!("  CRPIX1={:.2}, CRPIX2={:.2}", crpix1, crpix2);
        println!(
            "  Science center in full frame: col={:.1}, row={:.1}",
            sci_col_start as f64 + sci_width as f64 / 2.0,
            sci_row_start as f64 + sci_height as f64 / 2.0,
        );

        // Inverse CD matrix for (xi,eta) → (u,v) in pixel space
        let cd_inv_11 = cd22 / det_cd;
        let cd_inv_12 = -cd12 / det_cd;
        let cd_inv_21 = -cd21 / det_cd;
        let cd_inv_22 = cd11 / det_cd;

        let ra0_rad = crval_ra.to_radians();
        let dec0_rad = crval_dec.to_radians();

        // ── Validate centroids by projecting catalog stars using known WCS ──
        let wcs_q = wcs_to_quaternion(image_hdu);
        let wcs_boresight_uvec = wcs_q.inverse() * Vector3::new(0.0, 0.0, 1.0);
        let half_fov = (sci_fov_diag_deg as f32 / 2.0).to_radians();
        let nearby_indices = db
            .star_catalog
            .query_indices_from_uvec(wcs_boresight_uvec, half_fov * 1.2);

        // Project catalog stars into camera frame
        let mut catalog_centroids: Vec<(f32, f32, f32, usize)> = Vec::new(); // (x_rad, y_rad, mag, star_idx)
        for &idx in &nearby_indices {
            let sv = &db.star_vectors[idx];
            let icrs_v = Vector3::new(sv[0], sv[1], sv[2]);
            let cam_v = wcs_q * icrs_v;
            if cam_v.z > 0.01 {
                let cx = cam_v.x / cam_v.z;
                let cy = cam_v.y / cam_v.z;
                if cx.abs() < half_fov && cy.abs() < half_fov {
                    catalog_centroids.push((cx, cy, db.star_catalog.stars()[idx].mag, idx));
                }
            }
        }
        catalog_centroids.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        println!(
            "  Catalog stars in FOV: {} (via WCS projection)",
            catalog_centroids.len()
        );
        println!("  Brightest 10 catalog stars (projected vs SIP-corrected centroids):");
        for (i, &(cx, cy, mag, _idx)) in catalog_centroids.iter().take(10).enumerate() {
            // Find closest SIP-corrected centroid
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
            println!(
                "    [{:2}] mag={:.1} pos=({:+.4}°, {:+.4}°) → closest centroid [{:2}] dist={:.1}' ({:.1}\")",
                i,
                mag,
                cx.to_degrees(),
                cy.to_degrees(),
                best_j,
                best_dist.to_degrees() * 60.0,
                best_dist.to_degrees() * 3600.0,
            );
        }

        // ── Pixel-space comparison (bypass quaternion) ──
        // Project catalog stars to pixel coords via TAN + CD^-1, compare with centroids
        println!("  Pixel-space comparison (TAN+CD^-1, brightest 10):");
        for (i, &(_, _, mag, idx)) in catalog_centroids.iter().take(10).enumerate() {
            let sv = &db.star_vectors[idx];
            let star_ra = (sv[1] as f64).atan2(sv[0] as f64);
            let star_dec = (sv[2] as f64).asin();

            // TAN projection: star → tangent plane at CRVAL
            let dra = star_ra - ra0_rad;
            let cos_dec = star_dec.cos();
            let sin_dec = star_dec.sin();
            let cos_dec0 = dec0_rad.cos();
            let sin_dec0 = dec0_rad.sin();
            let cos_dra = dra.cos();
            let denom = sin_dec * sin_dec0 + cos_dec * cos_dec0 * cos_dra;
            if denom <= 0.0 {
                continue; // behind tangent point
            }
            let xi_deg = (cos_dec * dra.sin() / denom).to_degrees();
            let eta_deg =
                ((cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_dra) / denom).to_degrees();

            // Inverse CD: tangent plane → pixel offsets from CRPIX
            let u = cd_inv_11 * xi_deg + cd_inv_12 * eta_deg;
            let v = cd_inv_21 * xi_deg + cd_inv_22 * eta_deg;

            // Pixel position in science region (0-indexed)
            let col_full = u + crpix1 - 1.0; // 0-indexed full frame
            let row_full = v + crpix2 - 1.0;
            let col_sci = col_full - sci_col_start as f64;
            let row_sci = row_full - sci_row_start as f64;

            // Convert to pixel-scaled radians (same frame as centroids)
            let cat_x = (col_sci - sci_width as f64 / 2.0) * pixel_size_rad;
            let cat_y = (row_sci - sci_height as f64 / 2.0) * pixel_size_rad;

            // Find nearest centroid
            let mut best_dist = f64::MAX;
            let mut best_j = 0;
            for (j, ec) in extraction.centroids.iter().enumerate() {
                let dx = ec.x as f64 - cat_x;
                let dy = ec.y as f64 - cat_y;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < best_dist {
                    best_dist = dist;
                    best_j = j;
                }
            }
            println!(
                "    [{:2}] mag={:.1} pixel=({:.1},{:.1}) x={:+.4}° y={:+.4}° → centroid [{:2}] dist={:.1}' ({:.1}\")",
                i,
                mag,
                col_sci,
                row_sci,
                cat_x.to_degrees(),
                cat_y.to_degrees(),
                best_j,
                best_dist.to_degrees() * 60.0,
                best_dist.to_degrees() * 3600.0,
            );
        }

        if extraction.centroids.len() < 4 {
            println!(
                "  SKIP: Too few centroids ({}) to attempt solve",
                extraction.centroids.len()
            );
            continue;
        }

        // ── Reference solve: use catalog stars projected through WCS ──
        // This verifies the database works for this sky region
        let mut ref_centroids: Vec<tetra3::Centroid> = catalog_centroids
            .iter()
            .map(|&(cx, cy, mag, _idx)| tetra3::Centroid {
                x: cx,
                y: cy,
                mass: Some(10.0 - mag),
                cov: None,
            })
            .collect();
        // Keep only stars within the image FOV
        let half_fov_h = (sci_fov_h_deg as f32 / 2.0).to_radians();
        let half_fov_v = (sci_fov_v_deg as f32 / 2.0).to_radians();
        ref_centroids.retain(|c| c.x.abs() < half_fov_h && c.y.abs() < half_fov_v);

        let ref_solve_config = SolveConfig {
            fov_estimate_rad: (sci_fov_h_deg as f32).to_radians(),
            fov_max_error_rad: Some((3.0_f32).to_radians()),
            match_radius: 0.01,
            match_threshold: 1e-5,
            solve_timeout_ms: Some(30_000),
            match_max_error: None,
        };

        let ref_result = db.solve_from_centroids(&ref_centroids, &ref_solve_config);
        println!(
            "  Reference solve (WCS catalog centroids): {:?} ({} centroids, {:.1}ms)",
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

        // ── Solve from extracted centroids ──
        // Use horizontal FOV as the estimate (matches the database pattern scale)
        let solve_config = SolveConfig {
            fov_estimate_rad: (sci_fov_h_deg as f32).to_radians(),
            fov_max_error_rad: Some((3.0_f32).to_radians()),
            match_radius: 0.01,
            match_threshold: 1e-5,
            solve_timeout_ms: Some(60_000),
            match_max_error: Some(0.02),
        };

        let result = db.solve_from_centroids(&corrected_centroids, &solve_config);

        println!("  Solve status: {:?}", result.status);
        println!("  Solve time:   {:.1} ms", result.solve_time_ms);

        if result.status == SolveStatus::MatchFound {
            let solved_q = result.qicrs2cam.unwrap();
            let solved_boresight = solved_q.inverse() * Vector3::new(0.0, 0.0, 1.0);
            let (solved_ra, solved_dec) = uvec_to_radec(&solved_boresight);

            // Boresight error
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

            // Success criterion: boresight within 30 arcminutes of WCS solution
            // (generous given that CRVAL is the tangent point which may not be
            // exactly at the optical center, and the WCS includes SIP distortion
            // that our simple tangent-plane centroid model ignores)
            if error_arcmin > 30.0 {
                println!(
                    "  *** FAIL: boresight error {:.1}' exceeds 30' ***",
                    error_arcmin
                );
                all_passed = false;
            } else {
                println!("  PASS");
            }
        } else {
            println!("  *** FAIL: no match found ***");
            all_passed = false;
        }
    }

    println!("\n══════════════════════════════════════════════════════════════");
    assert!(all_passed, "One or more TESS solve tests failed");
}
