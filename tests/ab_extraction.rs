//! A/B comparison harness for centroid extraction.
//!
//! Runs extraction on a set of representative FITS test images with three
//! configurations and prints a table comparing raw blob counts, kept
//! centroids, and wall-clock time. Not a pass/fail test — run with
//! `--nocapture` to see the table:
//!
//! ```sh
//! cargo test --release --features image --test ab_extraction -- --nocapture
//! ```

mod test_data;

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;
use tetra3::{centroid_extraction::extract_centroids_from_raw, CentroidExtractionConfig};

// ── Minimal FITS reader (HDU 0, float32 BITPIX=-32 or int16 BITPIX=16) ──

enum FitsValue {
    Float(f64),
    Int(i64),
}

struct FitsInfo {
    headers: HashMap<String, FitsValue>,
    data_offset: u64,
    bitpix: i64,
    naxis1: u32,
    naxis2: u32,
    bzero: f64,
    bscale: f64,
}

fn parse_card(card: &[u8; 80]) -> Option<(String, FitsValue)> {
    let s = String::from_utf8_lossy(card);
    let kw = s[..8].trim().to_string();
    if kw.is_empty() || kw == "COMMENT" || kw == "HISTORY" || kw == "END" {
        return None;
    }
    if s.len() < 10 || &s[8..10] != "= " {
        return None;
    }
    let v = s[10..].trim();
    let num_part = v.split('/').next().unwrap_or(v).trim();
    if num_part.starts_with('\'') {
        return None;
    }
    if let Ok(i) = num_part.parse::<i64>() {
        Some((kw, FitsValue::Int(i)))
    } else if let Ok(f) = num_part.parse::<f64>() {
        Some((kw, FitsValue::Float(f)))
    } else {
        None
    }
}

/// Read HDUs until one contains a 2D image (NAXIS==2). TESS FFIs have the
/// image in HDU 1; SkyView files have it in HDU 0.
fn read_first_image_hdu(path: &str) -> (FitsInfo, File) {
    let mut file = File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path, e));
    let mut offset: u64 = 0;

    loop {
        let mut headers = HashMap::new();
        loop {
            let mut block = [0u8; 2880];
            file.read_exact(&mut block).expect("truncated FITS");
            offset += 2880;
            let mut done = false;
            for i in 0..36 {
                let card: &[u8; 80] = block[i * 80..(i + 1) * 80].try_into().unwrap();
                let cs = String::from_utf8_lossy(card);
                if cs.starts_with("END") {
                    done = true;
                    break;
                }
                if let Some((k, v)) = parse_card(card) {
                    headers.insert(k, v);
                }
            }
            if done {
                break;
            }
        }
        let get_i = |hs: &HashMap<String, FitsValue>, k: &str| match hs.get(k) {
            Some(FitsValue::Int(i)) => Some(*i),
            Some(FitsValue::Float(f)) => Some(*f as i64),
            None => None,
        };
        let get_f = |hs: &HashMap<String, FitsValue>, k: &str| match hs.get(k) {
            Some(FitsValue::Float(f)) => Some(*f),
            Some(FitsValue::Int(i)) => Some(*i as f64),
            None => None,
        };

        let naxis = get_i(&headers, "NAXIS").unwrap_or(0);
        let bitpix = get_i(&headers, "BITPIX").unwrap_or(8);

        if naxis == 2 {
            let naxis1 = get_i(&headers, "NAXIS1").unwrap() as u32;
            let naxis2 = get_i(&headers, "NAXIS2").unwrap() as u32;
            let bzero = get_f(&headers, "BZERO").unwrap_or(0.0);
            let bscale = get_f(&headers, "BSCALE").unwrap_or(1.0);
            return (
                FitsInfo {
                    headers,
                    data_offset: offset,
                    bitpix,
                    naxis1,
                    naxis2,
                    bzero,
                    bscale,
                },
                file,
            );
        }

        // Not an image HDU — skip past its data (if any) and keep looking.
        let mut data_len: u64 = if naxis > 0 {
            let bpp = (bitpix.unsigned_abs() as u64) / 8;
            let mut n: u64 = 1;
            for i in 1..=naxis as usize {
                if let Some(v) = get_i(&headers, &format!("NAXIS{}", i)) {
                    n *= v as u64;
                }
            }
            n * bpp
        } else {
            0
        };
        if let Some(pc) = get_i(&headers, "PCOUNT") {
            data_len += pc as u64;
        }
        let padded = data_len.div_ceil(2880) * 2880;
        offset += padded;
        file.seek(SeekFrom::Start(offset)).unwrap();
    }
}

fn load_fits_f32(path: &str) -> (Vec<f32>, u32, u32) {
    let (info, mut file) = read_first_image_hdu(path);
    let npixels = (info.naxis1 as usize) * (info.naxis2 as usize);
    file.seek(SeekFrom::Start(info.data_offset)).unwrap();
    let pixels = match info.bitpix {
        -32 => {
            let mut buf = vec![0u8; npixels * 4];
            file.read_exact(&mut buf).unwrap();
            (0..npixels)
                .map(|i| {
                    let b: [u8; 4] = buf[i * 4..(i + 1) * 4].try_into().unwrap();
                    f32::from_be_bytes(b)
                })
                .collect()
        }
        16 => {
            let mut buf = vec![0u8; npixels * 2];
            file.read_exact(&mut buf).unwrap();
            (0..npixels)
                .map(|i| {
                    let b: [u8; 2] = buf[i * 2..(i + 1) * 2].try_into().unwrap();
                    let raw = i16::from_be_bytes(b);
                    (raw as f64 * info.bscale + info.bzero) as f32
                })
                .collect()
        }
        32 => {
            let mut buf = vec![0u8; npixels * 4];
            file.read_exact(&mut buf).unwrap();
            (0..npixels)
                .map(|i| {
                    let b: [u8; 4] = buf[i * 4..(i + 1) * 4].try_into().unwrap();
                    let raw = i32::from_be_bytes(b);
                    (raw as f64 * info.bscale + info.bzero) as f32
                })
                .collect()
        }
        other => panic!("unsupported BITPIX {} (in {})", other, path),
    };
    let _ = info.headers; // silence unused
    (pixels, info.naxis1, info.naxis2)
}

struct ImageCase {
    path: &'static str,
    label: &'static str,
}

const CASES: &[ImageCase] = &[
    ImageCase {
        path: "data/skyview_10deg_test_images/orion_region_10deg.fits",
        label: "orion (nebulous)",
    },
    ImageCase {
        path: "data/skyview_10deg_test_images/cassiopeia_10deg.fits",
        label: "cassiopeia",
    },
    ImageCase {
        path: "data/skyview_10deg_test_images/sagittarius_10deg.fits",
        label: "sagittarius (gal. ctr)",
    },
    ImageCase {
        path: "data/tess_test_images/dense_galactic_plane.fits",
        label: "tess dense plane",
    },
    ImageCase {
        path: "data/tess_test_images/sparse_field_north_ecliptic.fits",
        label: "tess sparse",
    },
];

struct ConfigCase {
    label: &'static str,
    config: CentroidExtractionConfig,
}

fn build_configs() -> Vec<ConfigCase> {
    // Note: `num_blobs_raw` in CentroidExtractionResult counts blobs *after*
    // size/elongation/SNR filtering (despite the name), so it equals the
    // final centroid count when `max_centroids` is None. To expose the
    // impact of the SNR cut via the `kept` column, we leave `max_centroids`
    // unset and compare across configs.
    vec![
        ConfigCase {
            label: "fast (no local bg)",
            config: CentroidExtractionConfig {
                local_bg_block_size: None,
                snr_min: None,
                ..Default::default()
            },
        },
        ConfigCase {
            label: "baseline (pre-change)",
            config: CentroidExtractionConfig {
                snr_min: None,
                ..Default::default()
            },
        },
        ConfigCase {
            label: "default (snr>=5)",
            config: CentroidExtractionConfig::default(),
        },
        ConfigCase {
            label: "permissive+snr",
            config: CentroidExtractionConfig {
                sigma_threshold: 2.5,
                snr_min: Some(5.0),
                ..Default::default()
            },
        },
        ConfigCase {
            label: "matched-filter+snr",
            config: CentroidExtractionConfig {
                sigma_threshold: 3.0,
                snr_min: Some(5.0),
                matched_filter_sigma: Some(1.5),
                ..Default::default()
            },
        },
    ]
}

struct Stats {
    kept: usize,
    ms: f64,
}

fn bench_once(pixels: &[f32], w: u32, h: u32, cfg: &CentroidExtractionConfig, n_iter: usize) -> Stats {
    // Warmup (not timed) — first call pays page-fault costs.
    let warm = extract_centroids_from_raw(pixels, w, h, cfg).unwrap();
    let kept = warm.centroids.len();

    let start = Instant::now();
    for _ in 0..n_iter {
        let r = extract_centroids_from_raw(pixels, w, h, cfg).unwrap();
        std::hint::black_box(r);
    }
    let ms = start.elapsed().as_secs_f64() * 1000.0 / n_iter as f64;
    Stats { kept, ms }
}

#[test]
fn ab_extraction_table() {
    let configs = build_configs();
    const N_ITER: usize = 5;

    println!();
    println!(
        "{:<30} {:<24} {:>8} {:>10}",
        "image", "config", "kept", "ms/iter"
    );
    println!("{}", "─".repeat(76));

    for case in CASES {
        let path = test_data::ensure_test_file(case.path);
        let (pixels, w, h) = load_fits_f32(&path);

        // Pick the "fast" config as reference time for relative comparison.
        let mut ref_ms: Option<f64> = None;
        for (i, cc) in configs.iter().enumerate() {
            let s = bench_once(&pixels, w, h, &cc.config, N_ITER);
            if ref_ms.is_none() {
                ref_ms = Some(s.ms);
            }
            let rel = s.ms / ref_ms.unwrap();
            let img_label = if i == 0 {
                format!("{} ({}x{})", case.label, w, h)
            } else {
                String::new()
            };
            println!(
                "{:<30} {:<24} {:>8} {:>10.2}  ({:+.0}%)",
                img_label,
                cc.label,
                s.kept,
                s.ms,
                (rel - 1.0) * 100.0
            );
        }
        println!();
    }
}
