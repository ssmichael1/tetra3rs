//! Criterion microbenchmarks for centroid extraction.
//!
//! Run with:
//! ```sh
//! cargo bench --features image --bench extract
//! ```
//!
//! Benches a single representative FITS image across four configs:
//!   * `fast` — no local bg, no SNR cut (legacy quick path)
//!   * `baseline` — local bg, global sigma, no SNR cut (pre-change default)
//!   * `default` — local bg + sigma map + SNR cut (current default)
//!   * `permissive` — lowered per-pixel threshold with SNR cut doing selection
//!
//! The image is loaded once, then extraction is timed per iteration.

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use tetra3::{centroid_extraction::extract_centroids_from_raw, CentroidExtractionConfig};

// ── Minimal inline FITS loader (primary image HDU, BITPIX -32 / 16 / 32) ──

enum FitsValue {
    Float(f64),
    Int(i64),
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

fn load_fits_f32(path: &str) -> (Vec<f32>, u32, u32) {
    let mut file = File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path, e));
    let mut offset: u64 = 0;
    loop {
        let mut headers: HashMap<String, FitsValue> = HashMap::new();
        loop {
            let mut block = [0u8; 2880];
            file.read_exact(&mut block).unwrap();
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
        let get_i = |k: &str| match headers.get(k) {
            Some(FitsValue::Int(i)) => Some(*i),
            Some(FitsValue::Float(f)) => Some(*f as i64),
            None => None,
        };
        let get_f = |k: &str| match headers.get(k) {
            Some(FitsValue::Float(f)) => Some(*f),
            Some(FitsValue::Int(i)) => Some(*i as f64),
            None => None,
        };
        let naxis = get_i("NAXIS").unwrap_or(0);
        let bitpix = get_i("BITPIX").unwrap_or(8);
        if naxis == 2 {
            let n1 = get_i("NAXIS1").unwrap() as u32;
            let n2 = get_i("NAXIS2").unwrap() as u32;
            let bzero = get_f("BZERO").unwrap_or(0.0);
            let bscale = get_f("BSCALE").unwrap_or(1.0);
            let npixels = (n1 as usize) * (n2 as usize);
            file.seek(SeekFrom::Start(offset)).unwrap();
            let pixels = match bitpix {
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
                            (i16::from_be_bytes(b) as f64 * bscale + bzero) as f32
                        })
                        .collect()
                }
                32 => {
                    let mut buf = vec![0u8; npixels * 4];
                    file.read_exact(&mut buf).unwrap();
                    (0..npixels)
                        .map(|i| {
                            let b: [u8; 4] = buf[i * 4..(i + 1) * 4].try_into().unwrap();
                            (i32::from_be_bytes(b) as f64 * bscale + bzero) as f32
                        })
                        .collect()
                }
                other => panic!("unsupported BITPIX {} in {}", other, path),
            };
            return (pixels, n1, n2);
        }

        // Skip non-image HDU data and scan the next header.
        let mut data_len: u64 = if naxis > 0 {
            let bpp = (bitpix.unsigned_abs() as u64) / 8;
            let mut n: u64 = 1;
            for i in 1..=naxis as usize {
                if let Some(v) = get_i(&format!("NAXIS{}", i)) {
                    n *= v as u64;
                }
            }
            n * bpp
        } else {
            0
        };
        if let Some(pc) = get_i("PCOUNT") {
            data_len += pc as u64;
        }
        let padded = data_len.div_ceil(2880) * 2880;
        offset += padded;
        file.seek(SeekFrom::Start(offset)).unwrap();
    }
}

fn extract_bench(c: &mut Criterion) {
    // Orion — nebulous 2048×2048 SkyView image. Good stress case for local
    // bg subtraction and threshold behavior. Skip the bench if the file is
    // missing (run integration tests first to download test data).
    let path = "data/skyview_10deg_test_images/orion_region_10deg.fits";
    if !Path::new(path).exists() {
        eprintln!(
            "skipping: {} not found. run `cargo test --features image --test skyview_solve_test` once to download.",
            path
        );
        return;
    }
    let (pixels, w, h) = load_fits_f32(path);

    let mut group = c.benchmark_group("extract/orion_2048");
    group.sample_size(30);

    let fast = CentroidExtractionConfig {
        local_bg_block_size: None,
        snr_min: None,
        ..Default::default()
    };
    let baseline = CentroidExtractionConfig {
        snr_min: None,
        ..Default::default()
    };
    let default_cfg = CentroidExtractionConfig::default();
    let permissive = CentroidExtractionConfig {
        sigma_threshold: 2.5,
        snr_min: Some(5.0),
        ..Default::default()
    };

    group.bench_function("fast", |b| {
        b.iter(|| extract_centroids_from_raw(black_box(&pixels), w, h, &fast).unwrap())
    });
    group.bench_function("baseline", |b| {
        b.iter(|| extract_centroids_from_raw(black_box(&pixels), w, h, &baseline).unwrap())
    });
    group.bench_function("default", |b| {
        b.iter(|| extract_centroids_from_raw(black_box(&pixels), w, h, &default_cfg).unwrap())
    });
    group.bench_function("permissive", |b| {
        b.iter(|| extract_centroids_from_raw(black_box(&pixels), w, h, &permissive).unwrap())
    });

    group.finish();
}

criterion_group!(benches, extract_bench);
criterion_main!(benches);
