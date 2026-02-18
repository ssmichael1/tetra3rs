//! Shared test data download helper.
//!
//! Downloads test data files from Google Cloud Storage if not present locally.
//! Files are cached in `data/` so they only need to be downloaded once.

use std::fs;
use std::io::{Read, Write};
use std::path::Path;

/// Base URL for test data on Google Cloud Storage.
/// The bucket has the contents of `data/` at the top level (no `data/` prefix).
const GCS_BASE_URL: &str = "https://storage.googleapis.com/tetra3rs-testvecs";

/// Ensure a test data file exists locally, downloading from GCS if needed.
///
/// `local_path` is relative to the repo root (e.g. "data/hip2.dat").
/// Returns the local path string for convenience.
pub fn ensure_test_file(local_path: &str) -> String {
    let path = Path::new(local_path);
    if path.exists() {
        return local_path.to_string();
    }

    // Create parent directories
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("Failed to create directory {:?}: {}", parent, e));
    }

    // Strip the "data/" prefix for the GCS key since the bucket
    // has the contents of data/ at the top level.
    let gcs_key = local_path.strip_prefix("data/").unwrap_or(local_path);
    let url = format!("{}/{}", GCS_BASE_URL, gcs_key);
    println!("Downloading {} ...", url);

    let resp = ureq::get(&url)
        .call()
        .unwrap_or_else(|e| panic!("Failed to download {}: {}", url, e));

    let mut reader = resp.into_body().into_reader();
    let mut file = fs::File::create(path)
        .unwrap_or_else(|e| panic!("Failed to create {:?}: {}", path, e));

    let mut buf = vec![0u8; 1024 * 1024]; // 1MB buffer
    loop {
        let n = reader
            .read(&mut buf)
            .unwrap_or_else(|e| panic!("Failed to read response for {}: {}", url, e));
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .unwrap_or_else(|e| panic!("Failed to write {:?}: {}", path, e));
    }

    println!("Downloaded {} -> {}", url, local_path);
    local_path.to_string()
}
