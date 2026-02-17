//! Extract star centroids from an astronomical image.
//!
//! This module provides functions to detect and locate stars in an image by:
//! 1. Converting the image to grayscale floating-point values
//! 2. Estimating and subtracting the background (sigma-clipped median)
//! 3. Thresholding to identify bright pixels
//! 4. Labeling connected components (blobs)
//! 5. Computing intensity-weighted centroids for each blob
//! 6. Converting pixel positions to angular offsets (radians from boresight)
//!
//! Requires the `image` feature to be enabled.
//!
//! # Example
//!
//! ```no_run
//! use tetra3::centroid_extraction::{CentroidExtractionConfig, extract_centroids};
//!
//! let config = CentroidExtractionConfig {
//!     fov_horizontal_deg: 10.0,
//!     ..Default::default()
//! };
//!
//! let centroids = extract_centroids("my_star_image.png", &config).unwrap();
//! println!("Found {} stars", centroids.len());
//! ```

use crate::centroid::Centroid;
use anyhow::{Context, Result};
use image::GenericImageView;

/// Configuration for centroid extraction from an image.
#[derive(Debug, Clone)]
pub struct CentroidExtractionConfig {
    /// Horizontal field of view of the camera in degrees.
    /// This is used to convert pixel positions to angular offsets.
    pub fov_horizontal_deg: f32,

    /// Number of sigma above background to use as the detection threshold.
    /// Stars brighter than `background + sigma_threshold * noise` are detected.
    /// Default: 5.0
    pub sigma_threshold: f32,

    /// Minimum number of pixels in a blob to be considered a star.
    /// Helps filter out hot pixels and noise.
    /// Default: 3
    pub min_pixels: usize,

    /// Maximum number of pixels in a blob to be considered a star.
    /// Helps filter out very large extended objects.
    /// Set high enough to include saturated bright stars with large halos.
    /// Default: 10000
    pub max_pixels: usize,

    /// Maximum number of centroids to return, sorted by brightness (mass).
    /// If `None`, all detected centroids are returned.
    /// Default: None
    pub max_centroids: Option<usize>,

    /// Number of iterations for sigma-clipped background estimation.
    /// Default: 5
    pub sigma_clip_iterations: usize,

    /// Sigma clipping factor for background estimation.
    /// Pixels more than this many sigma from the mean are excluded.
    /// Default: 3.0
    pub sigma_clip_factor: f32,

    /// Whether to use 8-connectivity (true) or 4-connectivity (false) for
    /// connected component labeling.
    /// Default: true (8-connectivity)
    pub use_8_connectivity: bool,

    /// Block size (in pixels) for local background estimation.
    ///
    /// When set to `Some(n)`, the image is divided into `n×n` blocks and
    /// the median value in each block is computed. A smooth background
    /// model is created by bilinear interpolation between block centers
    /// and subtracted before star detection. This removes large-scale
    /// gradients from nebulosity, Milky Way emission, or vignetting.
    ///
    /// A good starting value is 32-128 pixels, or roughly 1-3% of the
    /// image width. Smaller blocks follow finer structure but risk
    /// subtracting real stars.
    ///
    /// When `None`, only global background subtraction is used (original
    /// behavior).
    ///
    /// Default: Some(64)
    pub local_bg_block_size: Option<u32>,

    /// Maximum allowed elongation ratio (major/minor axis) for a detected
    /// blob. Blobs more elongated than this are rejected as non-stellar
    /// (e.g. cosmic rays, satellite trails, diffraction spikes).
    ///
    /// A value of 2.0 means the blob can be at most 2× longer than wide.
    /// Set to a large value (e.g. 100) or `None` to disable.
    ///
    /// Default: None (disabled)
    pub max_elongation: Option<f32>,
}

impl Default for CentroidExtractionConfig {
    fn default() -> Self {
        Self {
            fov_horizontal_deg: 10.0,
            sigma_threshold: 5.0,
            min_pixels: 3,
            max_pixels: 10000,
            max_centroids: None,
            sigma_clip_iterations: 5,
            sigma_clip_factor: 3.0,
            use_8_connectivity: true,
            local_bg_block_size: Some(64),
            max_elongation: Some(3.0),
        }
    }
}

/// Result of centroid extraction, containing the centroids and diagnostic info.
#[derive(Debug, Clone)]
pub struct CentroidExtractionResult {
    /// Extracted centroids in angular coordinates (radians from boresight).
    pub centroids: Vec<Centroid>,

    /// Image width in pixels.
    pub image_width: u32,

    /// Image height in pixels.
    pub image_height: u32,

    /// Estimated background level (in image intensity units).
    pub background_mean: f32,

    /// Estimated background noise standard deviation.
    pub background_sigma: f32,

    /// Detection threshold used (background_mean + sigma_threshold * background_sigma).
    pub threshold: f32,

    /// Number of blobs found before size filtering.
    pub num_blobs_raw: usize,
}

/// Extract star centroids from an image file.
///
/// Loads the image from `path`, performs background subtraction, blob detection,
/// and centroid computation. Returns centroids in angular coordinates (radians
/// from boresight), suitable for use with [`SolverDatabase::solve_from_centroids`].
///
/// # Arguments
///
/// * `path` - Path to the image file (supports PNG, JPEG, TIFF, FITS, etc.)
/// * `config` - Extraction configuration parameters
///
/// # Returns
///
/// A [`CentroidExtractionResult`] containing the detected centroids and diagnostics.
pub fn extract_centroids(
    path: impl AsRef<std::path::Path>,
    config: &CentroidExtractionConfig,
) -> Result<CentroidExtractionResult> {
    let img = image::open(path.as_ref())
        .with_context(|| format!("Failed to open image: {}", path.as_ref().display()))?;
    extract_centroids_from_image(&img, config)
}

/// Extract star centroids from an already-loaded [`image::DynamicImage`].
///
/// Same algorithm as [`extract_centroids`] but operates on an in-memory image.
pub fn extract_centroids_from_image(
    img: &image::DynamicImage,
    config: &CentroidExtractionConfig,
) -> Result<CentroidExtractionResult> {
    let (width, height) = img.dimensions();
    let gray = to_grayscale_f32(img);
    extract_from_gray(&gray, width, height, config)
}

/// Extract star centroids from raw grayscale pixel data.
///
/// This is useful when you have pixel data that isn't in a standard image format,
/// e.g. from a camera SDK or FITS file parsed externally.
///
/// # Arguments
///
/// * `pixels` - Row-major grayscale pixel values (length must equal `width * height`)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `config` - Extraction configuration parameters
pub fn extract_centroids_from_raw(
    pixels: &[f32],
    width: u32,
    height: u32,
    config: &CentroidExtractionConfig,
) -> Result<CentroidExtractionResult> {
    anyhow::ensure!(
        pixels.len() == (width as usize) * (height as usize),
        "Pixel data length ({}) does not match width*height ({}x{}={})",
        pixels.len(),
        width,
        height,
        width as usize * height as usize
    );
    extract_from_gray(pixels, width, height, config)
}

// ─── Internal helpers ──────────────────────────────────────────────────────

/// Shared extraction pipeline for both image and raw-pixel entry points.
fn extract_from_gray(
    gray_input: &[f32],
    width: u32,
    height: u32,
    config: &CentroidExtractionConfig,
) -> Result<CentroidExtractionResult> {
    let _w = width as usize;
    let _h = height as usize;

    // ── Step 1: local background subtraction ──
    // If local_bg_block_size is set, estimate and subtract a spatially varying
    // background model. This is critical for images with nebulosity, Milky Way
    // emission, vignetting, or other large-scale intensity gradients.
    let gray: Vec<f32>;
    let local_bg: Option<Vec<f32>>;
    if let Some(block_size) = config.local_bg_block_size {
        let bg = estimate_local_background(gray_input, width, height, block_size);
        gray = gray_input
            .iter()
            .zip(bg.iter())
            .map(|(&v, &b)| (v - b).max(0.0))
            .collect();
        local_bg = Some(bg);
    } else {
        gray = gray_input.to_vec();
        local_bg = None;
    }

    // ── Step 2: estimate residual background noise ──
    let (bg_mean, bg_sigma) = estimate_background(&gray, width, height, config);
    let threshold = bg_mean + config.sigma_threshold * bg_sigma;

    // ── Step 3: threshold and label blobs ──
    let mask: Vec<bool> = gray.iter().map(|&v| v > threshold).collect();
    let labels = label_connected_components(&mask, width, height, config.use_8_connectivity);
    let num_labels = *labels.iter().max().unwrap_or(&0) as usize;

    // ── Step 4: compute centroids ──
    // Use the local-background-subtracted image for centroid weighting so that
    // the intensity weights reflect only the stellar signal, not the gradient.
    let bg_for_centroids = if local_bg.is_some() {
        // Already subtracted — use 0 as the level
        0.0
    } else {
        bg_mean
    };
    let raw_centroids = compute_blob_centroids(
        &gray,
        &labels,
        num_labels,
        width,
        height,
        bg_for_centroids,
        config,
    );
    let num_blobs_raw = raw_centroids.len();

    // ── Step 5: convert to angular coordinates ──
    let fov_h_rad = (config.fov_horizontal_deg as f64).to_radians() as f32;
    let pixel_scale = fov_h_rad / width as f32;
    let cx = width as f32 / 2.0;
    let cy = height as f32 / 2.0;

    let mut centroids: Vec<Centroid> = raw_centroids
        .into_iter()
        .map(|rc| Centroid {
            x: (rc.x_px - cx) * pixel_scale,
            y: (rc.y_px - cy) * pixel_scale,
            mass: Some(rc.mass),
            cov: None,
        })
        .collect();

    // Sort by brightness (descending)
    centroids.sort_by(|a, b| {
        b.mass
            .unwrap_or(0.0)
            .partial_cmp(&a.mass.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(max) = config.max_centroids {
        centroids.truncate(max);
    }

    Ok(CentroidExtractionResult {
        centroids,
        image_width: width,
        image_height: height,
        background_mean: bg_mean,
        background_sigma: bg_sigma,
        threshold,
        num_blobs_raw,
    })
}

/// Estimate a spatially varying background by computing block medians and
/// interpolating between block centers.
///
/// The image is divided into `block_size × block_size` tiles. For each tile,
/// the median pixel value is computed (ignoring zeros). A smooth background
/// surface is then reconstructed via bilinear interpolation between tile
/// centers.
///
/// This effectively removes large-scale structure (nebulosity, Milky Way
/// emission, vignetting) while preserving point sources (stars).
fn estimate_local_background(pixels: &[f32], width: u32, height: u32, block_size: u32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let bs = block_size as usize;

    // Number of blocks in each dimension
    let nx = (w + bs - 1) / bs;
    let ny = (h + bs - 1) / bs;

    // Compute median for each block
    let mut block_medians = vec![0.0f32; nx * ny];
    for by in 0..ny {
        for bx in 0..nx {
            let x0 = bx * bs;
            let y0 = by * bs;
            let x1 = (x0 + bs).min(w);
            let y1 = (y0 + bs).min(h);

            let mut vals: Vec<f32> = Vec::with_capacity(bs * bs);
            for y in y0..y1 {
                for x in x0..x1 {
                    let v = pixels[y * w + x];
                    if v > 0.0 && v.is_finite() {
                        vals.push(v);
                    }
                }
            }

            if vals.is_empty() {
                block_medians[by * nx + bx] = 0.0;
            } else {
                vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                block_medians[by * nx + bx] = vals[vals.len() / 2];
            }
        }
    }

    // Bilinearly interpolate between block centers to produce a smooth
    // background estimate at every pixel.
    let mut background = vec![0.0f32; w * h];
    let half_bs = bs as f32 / 2.0;

    for y in 0..h {
        for x in 0..w {
            // Position in block-center coordinates
            let bx_f = (x as f32 - half_bs) / bs as f32;
            let by_f = (y as f32 - half_bs) / bs as f32;

            let bx0 = (bx_f.floor() as isize).max(0).min(nx as isize - 1) as usize;
            let by0 = (by_f.floor() as isize).max(0).min(ny as isize - 1) as usize;
            let bx1 = (bx0 + 1).min(nx - 1);
            let by1 = (by0 + 1).min(ny - 1);

            let fx = (bx_f - bx0 as f32).clamp(0.0, 1.0);
            let fy = (by_f - by0 as f32).clamp(0.0, 1.0);

            let m00 = block_medians[by0 * nx + bx0];
            let m10 = block_medians[by0 * nx + bx1];
            let m01 = block_medians[by1 * nx + bx0];
            let m11 = block_medians[by1 * nx + bx1];

            background[y * w + x] = m00 * (1.0 - fx) * (1.0 - fy)
                + m10 * fx * (1.0 - fy)
                + m01 * (1.0 - fx) * fy
                + m11 * fx * fy;
        }
    }

    background
}

/// Convert a DynamicImage to a Vec<f32> of grayscale values.
fn to_grayscale_f32(img: &image::DynamicImage) -> Vec<f32> {
    use image::DynamicImage;
    match img {
        // 16-bit images: normalize to [0, 65535] range as f32
        DynamicImage::ImageLuma16(g) => g.as_raw().iter().map(|&v| v as f32).collect(),
        DynamicImage::ImageLumaA16(g) => g.pixels().map(|p| p.0[0] as f32).collect(),
        DynamicImage::ImageRgb16(rgb) => rgb
            .pixels()
            .map(|p| {
                let [r, g, b] = p.0;
                0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32
            })
            .collect(),
        DynamicImage::ImageRgba16(rgba) => rgba
            .pixels()
            .map(|p| {
                let [r, g, b, _] = p.0;
                0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32
            })
            .collect(),
        // For 32-bit float images
        DynamicImage::ImageRgb32F(rgb) => rgb
            .pixels()
            .map(|p| {
                let [r, g, b] = p.0;
                0.2126 * r + 0.7152 * g + 0.0722 * b
            })
            .collect(),
        DynamicImage::ImageRgba32F(rgba) => rgba
            .pixels()
            .map(|p| {
                let [r, g, b, _] = p.0;
                0.2126 * r + 0.7152 * g + 0.0722 * b
            })
            .collect(),
        // 8-bit and other formats: convert via luma8
        _ => {
            let gray = img.to_luma8();
            gray.as_raw().iter().map(|&v| v as f32).collect()
        }
    }
}

/// Estimate background level and noise using sigma-clipped statistics.
fn estimate_background(
    gray: &[f32],
    _width: u32,
    _height: u32,
    config: &CentroidExtractionConfig,
) -> (f32, f32) {
    let mut values: Vec<f32> = gray.to_vec();

    let mut mean = 0.0_f32;
    let mut sigma = 0.0_f32;

    for _ in 0..config.sigma_clip_iterations {
        if values.is_empty() {
            break;
        }

        // Compute mean
        let sum: f64 = values.iter().map(|&v| v as f64).sum();
        mean = (sum / values.len() as f64) as f32;

        // Compute sigma
        let var_sum: f64 = values.iter().map(|&v| ((v - mean) as f64).powi(2)).sum();
        sigma = (var_sum / values.len() as f64).sqrt() as f32;

        if sigma < 1e-10 {
            break;
        }

        // Clip outliers
        let lo = mean - config.sigma_clip_factor * sigma;
        let hi = mean + config.sigma_clip_factor * sigma;
        values.retain(|&v| v >= lo && v <= hi);
    }

    // Use median as a more robust background estimator
    if !values.is_empty() {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if values.len() % 2 == 0 {
            (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
        } else {
            values[values.len() / 2]
        };
        (median, sigma)
    } else {
        (mean, sigma)
    }
}

/// Label connected components in a binary mask using two-pass union-find.
fn label_connected_components(
    mask: &[bool],
    width: u32,
    height: u32,
    use_8_connectivity: bool,
) -> Vec<u32> {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;

    let mut labels = vec![0u32; n];
    let mut parent: Vec<u32> = Vec::new();
    let mut next_label = 1u32;

    // Find root with path compression
    fn find(parent: &mut Vec<u32>, mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }

    // Union two labels
    fn union(parent: &mut Vec<u32>, a: u32, b: u32) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            // Merge higher into lower to keep labels stable
            if ra < rb {
                parent[rb as usize] = ra;
            } else {
                parent[ra as usize] = rb;
            }
        }
    }

    // Reserve index 0 as background
    parent.push(0);

    // First pass: assign provisional labels
    for row in 0..h {
        for col in 0..w {
            let idx = row * w + col;
            if !mask[idx] {
                continue;
            }

            // Collect labeled neighbors
            let mut neighbor_labels = Vec::with_capacity(4);

            // Left
            if col > 0 && labels[idx - 1] > 0 {
                neighbor_labels.push(labels[idx - 1]);
            }
            // Above
            if row > 0 && labels[idx - w] > 0 {
                neighbor_labels.push(labels[idx - w]);
            }

            if use_8_connectivity {
                // Above-left
                if row > 0 && col > 0 && labels[idx - w - 1] > 0 {
                    neighbor_labels.push(labels[idx - w - 1]);
                }
                // Above-right
                if row > 0 && col + 1 < w && labels[idx - w + 1] > 0 {
                    neighbor_labels.push(labels[idx - w + 1]);
                }
            }

            if neighbor_labels.is_empty() {
                // New label
                parent.push(next_label);
                labels[idx] = next_label;
                next_label += 1;
            } else {
                // Use minimum label
                let min_label = *neighbor_labels.iter().min().unwrap();
                labels[idx] = min_label;
                // Union all neighbor labels
                for &nl in &neighbor_labels {
                    union(&mut parent, min_label, nl);
                }
            }
        }
    }

    // Second pass: flatten labels
    // Build a mapping from root -> sequential label
    let mut root_map = std::collections::HashMap::new();
    let mut seq = 1u32;

    for label in labels.iter_mut() {
        if *label > 0 {
            let root = find(&mut parent, *label);
            let mapped = root_map.entry(root).or_insert_with(|| {
                let s = seq;
                seq += 1;
                s
            });
            *label = *mapped;
        }
    }

    labels
}

/// Raw pixel-coordinate centroid with mass.
struct RawCentroid {
    x_px: f32,
    y_px: f32,
    mass: f32,
}

/// Compute intensity-weighted centroids for each labeled blob.
///
/// When `max_elongation` is set in config, blobs with elongation ratio
/// (major/minor axis) exceeding the threshold are rejected as non-stellar.
fn compute_blob_centroids(
    gray: &[f32],
    labels: &[u32],
    num_labels: usize,
    width: u32,
    _height: u32,
    bg_level: f32,
    config: &CentroidExtractionConfig,
) -> Vec<RawCentroid> {
    let w = width as usize;

    // Accumulators for each label: intensity-weighted moments
    struct BlobAccum {
        sum_x: f64,
        sum_y: f64,
        sum_intensity: f64,
        // Second-order moments for elongation / covariance
        sum_xx: f64,
        sum_yy: f64,
        sum_xy: f64,
        pixel_count: usize,
        // Bounding box for compactness check
        min_col: usize,
        max_col: usize,
        min_row: usize,
        max_row: usize,
    }

    let mut accums: Vec<BlobAccum> = (0..=num_labels)
        .map(|_| BlobAccum {
            sum_x: 0.0,
            sum_y: 0.0,
            sum_intensity: 0.0,
            sum_xx: 0.0,
            sum_yy: 0.0,
            sum_xy: 0.0,
            pixel_count: 0,
            min_col: usize::MAX,
            max_col: 0,
            min_row: usize::MAX,
            max_row: 0,
        })
        .collect();

    for (idx, (&label, &pixel_val)) in labels.iter().zip(gray.iter()).enumerate() {
        if label == 0 {
            continue;
        }
        let col = idx % w;
        let row = idx / w;
        let intensity = (pixel_val - bg_level).max(0.0) as f64;

        let acc = &mut accums[label as usize];
        let cf = col as f64;
        let rf = row as f64;
        acc.sum_x += cf * intensity;
        acc.sum_y += rf * intensity;
        acc.sum_xx += cf * cf * intensity;
        acc.sum_yy += rf * rf * intensity;
        acc.sum_xy += cf * rf * intensity;
        acc.sum_intensity += intensity;
        acc.pixel_count += 1;
        acc.min_col = acc.min_col.min(col);
        acc.max_col = acc.max_col.max(col);
        acc.min_row = acc.min_row.min(row);
        acc.max_row = acc.max_row.max(row);
    }

    accums
        .into_iter()
        .skip(1) // skip label 0 (background)
        .filter(|acc| {
            if acc.pixel_count < config.min_pixels
                || acc.pixel_count > config.max_pixels
                || acc.sum_intensity <= 0.0
            {
                return false;
            }

            // Elongation filter: compute the ratio of major to minor axis
            // from the intensity-weighted covariance matrix.
            if let Some(max_elong) = config.max_elongation {
                let xbar = acc.sum_x / acc.sum_intensity;
                let ybar = acc.sum_y / acc.sum_intensity;
                let cxx = acc.sum_xx / acc.sum_intensity - xbar * xbar;
                let cyy = acc.sum_yy / acc.sum_intensity - ybar * ybar;
                let cxy = acc.sum_xy / acc.sum_intensity - xbar * ybar;

                // Eigenvalues of [[cxx, cxy], [cxy, cyy]]
                let trace = cxx + cyy;
                let det = cxx * cyy - cxy * cxy;
                let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
                let lambda_max = (trace + disc) / 2.0;
                let lambda_min = (trace - disc).max(1e-12) / 2.0;
                let elongation = (lambda_max / lambda_min).sqrt() as f32;
                if elongation > max_elong {
                    return false;
                }
            }

            true
        })
        .map(|acc| RawCentroid {
            x_px: (acc.sum_x / acc.sum_intensity) as f32,
            y_px: (acc.sum_y / acc.sum_intensity) as f32,
            mass: acc.sum_intensity as f32,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_background_estimation() {
        // Uniform image: background should be ~100, sigma ~0
        let pixels = vec![100.0_f32; 100 * 100];
        let config = CentroidExtractionConfig::default();
        let (mean, sigma) = estimate_background(&pixels, 100, 100, &config);
        assert!((mean - 100.0).abs() < 1.0);
        assert!(sigma < 1.0);
    }

    #[test]
    fn test_connected_components_4conn() {
        // 5x5 image with two separate blobs
        let mask = vec![
            false, true, true, false, false, // row 0
            false, true, false, false, false, // row 1
            false, false, false, false, false, // row 2
            false, false, false, true, true, // row 3
            false, false, false, true, false, // row 4
        ];
        let labels = label_connected_components(&mask, 5, 5, false);
        // Blob 1: (0,1), (0,2), (1,1)
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[1], labels[6]);
        // Blob 2: (3,3), (3,4), (4,3)
        assert_eq!(labels[18], labels[19]);
        assert_eq!(labels[18], labels[23]);
        // Different blobs
        assert_ne!(labels[1], labels[18]);
    }

    #[test]
    fn test_extract_from_raw_single_star() {
        let width = 64u32;
        let height = 64u32;
        let mut pixels = vec![10.0_f32; (width * height) as usize];

        // Place a bright Gaussian-ish star near center
        let star_x = 32.0_f32;
        let star_y = 32.0_f32;
        let sigma_px = 2.0_f32;
        for row in 0..height {
            for col in 0..width {
                let dx = col as f32 - star_x;
                let dy = row as f32 - star_y;
                let r2 = dx * dx + dy * dy;
                pixels[(row * width + col) as usize] +=
                    1000.0 * (-r2 / (2.0 * sigma_px * sigma_px)).exp();
            }
        }

        let config = CentroidExtractionConfig {
            fov_horizontal_deg: 10.0,
            sigma_threshold: 3.0,
            min_pixels: 2,
            ..Default::default()
        };

        let result = extract_centroids_from_raw(&pixels, width, height, &config).unwrap();
        assert_eq!(result.centroids.len(), 1);

        // The centroid should be near the center of the image (0, 0 in angular coords)
        let c = &result.centroids[0];
        assert!(c.x.abs() < 0.001, "Expected x near 0, got {}", c.x);
        assert!(c.y.abs() < 0.001, "Expected y near 0, got {}", c.y);
        assert!(c.mass.unwrap() > 0.0);
    }

    #[test]
    fn test_extract_from_raw_multiple_stars() {
        let width = 128u32;
        let height = 128u32;
        let mut pixels = vec![10.0_f32; (width * height) as usize];

        // Place 3 stars at different positions
        let stars = [
            (30.0, 30.0, 800.0),
            (90.0, 50.0, 1200.0),
            (60.0, 100.0, 500.0),
        ];
        let sigma_px = 2.0_f32;

        for &(sx, sy, brightness) in &stars {
            for row in 0..height {
                for col in 0..width {
                    let dx = col as f32 - sx;
                    let dy = row as f32 - sy;
                    let r2 = dx * dx + dy * dy;
                    pixels[(row * width + col) as usize] +=
                        brightness * (-r2 / (2.0 * sigma_px * sigma_px)).exp();
                }
            }
        }

        let config = CentroidExtractionConfig {
            fov_horizontal_deg: 10.0,
            sigma_threshold: 3.0,
            min_pixels: 2,
            ..Default::default()
        };

        let result = extract_centroids_from_raw(&pixels, width, height, &config).unwrap();
        assert_eq!(
            result.centroids.len(),
            3,
            "Expected 3 stars, got {}",
            result.centroids.len()
        );

        // Centroids should be sorted by brightness (descending)
        assert!(result.centroids[0].mass.unwrap() >= result.centroids[1].mass.unwrap());
        assert!(result.centroids[1].mass.unwrap() >= result.centroids[2].mass.unwrap());
    }

    #[test]
    fn test_max_centroids_limit() {
        let width = 128u32;
        let height = 128u32;
        let mut pixels = vec![10.0_f32; (width * height) as usize];

        let stars = [
            (30.0, 30.0, 800.0),
            (90.0, 50.0, 1200.0),
            (60.0, 100.0, 500.0),
        ];
        let sigma_px = 2.0_f32;

        for &(sx, sy, brightness) in &stars {
            for row in 0..height {
                for col in 0..width {
                    let dx = col as f32 - sx;
                    let dy = row as f32 - sy;
                    let r2 = dx * dx + dy * dy;
                    pixels[(row * width + col) as usize] +=
                        brightness * (-r2 / (2.0 * sigma_px * sigma_px)).exp();
                }
            }
        }

        let config = CentroidExtractionConfig {
            fov_horizontal_deg: 10.0,
            sigma_threshold: 3.0,
            min_pixels: 2,
            max_centroids: Some(2),
            ..Default::default()
        };

        let result = extract_centroids_from_raw(&pixels, width, height, &config).unwrap();
        assert_eq!(result.centroids.len(), 2);
    }
}
