//! Extract star centroids from an astronomical image.
//!
//! This module provides functions to detect and locate stars in an image by:
//! 1. Converting the image to grayscale floating-point values
//! 2. Estimating and subtracting the background (sigma-clipped median)
//! 3. Thresholding to identify bright pixels
//! 4. Labeling connected components (blobs)
//! 5. Computing intensity-weighted centroids for each blob
//! 6. Converting pixel positions to centered coordinates (origin at image center)
//!
//! Requires the `image` feature to be enabled.
//!
//! # Example
//!
//! ```no_run
//! use tetra3::centroid_extraction::{CentroidExtractionConfig, extract_centroids};
//!
//! let config = CentroidExtractionConfig::default();
//!
//! let result = extract_centroids("my_star_image.png", &config).unwrap();
//! println!("Found {} stars", result.centroids.len());
//! ```

use crate::centroid::Centroid;
use anyhow::{Context, Result};
use image::GenericImageView;

/// Configuration for centroid extraction from an image.
#[derive(Debug, Clone)]
pub struct CentroidExtractionConfig {
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
    /// Extracted centroids in pixel coordinates, with (0, 0) at the image center.
    /// +X is right (increasing column), +Y is down (increasing row).
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
/// and centroid computation. Returns centroids in pixel coordinates centered at
/// the image center, suitable for use with [`SolverDatabase::solve_from_centroids`].
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
    // Use unclamped residuals for noise estimation so the lower half of the
    // distribution is preserved (clamping to 0 destroys it).
    let noise_input = if let Some(ref bg) = local_bg {
        gray_input
            .iter()
            .zip(bg.iter())
            .map(|(&v, &b)| v - b)
            .collect::<Vec<f32>>()
    } else {
        gray_input.to_vec()
    };
    let (bg_mean, bg_sigma) = estimate_background(&noise_input, width, height, config);
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

    // ── Step 5: convert to centered pixel coordinates ──
    // Origin at image center, +X right, +Y down
    let cx = width as f32 / 2.0;
    let cy = height as f32 / 2.0;

    let mut centroids: Vec<Centroid> = raw_centroids
        .into_iter()
        .map(|rc| Centroid {
            x: rc.x_px - cx,
            y: rc.y_px - cy,
            mass: Some(rc.mass),
            cov: Some(rc.cov),
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

/// Estimate background level and noise.
///
/// Uses the median as the background level and estimates noise from the
/// lower half of the pixel distribution (below the median). This is robust
/// to contamination from stars and nebulosity, which only bias upward.
///
/// The noise estimate uses sigma-clipping on the below-median pixels to
/// further reject any remaining outliers, then mirrors the lower-half RMS
/// to get the full Gaussian sigma.
fn estimate_background(
    gray: &[f32],
    _width: u32,
    _height: u32,
    config: &CentroidExtractionConfig,
) -> (f32, f32) {
    let mut values: Vec<f32> = gray.iter().copied().filter(|v| v.is_finite()).collect();
    if values.is_empty() {
        return (0.0, 0.0);
    }

    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();

    // Median as robust background level
    let median = if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    };

    // Estimate noise from pixels at or below the median (uncontaminated by stars).
    // These represent the "dark side" of the noise distribution.
    let mut low_half: Vec<f32> = values
        .iter()
        .copied()
        .filter(|&v| v <= median)
        .collect();

    // Sigma-clip the lower half to reject any remaining outliers
    let mut sigma = 0.0_f32;
    for _ in 0..config.sigma_clip_iterations {
        if low_half.is_empty() {
            break;
        }
        let sum: f64 = low_half.iter().map(|&v| v as f64).sum();
        let mean_low = (sum / low_half.len() as f64) as f32;
        let var_sum: f64 = low_half
            .iter()
            .map(|&v| ((v - mean_low) as f64).powi(2))
            .sum();
        sigma = (var_sum / low_half.len() as f64).sqrt() as f32;
        if sigma < 1e-10 {
            break;
        }
        let lo = mean_low - config.sigma_clip_factor * sigma;
        let hi = mean_low + config.sigma_clip_factor * sigma;
        let before = low_half.len();
        low_half.retain(|&v| v >= lo && v <= hi);
        if low_half.len() == before {
            break; // converged
        }
    }

    (median, sigma)
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

/// Raw pixel-coordinate centroid with mass and covariance.
struct RawCentroid {
    x_px: f32,
    y_px: f32,
    mass: f32,
    /// Intensity-weighted 2×2 covariance matrix [[cxx, cxy], [cxy, cyy]] in pixels².
    cov: crate::Matrix2,
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
    height: u32,
    bg_level: f32,
    config: &CentroidExtractionConfig,
) -> Vec<RawCentroid> {
    let w = width as usize;

    // Accumulators for each label: intensity-weighted moments.
    // Moments are computed relative to a reference pixel (ref_col, ref_row)
    // within each blob to avoid floating-point bias from large absolute
    // coordinates. The first pixel encountered sets the reference.
    struct BlobAccum {
        sum_x: f64,
        sum_y: f64,
        sum_intensity: f64,
        // Second-order moments for elongation / covariance
        sum_xx: f64,
        sum_yy: f64,
        sum_xy: f64,
        pixel_count: usize,
        // Reference pixel: moments are relative to this origin
        ref_col: usize,
        ref_row: usize,
        // Bounding box for compactness check
        min_col: usize,
        max_col: usize,
        min_row: usize,
        max_row: usize,
        // Peak pixel tracking
        peak_col: usize,
        peak_row: usize,
        peak_val: f32,
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
            ref_col: 0,
            ref_row: 0,
            min_col: usize::MAX,
            max_col: 0,
            min_row: usize::MAX,
            max_row: 0,
            peak_col: 0,
            peak_row: 0,
            peak_val: f32::NEG_INFINITY,
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

        // Set reference pixel on first encounter
        if acc.pixel_count == 0 {
            acc.ref_col = col;
            acc.ref_row = row;
        }

        // Accumulate moments relative to reference pixel (signed — blob pixels
        // can be in any direction from the first pixel encountered)
        let dx = col as f64 - acc.ref_col as f64;
        let dy = row as f64 - acc.ref_row as f64;
        acc.sum_x += dx * intensity;
        acc.sum_y += dy * intensity;
        acc.sum_xx += dx * dx * intensity;
        acc.sum_yy += dy * dy * intensity;
        acc.sum_xy += dx * dy * intensity;
        acc.sum_intensity += intensity;
        acc.pixel_count += 1;
        acc.min_col = acc.min_col.min(col);
        acc.max_col = acc.max_col.max(col);
        acc.min_row = acc.min_row.min(row);
        acc.max_row = acc.max_row.max(row);
        if pixel_val > acc.peak_val {
            acc.peak_val = pixel_val;
            acc.peak_col = col;
            acc.peak_row = row;
        }
    }

    let h = height as usize;

    accums
        .into_iter()
        .enumerate()
        .skip(1) // skip label 0 (background)
        .filter_map(|(blob_label, acc)| {
            if acc.pixel_count < config.min_pixels
                || acc.pixel_count > config.max_pixels
                || acc.sum_intensity <= 0.0
            {
                return None;
            }

            // --- Initial CoM for elongation filter (uses global bg) ---
            let dx_bar = acc.sum_x / acc.sum_intensity;
            let dy_bar = acc.sum_y / acc.sum_intensity;
            let cxx = acc.sum_xx / acc.sum_intensity - dx_bar * dx_bar;
            let cyy = acc.sum_yy / acc.sum_intensity - dy_bar * dy_bar;
            let cxy = acc.sum_xy / acc.sum_intensity - dx_bar * dy_bar;

            // Elongation filter
            if let Some(max_elong) = config.max_elongation {
                let trace = cxx + cyy;
                let det = cxx * cyy - cxy * cxy;
                let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
                let lambda_max = (trace + disc) / 2.0;
                let lambda_min = (trace - disc).max(1e-12) / 2.0;
                let elongation = (lambda_max / lambda_min).sqrt() as f32;
                if elongation > max_elong {
                    return None;
                }
            }

            // --- Per-blob local background from annulus ---
            // Expand bounding box by margin, collect non-blob pixels
            const ANNULUS_MARGIN: usize = 5;
            let r0 = acc.min_row.saturating_sub(ANNULUS_MARGIN);
            let r1 = (acc.max_row + ANNULUS_MARGIN + 1).min(h);
            let c0 = acc.min_col.saturating_sub(ANNULUS_MARGIN);
            let c1 = (acc.max_col + ANNULUS_MARGIN + 1).min(w);

            let mut annulus_vals: Vec<f32> = Vec::new();
            for r in r0..r1 {
                let row_off = r * w;
                for c in c0..c1 {
                    let idx = row_off + c;
                    if labels[idx] == 0 {
                        annulus_vals.push(gray[idx]);
                    }
                }
            }

            // Median of annulus (residual local background in bg-subtracted image)
            let local_bg = if annulus_vals.is_empty() {
                0.0_f64
            } else {
                annulus_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let mid = annulus_vals.len() / 2;
                if annulus_vals.len() % 2 == 0 {
                    (annulus_vals[mid - 1] + annulus_vals[mid]) as f64 / 2.0
                } else {
                    annulus_vals[mid] as f64
                }
            };

            // --- Re-accumulate moments with local background correction ---
            let ref_col = acc.ref_col;
            let ref_row = acc.ref_row;
            let mut sum_x = 0.0_f64;
            let mut sum_y = 0.0_f64;
            let mut sum_xx = 0.0_f64;
            let mut sum_yy = 0.0_f64;
            let mut sum_xy = 0.0_f64;
            let mut sum_i = 0.0_f64;

            for r in acc.min_row..=acc.max_row {
                let row_off = r * w;
                for c in acc.min_col..=acc.max_col {
                    let idx = row_off + c;
                    if labels[idx] as usize == blob_label {
                        let intensity = (gray[idx] as f64 - local_bg).max(0.0);
                        let dx = c as f64 - ref_col as f64;
                        let dy = r as f64 - ref_row as f64;
                        sum_x += dx * intensity;
                        sum_y += dy * intensity;
                        sum_xx += dx * dx * intensity;
                        sum_yy += dy * dy * intensity;
                        sum_xy += dx * dy * intensity;
                        sum_i += intensity;
                    }
                }
            }

            if sum_i <= 0.0 {
                return None;
            }

            let dx_bar = sum_x / sum_i;
            let dy_bar = sum_y / sum_i;
            let xbar = ref_col as f64 + dx_bar;
            let ybar = ref_row as f64 + dy_bar;
            let cxx = sum_xx / sum_i - dx_bar * dx_bar;
            let cyy = sum_yy / sum_i - dy_bar * dy_bar;
            let cxy = sum_xy / sum_i - dx_bar * dy_bar;

            // --- Quadratic peak refinement ---
            let mut final_x = xbar;
            let mut final_y = ybar;

            let pc = acc.peak_col;
            let pr = acc.peak_row;
            if acc.pixel_count >= 5 && pc >= 1 && pr >= 1 && pc + 1 < w && pr + 1 < h {
                // Build 3x3 grid of background-subtracted values around peak
                let effective_bg = local_bg;
                let v = |dy: isize, dx: isize| -> f64 {
                    let r = (pr as isize + dy) as usize;
                    let c = (pc as isize + dx) as usize;
                    gray[r * w + c] as f64 - effective_bg
                };

                let b = (v(0, 1) - v(0, -1)) / 2.0;
                let c_coeff = (v(1, 0) - v(-1, 0)) / 2.0;
                let d = (v(0, 1) + v(0, -1) - 2.0 * v(0, 0)) / 2.0;
                let f = (v(1, 0) + v(-1, 0) - 2.0 * v(0, 0)) / 2.0;
                let e = (v(1, 1) - v(1, -1) - v(-1, 1) + v(-1, -1)) / 4.0;

                let denom = 4.0 * d * f - e * e;
                if denom.abs() > 1e-10 {
                    let x_off = (e * c_coeff - 2.0 * f * b) / denom;
                    let y_off = (e * b - 2.0 * d * c_coeff) / denom;

                    // Only apply if offset is within half a pixel
                    if x_off.abs() <= 0.5 && y_off.abs() <= 0.5 {
                        let qx = pc as f64 + x_off;
                        let qy = pr as f64 + y_off;
                        // Only use quadratic when it agrees with CoM (within 0.5 px).
                        // For asymmetric or blended blobs, CoM is more reliable.
                        let dist_sq = (qx - xbar) * (qx - xbar) + (qy - ybar) * (qy - ybar);
                        if dist_sq < 0.25 {
                            final_x = qx;
                            final_y = qy;
                        }
                    }
                }
            }

            Some(RawCentroid {
                x_px: final_x as f32,
                y_px: final_y as f32,
                mass: sum_i as f32,
                cov: crate::Matrix2::new(cxx as f32, cxy as f32, cxy as f32, cyy as f32),
            })
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
            sigma_threshold: 3.0,
            min_pixels: 2,
            ..Default::default()
        };

        let result = extract_centroids_from_raw(&pixels, width, height, &config).unwrap();
        assert_eq!(result.centroids.len(), 1);

        // The centroid should be near the center of the image (0, 0 in pixel coords)
        let c = &result.centroids[0];
        assert!(c.x.abs() < 1.0, "Expected x near 0, got {}", c.x);
        assert!(c.y.abs() < 1.0, "Expected y near 0, got {}", c.y);
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
            sigma_threshold: 3.0,
            min_pixels: 2,
            max_centroids: Some(2),
            ..Default::default()
        };

        let result = extract_centroids_from_raw(&pixels, width, height, &config).unwrap();
        assert_eq!(result.centroids.len(), 2);
    }

    #[test]
    fn test_quadratic_refinement() {
        // Place a Gaussian star at a known sub-pixel offset on uniform background
        let width = 64u32;
        let height = 64u32;
        let bg = 100.0_f32;
        let true_x = 32.3_f32;
        let true_y = 32.7_f32;
        let sigma_px = 2.0_f32;
        let peak_brightness = 2000.0_f32;

        let mut pixels = vec![bg; (width * height) as usize];
        for row in 0..height {
            for col in 0..width {
                let dx = col as f32 - true_x;
                let dy = row as f32 - true_y;
                let r2 = dx * dx + dy * dy;
                pixels[(row * width + col) as usize] +=
                    peak_brightness * (-r2 / (2.0 * sigma_px * sigma_px)).exp();
            }
        }

        let config = CentroidExtractionConfig {
            sigma_threshold: 3.0,
            min_pixels: 3,
            ..Default::default()
        };

        let result = extract_centroids_from_raw(&pixels, width, height, &config).unwrap();
        assert_eq!(result.centroids.len(), 1, "Expected 1 star, got {}", result.centroids.len());

        // Centroid is in centered coords (origin at image center)
        let c = &result.centroids[0];
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        let abs_x = c.x + cx;
        let abs_y = c.y + cy;

        let err_x = (abs_x - true_x).abs();
        let err_y = (abs_y - true_y).abs();
        assert!(
            err_x < 0.15,
            "X error too large: centroid={abs_x:.4}, true={true_x}, err={err_x:.4}"
        );
        assert!(
            err_y < 0.15,
            "Y error too large: centroid={abs_y:.4}, true={true_y}, err={err_y:.4}"
        );
    }

    #[test]
    fn test_quadratic_refinement_with_gradient_background() {
        // Place a star on a gradient background to test local background correction
        let width = 128u32;
        let height = 128u32;
        let true_x = 64.4_f32;
        let true_y = 64.6_f32;
        let sigma_px = 2.0_f32;
        let peak_brightness = 2000.0_f32;

        let mut pixels = vec![0.0_f32; (width * height) as usize];
        // Add a gradient background: increases from left to right (50 to 150)
        for row in 0..height {
            for col in 0..width {
                let bg = 50.0 + 100.0 * (col as f32 / width as f32);
                pixels[(row * width + col) as usize] = bg;
            }
        }
        // Add Gaussian star
        for row in 0..height {
            for col in 0..width {
                let dx = col as f32 - true_x;
                let dy = row as f32 - true_y;
                let r2 = dx * dx + dy * dy;
                pixels[(row * width + col) as usize] +=
                    peak_brightness * (-r2 / (2.0 * sigma_px * sigma_px)).exp();
            }
        }

        let config = CentroidExtractionConfig {
            sigma_threshold: 5.0,
            min_pixels: 3,
            ..Default::default()
        };

        let result = extract_centroids_from_raw(&pixels, width, height, &config).unwrap();
        assert!(
            !result.centroids.is_empty(),
            "Should detect at least one star on gradient background"
        );

        // Find the centroid closest to our true position
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        let best = result
            .centroids
            .iter()
            .min_by(|a, b| {
                let da = (a.x + cx - true_x).powi(2) + (a.y + cy - true_y).powi(2);
                let db = (b.x + cx - true_x).powi(2) + (b.y + cy - true_y).powi(2);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();

        let abs_x = best.x + cx;
        let abs_y = best.y + cy;
        let err_x = (abs_x - true_x).abs();
        let err_y = (abs_y - true_y).abs();
        assert!(
            err_x < 0.3,
            "X error too large on gradient bg: centroid={abs_x:.4}, true={true_x}, err={err_x:.4}"
        );
        assert!(
            err_y < 0.3,
            "Y error too large on gradient bg: centroid={abs_y:.4}, true={true_y}, err={err_y:.4}"
        );
    }
}
