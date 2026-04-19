//! Extract star centroids from an astronomical image.
//!
//! This module provides functions to detect and locate stars in an image by:
//! 1. Converting the image to grayscale floating-point values
//! 2. Estimating and subtracting the background (sigma-clipped median)
//! 3. Thresholding to identify bright pixels
//! 4. Labeling connected components (blobs)
//! 5. Computing intensity-weighted centroids for each blob, with:
//!    - Per-blob local background from an annulus of non-blob pixels
//!    - Quadratic peak refinement (2D fit to 3×3 around peak pixel)
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

    /// Minimum per-blob signal-to-noise ratio to keep a detection.
    ///
    /// Computed as `sum_intensity / (sqrt(pixel_count) * sigma_at_peak)`,
    /// where `sigma_at_peak` is sampled from the local noise map (or the
    /// global background sigma when `local_bg_block_size` is `None`). This
    /// is the total integrated significance of the detection, distinct
    /// from `sigma_threshold` which is a per-pixel cut used only to form
    /// the detection mask.
    ///
    /// Pairing a permissive `sigma_threshold` (e.g. 2.5–3.0) with an
    /// `snr_min` around 5 is the recommended configuration: the per-pixel
    /// cut samples more of each star's PSF wings, and the per-blob SNR
    /// cut does the false-positive rejection.
    ///
    /// Default: Some(5.0). Set to `None` to disable.
    pub snr_min: Option<f32>,
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
            snr_min: Some(5.0),
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
    // ── Step 1: estimate background and noise ──
    // When `local_bg_block_size` is set, always build a local background
    // surface. The local *noise* surface is only used for per-pixel
    // thresholding when `snr_min` is also set — that gates the adaptive
    // per-pixel + per-blob-SNR pipeline as a coupled opt-in and leaves
    // single-scalar-sigma behavior untouched otherwise.
    let (bg_surface, sigma_surface, bg_mean, bg_sigma) = match config.local_bg_block_size {
        Some(block_size) => {
            let want_sigma = config.snr_min.is_some();
            let (bg, sig) =
                estimate_local_bg_sigma(gray_input, width, height, block_size, config, want_sigma);
            match sig {
                Some(sig) => {
                    let bg_mean = median_finite(&bg);
                    let bg_sigma = median_finite(&sig);
                    (Some(bg), Some(sig), bg_mean, bg_sigma)
                }
                None => {
                    // Legacy path: local bg, global-sigma from residual.
                    let noise_input: Vec<f32> =
                        gray_input.iter().zip(bg.iter()).map(|(&v, &b)| v - b).collect();
                    let (m, s) = estimate_background(&noise_input, width, height, config);
                    (Some(bg), None, m, s)
                }
            }
        }
        None => {
            let (m, s) = estimate_background(gray_input, width, height, config);
            (None, None, m, s)
        }
    };
    let threshold = bg_mean + config.sigma_threshold * bg_sigma;

    // ── Step 2: subtract background to produce the residual image ──
    // Always subtract, so downstream thresholding and centroiding treat the
    // bg as zero. Clamp negatives to zero for intensity-weighted moments.
    let gray: Vec<f32> = if let Some(bg) = &bg_surface {
        gray_input
            .iter()
            .zip(bg.iter())
            .map(|(&v, &b)| (v - b).max(0.0))
            .collect()
    } else {
        gray_input.iter().map(|&v| (v - bg_mean).max(0.0)).collect()
    };

    // ── Step 3: per-pixel threshold ──
    // In local mode, the threshold adapts to the noise surface; in global
    // mode every pixel shares the same sigma. Mathematically equivalent
    // to the previous `gray > bg_mean + N*bg_sigma` check.
    let mask: Vec<bool> = if let Some(sig) = &sigma_surface {
        gray.iter()
            .zip(sig.iter())
            .map(|(&v, &s)| v > config.sigma_threshold * s)
            .collect()
    } else {
        let cut = config.sigma_threshold * bg_sigma;
        gray.iter().map(|&v| v > cut).collect()
    };
    let labels = label_connected_components(&mask, width, height, config.use_8_connectivity);
    let num_labels = *labels.iter().max().unwrap_or(&0) as usize;

    // ── Step 4: compute centroids ──
    // The residual image already has bg subtracted, so bg_for_centroids = 0.
    // The per-blob SNR cut (when configured) samples the noise map at each
    // blob's peak pixel and short-circuits before the expensive annulus
    // and quadratic-refinement work.
    let raw_centroids = compute_blob_centroids(
        &gray,
        &labels,
        num_labels,
        width,
        height,
        0.0,
        bg_sigma,
        sigma_surface.as_deref(),
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

/// Estimate a spatially varying background surface (and optionally a noise
/// surface) by computing per-tile statistics and bilinearly interpolating
/// between tile centers.
///
/// The image is divided into `block_size × block_size` tiles. For each tile:
///   - The median pixel value estimates the local background level.
///   - When `want_sigma`, the lower-half RMS (sigma-clipped) of pixels at or
///     below the median estimates the local noise sigma. The lower half is
///     used because it is uncontaminated by stars, which only bias the
///     distribution upward.
///
/// The surface is reconstructed via bilinear interpolation between tile
/// centers, yielding a smooth per-pixel estimate. Scratch buffers are
/// allocated once up front and reused across tiles.
///
/// This removes large-scale structure (nebulosity, vignetting) and, when
/// the sigma surface is built, lets the detection threshold adapt to
/// regions of varying noise.
fn estimate_local_bg_sigma(
    pixels: &[f32],
    width: u32,
    height: u32,
    block_size: u32,
    config: &CentroidExtractionConfig,
    want_sigma: bool,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let w = width as usize;
    let h = height as usize;
    let bs = block_size as usize;

    let nx = w.div_ceil(bs);
    let ny = h.div_ceil(bs);

    let mut block_medians = vec![0.0f32; nx * ny];
    let mut block_sigmas = if want_sigma {
        vec![0.0f32; nx * ny]
    } else {
        Vec::new()
    };

    // Reusable scratch buffers: one allocation per call instead of one per tile.
    let tile_cap = bs * bs;
    let mut vals: Vec<f32> = Vec::with_capacity(tile_cap);
    let mut low_half: Vec<f32> = if want_sigma {
        Vec::with_capacity(tile_cap)
    } else {
        Vec::new()
    };

    for by in 0..ny {
        for bx in 0..nx {
            let x0 = bx * bs;
            let y0 = by * bs;
            let x1 = (x0 + bs).min(w);
            let y1 = (y0 + bs).min(h);

            vals.clear();
            for y in y0..y1 {
                for x in x0..x1 {
                    let v = pixels[y * w + x];
                    if v > 0.0 && v.is_finite() {
                        vals.push(v);
                    }
                }
            }

            let tile_idx = by * nx + bx;
            if vals.is_empty() {
                // medians and (optional) sigmas already zero-initialized
                continue;
            }

            // Quickselect for the median — O(N) vs the old O(N log N) sort.
            let k = vals.len() / 2;
            vals.select_nth_unstable_by(k, f32::total_cmp);
            let median = vals[k];
            block_medians[tile_idx] = median;

            if !want_sigma {
                continue;
            }

            // Lower-half sigma-clipped RMS, mirrored to yield Gaussian σ.
            // After select_nth_unstable, `vals` is partially ordered — the
            // `<= median` filter still gives the correct half.
            low_half.clear();
            low_half.extend(vals.iter().copied().filter(|&v| v <= median));
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
                    break;
                }
            }
            block_sigmas[tile_idx] = sigma;
        }
    }

    let bg_surface = bilinear_upscale(&block_medians, nx, ny, w, h, bs);
    let sigma_surface = if want_sigma {
        Some(bilinear_upscale(&block_sigmas, nx, ny, w, h, bs))
    } else {
        None
    };
    (bg_surface, sigma_surface)
}

/// Median of the finite values in a slice. Returns 0.0 for empty input.
/// Uses quickselect (O(N) expected) rather than a full sort.
fn median_finite(values: &[f32]) -> f32 {
    let mut v: Vec<f32> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return 0.0;
    }
    let k = v.len() / 2;
    v.select_nth_unstable_by(k, f32::total_cmp);
    v[k]
}

/// Bilinearly upsample a tile-resolution value grid to full pixel resolution.
///
/// `tile_values` is `nx × ny` row-major. Output is `w × h` row-major. Tile
/// centers are located at `((bx + 0.5) * bs, (by + 0.5) * bs)`.
fn bilinear_upscale(
    tile_values: &[f32],
    nx: usize,
    ny: usize,
    w: usize,
    h: usize,
    bs: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    let half_bs = bs as f32 / 2.0;

    for y in 0..h {
        for x in 0..w {
            let bx_f = (x as f32 - half_bs) / bs as f32;
            let by_f = (y as f32 - half_bs) / bs as f32;

            let bx0 = (bx_f.floor() as isize).max(0).min(nx as isize - 1) as usize;
            let by0 = (by_f.floor() as isize).max(0).min(ny as isize - 1) as usize;
            let bx1 = (bx0 + 1).min(nx - 1);
            let by1 = (by0 + 1).min(ny - 1);

            let fx = (bx_f - bx0 as f32).clamp(0.0, 1.0);
            let fy = (by_f - by0 as f32).clamp(0.0, 1.0);

            let m00 = tile_values[by0 * nx + bx0];
            let m10 = tile_values[by0 * nx + bx1];
            let m01 = tile_values[by1 * nx + bx0];
            let m11 = tile_values[by1 * nx + bx1];

            out[y * w + x] = m00 * (1.0 - fx) * (1.0 - fy)
                + m10 * fx * (1.0 - fy)
                + m01 * (1.0 - fx) * fy
                + m11 * fx * fy;
        }
    }

    out
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

    // Quickselect median — O(N) expected, vs O(N log N) for a full sort. The
    // lower-half noise estimate below doesn't need sortedness; only the
    // value at index n/2 matters.
    let n = values.len();
    let k = n / 2;
    values.select_nth_unstable_by(k, f32::total_cmp);
    let median = values[k];

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

    // Second pass: flatten labels via a Vec<u32> remap (indexed by root
    // label). Faster than a HashMap — no hashing, single cache-friendly
    // read/write per pixel. Slot 0 is sentinel for "unmapped".
    let mut root_to_seq = vec![0u32; parent.len()];
    let mut seq = 1u32;

    for label in labels.iter_mut() {
        if *label > 0 {
            let root = find(&mut parent, *label);
            let slot = &mut root_to_seq[root as usize];
            if *slot == 0 {
                *slot = seq;
                seq += 1;
            }
            *label = *slot;
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
/// For each blob that passes size, elongation, and per-blob SNR filters:
/// 1. A local background is estimated from the median of non-blob pixels in a
///    5-pixel annulus around the blob's bounding box.
/// 2. Intensity-weighted moments are re-accumulated with the local background
///    subtracted, yielding a center-of-mass (CoM) position.
/// 3. A 2D quadratic is fit to the 3×3 neighborhood around the peak pixel to
///    interpolate the sub-pixel intensity maximum. The quadratic position is
///    used only when it agrees with the CoM (within 0.5 px); otherwise the CoM
///    is kept as-is.
///
/// When `max_elongation` is set in config, blobs with elongation ratio
/// (major/minor axis) exceeding the threshold are rejected as non-stellar.
/// When `snr_min` is set, blobs whose integrated SNR (sum_intensity /
/// (sqrt(pixel_count) * sigma_at_peak)) falls below the threshold are
/// rejected *before* the annulus and quadratic-fit work — the common case
/// on noisy images where most raw blobs are spurious.
fn compute_blob_centroids(
    gray: &[f32],
    labels: &[u32],
    num_labels: usize,
    width: u32,
    height: u32,
    bg_level: f32,
    global_sigma: f32,
    sigma_surface: Option<&[f32]>,
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

    // Reusable scratch for per-blob annulus sampling. One allocation,
    // `clear()`-ed per blob, sized to a reasonable upper bound so we
    // rarely reallocate. Dense fields can see tens of thousands of
    // blobs; allocating per-blob is a measurable cost otherwise.
    let mut annulus_vals: Vec<f32> = Vec::with_capacity(256);

    let mut result: Vec<RawCentroid> = Vec::with_capacity(accums.len().saturating_sub(1));

    for (blob_label, acc) in accums.into_iter().enumerate().skip(1) {
        if acc.pixel_count < config.min_pixels
            || acc.pixel_count > config.max_pixels
            || acc.sum_intensity <= 0.0
        {
            continue;
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
                continue;
            }
        }

        // Per-blob SNR filter — runs before annulus sampling and
        // quadratic refinement so spurious blobs are rejected cheaply.
        // sigma_at_peak is sampled from the local noise map when
        // available, else falls back to the global background sigma.
        if let Some(snr_min) = config.snr_min {
            let peak_idx = acc.peak_row * w + acc.peak_col;
            let sigma_at_peak = sigma_surface
                .map(|s| s[peak_idx])
                .unwrap_or(global_sigma);
            if sigma_at_peak > 0.0 {
                let noise = (acc.pixel_count as f64).sqrt() * sigma_at_peak as f64;
                let snr = (acc.sum_intensity / noise) as f32;
                if snr < snr_min {
                    continue;
                }
            }
        }

        // --- Per-blob local background from annulus ---
        // Expand bounding box by margin, collect non-blob pixels.
        const ANNULUS_MARGIN: usize = 5;
        let r0 = acc.min_row.saturating_sub(ANNULUS_MARGIN);
        let r1 = (acc.max_row + ANNULUS_MARGIN + 1).min(h);
        let c0 = acc.min_col.saturating_sub(ANNULUS_MARGIN);
        let c1 = (acc.max_col + ANNULUS_MARGIN + 1).min(w);

        annulus_vals.clear();
        for r in r0..r1 {
            let row_off = r * w;
            for c in c0..c1 {
                let idx = row_off + c;
                if labels[idx] == 0 {
                    annulus_vals.push(gray[idx]);
                }
            }
        }

        // Median of annulus (residual local background in bg-subtracted image).
        // Quickselect — O(N) expected — matches what the global and per-tile
        // medians do. Uses the single n/2 element rather than averaging two
        // middle elements; the difference is sub-DN for typical annulus sizes
        // and matters less than the overall pedestal accuracy.
        let local_bg = if annulus_vals.is_empty() {
            0.0_f64
        } else {
            let mid = annulus_vals.len() / 2;
            annulus_vals.select_nth_unstable_by(mid, f32::total_cmp);
            annulus_vals[mid] as f64
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
            continue;
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

        result.push(RawCentroid {
            x_px: final_x as f32,
            y_px: final_y as f32,
            mass: sum_i as f32,
            cov: crate::Matrix2::new([[cxx as f32, cxy as f32], [cxy as f32, cyy as f32]]),
        });
    }

    result
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
