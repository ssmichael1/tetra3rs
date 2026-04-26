//! Extract star centroids from an astronomical image.
//!
//! This module provides functions to detect and locate stars in pixel data by:
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
//! Two entry points are provided:
//! - [`extract_centroids_from_image`] for an already-decoded
//!   [`image::DynamicImage`]. The caller is responsible for decoding the
//!   file (using whichever `image` feature flags suit their needs).
//! - [`extract_centroids_from_raw`] for raw grayscale `f32` pixel data —
//!   useful for FITS, camera SDK output, or any other non-standard source.
//!
//! # Example
//!
//! ```no_run
//! use tetra3::centroid_extraction::{CentroidExtractionConfig, extract_centroids_from_image};
//!
//! let img = image::open("my_star_image.png").unwrap();
//! let config = CentroidExtractionConfig::default();
//! let result = extract_centroids_from_image(&img, &config).unwrap();
//! println!("Found {} stars", result.centroids.len());
//! ```

use crate::centroid::Centroid;
use crate::error::{Error, Result};
use image::GenericImageView;
use numeris::imageproc::{
    connected_components_with_label_buffer, Component, Connectivity,
};
use numeris::DynMatrix;

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

/// Extract star centroids from an already-decoded [`image::DynamicImage`].
///
/// Performs background subtraction, blob detection, and centroid computation
/// on an in-memory image. Centroids are returned in pixel coordinates with the
/// origin at the image center, suitable for use with
/// [`SolverDatabase::solve_from_centroids`].
///
/// To load from a file, decode it with `image::open(path)?` (which requires
/// the appropriate `image` crate format features in your own `Cargo.toml`)
/// and pass the resulting `DynamicImage` here.
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
    let expected = (width as usize) * (height as usize);
    if pixels.len() != expected {
        return Err(Error::InvalidInput(format!(
            "Pixel data length ({}) does not match width*height ({}x{}={})",
            pixels.len(),
            width,
            height,
            expected,
        )));
    }
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
    // Build a u8 mask in a row-major DynMatrix (numeris's CCL takes a MatrixRef).
    // Foreground = 1, background = 0.
    let w = width as usize;
    let h = height as usize;
    let mut mask = DynMatrix::<u8>::zeros(h, w);
    for r in 0..h {
        for c in 0..w {
            if gray[r * w + c] > threshold {
                mask[(r, c)] = 1;
            }
        }
    }
    let connectivity = if config.use_8_connectivity {
        Connectivity::Eight
    } else {
        Connectivity::Four
    };
    let (labels, components) =
        connected_components_with_label_buffer(&mask, connectivity, 0u8);

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
        &components,
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
/// Consumes [`numeris::imageproc::Component`]s for area / bounding box, plus
/// the row-major labels buffer for per-pixel masking. For each blob that
/// passes size and elongation filters:
/// 1. A local background is estimated from the median of non-blob pixels in a
///    5-pixel annulus around the blob's bounding box.
/// 2. Intensity-weighted moments are accumulated with the local background
///    subtracted, yielding a center-of-mass (CoM) position. Peak pixel is
///    tracked in the same pass.
/// 3. A 2D quadratic is fit to the 3×3 neighborhood around the peak pixel to
///    interpolate the sub-pixel intensity maximum. The quadratic position is
///    used only when it agrees with the CoM (within 0.5 px); otherwise the CoM
///    is kept as-is.
///
/// When `max_elongation` is set in config, blobs with elongation ratio
/// (major/minor axis) exceeding the threshold are rejected as non-stellar.
/// The elongation test uses **intensity-weighted** second moments (with the
/// global background subtracted), matching the original behavior — geometric
/// moments admit a slightly different set of marginal blobs (saturated stars
/// with large halos, etc.), which destabilizes downstream calibration on
/// dense fields like TESS.
fn compute_blob_centroids(
    gray: &[f32],
    labels: &[u32],
    components: &[Component],
    width: u32,
    height: u32,
    bg_level: f32,
    config: &CentroidExtractionConfig,
) -> Vec<RawCentroid> {
    let w = width as usize;
    let h = height as usize;
    let bg_level_f64 = bg_level as f64;

    components
        .iter()
        .enumerate()
        .filter_map(|(idx, comp)| {
            let blob_label = (idx + 1) as u32;
            let pixel_count = comp.area as usize;
            if pixel_count < config.min_pixels || pixel_count > config.max_pixels {
                return None;
            }

            // Bounding box (numeris uses (row, col) with inclusive max).
            let min_row = comp.bbox_min.0 as usize;
            let max_row = comp.bbox_max.0 as usize;
            let min_col = comp.bbox_min.1 as usize;
            let max_col = comp.bbox_max.1 as usize;

            // Reference pixel = bbox top-left, to keep moments numerically stable.
            let ref_col = min_col;
            let ref_row = min_row;

            // --- Pass 1: intensity-weighted moments with global bg + peak ---
            let mut sum_x = 0.0_f64;
            let mut sum_y = 0.0_f64;
            let mut sum_xx = 0.0_f64;
            let mut sum_yy = 0.0_f64;
            let mut sum_xy = 0.0_f64;
            let mut sum_i = 0.0_f64;
            let mut peak_val = f32::NEG_INFINITY;
            let mut peak_col: usize = ref_col;
            let mut peak_row: usize = ref_row;

            for r in min_row..=max_row {
                let row_off = r * w;
                for c in min_col..=max_col {
                    let i = row_off + c;
                    if labels[i] != blob_label {
                        continue;
                    }
                    let raw = gray[i];
                    if raw > peak_val {
                        peak_val = raw;
                        peak_col = c;
                        peak_row = r;
                    }
                    let intensity = (raw as f64 - bg_level_f64).max(0.0);
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

            if sum_i <= 0.0 {
                return None;
            }

            // Elongation filter on intensity-weighted moments
            if let Some(max_elong) = config.max_elongation {
                let dx_bar = sum_x / sum_i;
                let dy_bar = sum_y / sum_i;
                let cxx = sum_xx / sum_i - dx_bar * dx_bar;
                let cyy = sum_yy / sum_i - dy_bar * dy_bar;
                let cxy = sum_xy / sum_i - dx_bar * dy_bar;
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
            let r0 = min_row.saturating_sub(ANNULUS_MARGIN);
            let r1 = (max_row + ANNULUS_MARGIN + 1).min(h);
            let c0 = min_col.saturating_sub(ANNULUS_MARGIN);
            let c1 = (max_col + ANNULUS_MARGIN + 1).min(w);

            let mut annulus_vals: Vec<f32> = Vec::new();
            for r in r0..r1 {
                let row_off = r * w;
                for c in c0..c1 {
                    let i = row_off + c;
                    if labels[i] == 0 {
                        annulus_vals.push(gray[i]);
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

            // --- Pass 2: re-accumulate intensity-weighted moments with local bg ---
            sum_x = 0.0;
            sum_y = 0.0;
            sum_xx = 0.0;
            sum_yy = 0.0;
            sum_xy = 0.0;
            sum_i = 0.0;

            for r in min_row..=max_row {
                let row_off = r * w;
                for c in min_col..=max_col {
                    let i = row_off + c;
                    if labels[i] != blob_label {
                        continue;
                    }
                    let intensity = (gray[i] as f64 - local_bg).max(0.0);
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

            let pc = peak_col;
            let pr = peak_row;
            if pixel_count >= 5 && pc >= 1 && pr >= 1 && pc + 1 < w && pr + 1 < h {
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
                cov: crate::Matrix2::new([[cxx as f32, cxy as f32], [cxy as f32, cyy as f32]]),
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
