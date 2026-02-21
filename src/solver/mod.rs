//! Star plate solver based on the tetra3/cedar-solve algorithm.
//!
//! This module implements a "lost-in-space" plate solver that identifies star patterns
//! in images and determines the camera pointing direction. The algorithm:
//!
//! 1. **Database generation**: Precomputes 4-star geometric patterns from a star catalog,
//!    hashing their edge ratios into a lookup table.
//! 2. **Solving**: Given image centroids and an approximate FOV, tries 4-centroid
//!    combinations, looks up matching catalog patterns, estimates rotation via SVD,
//!    and verifies by counting star matches.
//!
//! Reference: cedar-solve / tetra3 by Gustav Pettersson (ESA) and Steven Rosenthal.

pub mod combinations;
pub mod database;
pub mod pattern;
pub mod solve;
pub mod wcs_refine;

use rkyv::{Archive, Deserialize, Serialize};

use crate::camera_model::CameraModel;
use crate::distortion::Distortion;
use crate::{Quaternion, StarCatalog};

// ── Status codes (matching tetra3) ──────────────────────────────────────────

/// Outcome of a plate-solve attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStatus {
    /// A valid match was found.
    MatchFound,
    /// All pattern combinations were exhausted without a match.
    NoMatch,
    /// The solve timeout was reached before a match was found.
    Timeout,
    /// Too few centroids were provided to form a pattern.
    TooFew,
}

// ── Database properties ─────────────────────────────────────────────────────

/// Metadata describing how a solver database was built.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct DatabaseProperties {
    /// Number of quantization bins per edge-ratio dimension.
    /// Computed as round(0.25 / pattern_max_error).
    pub pattern_bins: u32,
    /// Maximum tolerated error in edge ratios for a match.
    pub pattern_max_error: f32,
    /// Maximum FOV the database was built for (radians).
    pub max_fov_rad: f32,
    /// Minimum FOV the database was built for (radians).
    pub min_fov_rad: f32,
    /// Faintest star magnitude included.
    pub star_max_magnitude: f32,
    /// Total number of distinct patterns stored.
    pub num_patterns: u32,
    /// Catalog coordinate epoch (e.g. 2000 for J2000/ICRS).
    pub epoch_equinox: u16,
    /// Year to which proper motions were propagated.
    pub epoch_proper_motion_year: f32,
    /// Max catalog stars per FOV used for verification.
    pub verification_stars_per_fov: u32,
    /// Lattice field oversampling factor during generation.
    pub lattice_field_oversampling: u32,
    /// Number of patterns generated per lattice field.
    pub patterns_per_lattice_field: u32,
}

// ── The solver database ─────────────────────────────────────────────────────

/// Complete solver database, serializable with rkyv.
///
/// Contains a spatial star catalog (brightness-sorted), precomputed unit vectors,
/// and a pattern hash table for fast geometric matching.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct SolverDatabase {
    /// Spatial star catalog. Stars are sorted brightest-first so that
    /// index order equals brightness order.
    pub star_catalog: StarCatalog,

    /// Precomputed ICRS unit vectors for each star, matching star_catalog.stars ordering.
    /// Stored as [x, y, z] where x=cos(ra)cos(dec), y=sin(ra)cos(dec), z=sin(dec).
    pub star_vectors: Vec<[f32; 3]>,

    /// Original catalog IDs (e.g. HIP number) for each star.
    pub star_catalog_ids: Vec<u64>,

    /// Pattern hash table (open addressing, quadratic probing).
    /// Each slot holds [star_idx0, star_idx1, star_idx2, star_idx3].
    /// Empty slots are [0, 0, 0, 0]. Since patterns always have 4 *distinct*
    /// star indices, all-zero is an unambiguous empty marker.
    pub pattern_catalog: Vec<[u32; 4]>,

    /// Largest edge angle (radians) for each pattern slot.
    /// Used for fast FOV-consistency filtering during solve.
    pub pattern_largest_edge: Vec<f32>,

    /// Lower 16 bits of the pattern key hash for each slot.
    /// Used as a fast pre-filter before full edge-ratio comparison.
    pub pattern_key_hashes: Vec<u16>,

    /// Database generation parameters.
    pub props: DatabaseProperties,
}

// ── Configuration for database generation ───────────────────────────────────

/// Parameters controlling database generation.
pub struct GenerateDatabaseConfig {
    /// Maximum FOV in degrees.
    pub max_fov_deg: f32,
    /// Minimum FOV in degrees. If None, equals max_fov (single-scale database).
    pub min_fov_deg: Option<f32>,
    /// Faintest star magnitude to include. None = auto-compute from star density.
    pub star_max_magnitude: Option<f32>,
    /// Maximum edge-ratio error for pattern matching.
    /// Determines bin count: bins = round(0.25 / pattern_max_error).
    /// Default 0.001 → 250 bins.
    pub pattern_max_error: f32,
    /// Oversampling factor for lattice field distribution. Default 100.
    pub lattice_field_oversampling: u32,
    /// Patterns to generate per lattice field. Default 50.
    pub patterns_per_lattice_field: u32,
    /// Max catalog stars per FOV for verification. Default 150.
    pub verification_stars_per_fov: u32,
    /// Multiscale FOV step ratio. Default 1.5.
    pub multiscale_step: f32,
    /// Year for proper motion propagation. None = don't propagate.
    pub epoch_proper_motion_year: Option<f64>,
    /// HEALPix nside for the spatial star catalog index. Default 16.
    pub catalog_nside: u32,
}

impl Default for GenerateDatabaseConfig {
    fn default() -> Self {
        Self {
            max_fov_deg: 30.0,
            min_fov_deg: None,
            star_max_magnitude: None,
            pattern_max_error: 0.001,
            lattice_field_oversampling: 100,
            patterns_per_lattice_field: 50,
            verification_stars_per_fov: 150,
            multiscale_step: 1.5,
            epoch_proper_motion_year: Some(2025.0),
            catalog_nside: 16,
        }
    }
}

// ── Configuration for plate solving ─────────────────────────────────────────

/// Parameters controlling the plate-solve attempt.
pub struct SolveConfig {
    /// Estimated horizontal field of view in radians (along columns / image width).
    /// This is used together with `image_width` to compute the pixel scale.
    pub fov_estimate_rad: f32,
    /// Image width in pixels (number of columns).
    /// Together with `fov_estimate_rad`, defines the pixel scale:
    /// `pixel_scale = fov_estimate_rad / image_width`.
    pub image_width: u32,
    /// Image height in pixels (number of rows).
    pub image_height: u32,
    /// Maximum acceptable FOV error in radians. None = no FOV filtering.
    pub fov_max_error_rad: Option<f32>,
    /// Maximum match distance as a fraction of the FOV. Default 0.01.
    pub match_radius: f32,
    /// False-positive probability threshold. Default 1e-5.
    pub match_threshold: f64,
    /// Timeout in milliseconds. None = no timeout. Default 5000.
    pub solve_timeout_ms: Option<u64>,
    /// Maximum edge-ratio error for matching. None = use database value.
    pub match_max_error: Option<f32>,
    /// Number of iterative SVD refinement passes after the initial match.
    ///
    /// Each pass re-projects catalog stars using the refined rotation and
    /// re-matches centroids, potentially discovering additional inliers at
    /// the edges of the match radius.  Terminates early if no new inliers
    /// are found.
    ///
    /// - `1` = single refinement pass (original behavior)
    /// - `2` = one additional re-match after the first refinement (default)
    ///
    /// Default: 2.
    pub refine_iterations: u32,
    /// Camera intrinsics model (focal length, optical center, parity, distortion).
    ///
    /// Encapsulates the lens distortion model, optical center offset (CRPIX),
    /// and parity flip into a single struct. The solver uses this to preprocess
    /// centroids before pattern matching and WCS refinement.
    ///
    /// If not explicitly set, uses a simple pinhole model with no distortion,
    /// crpix=[0,0], and no parity flip.
    pub camera_model: CameraModel,
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            fov_estimate_rad: 0.0,
            image_width: 0,
            image_height: 0,
            fov_max_error_rad: None,
            match_radius: 0.01,
            match_threshold: 1e-5,
            solve_timeout_ms: Some(5000),
            match_max_error: None,
            refine_iterations: 2,
            camera_model: CameraModel {
                focal_length_px: 1.0,
                crpix: [0.0, 0.0],
                parity_flip: false,
                distortion: Distortion::None,
            },
        }
    }
}

impl SolveConfig {
    /// Create a solve configuration with the given FOV estimate (radians) and image dimensions.
    pub fn new(fov_estimate_rad: f32, image_width: u32, image_height: u32) -> Self {
        Self {
            fov_estimate_rad,
            image_width,
            image_height,
            camera_model: CameraModel::from_fov(fov_estimate_rad as f64, image_width),
            ..Default::default()
        }
    }

    /// Pixel scale in radians per pixel (horizontal).
    pub fn pixel_scale(&self) -> f32 {
        if self.image_width > 0 {
            self.fov_estimate_rad / self.image_width as f32
        } else {
            0.0
        }
    }
}

// ── Solve result ────────────────────────────────────────────────────────────

/// Result of a plate-solve attempt.
#[derive(Debug, Clone)]
pub struct SolveResult {
    /// Quaternion rotating ICRS vectors to camera-frame vectors.
    /// Camera frame: +X right, +Y down, +Z boresight.
    /// Usage: `camera_vec = qicrs2cam * icrs_vec`
    pub qicrs2cam: Option<Quaternion>,
    /// Estimated field of view (radians).
    pub fov_rad: Option<f32>,
    /// Number of matched stars in the verification step.
    pub num_matches: Option<u32>,
    /// RMS angular residual (radians) of matched stars.
    pub rmse_rad: Option<f32>,
    /// 90th-percentile angular residual (radians).
    pub p90e_rad: Option<f32>,
    /// Maximum angular residual (radians).
    pub max_err_rad: Option<f32>,
    /// False-positive probability (lower is better).
    pub prob: Option<f64>,
    /// Wall-clock time spent solving, in milliseconds.
    pub solve_time_ms: f32,
    /// Outcome status.
    pub status: SolveStatus,
    /// Whether the image x-axis was flipped to achieve a proper rotation.
    ///
    /// When `true`, the rotation matrix assumes negated x-coordinates.
    /// Pixel↔sky conversions must account for this flip.
    pub parity_flip: bool,
    /// Catalog IDs of matched stars (only populated on success).
    pub matched_catalog_ids: Vec<u64>,
    /// Indices into the input centroid slice for each match.
    pub matched_centroid_indices: Vec<usize>,
    /// Image width in pixels (used for coordinate transforms).
    pub image_width: u32,
    /// Image height in pixels (used for coordinate transforms).
    pub image_height: u32,
    /// WCS CD matrix: `[[CD11, CD12], [CD21, CD22]]` in tangent-plane radians per pixel.
    ///
    /// Maps pixel offsets from the optical center (CRPIX) to gnomonic tangent-plane
    /// coordinates at the reference point (CRVAL). Follows the FITS WCS TAN convention.
    /// Only populated on successful solve (`MatchFound`).
    pub cd_matrix: Option<[[f64; 2]; 2]>,
    /// WCS reference point `[RA, Dec]` in radians.
    ///
    /// The tangent point of the gnomonic (TAN) projection, typically very close to
    /// the camera boresight. Only populated on successful solve (`MatchFound`).
    pub crval_rad: Option<[f64; 2]>,
    /// Camera model used during solving (stored for coordinate transforms).
    ///
    /// On success, this contains the camera model with the refined focal length
    /// (from the matched FOV) and detected parity. The distortion model and CRPIX
    /// are copied from the input [`SolveConfig::camera_model`].
    pub camera_model: Option<CameraModel>,
    /// Fitted rotation angle in radians (camera roll in tangent plane).
    ///
    /// The angle from the tangent-plane ξ (East) axis to the camera +X axis,
    /// measured counter-clockwise. Only populated on successful solve.
    pub theta_rad: Option<f64>,
}

impl SolveResult {
    /// Create a failure result with the given status and elapsed time.
    pub(crate) fn failure(status: SolveStatus, solve_time_ms: f32) -> Self {
        Self {
            qicrs2cam: None,
            fov_rad: None,
            num_matches: None,
            rmse_rad: None,
            p90e_rad: None,
            max_err_rad: None,
            prob: None,
            solve_time_ms,
            status,
            parity_flip: false,
            matched_catalog_ids: Vec::new(),
            matched_centroid_indices: Vec::new(),
            image_width: 0,
            image_height: 0,
            cd_matrix: None,
            crval_rad: None,
            camera_model: None,
            theta_rad: None,
        }
    }

    /// Convert centered pixel coordinates to world coordinates (RA, Dec) in degrees.
    ///
    /// Pixel coordinates use the same convention as solver centroids:
    /// origin at the geometric image center, +X right, +Y down.
    ///
    /// When a CameraModel and theta are available (from the constrained WCS refinement),
    /// the pipeline is:
    /// 1. CameraModel.pixel_to_tanplane: crpix subtract → undistort → parity → divide by f
    /// 2. Rotate from camera frame to sky frame using theta
    /// 3. Inverse TAN projection at CRVAL → (RA, Dec)
    ///
    /// Falls back to the CD matrix path, then to the quaternion + FOV path.
    ///
    /// Returns `None` if the solve was unsuccessful.
    pub fn pixel_to_world(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        if let (Some(ref cam), Some(crval), Some(theta)) =
            (&self.camera_model, &self.crval_rad, self.theta_rad)
        {
            // ── CameraModel + theta path ──
            let (xi_cam, eta_cam) = cam.pixel_to_tanplane(x, y);
            // Rotate from camera frame to sky frame: R(θ) * [xi_cam, eta_cam]
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let xi = cos_t * xi_cam - sin_t * eta_cam;
            let eta = sin_t * xi_cam + cos_t * eta_cam;
            let (ra, dec) = wcs_refine::inverse_tan_project(xi, eta, crval[0], crval[1]);
            Some((ra.to_degrees().rem_euclid(360.0), dec.to_degrees()))
        } else if let (Some(cd), Some(crval)) = (&self.cd_matrix, &self.crval_rad) {
            // ── Legacy CD-matrix fallback (no CameraModel) ──
            let xi = cd[0][0] * x + cd[0][1] * y;
            let eta = cd[1][0] * x + cd[1][1] * y;
            let (ra, dec) = wcs_refine::inverse_tan_project(xi, eta, crval[0], crval[1]);
            Some((ra.to_degrees().rem_euclid(360.0), dec.to_degrees()))
        } else {
            // ── Fallback: quaternion + FOV ──
            let q = self.qicrs2cam.as_ref()?;
            let fov = self.fov_rad? as f64;
            let pixel_scale = fov / self.image_width.max(1) as f64;

            let ps = if self.parity_flip { -1.0 } else { 1.0 };
            let xr = ps * x * pixel_scale;
            let yr = y * pixel_scale;

            let norm = (xr * xr + yr * yr + 1.0).sqrt();
            let cx = xr / norm;
            let cy = yr / norm;
            let cz = 1.0 / norm;

            let rot = q.to_rotation_matrix();
            let m = rot.matrix();
            let ix = m[(0, 0)] as f64 * cx + m[(1, 0)] as f64 * cy + m[(2, 0)] as f64 * cz;
            let iy = m[(0, 1)] as f64 * cx + m[(1, 1)] as f64 * cy + m[(2, 1)] as f64 * cz;
            let iz = m[(0, 2)] as f64 * cx + m[(1, 2)] as f64 * cy + m[(2, 2)] as f64 * cz;

            let dec = iz.asin();
            let ra = iy.atan2(ix);
            Some((ra.to_degrees().rem_euclid(360.0), dec.to_degrees()))
        }
    }

    /// Convert world coordinates (RA, Dec in degrees) to centered pixel coordinates.
    ///
    /// Returns pixel coordinates in the same convention as solver centroids:
    /// origin at the geometric image center, +X right, +Y down.
    ///
    /// When a CameraModel and theta are available, the pipeline is:
    /// 1. TAN project (RA, Dec) at CRVAL → sky tangent-plane (ξ, η)
    /// 2. Rotate from sky frame to camera frame using -theta
    /// 3. CameraModel.tanplane_to_pixel: multiply by f → parity → distort → add crpix
    ///
    /// Falls back to the CD matrix path, then to the quaternion + FOV path.
    ///
    /// Returns `None` if the solve was unsuccessful or the point is behind the camera.
    pub fn world_to_pixel(&self, ra_deg: f64, dec_deg: f64) -> Option<(f64, f64)> {
        if let (Some(ref cam), Some(crval), Some(theta)) =
            (&self.camera_model, &self.crval_rad, self.theta_rad)
        {
            // ── CameraModel + theta path ──
            let ra = ra_deg.to_radians();
            let dec = dec_deg.to_radians();
            let (xi, eta) = wcs_refine::tan_project(ra, dec, crval[0], crval[1])?;
            // Rotate from sky frame to camera frame: R(-θ) * [xi, eta]
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let xi_cam = cos_t * xi + sin_t * eta;
            let eta_cam = -sin_t * xi + cos_t * eta;
            let (px, py) = cam.tanplane_to_pixel(xi_cam, eta_cam);
            Some((px, py))
        } else if let (Some(cd), Some(crval)) = (&self.cd_matrix, &self.crval_rad) {
            // ── Legacy CD-matrix fallback ──
            let ra = ra_deg.to_radians();
            let dec = dec_deg.to_radians();
            let (xi, eta) = wcs_refine::tan_project(ra, dec, crval[0], crval[1])?;
            let cd_inv = wcs_refine::cd_inverse(cd)?;
            let px = cd_inv[0][0] * xi + cd_inv[0][1] * eta;
            let py = cd_inv[1][0] * xi + cd_inv[1][1] * eta;
            Some((px, py))
        } else {
            // ── Fallback: quaternion + FOV ──
            let q = self.qicrs2cam.as_ref()?;
            let fov = self.fov_rad? as f64;
            let pixel_scale = fov / self.image_width.max(1) as f64;

            let ra = ra_deg.to_radians();
            let dec = dec_deg.to_radians();
            let cos_dec = dec.cos();
            let ix = ra.cos() * cos_dec;
            let iy = ra.sin() * cos_dec;
            let iz = dec.sin();

            let rot = q.to_rotation_matrix();
            let m = rot.matrix();
            let cx = m[(0, 0)] as f64 * ix + m[(0, 1)] as f64 * iy + m[(0, 2)] as f64 * iz;
            let cy = m[(1, 0)] as f64 * ix + m[(1, 1)] as f64 * iy + m[(1, 2)] as f64 * iz;
            let cz = m[(2, 0)] as f64 * ix + m[(2, 1)] as f64 * iy + m[(2, 2)] as f64 * iz;

            if cz <= 0.0 {
                return None;
            }

            let xr = cx / cz;
            let yr = cy / cz;

            let ps = if self.parity_flip { -1.0 } else { 1.0 };
            let ux = ps * xr / pixel_scale;
            let uy = yr / pixel_scale;

            Some((ux, uy))
        }
    }
}
