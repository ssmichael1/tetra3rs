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

use rkyv::{Archive, Deserialize, Serialize};

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
        }
    }
}
