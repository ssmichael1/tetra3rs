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

/// Time `$e`, attributing the elapsed nanoseconds (and one operation) to
/// `$bucket`. With the `profile` feature off this expands to just `$e`.
macro_rules! timed {
    ($bucket:expr, $e:expr) => {{
        #[cfg(feature = "profile")]
        let __t0 = std::time::Instant::now();
        let __res = $e;
        #[cfg(feature = "profile")]
        $crate::solver::profiling::record($bucket, __t0.elapsed().as_nanos(), 1);
        __res
    }};
}

pub(crate) mod clock;
pub(crate) mod combinations;
pub(crate) mod database;
pub(crate) mod matching;
pub(crate) mod pattern;
#[cfg(feature = "profile")]
pub mod profiling;
pub(crate) mod solve;
pub(crate) mod track;
pub(crate) mod wcs_refine;

use serde::{Deserialize, Serialize};

use crate::camera_model::CameraModel;
use crate::distortion::Distortion;
use crate::{Quaternion, StarCatalog};

// ── Pattern hash table entry ────────────────────────────────────────────────

/// A single slot in the pattern hash table.
///
/// Packing star indices, largest-edge angle, and key hash into one struct
/// means a single cache-line fetch per quadratic-probe step instead of three.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct PatternEntry {
    /// Indices into the star catalog for the 4 stars in this pattern.
    pub star_indices: [u32; 4],
    /// Largest pairwise edge angle (radians) — used for FOV consistency filtering.
    pub largest_edge: f32,
    /// Lower 16 bits of the pattern key hash — fast pre-filter.
    pub key_hash: u16,
    /// Explicit padding to 24 bytes (8-byte aligned).
    _pad: u16,
}

impl PatternEntry {
    /// Sentinel value for an empty hash-table slot.
    pub const EMPTY: Self = Self {
        star_indices: [0, 0, 0, 0],
        largest_edge: 0.0,
        key_hash: 0,
        _pad: 0,
    };

    /// Create a new pattern entry.
    #[inline]
    pub fn new(star_indices: [u32; 4], largest_edge: f32, key_hash: u16) -> Self {
        Self {
            star_indices,
            largest_edge,
            key_hash,
            _pad: 0,
        }
    }

    /// Returns `true` if this slot is empty (sentinel).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.star_indices == [0, 0, 0, 0]
    }
}

// ── Pattern catalog (flat hash table) ───────────────────────────────────────

/// Pattern hash table backed by a single flat `Vec<PatternEntry>`.
///
/// Open addressing with quadratic probing; empty slots have
/// `star_indices == [0, 0, 0, 0]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternCatalog {
    pub entries: Vec<PatternEntry>,
}

impl PatternCatalog {
    /// Allocate a catalog of `capacity` empty entries.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: vec![PatternEntry::EMPTY; capacity],
        }
    }

    /// Total number of slots.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the catalog has no slots.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Immutable access to slot `idx`. Panics if `idx >= len()`.
    #[inline]
    pub fn get(&self, idx: usize) -> &PatternEntry {
        &self.entries[idx]
    }

    /// Mutable access to slot `idx`. Panics if `idx >= len()`.
    #[inline]
    pub fn get_mut(&mut self, idx: usize) -> &mut PatternEntry {
        &mut self.entries[idx]
    }
}

#[cfg(test)]
mod pattern_catalog_tests {
    use super::*;

    #[test]
    fn small_catalog() {
        let mut cat = PatternCatalog::with_capacity(100);
        assert_eq!(cat.len(), 100);

        *cat.get_mut(42) = PatternEntry::new([1, 2, 3, 4], 0.5, 0xabcd);
        let e = cat.get(42);
        assert_eq!(e.star_indices, [1, 2, 3, 4]);
        assert!((e.largest_edge - 0.5).abs() < 1e-6);
        assert_eq!(e.key_hash, 0xabcd);
        assert!(cat.get(0).is_empty());
    }

    #[test]
    fn empty_catalog() {
        let cat = PatternCatalog::with_capacity(0);
        assert_eq!(cat.len(), 0);
        assert!(cat.is_empty());
    }

    #[test]
    fn postcard_roundtrip_small() {
        let mut cat = PatternCatalog::with_capacity(1024);
        *cat.get_mut(0) = PatternEntry::new([10, 20, 30, 40], 0.1, 0x1111);
        *cat.get_mut(1023) = PatternEntry::new([1, 2, 3, 4], 0.9, 0xffff);

        let bytes = postcard::to_allocvec(&cat).expect("serialize");
        let restored: PatternCatalog = postcard::from_bytes(&bytes).expect("deserialize");

        assert_eq!(restored.len(), 1024);
        assert_eq!(restored.get(0).star_indices, [10, 20, 30, 40]);
        assert_eq!(restored.get(1023).key_hash, 0xffff);
        assert!(restored.get(500).is_empty());
    }
}

// ── Status codes (matching tetra3) ──────────────────────────────────────────

/// Why a plate-solve attempt produced no solution.
///
/// A successful solve returns a [`Solution`] instead; see [`SolveResult`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolveStatus {
    /// All pattern combinations were exhausted without a match.
    NoMatch,
    /// The solve timeout was reached before a match was found.
    Timeout,
    /// Too few centroids were provided to form a pattern.
    TooFew,
}

/// A failed plate-solve attempt: why it failed and how long it took.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SolveFailure {
    /// Why the solve produced no solution.
    pub status: SolveStatus,
    /// Wall-clock time spent before giving up, in milliseconds.
    pub solve_time_ms: f32,
}

// ── Database properties ─────────────────────────────────────────────────────

/// Metadata describing how a solver database was built.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Complete solver database, serializable with postcard.
///
/// Contains a spatial star catalog (brightness-sorted), precomputed unit vectors,
/// and a pattern hash table for fast geometric matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverDatabase {
    /// Spatial star catalog. Stars are sorted brightest-first so that
    /// index order equals brightness order.
    pub star_catalog: StarCatalog,

    /// Precomputed ICRS unit vectors for each star, matching star_catalog.stars ordering.
    /// Stored as [x, y, z] where x=cos(ra)cos(dec), y=sin(ra)cos(dec), z=sin(dec).
    pub star_vectors: Vec<[f32; 3]>,

    /// Original catalog IDs (e.g. HIP number) for each star.
    pub star_catalog_ids: Vec<i64>,

    /// Pattern hash table (open addressing, quadratic probing).
    /// Each slot packs the 4 star indices, largest edge angle, and a 16-bit
    /// key hash prefilter into a single cache-friendly struct. Empty slots
    /// have `star_indices == [0, 0, 0, 0]`.
    pub pattern_catalog: PatternCatalog,

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

impl GenerateDatabaseConfig {
    /// Validate that the generation parameters are self-consistent, before they
    /// drive index arithmetic that would otherwise overflow, divide by zero, or
    /// silently corrupt every pattern key. Returns [`crate::Error::InvalidInput`]
    /// with a specific message for the first offending field.
    pub fn validate(&self) -> crate::Result<()> {
        use crate::Error::InvalidInput;
        if !(self.max_fov_deg.is_finite() && self.max_fov_deg > 0.0 && self.max_fov_deg < 180.0) {
            return Err(InvalidInput(format!(
                "max_fov_deg must be in (0, 180), got {}",
                self.max_fov_deg
            )));
        }
        if let Some(min_fov) = self.min_fov_deg {
            if !(min_fov.is_finite() && min_fov > 0.0 && min_fov <= self.max_fov_deg) {
                return Err(InvalidInput(format!(
                    "min_fov_deg must be in (0, max_fov_deg={}], got {}",
                    self.max_fov_deg, min_fov
                )));
            }
        }
        // bins = round(0.25 / pattern_max_error). Values above 0.25 round to a
        // single bin (degenerate hash — every key dimension collapses), so the
        // documented valid range is (0, 0.25].
        if !(self.pattern_max_error.is_finite()
            && self.pattern_max_error > 0.0
            && self.pattern_max_error <= 0.25)
        {
            return Err(InvalidInput(format!(
                "pattern_max_error must be finite and in (0, 0.25], got {}",
                self.pattern_max_error
            )));
        }
        // fov_divisions uses ln(multiscale_step) as a divisor; step <= 1 gives a
        // zero/negative log and an infinite/NaN division count.
        if !(self.multiscale_step.is_finite() && self.multiscale_step > 1.0) {
            return Err(InvalidInput(format!(
                "multiscale_step must be finite and > 1.0, got {}",
                self.multiscale_step
            )));
        }
        if self.verification_stars_per_fov == 0 {
            return Err(InvalidInput(
                "verification_stars_per_fov must be >= 1".into(),
            ));
        }
        if self.catalog_nside == 0 {
            return Err(InvalidInput("catalog_nside must be >= 1".into()));
        }
        Ok(())
    }
}

// ── Configuration for plate solving ─────────────────────────────────────────

/// Parameters controlling the plate-solve attempt.
///
/// One config serves both solve modes. `solve_from_centroids` runs
/// **lost-in-space** (pattern-hash search) unless an [`attitude_hint`] is set,
/// in which case it runs **tracking** (direct correspondence from the hint)
/// with automatic fallback to lost-in-space. The fields are grouped by which
/// mode reads them — fields marked for one mode are ignored in the other:
///
/// - **Camera geometry** ([`camera_model`]) and **matching/verification**
///   ([`match_radius`], [`match_threshold`], [`solve_timeout_ms`]) apply to
///   both modes.
/// - **Lost-in-space** ([`fov_max_error_rad`], [`match_max_error`]) tune the
///   pattern search; ignored in tracking.
/// - **Tracking** ([`attitude_hint`], [`hint_uncertainty_rad`],
///   [`strict_hint`]) are inert unless `attitude_hint` is set.
/// - **Stellar aberration** ([`observer_velocity_km_s`]) applies to both
///   modes.
///
/// [`camera_model`]: SolveConfig::camera_model
/// [`match_radius`]: SolveConfig::match_radius
/// [`match_threshold`]: SolveConfig::match_threshold
/// [`solve_timeout_ms`]: SolveConfig::solve_timeout_ms
/// [`fov_max_error_rad`]: SolveConfig::fov_max_error_rad
/// [`match_max_error`]: SolveConfig::match_max_error
/// [`attitude_hint`]: SolveConfig::attitude_hint
/// [`hint_uncertainty_rad`]: SolveConfig::hint_uncertainty_rad
/// [`strict_hint`]: SolveConfig::strict_hint
/// [`observer_velocity_km_s`]: SolveConfig::observer_velocity_km_s
pub struct SolveConfig {
    // ── Camera geometry (both modes) ──
    /// Camera intrinsics model — the single source of camera geometry.
    ///
    /// The model's focal length and image width define the solver's FOV
    /// estimate ([`SolveConfig::fov_estimate_rad`]); its image dimensions,
    /// optical center (CRPIX), parity, and distortion drive centroid
    /// preprocessing and WCS refinement. Build it with
    /// [`CameraModel::from_fov`] (or use [`SolveConfig::new`]) when only a
    /// FOV estimate is known, or pass a calibrated model from a previous
    /// solve / [`crate::calibrate_camera`] run.
    pub camera_model: CameraModel,

    // ── Matching & verification (both modes) ──
    /// Maximum match distance as a fraction of the FOV. Default 0.01.
    pub match_radius: f32,
    /// False-positive probability budget for accepting a solution.
    /// Default 1e-5.
    ///
    /// Each candidate attitude's verification yields a binomial p-value (the
    /// probability that a wrong attitude would coincidentally match as many
    /// stars); candidate `k` of a lost-in-space search is accepted when
    /// `p·k < match_threshold` — a sequential Bonferroni correction over the
    /// candidates actually tested, so the total false-accept probability of a
    /// full search stays within a small (logarithmic) multiple of this
    /// budget. Tracking tests a single hinted candidate against the budget
    /// directly. Raising this (e.g. `1e-3`) accepts weaker evidence — useful
    /// for very sparse fields (≲7 stars) at increased false-positive risk.
    pub match_threshold: f64,
    /// Timeout in milliseconds. None = no timeout. Default 5000.
    pub solve_timeout_ms: Option<u64>,

    // ── Lost-in-space pattern search (ignored in tracking mode) ──
    /// Maximum acceptable FOV error in radians. None = no FOV filtering.
    ///
    /// When set, the solver sweeps FOV values across
    /// `fov_estimate ± fov_max_error` and rejects pattern matches implying a
    /// FOV outside the range. *Lost-in-space only* — tracking takes its scale
    /// from the hint and camera model.
    pub fov_max_error_rad: Option<f32>,
    /// Maximum edge-ratio error for matching. None = use database value.
    ///
    /// Values below the database's `pattern_max_error` are clamped up to it —
    /// patterns were quantized at that tolerance during generation, so a
    /// tighter match tolerance cannot be honored. *Lost-in-space only* —
    /// tracking does not use the pattern hash.
    pub match_max_error: Option<f32>,

    // ── Tracking (ignored unless `attitude_hint` is set) ──
    /// Optional attitude hint for tracking-mode solving.
    ///
    /// When set, the solver first attempts a fast direct-correspondence solve
    /// using the hint (project catalog stars near the hinted boresight, match
    /// them to centroids by nearest-neighbor in pixel space, run Wahba SVD).
    /// On success, returns the result; on failure, falls back to the full
    /// lost-in-space pattern-hash search unless [`SolveConfig::strict_hint`]
    /// is set.
    ///
    /// The quaternion uses the same convention as [`Solution::qicrs2cam`]:
    /// it rotates ICRS vectors into the camera frame. Pass the previous
    /// frame's `qicrs2cam` to chain solves at video rate.
    ///
    /// Tracking mode can succeed with as few as 3 matched stars (LIS needs 4)
    /// and is robust against pattern-hash failures from low-SNR or sparse
    /// fields. The main robustness assumption is that the hint is within
    /// [`SolveConfig::hint_uncertainty_rad`] of the true attitude. Setting this
    /// is what switches `solve_from_centroids` into tracking mode; `None`
    /// (default) runs lost-in-space.
    pub attitude_hint: Option<Quaternion>,

    /// Angular uncertainty (radians) of [`SolveConfig::attitude_hint`].
    ///
    /// Used to size the catalog cone search and the pixel-space match radius
    /// for centroid–star correspondence. Larger values widen the search and
    /// allow staler hints, at the cost of more spurious candidates.
    ///
    /// Ignored unless `attitude_hint` is set. Default: 1°.
    pub hint_uncertainty_rad: f32,

    /// When `true`, do not fall back to lost-in-space if the hinted solve
    /// fails. Returns the hinted-solve failure status directly.
    ///
    /// Useful when downstream logic must distinguish "hint was wrong" from
    /// "fell back to LIS and succeeded with a different attitude than expected."
    /// Ignored unless `attitude_hint` is set. Default: `false`.
    pub strict_hint: bool,

    // ── Stellar aberration (both modes) ──
    /// Observer's barycentric velocity in km/s (ICRS/GCRF Cartesian).
    ///
    /// When set, catalog star positions are aberration-corrected to apparent
    /// positions before matching and refinement, removing the ~20" systematic
    /// bias from Earth's orbital velocity.
    ///
    /// For ground-based or Earth-orbiting observers, this is dominated by
    /// Earth's orbital velocity (~30 km/s). Default: `None` (no correction).
    pub observer_velocity_km_s: Option<[f64; 3]>,
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            fov_max_error_rad: None,
            match_radius: 0.01,
            match_threshold: 1e-5,
            solve_timeout_ms: Some(5000),
            match_max_error: None,
            camera_model: CameraModel {
                focal_length_px: 1.0,
                image_width: 0,
                image_height: 0,
                crpix: [0.0, 0.0],
                parity_flip: false,
                distortion: Distortion::None,
            },
            observer_velocity_km_s: None,
            attitude_hint: None,
            hint_uncertainty_rad: 1.0_f32.to_radians(),
            strict_hint: false,
        }
    }
}

impl SolveConfig {
    /// Create a solve configuration with the given FOV estimate (radians) and
    /// image dimensions, using a simple pinhole camera model (no distortion,
    /// centered optical axis, no parity flip).
    pub fn new(fov_estimate_rad: f32, image_width: u32, image_height: u32) -> Self {
        Self::with_camera_model(CameraModel::from_fov(
            fov_estimate_rad as f64,
            image_width,
            image_height,
        ))
    }

    /// Create a solve configuration from an existing camera model, e.g. one
    /// refined by a previous solve or fitted by [`crate::calibrate_camera`].
    pub fn with_camera_model(camera_model: CameraModel) -> Self {
        Self {
            camera_model,
            ..Default::default()
        }
    }

    /// Estimated horizontal field of view in radians, derived from the camera
    /// model's focal length and image width.
    pub fn fov_estimate_rad(&self) -> f32 {
        self.camera_model.fov_rad() as f32
    }

    /// Image width in pixels (from the camera model).
    pub fn image_width(&self) -> u32 {
        self.camera_model.image_width
    }

    /// Image height in pixels (from the camera model).
    pub fn image_height(&self) -> u32 {
        self.camera_model.image_height
    }

    /// Pixel scale in radians per pixel (horizontal): `1 / focal_length_px`.
    ///
    /// Returns `0.0` when the camera model is the unconfigured
    /// `Default::default()` placeholder (zero image width). Crate-internal: the
    /// `0.0`-means-unconfigured sentinel is an implementation detail (Python
    /// exposes pixel scale via `CameraModel.pixel_scale`).
    pub(crate) fn pixel_scale(&self) -> f32 {
        if self.camera_model.image_width > 0 && self.camera_model.focal_length_px > 0.0 {
            self.camera_model.pixel_scale() as f32
        } else {
            0.0
        }
    }
}

// ── Solve result ────────────────────────────────────────────────────────────

/// Outcome of a plate-solve attempt: a [`Solution`] on success, or a
/// [`SolveFailure`] carrying the reason and elapsed time.
pub type SolveResult = Result<Solution, SolveFailure>;

/// A successful plate solve. Every field is guaranteed present.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    /// Quaternion rotating ICRS vectors to camera-frame vectors.
    /// Camera frame: +X right, +Y down, +Z boresight.
    /// Usage: `camera_vec = qicrs2cam * icrs_vec`
    pub qicrs2cam: Quaternion,
    /// Estimated field of view (radians).
    pub fov_rad: f32,
    /// Number of matched stars in the verification step.
    pub num_matches: u32,
    /// RMS angular residual (radians) of matched stars.
    pub rmse_rad: f32,
    /// 90th-percentile angular residual (radians).
    pub p90e_rad: f32,
    /// Maximum angular residual (radians).
    pub max_err_rad: f32,
    /// False-positive probability of the accepted match (lower is better):
    /// the verification p-value after the multiple-comparison correction
    /// that was tested against [`SolveConfig::match_threshold`].
    pub prob: f64,
    /// Wall-clock time spent solving, in milliseconds.
    pub solve_time_ms: f32,
    /// Whether the image x-axis was flipped to achieve a proper rotation.
    ///
    /// When `true`, the rotation matrix assumes negated x-coordinates.
    /// Pixel↔sky conversions must account for this flip.
    pub parity_flip: bool,
    /// Catalog IDs of matched stars.
    pub matched_catalog_ids: Vec<i64>,
    /// Indices into the input centroid slice for each match.
    pub matched_centroid_indices: Vec<usize>,
    /// WCS CD matrix: `[[CD11, CD12], [CD21, CD22]]` in tangent-plane radians per pixel.
    ///
    /// Maps pixel offsets from the optical center (CRPIX) to gnomonic tangent-plane
    /// coordinates at the reference point (CRVAL). Follows the FITS WCS TAN convention.
    pub cd_matrix: [[f64; 2]; 2],
    /// WCS reference point `[RA, Dec]` in radians.
    ///
    /// The tangent point of the gnomonic (TAN) projection, typically very close to
    /// the camera boresight.
    pub crval_rad: [f64; 2],
    /// Camera model used during solving (stored for coordinate transforms).
    ///
    /// The camera model with the refined focal length (from the matched FOV)
    /// and detected parity. The distortion model and CRPIX are copied from
    /// the input [`SolveConfig::camera_model`].
    pub camera_model: CameraModel,
    /// Fitted rotation angle in radians (camera roll in tangent plane).
    ///
    /// The angle from the tangent-plane ξ (East) axis to the camera +X axis,
    /// measured counter-clockwise. When [`Solution::parity_flip`] is `true`,
    /// "camera +X" means the x-negated (mirror-corrected) axis — the same
    /// frame [`Solution::qicrs2cam`] rotates into.
    pub theta_rad: f64,
}

impl Solution {
    /// Convert centered pixel coordinates to world coordinates (RA, Dec) in degrees.
    ///
    /// Pixel coordinates use the same convention as solver centroids:
    /// origin at the geometric image center, +X right, +Y down.
    ///
    /// Pipeline:
    /// 1. CameraModel.pixel_to_tanplane: crpix subtract → undistort → parity → divide by f
    /// 2. Rotate from camera frame to sky frame using theta
    /// 3. Inverse TAN projection at CRVAL → (RA, Dec)
    pub fn pixel_to_world(&self, x: f64, y: f64) -> (f64, f64) {
        let (xi_cam, eta_cam) = self.camera_model.pixel_to_tanplane(x, y);
        // Rotate from camera frame to sky frame: R(θ) * [xi_cam, eta_cam]
        let cos_t = self.theta_rad.cos();
        let sin_t = self.theta_rad.sin();
        let xi = cos_t * xi_cam - sin_t * eta_cam;
        let eta = sin_t * xi_cam + cos_t * eta_cam;
        let (ra, dec) =
            wcs_refine::inverse_tan_project(xi, eta, self.crval_rad[0], self.crval_rad[1]);
        (ra.to_degrees().rem_euclid(360.0), dec.to_degrees())
    }

    /// Convert world coordinates (RA, Dec in degrees) to centered pixel coordinates.
    ///
    /// Returns pixel coordinates in the same convention as solver centroids:
    /// origin at the geometric image center, +X right, +Y down.
    ///
    /// Pipeline:
    /// 1. TAN project (RA, Dec) at CRVAL → sky tangent-plane (ξ, η)
    /// 2. Rotate from sky frame to camera frame using -theta
    /// 3. CameraModel.tanplane_to_pixel: multiply by f → parity → distort → add crpix
    ///
    /// Returns `None` if the point is behind the camera.
    pub fn world_to_pixel(&self, ra_deg: f64, dec_deg: f64) -> Option<(f64, f64)> {
        let ra = ra_deg.to_radians();
        let dec = dec_deg.to_radians();
        let (xi, eta) = wcs_refine::tan_project(ra, dec, self.crval_rad[0], self.crval_rad[1])?;
        // Rotate from sky frame to camera frame: R(-θ) * [xi, eta]
        let cos_t = self.theta_rad.cos();
        let sin_t = self.theta_rad.sin();
        let xi_cam = cos_t * xi + sin_t * eta;
        let eta_cam = -sin_t * xi + cos_t * eta;
        Some(self.camera_model.tanplane_to_pixel(xi_cam, eta_cam))
    }
}

/// Pinhole focal length in pixels from an angular FOV and image width:
/// `f = (image_width / 2) / tan(fov / 2)`.
pub(crate) fn focal_length_from_fov(image_width: u32, fov_rad: f64) -> f64 {
    (image_width.max(1) as f64 / 2.0) / (fov_rad / 2.0).tan()
}

/// Pinhole pixel scale (radians per pixel) from an angular FOV and image width:
/// `pixel_scale = 1 / focal_length_from_fov(...)`.
pub(crate) fn pixel_scale_from_fov(image_width: u32, fov_rad: f64) -> f64 {
    1.0 / focal_length_from_fov(image_width, fov_rad)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_config_validate_accepts_default() {
        assert!(GenerateDatabaseConfig::default().validate().is_ok());
    }

    #[test]
    fn generate_config_validate_rejects_bad_values() {
        let bad = |f: fn(&mut GenerateDatabaseConfig)| {
            let mut c = GenerateDatabaseConfig::default();
            f(&mut c);
            assert!(c.validate().is_err());
        };
        bad(|c| c.multiscale_step = 1.0); // ln(1) == 0 → infinite divisions
        bad(|c| c.pattern_max_error = 0.0); // bins → ∞
        bad(|c| c.pattern_max_error = -0.001);
        bad(|c| c.verification_stars_per_fov = 0); // divide by zero
        bad(|c| c.catalog_nside = 0);
        bad(|c| c.max_fov_deg = 0.0);
        bad(|c| c.min_fov_deg = Some(40.0)); // > max_fov_deg (30)
    }
}
