//! Tracking-mode plate solving: solve using an attitude hint instead of
//! the lost-in-space pattern hash.
//!
//! When the caller provides an [`attitude_hint`](super::SolveConfig::attitude_hint)
//! (typically the previous frame's quaternion), the solver can skip pattern-hash
//! lookup entirely:
//!
//! 1. Query catalog stars within a cone around the hinted boresight.
//! 2. Project them to pixel coordinates using the hint rotation.
//! 3. Match each centroid to its nearest predicted star (within a radius set by
//!    hint uncertainty).
//! 4. If enough unique matches exist, run Wahba SVD for a refined rotation.
//! 5. Hand off to the same verification + WCS refine path used by the LIS solver.
//!
//! This succeeds with as few as 3 matched stars (LIS needs 4) and is robust to
//! pattern-hash failures from sparse / low-SNR fields.

use super::clock::Instant;

use numeris::Vector3;
use tracing::debug;

use crate::{Centroid, Quaternion};

use super::preprocess::{centroid_unit_vectors, sort_indices_by_brightness, CentroidVectors};
use super::solve::{failure, wahba_rotation};
use super::verify::{diagonal_factor, find_centroid_matches};
use super::{SolveConfig, SolveResult, SolveStatus, SolverDatabase};

/// Minimum unique correspondences required to attempt the SVD step.
const MIN_HINT_MATCHES: usize = 3;

impl SolverDatabase {
    /// Tracking solve using an attitude hint. See [`SolveConfig::attitude_hint`].
    ///
    /// `star_vectors` is the (possibly aberration-corrected) catalog
    /// unit-vector slice prepared by [`SolverDatabase::solve_from_centroids`]
    /// — the same slice the LIS path matches and refines against.
    ///
    /// Returns a [`SolveResult`] with the same shape as the LIS path. On failure
    /// the status is [`SolveStatus::NoMatch`] (or [`SolveStatus::TooFew`] if there
    /// aren't enough centroids).
    pub(crate) fn solve_with_hint(
        &self,
        preprocessed: &[Centroid],
        star_vectors: &[[f32; 3]],
        config: &SolveConfig,
        hint: &Quaternion,
        t0: Instant,
    ) -> SolveResult {
        let cam = &config.camera_model;
        let parity_flip = cam.parity_flip;
        let parity_sign: f32 = if parity_flip { -1.0 } else { 1.0 };

        // True pinhole pixel scale (1/f) from the camera model — the single
        // source of camera geometry. Zero means the model is the unconfigured
        // placeholder, so a hinted solve is impossible.
        let pixel_scale: f32 = config.pixel_scale();
        if pixel_scale <= 0.0 {
            return failure(SolveStatus::NoMatch, t0);
        }
        let fov_rad = config.fov_estimate_rad();

        if preprocessed.len() < MIN_HINT_MATCHES {
            return failure(SolveStatus::TooFew, t0);
        }

        // ── Hint geometry ──
        let r_hint = hint.to_rotation_matrix();
        // Boresight in ICRS = R^T * [0,0,1] = third row of R
        let boresight_icrs = Vector3::from_array([r_hint[(2, 0)], r_hint[(2, 1)], r_hint[(2, 2)]]);

        // Cone radius: half-FOV (use diagonal for safety) + hint uncertainty + small margin
        let fov_diagonal = fov_rad * diagonal_factor(config);
        let cone_radius = fov_diagonal / 2.0 + config.hint_uncertainty_rad + 2.0 * pixel_scale;
        let nearby_inds = self.star_catalog.query_indices_from_uvec_cached(
            boresight_icrs,
            cone_radius,
            &self.star_vectors,
        );

        debug!(
            "Tracking: hint cone {:.3}° → {} catalog stars",
            cone_radius.to_degrees(),
            nearby_inds.len()
        );

        if nearby_inds.len() < MIN_HINT_MATCHES {
            return failure(SolveStatus::NoMatch, t0);
        }

        // ── Sort centroids by brightness (mirrors LIS path) ──
        let sorted_indices = sort_indices_by_brightness(preprocessed);

        // Trim to verification limit (same as LIS).
        let verification_stars = self.props.verification_stars_per_fov as usize;
        let match_centroid_count = preprocessed.len().min(verification_stars);

        // ── Build centroid unit vectors in the camera frame, parity-applied ──
        let centroid_vectors =
            centroid_unit_vectors(preprocessed, &sorted_indices, pixel_scale, parity_sign);

        // ── Project candidate catalog stars to camera-plane angles via the hint ──
        // Note: r_hint maps ICRS→camera, so cam_v = r_hint * icrs_v.
        let half_w = (config.image_width() as f32 / 2.0 + 4.0) * pixel_scale;
        let half_h = (config.image_height() as f32 / 2.0 + 4.0) * pixel_scale;
        let mut projected: Vec<(usize, f32, f32)> = Vec::with_capacity(nearby_inds.len());
        for &cat_idx in &nearby_inds {
            let sv = &star_vectors[cat_idx];
            let icrs_v = Vector3::from_array([sv[0], sv[1], sv[2]]);
            let cam_v = r_hint * icrs_v;
            if cam_v[2] > 0.0 {
                let cx = cam_v[0] / cam_v[2];
                let cy = cam_v[1] / cam_v[2];
                // Only keep stars geometrically inside the (slightly padded) image
                if cx.abs() <= half_w && cy.abs() <= half_h {
                    projected.push((cat_idx, cx, cy));
                }
            }
        }

        if projected.len() < MIN_HINT_MATCHES {
            return failure(SolveStatus::NoMatch, t0);
        }

        // ── Initial centroid → catalog star matching ──
        // Match radius covers (a) hint angular uncertainty and (b) the LIS-equivalent
        // fractional match radius. Whichever is larger.
        let hint_match_radius = (config.hint_uncertainty_rad).max(config.match_radius * fov_rad);

        // Matching working buffers, shared by the initial match and the
        // verification below (a single-shot solve, but the shared signature
        // requires them).
        let mut match_xy: Vec<(f32, f32)> = Vec::new();
        let mut match_scratch = super::matching::MatchScratch::<f32>::default();

        let initial_matches = find_centroid_matches(
            &centroid_vectors[..match_centroid_count.min(centroid_vectors.len())],
            &projected,
            hint_match_radius,
            &mut match_xy,
            &mut match_scratch,
        );

        debug!(
            "Tracking: initial NN match → {} pairs (radius {:.1}″)",
            initial_matches.len(),
            hint_match_radius.to_degrees() * 3600.0
        );

        if initial_matches.len() < MIN_HINT_MATCHES {
            return failure(SolveStatus::NoMatch, t0);
        }

        // ── Wahba SVD on the initial correspondence set ──
        // A failed SVD (degenerate cross-covariance) or a negative determinant
        // (likely parity mismatch) both bail — the caller may still fall back
        // to LIS.
        let Some(rotation_matrix) = wahba_rotation(
            initial_matches
                .iter()
                .map(|&(cent_idx, cat_idx)| (&centroid_vectors[cent_idx], &star_vectors[cat_idx])),
        ) else {
            return failure(SolveStatus::NoMatch, t0);
        };
        let det = rotation_matrix.det();
        if det.is_nan() || det <= 0.0 {
            return failure(SolveStatus::NoMatch, t0);
        }

        // ── Verification (same path as LIS) ──
        // `hypothesis_matches = 0`: the attitude hypothesis comes from the
        // caller's hint, not from any of the tested centroids, so every match
        // is independent evidence.
        let (verify_matches, prob_mismatch) = self.verify_attitude(
            &rotation_matrix,
            CentroidVectors {
                pixel_scale,
                parity_flip,
                data: &centroid_vectors,
            },
            match_centroid_count,
            fov_rad,
            config,
            star_vectors,
            0,
            None,
            &mut match_xy,
            &mut match_scratch,
        );

        // Same false-positive probability test as LIS, but without any
        // multiple-comparison correction (a single candidate is tested).
        if prob_mismatch >= config.match_threshold {
            debug!(
                "Tracking: verification rejected (matches={}, prob={:.2e})",
                verify_matches.len(),
                prob_mismatch
            );
            return failure(SolveStatus::NoMatch, t0);
        }

        debug!(
            "Tracking: VERIFIED — {} matches, prob={:.2e}",
            verify_matches.len(),
            prob_mismatch
        );

        // ── WCS refinement + finalization (same path as LIS) ──
        match self.refine_and_finalize(
            &rotation_matrix,
            &verify_matches,
            preprocessed,
            &sorted_indices,
            star_vectors,
            config,
            parity_flip,
            fov_rad,
            pixel_scale as f64,
            match_centroid_count,
            MIN_HINT_MATCHES,
            prob_mismatch,
            t0,
        ) {
            Some(solution) => Ok(solution),
            None => failure(SolveStatus::NoMatch, t0),
        }
    }
}
