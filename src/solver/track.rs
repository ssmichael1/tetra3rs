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

use std::time::Instant;

use numeris::{Matrix3, Vector3};
use tracing::debug;

use crate::{Centroid, Quaternion};

use super::solve::{aberration_correct, binomial_cdf, find_centroid_matches};
use super::wcs_refine;
use super::{SolveConfig, SolveResult, SolveStatus, SolverDatabase};

/// Speed of light in km/s (duplicate of solve.rs C_KM_S; kept private here).
const C_KM_S: f64 = 299_792.458;

/// Minimum unique correspondences required to attempt the SVD step.
const MIN_HINT_MATCHES: usize = 3;

impl SolverDatabase {
    /// Tracking solve using an attitude hint. See [`SolveConfig::attitude_hint`].
    ///
    /// Returns a [`SolveResult`] with the same shape as the LIS path. On failure
    /// the status is [`SolveStatus::NoMatch`] (or [`SolveStatus::TooFew`] if there
    /// aren't enough centroids).
    pub(crate) fn solve_with_hint(
        &self,
        preprocessed: &[Centroid],
        config: &SolveConfig,
        hint: &Quaternion,
        t0: Instant,
    ) -> SolveResult {
        let cam = &config.camera_model;
        let parity_flip = cam.parity_flip;
        let parity_sign: f32 = if parity_flip { -1.0 } else { 1.0 };

        // True pinhole pixel scale (1/f). Prefer the camera model's focal length
        // when it was explicitly set; otherwise fall back to fov_estimate so a
        // default-constructed `SolveConfig` still works.
        let camera_model_initialized =
            cam.image_width == config.image_width && cam.focal_length_px > 2.0;
        let pixel_scale: f32 = if camera_model_initialized {
            (1.0 / cam.focal_length_px) as f32
        } else if config.fov_estimate_rad > 0.0 && config.image_width > 0 {
            let f = (config.image_width as f32 / 2.0)
                / (config.fov_estimate_rad / 2.0).tan();
            1.0 / f
        } else {
            return SolveResult::failure(SolveStatus::NoMatch, elapsed_ms(t0));
        };
        // Angular FOV derived from pixel scale.
        let fov_rad = 2.0 * (config.image_width as f32 / 2.0 * pixel_scale).atan();

        if preprocessed.len() < MIN_HINT_MATCHES {
            return SolveResult::failure(SolveStatus::TooFew, elapsed_ms(t0));
        }

        // ── Hint geometry ──
        let r_hint = hint.to_rotation_matrix();
        // Boresight in ICRS = R^T * [0,0,1] = third row of R
        let boresight_icrs = Vector3::from_array([
            r_hint[(2, 0)],
            r_hint[(2, 1)],
            r_hint[(2, 2)],
        ]);

        // Cone radius: half-FOV (use diagonal for safety) + hint uncertainty + small margin
        let fov_diagonal = fov_rad * 1.42;
        let cone_radius =
            fov_diagonal / 2.0 + config.hint_uncertainty_rad + 2.0 * pixel_scale;
        let nearby_inds = self
            .star_catalog
            .query_indices_from_uvec(boresight_icrs, cone_radius);

        debug!(
            "Tracking: hint cone {:.3}° → {} catalog stars",
            cone_radius.to_degrees(),
            nearby_inds.len()
        );

        if nearby_inds.len() < MIN_HINT_MATCHES {
            return SolveResult::failure(SolveStatus::NoMatch, elapsed_ms(t0));
        }

        // ── Aberration correction (only on the candidates) ──
        let beta = config
            .observer_velocity_km_s
            .map(|v| [v[0] / C_KM_S, v[1] / C_KM_S, v[2] / C_KM_S]);
        let candidate_vecs: Vec<[f32; 3]> = nearby_inds
            .iter()
            .map(|&idx| {
                let raw = &self.star_vectors[idx];
                match beta {
                    Some(b) => aberration_correct(raw, &b),
                    None => *raw,
                }
            })
            .collect();

        // ── Sort centroids by brightness (mirrors LIS path) ──
        let mut sorted_indices: Vec<usize> = (0..preprocessed.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            let ma = preprocessed[a].mass.unwrap_or(f32::MIN);
            let mb = preprocessed[b].mass.unwrap_or(f32::MIN);
            mb.partial_cmp(&ma).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Trim to verification limit (same as LIS).
        let verification_stars = self.props.verification_stars_per_fov as usize;
        let match_centroid_count = preprocessed.len().min(verification_stars);

        // ── Build centroid unit vectors in the camera frame, parity-applied ──
        let centroid_vectors: Vec<[f32; 3]> = sorted_indices
            .iter()
            .map(|&i| {
                let x = parity_sign * preprocessed[i].x * pixel_scale;
                let y = preprocessed[i].y * pixel_scale;
                let z = 1.0f32;
                let norm = (x * x + y * y + z * z).sqrt();
                [x / norm, y / norm, z / norm]
            })
            .collect();

        // ── Project candidate catalog stars to camera-plane angles via the hint ──
        // Note: r_hint maps ICRS→camera, so cam_v = r_hint * icrs_v.
        let mut projected: Vec<(usize, f32, f32)> = Vec::with_capacity(candidate_vecs.len());
        for (local_i, sv) in candidate_vecs.iter().enumerate() {
            let icrs_v = Vector3::from_array([sv[0], sv[1], sv[2]]);
            let cam_v = r_hint * icrs_v;
            if cam_v[2] > 0.0 {
                let cx = cam_v[0] / cam_v[2];
                let cy = cam_v[1] / cam_v[2];
                // Only keep stars geometrically inside the (slightly padded) image
                let half_w = (config.image_width as f32 / 2.0 + 4.0) * pixel_scale;
                let half_h = (config.image_height as f32 / 2.0 + 4.0) * pixel_scale;
                if cx.abs() <= half_w && cy.abs() <= half_h {
                    let cat_star_idx = nearby_inds[local_i];
                    projected.push((cat_star_idx, cx, cy));
                }
            }
        }

        if projected.len() < MIN_HINT_MATCHES {
            return SolveResult::failure(SolveStatus::NoMatch, elapsed_ms(t0));
        }

        // ── Initial centroid → catalog star matching ──
        // Match radius covers (a) hint angular uncertainty and (b) the LIS-equivalent
        // fractional match radius. Whichever is larger.
        let hint_match_radius =
            (config.hint_uncertainty_rad).max(config.match_radius * fov_rad);

        let initial_matches = find_centroid_matches(
            &centroid_vectors[..match_centroid_count.min(centroid_vectors.len())],
            &projected,
            hint_match_radius,
        );

        debug!(
            "Tracking: initial NN match → {} pairs (radius {:.1}″)",
            initial_matches.len(),
            hint_match_radius.to_degrees() * 3600.0
        );

        if initial_matches.len() < MIN_HINT_MATCHES {
            return SolveResult::failure(SolveStatus::NoMatch, elapsed_ms(t0));
        }

        // ── Wahba SVD on the initial correspondence set ──
        let (rotation_matrix, det_sign_ok) =
            wahba_svd_dynamic(&centroid_vectors, &candidate_vecs, &nearby_inds, &initial_matches);
        if !det_sign_ok {
            // Parity mismatch — bail (caller may still fall back to LIS).
            return SolveResult::failure(SolveStatus::NoMatch, elapsed_ms(t0));
        }

        // ── Verification (mirrors solve.rs verify step) ──
        let match_radius_rad = config.match_radius * fov_rad;
        let image_center_icrs =
            rotation_matrix.transpose() * Vector3::from_array([0.0, 0.0, 1.0]);
        let verify_inds = self
            .star_catalog
            .query_indices_from_uvec(image_center_icrs, fov_diagonal / 2.0);

        let mut verify_positions: Vec<(usize, f32, f32)> = Vec::new();
        for &cat_idx in &verify_inds {
            let raw = &self.star_vectors[cat_idx];
            let sv = match beta {
                Some(b) => aberration_correct(raw, &b),
                None => *raw,
            };
            let icrs_v = Vector3::from_array([sv[0], sv[1], sv[2]]);
            let cam_v = rotation_matrix * icrs_v;
            if cam_v[2] > 0.0 {
                verify_positions.push((cat_idx, cam_v[0] / cam_v[2], cam_v[1] / cam_v[2]));
            }
        }
        verify_positions.truncate(2 * match_centroid_count);
        let num_nearby = verify_positions.len();

        let verify_matches = find_centroid_matches(
            &centroid_vectors[..match_centroid_count.min(centroid_vectors.len())],
            &verify_positions,
            match_radius_rad,
        );
        let current_num_matches = verify_matches.len();

        // Same false-positive probability test as LIS, but without the
        // /num_patterns Bonferroni division (no pattern-hash trials happened).
        let prob_single = num_nearby as f64 * (config.match_radius as f64).powi(2);
        let prob_mismatch = binomial_cdf(
            (match_centroid_count as i64 - (current_num_matches as i64 - 2)).max(0) as u32,
            match_centroid_count as u32,
            1.0 - prob_single.min(1.0),
        );

        if prob_mismatch >= config.match_threshold {
            debug!(
                "Tracking: verification rejected (matches={}, prob={:.2e})",
                current_num_matches, prob_mismatch
            );
            return SolveResult::failure(SolveStatus::NoMatch, elapsed_ms(t0));
        }

        debug!(
            "Tracking: VERIFIED — {} matches, prob={:.2e}",
            current_num_matches, prob_mismatch
        );

        // ── WCS refinement (same path as LIS) ──
        let centroids_px: Vec<(f64, f64)> = sorted_indices
            .iter()
            .map(|&i| {
                let px = parity_sign as f64 * preprocessed[i].x as f64;
                let py = preprocessed[i].y as f64;
                (px, py)
            })
            .collect();

        // True pinhole pixel scale for wcs_refine (1/f).
        let ps_refine = pixel_scale as f64;

        let wcs_result = wcs_refine::wcs_refine(
            &rotation_matrix,
            &verify_matches,
            &centroids_px,
            &self.star_vectors,
            &self.star_catalog,
            ps_refine,
            parity_flip,
            match_radius_rad,
            match_centroid_count,
            10,
        );

        if wcs_result.matches.len() < MIN_HINT_MATCHES {
            return SolveResult::failure(SolveStatus::NoMatch, elapsed_ms(t0));
        }

        let (refined_rotation, refined_fov, _) = wcs_refine::wcs_to_rotation(
            &wcs_result.cd_matrix,
            wcs_result.crval_rad[0],
            wcs_result.crval_rad[1],
            config.image_width,
        );

        // ── Build matched IDs / residuals ──
        // True pinhole pixel scale derived from the angular `refined_fov`.
        let ps = {
            let f = (config.image_width.max(1) as f32 / 2.0) / (refined_fov / 2.0).tan();
            1.0 / f
        };
        let mut matched_cat_ids: Vec<i64> = Vec::with_capacity(wcs_result.matches.len());
        let mut matched_cent_inds: Vec<usize> = Vec::with_capacity(wcs_result.matches.len());
        let mut angular_residuals: Vec<f32> = Vec::with_capacity(wcs_result.matches.len());
        for &(cent_local_idx, cat_star_idx) in &wcs_result.matches {
            matched_cat_ids.push(self.star_catalog_ids[cat_star_idx]);
            matched_cent_inds.push(sorted_indices[cent_local_idx]);
            let (px, py) = centroids_px[cent_local_idx];
            let ix = px as f32 * ps;
            let iy = py as f32 * ps;
            let iz = 1.0f32;
            let norm = (ix * ix + iy * iy + iz * iz).sqrt();
            let img_v = refined_rotation.transpose()
                * Vector3::from_array([ix / norm, iy / norm, iz / norm]);
            let sv = &self.star_vectors[cat_star_idx];
            let cat_v = Vector3::from_array([sv[0], sv[1], sv[2]]);
            let cross = img_v.cross(&cat_v);
            let ang = cross.norm().atan2(img_v.dot(&cat_v));
            angular_residuals.push(ang);
        }
        angular_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let rmse = if angular_residuals.is_empty() {
            0.0
        } else {
            (angular_residuals.iter().map(|r| r * r).sum::<f32>() / angular_residuals.len() as f32)
                .sqrt()
        };
        let p90e = if angular_residuals.is_empty() {
            0.0
        } else {
            angular_residuals[(0.9 * (angular_residuals.len() - 1) as f32) as usize]
        };
        let max_err = angular_residuals.last().copied().unwrap_or(0.0);

        let quat = Quaternion::from_rotation_matrix(&refined_rotation);

        let mut result_cam = config.camera_model.clone();
        let refined_f = (config.image_width as f64 / 2.0) / (refined_fov as f64 / 2.0).tan();
        result_cam.focal_length_px = refined_f;
        result_cam.parity_flip = parity_flip;

        SolveResult {
            qicrs2cam: Some(quat),
            fov_rad: Some(refined_fov),
            num_matches: Some(wcs_result.matches.len() as u32),
            rmse_rad: Some(rmse),
            p90e_rad: Some(p90e),
            max_err_rad: Some(max_err),
            prob: Some(prob_mismatch),
            solve_time_ms: elapsed_ms(t0),
            status: SolveStatus::MatchFound,
            parity_flip,
            matched_catalog_ids: matched_cat_ids,
            matched_centroid_indices: matched_cent_inds,
            image_width: config.image_width,
            image_height: config.image_height,
            cd_matrix: Some(wcs_result.cd_matrix),
            crval_rad: Some(wcs_result.crval_rad),
            camera_model: Some(result_cam),
            theta_rad: Some(wcs_result.theta_rad),
        }
    }
}

/// Run Wahba SVD on a dynamic-sized correspondence set.
///
/// `centroid_vectors` is indexed by sorted (brightness) centroid index; the
/// match pairs reference this same index space. `candidate_vecs` is indexed
/// by position within `nearby_inds` — match pairs reference the catalog star
/// index, so we look it up.
///
/// Returns the rotation matrix and a flag indicating whether the determinant
/// is positive (true) or negative (false → likely parity mismatch).
fn wahba_svd_dynamic(
    centroid_vectors: &[[f32; 3]],
    candidate_vecs: &[[f32; 3]],
    nearby_inds: &[usize],
    matches: &[(usize, usize)],
) -> (Matrix3<f32>, bool) {
    // Build catalog-index → local-position map for candidate_vecs lookup.
    // Linear scan is fine — nearby_inds is typically <1000.
    let cat_to_local = |cat_idx: usize| -> Option<usize> {
        nearby_inds.iter().position(|&x| x == cat_idx)
    };

    // Collect paired vectors as Vec for find_rotation_matrix (which is generic
    // on N — we'd need a const N. Instead, build the cross-covariance directly).
    let mut h = numeris::Matrix3::<f64>::zeros();
    let mut n_pairs = 0u32;
    for &(cent_idx, cat_idx) in matches {
        let local_i = match cat_to_local(cat_idx) {
            Some(i) => i,
            None => continue,
        };
        let img = &centroid_vectors[cent_idx];
        let cat = &candidate_vecs[local_i];
        let img_v = numeris::Vector3::<f64>::from_array([img[0] as f64, img[1] as f64, img[2] as f64]);
        let cat_v = numeris::Vector3::<f64>::from_array([cat[0] as f64, cat[1] as f64, cat[2] as f64]);
        h = h + img_v.outer(&cat_v);
        n_pairs += 1;
    }

    if n_pairs < MIN_HINT_MATCHES as u32 {
        return (Matrix3::<f32>::zeros(), false);
    }

    let svd = h.svd().expect("SVD failed");
    let u = svd.u();
    let v_t = svd.vt();
    let r64 = *u * *v_t;
    let r = r64.cast::<f32>();
    let det_ok = r.det() > 0.0;
    (r, det_ok)
}

fn elapsed_ms(t0: Instant) -> f32 {
    t0.elapsed().as_secs_f32() * 1000.0
}
