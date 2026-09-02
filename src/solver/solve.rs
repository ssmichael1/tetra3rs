//! Plate solving: given image centroids and an approximate FOV, find the
//! camera pointing direction as a quaternion.
//!
//! The solve is a staged pipeline; this module is the driver and the tail:
//!
//! 1. **Preprocess** (`preprocess.rs`): CRPIX subtraction, undistortion,
//!    non-finite filtering, brightness ordering.
//! 2. **Hypothesis source** — one of:
//!    - lost-in-space **pattern search** (`pattern_search.rs`): 4-star
//!      edge-ratio hash lookups over an FOV sweep, Wahba SVD per match;
//!    - **tracking** (`track.rs`): correspondences from an attitude hint.
//! 3. **Verify** (`verify.rs`): project nearby catalog stars, match, score
//!    the match count against a coincidence model.
//! 4. **Refine + finalize** (here, with `wcs_refine.rs`): constrained WCS
//!    refinement on the verified matches, then `Solution` assembly.
//!
//! The lost-in-space acceptance stage (`accept_lis_candidate`) glues 3 and
//! 4 together per candidate: pre-gate, refine, re-verify the refined
//! attitude, and apply the sequential multiple-comparison correction.

use super::clock::Instant;

use numeris::{Matrix3, Quaternion, Vector3};
use tracing::{debug, warn};

use crate::Centroid;

use super::matching;
use super::pattern::PATTERN_SIZE;
use super::pattern_search::{Hypothesis, PatternSearch};
use super::preprocess::{
    centroid_sigma_px, preprocess, sort_indices_by_brightness, unit_vector_from_pixels,
    CentroidVectors,
};
use super::verify::VerifyStage;
use super::wcs_refine;
use super::{
    pixel_scale_from_fov, Solution, SolveConfig, SolveFailure, SolveResult, SolveStatus,
    SolverDatabase,
};

#[cfg(feature = "profile")]
use crate::solver::profiling::{self, buckets};

/// Floor on the refined-stage verification σ, in pixels.
const REFINED_SIGMA_FLOOR_PX: f32 = 0.5;

/// Speed of light in km/s.
pub(super) const C_KM_S: f64 = 299_792.458;

/// Catalog unit vectors as seen by one solve: the database's stored ICRS
/// vectors, optionally aberration-corrected for the observer's velocity.
///
/// The correction is applied *on access* rather than by copying the whole
/// catalog per solve: a solve touches a few hundred stars (the pattern
/// candidates, the verification cone, the refinement's re-association
/// set), while the catalog holds tens or hundreds of thousands, so the
/// former copy cost more than the solve itself whenever
/// `observer_velocity_km_s` was set. Each access applies exactly the
/// per-star formula the copy did, so results are bit-identical.
#[derive(Clone, Copy)]
pub(crate) struct StarVectors<'a> {
    base: &'a [[f32; 3]],
    /// Observer velocity / c in ICRS, when aberration is enabled.
    beta: Option<[f64; 3]>,
}

impl<'a> StarVectors<'a> {
    /// Vectors corrected for the observer's barycentric velocity (km/s,
    /// ICRS); `None` leaves them uncorrected.
    pub(crate) fn with_observer_velocity(
        base: &'a [[f32; 3]],
        velocity_km_s: Option<[f64; 3]>,
    ) -> Self {
        Self {
            base,
            beta: velocity_km_s.map(|v| [v[0] / C_KM_S, v[1] / C_KM_S, v[2] / C_KM_S]),
        }
    }

    /// Apparent unit vector of catalog star `idx`.
    #[inline]
    pub(crate) fn get(&self, idx: usize) -> [f32; 3] {
        match &self.beta {
            Some(beta) => aberration_correct(&self.base[idx], beta),
            None => self.base[idx],
        }
    }
}

/// Classical stellar aberration: true ICRS unit vector → apparent.
///
/// `beta` = observer barycentric velocity / c (dimensionless, ICRS frame).
/// Formula: `s' = (s + β) / |s + β|`.
///
/// This is the exact classical (non-relativistic) result: a photon with true
/// direction `−s` in the rest frame has apparent velocity `−c·s − v` in a
/// frame moving at `v = c·β`, so it appears to arrive from direction
/// `s + β` (then renormalized to unit length). The relativistic correction
/// is `O(β²)`, giving ~2 mas for Earth's orbital β ≈ 10⁻⁴ — well below
/// plate-solve precision.
pub(super) fn aberration_correct(sv: &[f32; 3], beta: &[f64; 3]) -> [f32; 3] {
    let ax = sv[0] as f64 + beta[0];
    let ay = sv[1] as f64 + beta[1];
    let az = sv[2] as f64 + beta[2];
    let norm = (ax * ax + ay * ay + az * az).sqrt();
    [(ax / norm) as f32, (ay / norm) as f32, (az / norm) as f32]
}

// ── Solve entry point ───────────────────────────────────────────────────────

/// Per-solve state of the lost-in-space acceptance stage, shared across
/// every candidate the pattern search produces.
struct LisContext<'a> {
    config: &'a SolveConfig,
    /// Preprocessed centroids (pixels; CRPIX-subtracted, undistorted).
    centroids: &'a [Centroid],
    /// Brightness-sorted centroid index order.
    sorted_indices: &'a [usize],
    /// Catalog unit vectors as seen by this solve.
    star_vectors: StarVectors<'a>,
    /// Number of brightest centroids the verification tests.
    match_centroid_count: usize,
    /// Per-centroid position σ in pixels (brightness order; 0 = unknown).
    sigma_px: Vec<f32>,
    t0: Instant,
    /// Candidate attitudes verified so far — the divisor of the sequential
    /// multiple-comparison correction in the acceptance test.
    candidates_tested: u64,
    /// Matching working buffers, reused across candidates (see
    /// `verify_attitude` / `find_centroid_matches`).
    match_xy: Vec<(f32, f32)>,
    match_scratch: matching::MatchScratch<f32>,
    /// Scratch for re-verify vectors rebuilt at the refined scale.
    reverify_buf: Vec<[f32; 3]>,
}

impl SolverDatabase {
    /// Solve for the camera pointing direction given image centroids.
    ///
    /// Centroids should have the `mass` field populated for brightness sorting.
    /// Centroid (x, y) are in pixel coordinates with (0, 0) at the image center.
    /// +X points right, +Y points down in the image.
    ///
    /// The `SolveConfig`'s camera model supplies all camera geometry: the FOV
    /// estimate (from its focal length and image width), the image dimensions,
    /// optical center, parity, and distortion. Use [`SolveConfig::new`] to
    /// build one from a FOV estimate and image dimensions.
    ///
    /// If `fov_max_error_rad` is set, the solver sweeps FOV values across the range
    /// `[fov_estimate - fov_max_error, fov_estimate + fov_max_error]`, trying the
    /// exact estimate first, then spiraling outward. This makes the solver robust
    /// to uncertain FOV estimates.
    ///
    /// Lost-in-space (no attitude hint) requires **at least 5 centroids**: 4
    /// form the geometric-hash pattern and at least one more is needed as
    /// independent verification evidence (the 4 pattern stars match by
    /// construction and are excluded from the acceptance statistic). A field of
    /// 4 or fewer finite centroids returns [`SolveStatus::TooFew`]. The
    /// tracking path (`SolveConfig::attitude_hint`) has no such floor.
    ///
    /// Returns a `SolveResult` with the ICRS→camera quaternion on success.
    pub fn solve_from_centroids(
        &self,
        centroids: &[Centroid],
        config: &SolveConfig,
    ) -> SolveResult {
        let t0 = Instant::now();

        // A config that cannot produce a meaningful solve — the
        // `SolveConfig::default()` placeholder camera model (zero image size),
        // NaN match parameters that silently disable all matching, etc. —
        // fails fast with `InvalidConfig` instead of burning the full search
        // to a guaranteed NoMatch.
        if let Err(e) = config.validate() {
            warn!("solve_from_centroids: {e}");
            return failure(SolveStatus::InvalidConfig, t0);
        }
        let cam = &config.camera_model;

        // Catalog vectors for this solve, aberration-corrected on access when
        // an observer velocity is set (see `StarVectors`).
        let star_vecs =
            StarVectors::with_observer_velocity(&self.star_vectors, config.observer_velocity_km_s);

        let pre = preprocess(centroids, cam);
        let working_centroids: &[Centroid] = &pre.centroids;
        let orig_indices = &pre.orig_indices;

        // ── Tracking-mode shortcut: if a hint is provided, try direct correspondence first ──
        if let Some(ref hint) = config.attitude_hint {
            match self.solve_with_hint(working_centroids, star_vecs, config, hint, t0) {
                Ok(mut solution) => {
                    remap_matched_indices(&mut solution, orig_indices);
                    debug!(
                        "Hinted solve succeeded in {:.1} ms ({} matches)",
                        solution.solve_time_ms, solution.num_matches
                    );
                    return Ok(solution);
                }
                Err(fail) => {
                    if config.strict_hint {
                        debug!("Hinted solve failed and strict_hint is set — returning failure");
                        return Err(fail);
                    }
                    debug!("Hinted solve failed; falling back to lost-in-space");
                }
            }
        }

        // LIS needs at least PATTERN_SIZE + 1 centroids. PATTERN_SIZE form the
        // 4-star hypothesis pattern; verification excludes those hypothesis
        // stars from the binomial (they match by construction — zero
        // independent evidence), so a field of exactly PATTERN_SIZE centroids
        // leaves zero verification trials and can never pass acceptance at any
        // `match_threshold`. Gate it here as TooFew rather than letting it burn
        // the whole FOV sweep only to return a silent NoMatch. (The tracking
        // path, tried above, has no such floor — its hypothesis comes from the
        // hint, not the centroids.) This is FOV-independent, unlike the
        // post-thinning TooFew inside the pattern search, so it ends the solve.
        if working_centroids.len() <= PATTERN_SIZE {
            return failure(SolveStatus::TooFew, t0);
        }

        // Sort centroids by brightness. FOV-independent, so computed once for
        // the whole search.
        let sorted_indices = sort_indices_by_brightness(working_centroids);
        // Trim the verification set to the database's per-FOV star budget.
        let match_centroid_count = working_centroids
            .len()
            .min(self.props.verification_stars_per_fov as usize);

        let mut ctx = LisContext {
            config,
            centroids: working_centroids,
            sorted_indices: &sorted_indices,
            star_vectors: star_vecs,
            match_centroid_count,
            sigma_px: centroid_sigma_px(working_centroids, &sorted_indices),
            t0,
            candidates_tested: 0,
            match_xy: Vec::new(),
            match_scratch: matching::MatchScratch::<f32>::default(),
            reverify_buf: Vec::new(),
        };
        let mut search = PatternSearch::new(
            self,
            working_centroids,
            &sorted_indices,
            config,
            star_vecs,
            t0,
        );
        let mut solution = search.run(&mut |h| self.accept_lis_candidate(h, &mut ctx))?;
        remap_matched_indices(&mut solution, orig_indices);
        Ok(solution)
    }

    /// Acceptance stage for a lost-in-space [`Hypothesis`]: verify at the
    /// search radius, pre-gate, refine, re-verify the refined attitude at a
    /// tightened radius, and apply the sequential multiple-comparison
    /// correction. Returns the finished [`Solution`] when the candidate is
    /// accepted, `None` to let the search continue.
    fn accept_lis_candidate(
        &self,
        h: &Hypothesis<'_>,
        ctx: &mut LisContext<'_>,
    ) -> Option<Solution> {
        let config = ctx.config;
        let centroids = ctx.centroids;
        let sorted_indices = ctx.sorted_indices;
        let star_vectors = ctx.star_vectors;
        let match_centroid_count = ctx.match_centroid_count;
        let t0 = ctx.t0;
        let fov = h.fov;

        let (current_matches, prob_mismatch) = self.verify_attitude(
            &h.rotation,
            h.vectors,
            match_centroid_count,
            fov,
            config,
            star_vectors,
            &h.pattern,
            &ctx.sigma_px,
            VerifyStage::Coarse,
            &mut ctx.match_xy,
            &mut ctx.match_scratch,
        );

        // ── Pre-gate ──
        // Every verified candidate consumes one unit of the
        // multiple-comparison budget (`candidates_tested` is the
        // sequential-Bonferroni divisor of the acceptance test
        // below). The pre-gate itself is deliberately loose: its
        // only job is to keep hopeless candidates away from the
        // (relatively expensive) refinement — acceptance is
        // decided by re-verifying the *refined* attitude at a
        // tightened radius, which separates true from false
        // candidates far more sharply than this coarse-radius
        // p-value can. Hostile-but-solvable fields depend on the
        // slack: e.g. a galaxy-cluster frame where only ~8 of the
        // 50 brightest centroids are catalog stars scores ~1e-2
        // here yet re-verifies decisively once refined. The
        // ceiling never tightens below the user's budget.
        const PREGATE_CEILING: f64 = 1e-2;
        ctx.candidates_tested += 1;
        if prob_mismatch >= config.match_threshold.max(PREGATE_CEILING) {
            return None;
        }

        debug!(
            "Candidate {}: {} matches, p={:.2e}, fov={:.3}° — refining",
            ctx.candidates_tested,
            current_matches.len(),
            prob_mismatch,
            fov.to_degrees()
        );

        // ── WCS TAN-projection refinement ──
        // The refinement locks its pixel scale to the
        // pattern-match refined FOV, NOT the camera model's focal
        // length — deliberately asymmetric with the tracking path
        // (which trusts the model's 1/f). Lost-in-space must stay
        // robust to a wrong focal-length estimate; the pattern
        // match measures the true scale, and the model's f is only
        // a search seed here. Fewer than 4 surviving matches → try
        // next candidate.
        let ps_fov = pixel_scale_from_fov(config.image_width(), fov as f64);
        let mut result = self.refine_and_finalize(
            &h.rotation,
            &current_matches,
            centroids,
            sorted_indices,
            star_vectors,
            config,
            h.vectors.parity_flip,
            fov,
            ps_fov,
            match_centroid_count,
            4,
            prob_mismatch,
            t0,
        )?;

        // ── Post-refinement verification (the acceptance test) ──
        // A 4-star SVD attitude can be several match-radii off at
        // the field edges (small quads amplify centroid noise), so
        // a *true* candidate in a dense field may verify weakly
        // (e.g. 14 of 50) — indistinguishable from a lucky false
        // candidate by p-value alone. Re-verify with the refined
        // attitude: the 3-DOF fit converges to the true attitude
        // from those matches and the match count jumps (its
        // p-value collapses by tens of orders), while a false
        // candidate's coincidences cannot be aligned by 3 DOF and
        // its p-value stays ~1. Acceptance applies the sequential
        // Bonferroni correction over all candidates tested this
        // solve: p·k < match_threshold. The total false-accept
        // probability of a full search is bounded by
        // match_threshold·ln(k), while early candidates face a
        // threshold 5-7 orders looser than the previous
        // `/ num_patterns` over-correction, which made clean
        // sparse fields (< ~7 stars) mathematically unsolvable.
        let refined_rotation = wcs_refine::rotation_from_theta_crval(
            result.theta_rad,
            result.crval_rad[0],
            result.crval_rad[1],
        );
        // Re-verify at the refined fit's own residual scale instead of the
        // (coarse) search radius: true matches sit within a few RMSE of the
        // refined attitude, while a false candidate's coincidences are
        // uniform across the search radius, so the tighter σ multiplies each
        // match's evidence by (search/refined)² — this is what separates a
        // true 14-of-50 dense-field candidate (many bright centroids simply
        // absent from the catalog) from a lucky false one. σ is floored at
        // half a pixel (the refinement's own adaptive-radius floor is
        // 2.5 px ≈ 5σ).
        // The refinement was locked to the candidate's measured
        // scale (`ps_fov`), but the hypothesis vectors may still
        // sit at the *swept* scale when the search skipped its
        // rebuild (mismatch ≤ 0.25·match_radius). That slack is
        // harmless at the coarse search radius, not at the
        // tightened re-verify σ below (floored at half a pixel):
        // the edge residual ε·r reaches ~7 px on a 4096 px frame,
        // silently dropping true edge matches from the acceptance
        // statistic. The typed vectors carry their scale, so
        // rebuild exactly when it differs — only pre-gated
        // candidates reach here, so the cost is negligible.
        let reverify_vectors: CentroidVectors<'_> = if h.vectors.pixel_scale == ps_fov as f32 {
            h.vectors
        } else {
            let sign = if h.vectors.parity_flip { -1.0f32 } else { 1.0 };
            ctx.reverify_buf.clear();
            ctx.reverify_buf.extend(
                sorted_indices
                    .iter()
                    .map(|&i| unit_vector_from_pixels(&centroids[i], ps_fov as f32, sign)),
            );
            CentroidVectors {
                pixel_scale: ps_fov as f32,
                parity_flip: h.vectors.parity_flip,
                data: &ctx.reverify_buf,
            }
        };
        let ps_refined = (1.0 / result.camera_model.focal_length_px) as f32;
        let sigma_refined = result.rmse_rad.max(REFINED_SIGMA_FLOOR_PX * ps_refined);
        let (refined_matches, p_refined) = self.verify_attitude(
            &refined_rotation,
            reverify_vectors,
            match_centroid_count,
            result.fov_rad,
            config,
            star_vectors,
            &h.pattern,
            &ctx.sigma_px,
            VerifyStage::Refined {
                sigma_rad: sigma_refined,
            },
            &mut ctx.match_xy,
            &mut ctx.match_scratch,
        );
        let corrected_prob = p_refined * ctx.candidates_tested as f64;
        if corrected_prob >= config.match_threshold {
            debug!(
                "Candidate {} rejected after refinement: {} → {} matches, corrected p={:.2e}",
                ctx.candidates_tested,
                current_matches.len(),
                refined_matches.len(),
                corrected_prob,
            );
            return None;
        }

        debug!(
            "MATCH: {} verified matches (candidate {}), corrected p={:.2e}, fov={:.3}°",
            refined_matches.len(),
            ctx.candidates_tested,
            corrected_prob,
            result.fov_rad.to_degrees()
        );
        result.prob = corrected_prob;
        Some(result)
    }

    /// Run the WCS refinement on a verified match set and assemble the final
    /// [`SolveResult`].
    ///
    /// Shared by the lost-in-space and tracking paths: builds the
    /// parity-applied pixel coordinate list, runs the constrained WCS
    /// refinement, and finalizes. Returns `None` when refinement keeps fewer
    /// than `min_matches` stars — LIS treats that as "try the next
    /// candidate"; tracking treats it as a failed hint.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn refine_and_finalize(
        &self,
        rotation_matrix: &Matrix3<f32>,
        verify_matches: &[(usize, usize)],
        centroids: &[Centroid],
        sorted_indices: &[usize],
        star_vectors: StarVectors<'_>,
        config: &SolveConfig,
        parity_flip: bool,
        fov: f32,
        pixel_scale: f64,
        match_centroid_count: usize,
        min_matches: usize,
        prob: f64,
        t0: Instant,
    ) -> Option<Solution> {
        // Build pixel coordinates: centroids are already CRPIX-subtracted and
        // undistorted. Apply the detected parity.
        let parity_sign: f64 = if parity_flip { -1.0 } else { 1.0 };
        let centroids_px: Vec<(f64, f64)> = sorted_indices
            .iter()
            .map(|&i| (parity_sign * centroids[i].x as f64, centroids[i].y as f64))
            .collect();

        let match_radius_rad = config.match_radius * fov;

        #[cfg(feature = "profile")]
        profiling::count(buckets::WCS_REFINE, 1);
        let wcs_result = timed!(
            buckets::WCS_REFINE,
            wcs_refine::wcs_refine(
                rotation_matrix,
                verify_matches,
                &centroids_px,
                star_vectors,
                &self.star_catalog,
                pixel_scale,
                parity_flip,
                match_radius_rad,
                match_centroid_count,
                10,
            )
        );

        if wcs_result.matches.len() < min_matches {
            return None;
        }

        Some(self.finalize_solve_result(
            &wcs_result,
            star_vectors,
            sorted_indices,
            &centroids_px,
            config,
            parity_flip,
            prob,
            t0,
        ))
    }

    /// Assemble a [`Solution`] from a completed WCS refinement.
    ///
    /// Shared by the lost-in-space (`accept_lis_candidate`) and tracking
    /// (`solve_with_hint`) paths. `star_vectors` is the (possibly
    /// aberration-corrected) catalog unit-vector slice; `prob` is the caller's
    /// false-positive probability estimate. The match set, residual statistics,
    /// quaternion, and camera model are derived from `wcs_result`.
    #[allow(clippy::too_many_arguments)]
    fn finalize_solve_result(
        &self,
        wcs_result: &wcs_refine::WcsRefineResult,
        star_vectors: StarVectors<'_>,
        sorted_indices: &[usize],
        centroids_px: &[(f64, f64)],
        config: &SolveConfig,
        parity_flip: bool,
        prob: f64,
        t0: Instant,
    ) -> Solution {
        // Derive the rotation directly from the constrained-fit parameters
        // (θ, CRVAL). The pixel scale was locked during refinement, so it is
        // the exact scale of the solution — no CD-matrix decomposition
        // needed. θ describes the parity-applied working frame, so the
        // rotation is proper regardless of `parity_flip`; the residual loop
        // below consistently uses the parity-applied `centroids_px`.
        let refined_rotation = wcs_refine::rotation_from_theta_crval(
            wcs_result.theta_rad,
            wcs_result.crval_rad[0],
            wcs_result.crval_rad[1],
        );
        let ps = wcs_result.pixel_scale as f32;
        let refined_fov =
            (2.0 * ((wcs_result.pixel_scale * config.image_width() as f64) / 2.0).atan()) as f32;

        // Build matched catalog IDs, centroid indices, and angular residuals.
        let mut matched_cat_ids: Vec<i64> = Vec::with_capacity(wcs_result.matches.len());
        let mut matched_cent_inds: Vec<usize> = Vec::with_capacity(wcs_result.matches.len());
        let mut angular_residuals: Vec<f32> = Vec::with_capacity(wcs_result.matches.len());
        for &(cent_local_idx, cat_star_idx) in &wcs_result.matches {
            matched_cat_ids.push(self.star_catalog_ids[cat_star_idx]);
            matched_cent_inds.push(sorted_indices[cent_local_idx]);
            // Compute angular residual using rotation matrix
            let (px, py) = centroids_px[cent_local_idx];
            let ix = px as f32 * ps;
            let iy = py as f32 * ps;
            let iz = 1.0f32;
            let norm = (ix * ix + iy * iy + iz * iz).sqrt();
            let img_v = refined_rotation.transpose()
                * Vector3::from_array([ix / norm, iy / norm, iz / norm]);
            let sv = star_vectors.get(cat_star_idx);
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

        // Convert rotation to quaternion
        let quat = Quaternion::from_rotation_matrix(&refined_rotation);

        // Build result camera model: copy the input model (which carries the
        // image dimensions, CRPIX, and distortion), then update the focal
        // length from the refinement's locked pixel scale and record the
        // detected parity.
        let mut result_cam = config.camera_model.clone();
        result_cam.focal_length_px = 1.0 / wcs_result.pixel_scale;
        result_cam.parity_flip = parity_flip;

        Solution {
            qicrs2cam: quat,
            fov_rad: refined_fov,
            num_matches: wcs_result.matches.len() as u32,
            rmse_rad: rmse,
            p90e_rad: p90e,
            max_err_rad: max_err,
            prob,
            solve_time_ms: elapsed_ms(t0),
            attitude_cov_rad2: wcs_result.covariance,
            parity_flip,
            observer_velocity_km_s: config.observer_velocity_km_s,
            matched_catalog_ids: matched_cat_ids,
            matched_centroid_indices: matched_cent_inds,
            cd_matrix: wcs_result.cd_matrix,
            crval_rad: wcs_result.crval_rad,
            camera_model: result_cam,
            theta_rad: wcs_result.theta_rad,
        }
    }
}

// ── Helper functions ────────────────────────────────────────────────────────

pub(super) fn elapsed_ms(t0: Instant) -> f32 {
    t0.elapsed().as_secs_f32() * 1000.0
}

/// Translate a solution's `matched_centroid_indices` from the compacted
/// working-centroid frame back into the caller's original input slice.
///
/// The solve pipeline may drop non-finite centroids up front, which shifts
/// every index at or beyond the drop point. `orig_indices[i]` is the caller
/// index that working centroid `i` came from; without this remap, a caller
/// (e.g. `calibrate_camera`) would pair the wrong observed positions with
/// catalog stars. When nothing was dropped the map is the identity.
fn remap_matched_indices(solution: &mut Solution, orig_indices: &[usize]) {
    for idx in solution.matched_centroid_indices.iter_mut() {
        *idx = orig_indices[*idx];
    }
}

/// Build a failed [`SolveResult`] with the elapsed time since `t0`.
pub(super) fn failure(status: SolveStatus, t0: Instant) -> SolveResult {
    Err(SolveFailure {
        status,
        solve_time_ms: elapsed_ms(t0),
    })
}

/// Compute the least-squares rotation matrix from paired image/catalog unit
/// vectors (Wahba's problem).
///
/// Uses SVD of the cross-covariance matrix H = Σ(img_i ⊗ cat_i).
/// The resulting R satisfies: camera_vec ≈ R * icrs_vec.
///
/// The SVD is computed in f64 for precision, then the result is converted back
/// to f32. Returns `None` if the SVD fails (degenerate cross-covariance from
/// pathological input vectors). Serves both the fixed-size 4-star LIS pattern
/// and the tracking path's dynamic correspondence sets.
pub(super) fn wahba_rotation(
    pairs: impl IntoIterator<Item = ([f32; 3], [f32; 3])>,
) -> Option<Matrix3<f32>> {
    let mut h = numeris::Matrix3::<f64>::zeros();
    for (img, cat) in pairs {
        let img_v =
            numeris::Vector3::<f64>::from_array([img[0] as f64, img[1] as f64, img[2] as f64]);
        let cat_v =
            numeris::Vector3::<f64>::from_array([cat[0] as f64, cat[1] as f64, cat[2] as f64]);
        h += img_v.outer(&cat_v);
    }

    let svd = h.svd().ok()?;
    let u = svd.u();
    let v_t = svd.vt();
    let r64 = *u * *v_t;
    Some(r64.cast::<f32>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aberration_correct_shift_direction() {
        // Star along +X, velocity 30 km/s along +Y
        // Aberration should shift the apparent position toward +Y
        let star = [1.0f32, 0.0, 0.0];
        let beta = [0.0, 30.0 / C_KM_S, 0.0];
        let apparent = aberration_correct(&star, &beta);

        // Output should be normalized
        let norm = (apparent[0] as f64 * apparent[0] as f64
            + apparent[1] as f64 * apparent[1] as f64
            + apparent[2] as f64 * apparent[2] as f64)
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "output not unit length: {norm}");

        // Y component should be positive (shifted toward velocity direction)
        assert!(
            apparent[1] > 0.0,
            "expected positive Y shift, got {}",
            apparent[1]
        );

        // Shift magnitude should be ~v/c ≈ 1e-4 rad ≈ 20"
        let shift_rad = (apparent[1] as f64).atan2(apparent[0] as f64);
        let expected = 30.0 / C_KM_S; // ~1e-4 rad
        assert!(
            (shift_rad - expected).abs() < 1e-6,
            "shift {shift_rad:.2e} rad, expected ~{expected:.2e} rad"
        );
    }

    #[test]
    fn test_aberration_correct_zero_velocity() {
        // Zero velocity should return the original unit vector unchanged
        let s = 1.0f32 / 3.0f32.sqrt();
        let star = [s, s, s];
        let beta = [0.0, 0.0, 0.0];
        let apparent = aberration_correct(&star, &beta);
        for i in 0..3 {
            assert!(
                (apparent[i] - star[i]).abs() < 1e-6,
                "component {i} changed: {} -> {}",
                star[i],
                apparent[i]
            );
        }
    }

    #[test]
    fn test_aberration_correct_parallel_velocity() {
        // Velocity parallel to star direction should produce zero transverse shift
        let star = [1.0f32, 0.0, 0.0];
        let beta = [30.0 / C_KM_S, 0.0, 0.0];
        let apparent = aberration_correct(&star, &beta);

        // Y and Z should remain essentially zero
        assert!(apparent[1].abs() < 1e-7, "Y not zero: {}", apparent[1]);
        assert!(apparent[2].abs() < 1e-7, "Z not zero: {}", apparent[2]);
        // X should still be ~1.0 (normalized)
        assert!((apparent[0] - 1.0).abs() < 1e-6);
    }
}
