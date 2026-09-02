//! Attitude verification: project nearby catalog stars through a candidate
//! rotation, match them to image centroids, and score the field with a
//! per-star likelihood ratio against a measured-density coincidence model.
//!
//! Shared by every hypothesis source (lost-in-space pattern search and
//! hinted tracking). This is the stage that decides, per candidate, whether
//! the evidence justifies refinement and acceptance.
//!
//! ## The statistic
//!
//! Every tested centroid `i` (the brightest `match_centroid_count`, minus
//! the stars that formed the hypothesis) contributes a likelihood ratio
//! between "this attitude is right" (H₁) and "it is wrong" (H₀):
//!
//! - Under H₀ a centroid is unrelated to the predicted catalog positions:
//!   uniform over the frame region, density `1/A`.
//! - Under H₁ it is either the image of one of the `n` predicted catalog
//!   stars (probability `q`, position error Gaussian with per-star σᵢ) or a
//!   detection with no catalog counterpart (probability `1 − q`, uniform):
//!   density `q · (1/n) · Σⱼ N₂(xᵢ − pⱼ; σᵢ) + (1 − q)/A`.
//!
//! With the nearest prediction dominating the sum, the ratio for a centroid
//! matched at distance `r` is `q · A/(n·2πσᵢ²) · exp(−r²/2σᵢ²) + (1 − q)`,
//! and for an unmatched centroid `1 − q`. The product Λ over the tested
//! centroids has expectation 1 under H₀ (each factor is a true likelihood
//! ratio against the H₀ density), so by Markov's inequality
//! `P_H₀(Λ ≥ t) ≤ 1/t`: **`1/Λ` is a valid p-value**, and the callers keep
//! treating it as one (pre-gate, sequential multiple-comparison correction
//! against `match_threshold`).
//!
//! `q` — the fraction of detections that are catalog stars — is not known
//! a priori (a galaxy-cluster frame has bright detections that are not
//! stars; a deep camera has faint ones below the catalog cutoff), so it is
//! estimated from the field itself, separately for the *bright* centroids
//! (brightness rank below the number of catalog stars predicted in frame,
//! which should nearly all be catalog stars) and the *faint* remainder. A
//! bright detection with no counterpart then costs `log(1 − q_bright)`,
//! which is severe in a clean deep field and mild where the bright
//! detections are demonstrably not stars, while faint misses are always
//! cheap. Fitting two fractions from the data inflates Λ slightly; a BIC
//! penalty of `½·ln(n_tested)` nats per fitted fraction is subtracted so the
//! p-value stays conservative.
//!
//! σᵢ combines the stage's expected residual (the coarse 4-star attitude is
//! off by up to the match radius at the frame edges; the refined attitude by
//! its RMSE) with the centroid's own covariance when the extractor supplies
//! one.

use numeris::{Matrix3, Vector3};

use super::matching;
use super::preprocess::CentroidVectors;
use super::{pixel_scale_from_fov, SolveConfig, SolverDatabase};

#[cfg(feature = "profile")]
use crate::solver::profiling::{self, buckets};

/// Which attitude estimate is being verified — sets the expected residual
/// scale σ and the matching cutoff radius.
#[derive(Clone, Copy)]
pub(super) enum VerifyStage {
    /// A search-stage attitude (4-star SVD or a tracking hint): residuals
    /// are dominated by attitude error, up to the search radius
    /// `match_radius · fov` at the frame edges. Cutoff = that radius,
    /// σ = radius / [`COARSE_SIGMA_DIVISOR`].
    Coarse,
    /// A refined attitude: residuals are the fit's own scatter.
    /// σ = `sigma_rad`, cutoff = `SIGMA_CUTOFF · σ` (never beyond the
    /// coarse radius).
    Refined { sigma_rad: f32 },
}

/// Coarse-stage σ as a fraction of the search radius: a 4-star attitude's
/// residual grows linearly toward the frame edge, so the radius is roughly
/// a 2.5σ bound on it.
const COARSE_SIGMA_DIVISOR: f32 = 2.5;
/// Matching cutoff in units of σ for the refined stage.
const SIGMA_CUTOFF: f32 = 5.0;
/// Bounds on the estimated catalog-star fractions, so a class with every
/// (or no) member matched cannot drive a factor to 0 or ∞.
const Q_MIN: f64 = 0.05;
const Q_MAX: f64 = 0.95;

/// `ln(exp(a) + exp(b))` without overflow.
#[inline]
fn log_add_exp(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    // Below ~1e-16 relative the smaller term vanishes in f64 anyway; skip the
    // exp/ln_1p (the common case for a clean match against a tiny σ).
    if hi - lo > 37.0 {
        return hi;
    }
    hi + (lo - hi).exp().ln_1p()
}

impl SolverDatabase {
    /// Verify a candidate attitude by projecting nearby catalog stars into
    /// the camera frame, greedily matching them to image centroids, and
    /// scoring the field with the per-star likelihood ratio described in the
    /// module docs.
    ///
    /// Shared by the lost-in-space and tracking paths. `vectors` must be
    /// brightness-sorted with parity already applied; `star_vectors`
    /// is the (possibly aberration-corrected) catalog unit-vector slice. The
    /// cone query itself uses the stored raw vectors, which are bit-identical
    /// to `Star::uvec()` and aligned with the catalog, so the candidate set
    /// is unchanged by aberration.
    ///
    /// Returns the matches `(centroid_local_idx, catalog_star_idx)` and the
    /// false-positive probability bound `1/Λ` (a per-candidate p-value,
    /// before any multiple-comparison correction by the caller).
    ///
    /// `exclude` lists the *local* (brightness-rank) indices of the centroids
    /// that formed the attitude hypothesis (LIS passes its 4 pattern stars;
    /// tracking passes none — its hint is independent of the centroids).
    /// Those stars match by construction — with the measured-FOV rebuild
    /// they fit near-exactly, handing every ratio-passing candidate "free"
    /// matches — so they are left out of the statistic entirely.
    /// `sigma_px` is each tested centroid's own position uncertainty in
    /// pixels (brightness order; `0` when unknown), folded into σᵢ.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn verify_attitude(
        &self,
        rotation_matrix: &Matrix3<f32>,
        vectors: CentroidVectors<'_>,
        match_centroid_count: usize,
        fov: f32,
        config: &SolveConfig,
        star_vectors: &[[f32; 3]],
        exclude: &[usize],
        sigma_px: &[f32],
        stage: VerifyStage,
        centroid_xy: &mut Vec<(f32, f32)>,
        scratch: &mut matching::MatchScratch<f32>,
    ) -> (Vec<(usize, usize)>, f64) {
        let fov_diagonal = fov * diagonal_factor(config);
        let search_radius = config.match_radius * fov;
        let (sigma_rad, match_radius_rad) = match stage {
            VerifyStage::Coarse => (search_radius / COARSE_SIGMA_DIVISOR, search_radius),
            VerifyStage::Refined { sigma_rad } => {
                (sigma_rad, (SIGMA_CUTOFF * sigma_rad).min(search_radius))
            }
        };

        // Find catalog stars within the diagonal FOV
        let image_center_icrs = rotation_matrix.transpose() * Vector3::from_array([0.0, 0.0, 1.0]);
        let nearby_inds = timed!(
            buckets::VERIFY_QUERY,
            self.star_catalog.query_indices_from_uvec_cached(
                image_center_icrs,
                fov_diagonal / 2.0,
                &self.star_vectors,
            )
        );
        #[cfg(feature = "profile")]
        profiling::count(buckets::VERIFY_QUERY_STARS, nearby_inds.len() as u64);

        // Project catalog stars to camera frame; keep stars in front (z > 0).
        let mut nearby_cam_positions: Vec<(usize, f32, f32)> = Vec::new();
        for &cat_idx in &nearby_inds {
            let sv = &star_vectors[cat_idx];
            let icrs_v = Vector3::from_array([sv[0], sv[1], sv[2]]);
            let cam_v = *rotation_matrix * icrs_v;
            if cam_v[2] > 0.0 {
                nearby_cam_positions.push((cat_idx, cam_v[0] / cam_v[2], cam_v[1] / cam_v[2]));
            }
        }
        // Limit to 2x the number of image centroids (like tetra3)
        nearby_cam_positions.truncate(2 * match_centroid_count);

        // Match image centroids to projected catalog stars
        let matches = timed!(
            buckets::VERIFY_MATCH,
            find_centroid_matches(
                &vectors.data[..match_centroid_count.min(vectors.data.len())],
                &nearby_cam_positions,
                match_radius_rad,
                centroid_xy,
                scratch,
            )
        );

        // ── Coincidence model ──
        // The prediction density is *measured* rather than modeled: count the
        // predictions that could match an in-frame centroid (inside the frame
        // plus a match-radius margin) and divide by that region's area, all
        // in tangent-plane units. Measuring the density avoids two opposite
        // errors that both showed up empirically: upstream tetra3's
        // `num_nearby·mr²` (no π, cone counted as if spread over the frame)
        // under-predicts the coincidence rate ~2-3× per match — false
        // dense-field candidates scored as near-certainties — while
        // π·num_nearby·mr² over-predicts it by the cone/frame area ratio,
        // rejecting true candidates in catalog-saturated fields where the
        // expected coincidence count is a large fraction of the centroids.
        let half_x = ((fov as f64) / 2.0).tan();
        let half_y = half_x * (config.image_height() as f64 / config.image_width().max(1) as f64);
        let margin = match_radius_rad as f64;
        let n_in = nearby_cam_positions
            .iter()
            .filter(|&&(_, x, y)| {
                (x as f64).abs() <= half_x + margin && (y as f64).abs() <= half_y + margin
            })
            .count();
        let region_area = (2.0 * (half_x + margin)) * (2.0 * (half_y + margin));
        let n_tested = match_centroid_count.min(vectors.data.len());
        if n_in == 0 || n_tested == 0 {
            // Nothing to test against: no evidence either way.
            return (matches, 1.0);
        }

        // ── Per-star likelihood ratio (see module docs) ──
        // Squared match distance per tested centroid; NaN = unmatched.
        let mut r2_of: Vec<f32> = vec![f32::NAN; n_tested];
        for (&(pt_idx, _), &d2) in matches.iter().zip(scratch.matched_d2()) {
            if pt_idx < n_tested {
                r2_of[pt_idx] = d2;
            }
        }
        let ps = pixel_scale_from_fov(config.image_width(), fov as f64);
        let sigma2_stage = (sigma_rad as f64) * (sigma_rad as f64);
        let ln_sigma2_stage = sigma2_stage.ln();
        // ln(A / (n · 2π)); per-star ln G = this − ln σᵢ².
        let ln_a_over_n2pi = (region_area / (n_in as f64 * 2.0 * std::f64::consts::PI)).ln();

        // Brightness classes: rank < n_in ("bright" — as many detections as
        // catalog stars predicted, so these should nearly all be stars) and
        // the faint remainder. Estimate each class's catalog fraction from
        // its own matched count.
        let mut n_class = [0usize; 2];
        let mut m_class = [0usize; 2];
        for (i, r2) in r2_of.iter().enumerate() {
            if exclude.contains(&i) {
                continue;
            }
            let c = usize::from(i >= n_in);
            n_class[c] += 1;
            if r2.is_finite() {
                m_class[c] += 1;
            }
        }
        let q_class: [f64; 2] = std::array::from_fn(|c| {
            if n_class[c] == 0 {
                0.0
            } else {
                (m_class[c] as f64 / n_class[c] as f64).clamp(Q_MIN, Q_MAX)
            }
        });

        let ln_q: [f64; 2] = std::array::from_fn(|c| q_class[c].ln());
        let ln_1mq: [f64; 2] = std::array::from_fn(|c| (1.0 - q_class[c]).ln());
        let mut log_lambda = 0.0f64;
        for (i, &r2) in r2_of.iter().enumerate() {
            if exclude.contains(&i) {
                continue;
            }
            let c = usize::from(i >= n_in);
            let ln_unmatched = ln_1mq[c];
            if r2.is_finite() {
                let s_px = sigma_px.get(i).copied().unwrap_or(0.0) as f64;
                let (sigma2, ln_sigma2) = if s_px > 0.0 {
                    let v = sigma2_stage + (s_px * ps) * (s_px * ps);
                    (v, v.ln())
                } else {
                    (sigma2_stage, ln_sigma2_stage)
                };
                let ln_matched = ln_q[c] + ln_a_over_n2pi - ln_sigma2 - 0.5 * r2 as f64 / sigma2;
                log_lambda += log_add_exp(ln_matched, ln_unmatched);
            } else {
                log_lambda += ln_unmatched;
            }
        }
        // BIC penalty for the fitted catalog fractions.
        let n_used = (n_class[0] + n_class[1]).max(1) as f64;
        let n_fitted = n_class.iter().filter(|&&n| n > 0).count() as f64;
        log_lambda -= 0.5 * n_used.ln() * n_fitted;

        let prob = if log_lambda <= 0.0 {
            1.0
        } else {
            (-log_lambda).exp().max(f64::MIN_POSITIVE)
        };
        (matches, prob)
    }
}

/// Ratio of the image diagonal to the image width, used to size the
/// verification cone (`fov_diagonal = fov * factor`).
///
/// At least 1.42 (≳ √2, the historical square-image constant, kept as a
/// conservative floor); larger for portrait images where the height exceeds
/// the width-referenced FOV and √2 would under-query the corners.
pub(super) fn diagonal_factor(config: &SolveConfig) -> f32 {
    let aspect = config.image_height() as f32 / config.image_width().max(1) as f32;
    (1.0 + aspect * aspect).sqrt().max(1.42)
}

/// Find unique 1-to-1 matches between image centroids and projected catalog positions.
///
/// Returns Vec<(centroid_local_idx, catalog_star_idx)>. `centroid_xy` and
/// `scratch` are caller-owned working buffers, held across the LIS candidate
/// loop (this runs once per candidate, twice per refined one) so the projection
/// buffer and the scratch's three larger candidate/flag buffers are reused
/// instead of reallocated per candidate; their contents are fully overwritten
/// each call. The match list is moved out via `take_matches` (no copy).
pub(super) fn find_centroid_matches(
    centroid_vectors: &[[f32; 3]],
    catalog_positions: &[(usize, f32, f32)], // (star_idx, cam_x, cam_y) in radians
    match_radius: f32,
    centroid_xy: &mut Vec<(f32, f32)>,
    scratch: &mut matching::MatchScratch<f32>,
) -> Vec<(usize, usize)> {
    // For each centroid, project to camera-plane angular coordinates
    centroid_xy.clear();
    centroid_xy.extend(centroid_vectors.iter().map(|v| {
        if v[2] > 0.0 {
            (v[0] / v[2], v[1] / v[2])
        } else {
            (f32::MAX, f32::MAX)
        }
    }));

    let n = centroid_xy.len();
    matching::greedy_unique_matches(
        centroid_xy,
        n,
        catalog_positions,
        match_radius * match_radius,
        scratch,
    );
    scratch.take_matches()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_add_exp_matches_direct() {
        for (a, b) in [(0.0f64, 0.0f64), (1.0, -3.0), (-800.0, -801.0), (5.0, 5.0)] {
            let direct = (a.exp() + b.exp()).ln();
            let got = log_add_exp(a, b);
            if direct.is_finite() {
                assert!((got - direct).abs() < 1e-12, "{a} {b}: {got} vs {direct}");
            } else {
                assert!(got.is_finite());
            }
        }
        assert!(
            (log_add_exp(-800.0, -801.0) - (-800.0 + (1.0 + (-1.0f64).exp()).ln())).abs() < 1e-12
        );
    }
}
