//! Attitude verification: project nearby catalog stars through a candidate
//! rotation, match them to image centroids, and score the match count
//! against a measured-density coincidence model.
//!
//! Shared by every hypothesis source (lost-in-space pattern search and
//! hinted tracking). This is the stage that decides, per candidate, whether
//! the evidence justifies refinement and acceptance.

use numeris::{Matrix3, Vector3};

use super::matching;
use super::preprocess::CentroidVectors;
use super::{SolveConfig, SolverDatabase};

#[cfg(feature = "profile")]
use crate::solver::profiling::{self, buckets};

impl SolverDatabase {
    /// Verify a candidate attitude by projecting nearby catalog stars into
    /// the camera frame and greedily matching them to image centroids.
    ///
    /// Shared by the lost-in-space and tracking paths. `vectors` must be
    /// brightness-sorted with parity already applied; `star_vectors`
    /// is the (possibly aberration-corrected) catalog unit-vector slice. The
    /// cone query itself uses the stored raw vectors, which are bit-identical
    /// to `Star::uvec()` and aligned with the catalog, so the candidate set
    /// is unchanged by aberration.
    ///
    /// Returns the matches `(centroid_local_idx, catalog_star_idx)` and the
    /// binomial false-positive probability of the match count (a per-candidate
    /// p-value, before any multiple-comparison correction by the caller).
    ///
    /// `hypothesis_matches` is the number of *tested* centroids that were used
    /// to form the attitude hypothesis (LIS passes the pattern stars that fall
    /// inside the tested set; tracking passes 0 — its hint is independent of
    /// the centroids). Those stars match by construction — with the
    /// measured-FOV rebuild they fit near-exactly, handing every
    /// ratio-passing candidate "free" matches — so they are excluded from
    /// both the binomial trials and successes rather than counted as
    /// evidence. This replaces the upstream-tetra3 heuristic of discounting a
    /// flat 2 matches.
    /// `match_radius_rad_override` replaces the default matching radius
    /// (`config.match_radius · fov`) — the false-positive model scales with
    /// it, so a tighter radius (e.g. derived from a refined fit's RMSE) makes
    /// every true match worth quadratically more evidence.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn verify_attitude(
        &self,
        rotation_matrix: &Matrix3<f32>,
        vectors: CentroidVectors<'_>,
        match_centroid_count: usize,
        fov: f32,
        config: &SolveConfig,
        star_vectors: &[[f32; 3]],
        hypothesis_matches: usize,
        match_radius_rad_override: Option<f32>,
        centroid_xy: &mut Vec<(f32, f32)>,
        scratch: &mut matching::MatchScratch<f32>,
    ) -> (Vec<(usize, usize)>, f64) {
        let fov_diagonal = fov * diagonal_factor(config);
        let match_radius_rad = match_radius_rad_override.unwrap_or(config.match_radius * fov);

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

        // False-positive probability of this match count.
        //
        // `prob_single` is the chance that a *coincidental* (wrong-attitude)
        // centroid lands within `match_radius_rad` of some projected catalog
        // star. Each projected star owns a disc of area π·r², and the
        // prediction density is *measured* rather than modeled: count the
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
        let prob_single = n_in as f64 * std::f64::consts::PI * margin * margin / region_area;
        // Exclude hypothesis stars from trials and successes (see doc above).
        let h = hypothesis_matches.min(match_centroid_count);
        let trials = match_centroid_count - h;
        let evidence = matches.len().saturating_sub(h).min(trials);
        let prob_mismatch = binomial_cdf(
            (trials - evidence) as u32,
            trials as u32,
            1.0 - prob_single.min(1.0),
        );
        (matches, prob_mismatch)
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

// ── Binomial CDF (no external dependency) ───────────────────────────────────

/// Compute the binomial CDF: P(X <= k) where X ~ Binomial(n, p).
/// Uses iterative computation for numerical stability at typical sizes (n < 500).
pub(super) fn binomial_cdf(k: u32, n: u32, p: f64) -> f64 {
    if k >= n {
        return 1.0;
    }
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0; // k < n here (k >= n already returned above)
    }

    let q = 1.0 - p;

    // Start with P(X=0) = q^n, then iteratively compute P(X=i)
    let mut cdf = 0.0;
    let mut log_term = n as f64 * q.ln(); // log(P(X=0))
    cdf += log_term.exp();

    for i in 1..=k as u64 {
        log_term += ((n as u64 - i + 1) as f64).ln() - (i as f64).ln() + p.ln() - q.ln();
        cdf += log_term.exp();
    }

    cdf.min(1.0)
}
