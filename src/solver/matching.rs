//! Greedy 1-to-1 nearest-neighbor matching, shared by the LIS/tracking
//! verification path (f32, tangent-plane angles) and the WCS refinement loop
//! (f64, pixels). One algorithm: collect all pairs within a radius, sort by
//! distance, then assign greedily so each point and each prediction is used
//! at most once.

use std::ops::{Add, Mul, Sub};

/// Coordinate scalar for the matcher — implemented for `f32` and `f64` so each
/// caller keeps its native precision (and bit-identical results).
pub(super) trait MatchCoord:
    Copy + PartialOrd + Sub<Output = Self> + Mul<Output = Self> + Add<Output = Self>
{
}
impl MatchCoord for f32 {}
impl MatchCoord for f64 {}

/// Reusable buffers for [`greedy_unique_matches`].
///
/// Hot callers (the WCS refinement loop) hold one across iterations and the
/// four allocations (candidate list + two used-flags + output) happen once per
/// solve instead of once per pass. Contents are fully overwritten each call,
/// so reuse is behavior-identical to fresh allocation.
#[derive(Default)]
pub(super) struct MatchScratch<F = f64> {
    /// (dist², point_idx, pred_idx) candidate pairs within radius.
    candidates: Vec<(F, usize, usize)>,
    /// Per-point "already assigned" flags.
    used_point: Vec<bool>,
    /// Per-prediction "already assigned" flags.
    used_pred: Vec<bool>,
    /// Resulting `(point_idx, catalog_star_idx)` matches.
    matches: Vec<(usize, usize)>,
    /// Squared distance of each accepted match, aligned with `matches`.
    matched_d2: Vec<F>,
}

impl<F> MatchScratch<F> {
    /// Move the last computed match set out of the scratch, leaving an empty
    /// buffer behind for the next call. Lets a caller that needs an owned Vec
    /// avoid the copy `.to_vec()` on the returned slice would incur, while
    /// still reusing the (larger) candidate/flag buffers across calls.
    pub(super) fn take_matches(&mut self) -> Vec<(usize, usize)> {
        std::mem::take(&mut self.matches)
    }

    /// Squared distances of the last computed match set, aligned with the
    /// matches (still valid after [`Self::take_matches`]).
    pub(super) fn matched_d2(&self) -> &[F] {
        &self.matched_d2
    }
}

/// Greedy unique 1-to-1 matching between points and predicted catalog positions.
///
/// Considers the first `max_points` entries of `points`; `predicted` entries
/// are `(catalog_star_idx, x, y)`. Writes the matches
/// `(point_idx, catalog_star_idx)` within `radius` (squared: `radius_sq`) into
/// `scratch.matches` and returns a reference to it.
pub(super) fn greedy_unique_matches<'a, F: MatchCoord>(
    points: &[(F, F)],
    max_points: usize,
    predicted: &[(usize, F, F)],
    radius_sq: F,
    scratch: &'a mut MatchScratch<F>,
) -> &'a [(usize, usize)] {
    let n_points = points.len().min(max_points);

    // Collect all candidate pairs within radius. We track the *position* in
    // `predicted` (not the catalog id) so uniqueness can use a bitset instead
    // of a HashSet — `predicted` holds distinct catalog stars, so position ↔ id
    // is a bijection and the dedup result is identical.
    let candidates = &mut scratch.candidates;
    candidates.clear();
    for (pt_idx, &(cx, cy)) in points[..n_points].iter().enumerate() {
        for (pred_idx, &(_cat_idx, px, py)) in predicted.iter().enumerate() {
            let dx = cx - px;
            let dy = cy - py;
            let d2 = dx * dx + dy * dy;
            if d2 <= radius_sq {
                candidates.push((d2, pt_idx, pred_idx));
            }
        }
    }

    // Sort by distance (closest first)
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Greedy unique 1-to-1 assignment
    let used_point = &mut scratch.used_point;
    used_point.clear();
    used_point.resize(n_points, false);
    let used_pred = &mut scratch.used_pred;
    used_pred.clear();
    used_pred.resize(predicted.len(), false);
    let matches = &mut scratch.matches;
    matches.clear();
    let matched_d2 = &mut scratch.matched_d2;
    matched_d2.clear();

    for &(d2, pt_idx, pred_idx) in candidates.iter() {
        if !used_point[pt_idx] && !used_pred[pred_idx] {
            used_point[pt_idx] = true;
            used_pred[pred_idx] = true;
            matches.push((pt_idx, predicted[pred_idx].0));
            matched_d2.push(d2);
        }
    }

    matches
}
