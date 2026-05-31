//! WCS TAN-projection iterative refinement (constrained).
//!
//! After the initial 4-star pattern match provides a seed rotation via SVD (Wahba's problem),
//! this module refines the solution by fitting 3 parameters per image:
//! **rotation angle θ** and **tangent-plane offset (dξ₀, dη₀)**, with the pixel scale
//! locked from the CameraModel's focal length.
//!
//! This constrained approach (vs. the full 6-DOF CD matrix fit) avoids degeneracy
//! between the linear part of the distortion polynomial and the per-image attitude,
//! which is critical for multi-image calibration.
//!
//! ## Algorithm
//!
//! 1. Extract initial CRVAL (RA, Dec) and rotation angle θ from the SVD rotation matrix.
//! 2. Iteratively:
//!    a. TAN-project matched catalog stars at current CRVAL → (ξ, η) in radians.
//!    b. Compute predicted tangent-plane coords from pixel coords using θ and pixel_scale.
//!    c. Solve a 3-parameter linear system for `[δθ, dξ₀, dη₀]`.
//!    d. Update θ and CRVAL.
//!    e. MAD-based outlier rejection.
//!    f. Re-associate: project catalog stars to pixel space, match to centroids.
//!    g. Converge when updates vanish, no outliers rejected, and match set is stable.

use numeris::{Matrix3, Vector3};
use tracing::debug;

use crate::starcatalog::StarCatalog;

#[cfg(feature = "profile")]
use crate::solver::profiling::{self, buckets};

// ── TAN projection ─────────────────────────────────────────────────────────

/// Forward gnomonic (TAN) projection.
///
/// Projects celestial point `(ra, dec)` onto the tangent plane at `(crval_ra, crval_dec)`.
/// Returns `(ξ, η)` in radians, or `None` if the point is on or behind the tangent plane.
///
/// Reference: Calabretta & Greisen (2002), FITS WCS Paper II, §5.1.1.
#[inline]
pub fn tan_project(ra: f64, dec: f64, crval_ra: f64, crval_dec: f64) -> Option<(f64, f64)> {
    let da = ra - crval_ra;
    let sin_dec = dec.sin();
    let cos_dec = dec.cos();
    let sin_dec0 = crval_dec.sin();
    let cos_dec0 = crval_dec.cos();
    let cos_da = da.cos();

    let denom = sin_dec * sin_dec0 + cos_dec * cos_dec0 * cos_da;
    if denom <= 1e-12 {
        return None; // behind or on the tangent plane
    }

    let xi = cos_dec * da.sin() / denom;
    let eta = (sin_dec * cos_dec0 - cos_dec * sin_dec0 * cos_da) / denom;
    Some((xi, eta))
}

/// Inverse gnomonic (TAN) projection.
///
/// Given tangent-plane coordinates `(ξ, η)` in radians at reference point
/// `(crval_ra, crval_dec)`, returns celestial coordinates `(ra, dec)` in radians.
#[inline]
pub fn inverse_tan_project(xi: f64, eta: f64, crval_ra: f64, crval_dec: f64) -> (f64, f64) {
    let sin_dec0 = crval_dec.sin();
    let cos_dec0 = crval_dec.cos();
    let rho_sq = xi * xi + eta * eta;

    if rho_sq < 1e-30 {
        // On the reference point itself
        return (crval_ra, crval_dec);
    }

    let rho = rho_sq.sqrt();
    let c = rho.atan(); // for TAN projection, c = atan(rho)
    let sin_c = c.sin();
    let cos_c = c.cos();

    let dec = (cos_c * sin_dec0 + eta * sin_c * cos_dec0 / rho).asin();
    let ra = crval_ra + (xi * sin_c).atan2(rho * cos_dec0 * cos_c - eta * sin_dec0 * sin_c);
    (ra, dec)
}

// ── 2×2 matrix helpers ─────────────────────────────────────────────────────

/// Invert a 2×2 matrix. Returns `None` if singular (|det| < 1e-30).
#[inline]
pub fn cd_inverse(cd: &[[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
    if det.abs() < 1e-30 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [cd[1][1] * inv_det, -cd[0][1] * inv_det],
        [-cd[1][0] * inv_det, cd[0][0] * inv_det],
    ])
}

/// Synthesize a CD matrix from rotation angle, pixel scale, and parity.
///
/// The CD matrix maps pixel offsets to tangent-plane coordinates:
/// ```text
/// CD = ps * R(θ)  (if parity_flip=false, det > 0)
/// CD = ps * [[−cos θ, sin θ], [sin θ, cos θ]]  (if parity_flip=true, det < 0)
/// ```
pub fn cd_from_theta(theta: f64, pixel_scale: f64, parity_flip: bool) -> [[f64; 2]; 2] {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let ps = pixel_scale;
    if parity_flip {
        [[-ps * cos_t, ps * sin_t], [ps * sin_t, ps * cos_t]]
    } else {
        [[ps * cos_t, -ps * sin_t], [ps * sin_t, ps * cos_t]]
    }
}

/// Decompose a CD matrix into rotation angle, pixel scale (x and y), and parity.
///
/// Returns `(theta_rad, scale_x, scale_y, parity_flip)`.
#[cfg(test)]
pub fn decompose_cd(cd: &[[f64; 2]; 2]) -> (f64, f64, f64, bool) {
    let det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
    let parity_flip = det < 0.0;

    // Scale = norm of each column
    let scale_x = (cd[0][0] * cd[0][0] + cd[1][0] * cd[1][0]).sqrt();
    let scale_y = (cd[0][1] * cd[0][1] + cd[1][1] * cd[1][1]).sqrt();

    // Rotation angle from the first column (camera +X direction)
    // For no parity: CD11 = ps*cos θ, CD21 = ps*sin θ
    // For parity:    CD11 = -ps*cos θ, CD21 = ps*sin θ
    let theta = if parity_flip {
        // CD21 = ps*sin θ, CD11 = -ps*cos θ
        cd[1][0].atan2(-cd[0][0])
    } else {
        cd[1][0].atan2(cd[0][0])
    };

    (theta, scale_x, scale_y, parity_flip)
}

// ── 3×3 linear solve ────────────────────────────────────────────────────────

/// Solve a 3×3 linear system `Ax = b` via Gaussian elimination with partial pivoting.
///
/// The normal equations `(AᵀA)x = Aᵀb` for our 3-parameter LS problem are always 3×3,
/// so this avoids pulling in a general linear algebra solver.
// Index-based pivoting/elimination is clearer here than iterator adapters.
#[allow(clippy::needless_range_loop)]
fn solve_3x3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> Option<[f64; 3]> {
    // Work on copies
    let mut m = *a;
    let mut rhs = *b;

    // Forward elimination with partial pivoting
    for col in 0..3 {
        // Find pivot
        let mut max_abs = m[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..3 {
            let v = m[row][col].abs();
            if v > max_abs {
                max_abs = v;
                max_row = row;
            }
        }
        if max_abs < 1e-30 {
            return None; // singular
        }

        // Swap rows
        if max_row != col {
            m.swap(col, max_row);
            rhs.swap(col, max_row);
        }

        // Eliminate below
        let pivot = m[col][col];
        for row in (col + 1)..3 {
            let factor = m[row][col] / pivot;
            for j in col..3 {
                m[row][j] -= factor * m[col][j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    // Back substitution
    let mut x = [0.0f64; 3];
    for i in (0..3).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..3 {
            sum -= m[i][j] * x[j];
        }
        if m[i][i].abs() < 1e-30 {
            return None;
        }
        x[i] = sum / m[i][i];
    }

    Some(x)
}

// ── Pixel-space matching ────────────────────────────────────────────────────

/// Reusable scratch buffers for [`find_pixel_matches`]. Hoisted out of the outer
/// refinement loop and `.clear()`ed before each call so the four allocations
/// (candidate list + two used-flags + output) happen once per solve instead of
/// once per outer iteration. Contents are fully overwritten each call, so reuse
/// is behavior-identical to fresh allocation.
#[derive(Default)]
struct MatchScratch {
    /// (dist_sq, cent_idx, pred_idx) candidate pairs within radius.
    candidates: Vec<(f64, usize, usize)>,
    /// Per-centroid "already assigned" flags.
    used_cent: Vec<bool>,
    /// Per-prediction "already assigned" flags.
    used_pred: Vec<bool>,
    /// Resulting `(centroid_idx, catalog_star_idx)` matches.
    matches: Vec<(usize, usize)>,
}

/// Greedy 1-to-1 matching between centroid pixel positions and predicted catalog positions.
///
/// Writes the unique matches `(centroid_idx, catalog_star_idx)` within
/// `radius_px` pixels into `scratch.matches` and returns a reference to it. All
/// buffers live in `scratch` so repeated calls reuse the same allocations.
fn find_pixel_matches<'a>(
    centroid_pixels: &[(f64, f64)],
    max_centroids: usize,
    predicted: &[(usize, f64, f64)], // (catalog_star_idx, pred_x, pred_y)
    radius_px: f64,
    scratch: &'a mut MatchScratch,
) -> &'a [(usize, usize)] {
    let radius_sq = radius_px * radius_px;
    let n_cent = centroid_pixels.len().min(max_centroids);

    // Collect all candidate pairs within radius. We track the *position* in
    // `predicted` (not the catalog id) so uniqueness can use a bitset instead of
    // a HashSet — `predicted` holds distinct catalog stars, so position ↔ id is
    // a bijection and the dedup result is identical.
    let candidates = &mut scratch.candidates;
    candidates.clear();
    for (cent_idx, &(cx, cy)) in centroid_pixels[..n_cent].iter().enumerate() {
        for (pred_idx, &(_cat_idx, px, py)) in predicted.iter().enumerate() {
            let dx = cx - px;
            let dy = cy - py;
            let d2 = dx * dx + dy * dy;
            if d2 <= radius_sq {
                candidates.push((d2, cent_idx, pred_idx));
            }
        }
    }

    // Sort by distance (closest first)
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Greedy unique 1-to-1 assignment
    let used_cent = &mut scratch.used_cent;
    used_cent.clear();
    used_cent.resize(n_cent, false);
    let used_pred = &mut scratch.used_pred;
    used_pred.clear();
    used_pred.resize(predicted.len(), false);
    let matches = &mut scratch.matches;
    matches.clear();

    for &(_, cent_idx, pred_idx) in candidates.iter() {
        if !used_cent[cent_idx] && !used_pred[pred_idx] {
            used_cent[cent_idx] = true;
            used_pred[pred_idx] = true;
            matches.push((cent_idx, predicted[pred_idx].0));
        }
    }

    matches
}

// ── Constrained prediction helpers ──────────────────────────────────────────

/// Predict tangent-plane coords from pixel coords using rotation angle and pixel scale.
///
/// `ξ = ps·(cos θ · px - sin θ · py)`
/// `η = ps·(sin θ · px + cos θ · py)`
#[inline]
fn predict_tanplane(px: f64, py: f64, cos_t: f64, sin_t: f64, ps: f64) -> (f64, f64) {
    let xi = ps * (cos_t * px - sin_t * py);
    let eta = ps * (sin_t * px + cos_t * py);
    (xi, eta)
}

/// Precomputed per-star projection inputs: right ascension plus the sine and
/// cosine of declination. Decoded once from the ICRS unit vector and reused
/// across every refinement pass (the `atan2`/`asin`/`sin`/`cos` would otherwise
/// be recomputed for the same star on every iteration).
#[derive(Clone, Copy)]
struct StarRaDec {
    ra: f64,
    sin_dec: f64,
    cos_dec: f64,
}

/// Decode a catalog star's ICRS unit vector into [`StarRaDec`].
#[inline]
fn star_radec(sv: &[f32; 3]) -> StarRaDec {
    #[cfg(feature = "profile")]
    profiling::count(buckets::WCS_RADEC, 1);
    let ra = (sv[1] as f64).atan2(sv[0] as f64);
    let dec = (sv[2] as f64).asin();
    StarRaDec {
        ra,
        sin_dec: dec.sin(),
        cos_dec: dec.cos(),
    }
}

/// TAN projection from precomputed star coords and precomputed CRVAL sin/cos.
///
/// Equivalent to [`tan_project`] but takes a [`StarRaDec`] (star dec sin/cos
/// already known) and the CRVAL declination sin/cos hoisted out of the per-star
/// loop, leaving only `cos(da)`/`sin(da)` to compute per call.
#[inline]
fn tan_project_pre(
    s: &StarRaDec,
    crval_ra: f64,
    sin_dec0: f64,
    cos_dec0: f64,
) -> Option<(f64, f64)> {
    let da = s.ra - crval_ra;
    let cos_da = da.cos();
    let denom = s.sin_dec * sin_dec0 + s.cos_dec * cos_dec0 * cos_da;
    if denom <= 1e-12 {
        return None;
    }
    let xi = s.cos_dec * da.sin() / denom;
    let eta = (s.sin_dec * cos_dec0 - s.cos_dec * sin_dec0 * cos_da) / denom;
    Some((xi, eta))
}

/// Accumulate one matched star's contribution to the 3-parameter
/// `[δθ, dξ₀, dη₀]` normal equations `AᵀA x = Aᵀb`.
///
/// Jacobian rows are `ξ: [∂ξ/∂θ, 1, 0]` and `η: [∂η/∂θ, 0, 1]`, with
/// `∂ξ/∂θ = ps·(-sinθ·px − cosθ·py)` and `∂η/∂θ = ps·(cosθ·px − sinθ·py)`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn accumulate_normal_equations(
    ata: &mut [[f64; 3]; 3],
    atb: &mut [f64; 3],
    px: f64,
    py: f64,
    cos_t: f64,
    sin_t: f64,
    ps: f64,
    r_xi: f64,
    r_eta: f64,
) {
    let j_xi_theta = ps * (-sin_t * px - cos_t * py);
    let j_eta_theta = ps * (cos_t * px - sin_t * py);
    let jxi = [j_xi_theta, 1.0, 0.0];
    let jeta = [j_eta_theta, 0.0, 1.0];
    for i in 0..3 {
        for j in 0..3 {
            ata[i][j] += jxi[i] * jxi[j] + jeta[i] * jeta[j];
        }
        atb[i] += jxi[i] * r_xi + jeta[i] * r_eta;
    }
}

/// Predict pixel coords from tangent-plane coords (inverse of predict_tanplane).
///
/// `px = (1/ps)·(cos θ · ξ + sin θ · η)`
/// `py = (1/ps)·(-sin θ · ξ + cos θ · η)`
#[inline]
fn predict_pixel(xi: f64, eta: f64, cos_t: f64, sin_t: f64, inv_ps: f64) -> (f64, f64) {
    let px = inv_ps * (cos_t * xi + sin_t * eta);
    let py = inv_ps * (-sin_t * xi + cos_t * eta);
    (px, py)
}

// ── WCS refinement result ───────────────────────────────────────────────────

/// Result of the WCS TAN-projection iterative refinement.
pub struct WcsRefineResult {
    /// CD matrix: `[[CD11, CD12], [CD21, CD22]]` in tangent-plane radians per pixel.
    /// Derived from `(theta, pixel_scale)` for FITS compatibility.
    pub cd_matrix: [[f64; 2]; 2],
    /// Reference point `[RA, Dec]` in radians.
    pub crval_rad: [f64; 2],
    /// Fitted rotation angle in radians (camera roll in tangent plane).
    pub theta_rad: f64,
    /// Final matched pairs: `(centroid_local_idx, catalog_star_idx)`.
    pub matches: Vec<(usize, usize)>,
    /// RMSE of angular residuals in radians.
    pub rmse_rad: f64,
}

// ── Main refinement entry point ─────────────────────────────────────────────

/// MAD → σ scale factor for a Gaussian distribution.
const MAD_SCALE: f64 = 1.4826;

/// Robust statistics of a residual list: the median residual and the
/// MAD-derived standard-deviation estimate (`MAD_SCALE · MAD`).
///
/// Both the median and the median-absolute-deviation use the simple midpoint of
/// the sorted values (`v[len / 2]`) — the refinement's existing convention.
fn residual_median_sigma(residuals: &[(usize, f64)]) -> (f64, f64) {
    // Only the single midpoint order statistic `v[len/2]` is needed from each
    // list (the refinement's existing "median = sorted midpoint" convention), so
    // a partial selection yields the identical element with less work than a full
    // sort. `select_nth_unstable_by` places the k-th smallest at index k under
    // the given comparator — the same value `sort_by` would put there.
    let mut res_vals: Vec<f64> = residuals.iter().map(|&(_, r)| r).collect();
    let mid = res_vals.len() / 2;
    res_vals.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    let median = res_vals[mid];
    let mut abs_devs: Vec<f64> = res_vals.iter().map(|r| (r - median).abs()).collect();
    let mid_dev = abs_devs.len() / 2;
    abs_devs.select_nth_unstable_by(mid_dev, |a, b| a.partial_cmp(b).unwrap());
    let mad = abs_devs[mid_dev];
    (median, MAD_SCALE * mad)
}

/// Constrained iterative WCS TAN-projection refinement.
///
/// Starting from an initial rotation matrix (from the SVD pattern match) and an initial
/// match set (from verification), refines the WCS solution by fitting 3 parameters
/// (rotation angle θ, tangent-plane offset dξ₀, dη₀) with the pixel scale locked
/// from the CameraModel.
///
/// # Arguments
///
/// * `initial_rotation` — 3×3 ICRS→camera rotation from the initial SVD solve.
/// * `initial_matches` — initial matched pairs `(centroid_local_idx, catalog_star_idx)`.
/// * `centroids_px` — pixel coordinates of centroids after undistortion and CRPIX
///   subtraction, with parity already applied. Indexed by local_idx (brightness-sorted).
/// * `star_vectors` — catalog star ICRS unit vectors, indexed by catalog star index.
/// * `star_catalog` — spatial index for cone queries.
/// * `pixel_scale` — radians per pixel (1/focal_length_px from CameraModel).
/// * `parity_flip` — whether the image x-axis is flipped.
/// * `match_radius_rad` — initial match radius in radians (from `config.match_radius * fov`).
/// * `max_match_centroids` — maximum number of centroids to consider for matching.
/// * `max_iterations` — maximum outer-loop iterations.
///
/// # Returns
///
/// A [`WcsRefineResult`] with the refined CD matrix, CRVAL, theta, match set, and
/// residual stats.
#[allow(clippy::too_many_arguments)]
pub fn wcs_refine(
    initial_rotation: &Matrix3<f32>,
    initial_matches: &[(usize, usize)],
    centroids_px: &[(f64, f64)],
    star_vectors: &[[f32; 3]],
    star_catalog: &StarCatalog,
    pixel_scale: f64,
    parity_flip: bool,
    match_radius_rad: f32,
    max_match_centroids: usize,
    max_iterations: u32,
) -> WcsRefineResult {
    // ── Constants ────────────────────────────────────────────────────────
    const CLIP_NSIGMA: f64 = 3.0;
    const CONVERGENCE_RAD: f64 = 1e-12; // tangent-plane offset convergence

    let ps = pixel_scale;
    let inv_ps = 1.0 / ps; // focal_length_px

    // ── Step 0: Extract initial CRVAL and θ from SVD rotation ──────────
    // Boresight in ICRS = R^T * [0, 0, 1] = third row of R
    let bx = initial_rotation[(2, 0)] as f64;
    let by = initial_rotation[(2, 1)] as f64;
    let bz = initial_rotation[(2, 2)] as f64;
    let mut crval_ra = by.atan2(bx);
    let mut crval_dec = bz.asin();

    // Extract initial theta from rotation matrix
    // Camera +X direction in ICRS = first row of R
    let cam_x_icrs = Vector3::<f64>::from_array([
        initial_rotation[(0, 0)] as f64,
        initial_rotation[(0, 1)] as f64,
        initial_rotation[(0, 2)] as f64,
    ]);

    // Tangent-plane basis vectors at CRVAL
    let sin_a = crval_ra.sin();
    let cos_a = crval_ra.cos();
    let sin_d = crval_dec.sin();
    let cos_d = crval_dec.cos();
    let e_xi = Vector3::<f64>::from_array([-sin_a, cos_a, 0.0]);
    let e_eta = Vector3::<f64>::from_array([-sin_d * cos_a, -sin_d * sin_a, cos_d]);

    // theta = angle of camera X in the tangent plane
    let xi_comp = cam_x_icrs.dot(&e_xi);
    let eta_comp = cam_x_icrs.dot(&e_eta);
    let mut theta = eta_comp.atan2(xi_comp);

    debug!(
        "WCS refine: initial CRVAL = ({:.4}°, {:.4}°), θ = {:.4}°, ps = {:.6e} rad/px, {} matches, {} centroids",
        crval_ra.to_degrees(),
        crval_dec.to_degrees(),
        theta.to_degrees(),
        ps,
        initial_matches.len(),
        centroids_px.len(),
    );

    // ── Working state ───────────────────────────────────────────────────
    let mut current_matches: Vec<(usize, usize)> = initial_matches.to_vec();

    // Phase-D search geometry is constant across iterations (depends only on the
    // centroid positions and pixel scale), so compute it once.
    let max_cent_dist_px = centroids_px
        .iter()
        .map(|(x, y)| (x * x + y * y).sqrt())
        .fold(0.0f64, f64::max);
    let search_radius = (ps * max_cent_dist_px * 1.5).max(match_radius_rad as f64 * 2.0);

    // Phase-D re-association cache: the boresight barely moves between outer
    // iterations, so we query the catalog cone once (padded by REQUERY_MARGIN)
    // and reuse the star set + its precomputed `StarRaDec` until the boresight
    // drifts past the margin. The cached set is a superset of any single
    // iteration's query, and the extra (annulus) stars project well outside the
    // image so they never enter `find_pixel_matches` — results are unchanged.
    let requery_margin = match_radius_rad as f64 * 2.0;
    let requery_cos = requery_margin.cos();
    let mut reassoc_cache: Option<(Vector3<f64>, Vec<usize>, Vec<StarRaDec>)> = None;

    // Phase-D scratch reused across outer iterations: the projected-pixel list
    // and the greedy-matcher's working buffers. Cleared + refilled each pass, so
    // reuse is behavior-identical to fresh allocation but moves the allocations
    // out of the loop.
    let mut predicted: Vec<(usize, f64, f64)> = Vec::new();
    let mut match_scratch = MatchScratch::default();

    // ── Outer refinement loop ───────────────────────────────────────────
    for outer_iter in 0..max_iterations {
        #[cfg(feature = "profile")]
        profiling::count(buckets::WCS_OUTER, 1);
        // Precompute per-star (ra, sin_dec, cos_dec) for the current match set
        // once; reused by Phase A's inner loop and Phase B. The values depend
        // only on the star, not on θ/CRVAL.
        let match_radec: Vec<StarRaDec> = current_matches
            .iter()
            .map(|&(_, cat_idx)| star_radec(&star_vectors[cat_idx]))
            .collect();

        // ── Phase A: LS fit (δθ, dξ₀, dη₀) ──────────────────────────
        for inner_iter in 0..10 {
            if current_matches.len() < 3 {
                break;
            }
            #[cfg(feature = "profile")]
            profiling::count(buckets::WCS_INNER, 1);

            let cos_t = theta.cos();
            let sin_t = theta.sin();
            // CRVAL changes each inner iteration; hoist its sin/cos out of the
            // per-star loop (was recomputed inside tan_project for every star).
            let sin_dec0 = crval_dec.sin();
            let cos_dec0 = crval_dec.cos();

            // Build normal equations AᵀA x = Aᵀb for 3 unknowns: [δθ, dξ₀, dη₀]
            let mut ata = [[0.0f64; 3]; 3];
            let mut atb = [0.0f64; 3];
            let mut n_valid = 0u32;

            for (i, &(cent_idx, _)) in current_matches.iter().enumerate() {
                let Some((xi_cat, eta_cat)) =
                    tan_project_pre(&match_radec[i], crval_ra, sin_dec0, cos_dec0)
                else {
                    continue;
                };

                let (px, py) = centroids_px[cent_idx];
                let (xi_pred, eta_pred) = predict_tanplane(px, py, cos_t, sin_t, ps);

                // Residuals
                let r_xi = xi_cat - xi_pred;
                let r_eta = eta_cat - eta_pred;

                accumulate_normal_equations(
                    &mut ata, &mut atb, px, py, cos_t, sin_t, ps, r_xi, r_eta,
                );
                n_valid += 1;
            }

            if n_valid < 3 {
                break;
            }

            // Solve the 3×3 system
            let Some(sol) = solve_3x3(&ata, &atb) else {
                debug!("WCS refine: singular normal equations, aborting");
                break;
            };

            let d_theta = sol[0];
            let dxi_0 = sol[1];
            let deta_0 = sol[2];

            // Update theta and CRVAL
            theta += d_theta;
            let (new_ra, new_dec) = inverse_tan_project(dxi_0, deta_0, crval_ra, crval_dec);
            crval_ra = new_ra;
            crval_dec = new_dec;

            debug!(
                "  inner {}: δθ={:.3e}°, offset=({:.3e}, {:.3e}) rad",
                inner_iter,
                d_theta.to_degrees(),
                dxi_0,
                deta_0,
            );

            // Check convergence
            if d_theta.abs() < 1e-10 && dxi_0.abs() + deta_0.abs() < CONVERGENCE_RAD {
                break;
            }
        }

        // ── Phase B: Compute residuals ──────────────────────────────────
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let sin_dec0 = crval_dec.sin();
        let cos_dec0 = crval_dec.cos();

        let mut residuals: Vec<(usize, f64)> = Vec::with_capacity(current_matches.len());
        for (match_idx, &(cent_idx, _)) in current_matches.iter().enumerate() {
            if let Some((xi_cat, eta_cat)) =
                tan_project_pre(&match_radec[match_idx], crval_ra, sin_dec0, cos_dec0)
            {
                let (px, py) = centroids_px[cent_idx];
                let (xi_pred, eta_pred) = predict_tanplane(px, py, cos_t, sin_t, ps);
                let dxi = xi_pred - xi_cat;
                let deta = eta_pred - eta_cat;
                let residual = (dxi * dxi + deta * deta).sqrt();
                residuals.push((match_idx, residual));
            }
        }

        // Robust residual statistics (median, MAD-derived σ), computed once per
        // iteration and reused by both Phase C clipping and Phase D's adaptive
        // match radius.
        let mad_stats = if residuals.len() >= 6 {
            Some(residual_median_sigma(&residuals))
        } else {
            None
        };

        // ── Phase C: MAD-based outlier rejection ────────────────────────
        if let Some((median, sigma_est)) = mad_stats {
            let clip_threshold = median + CLIP_NSIGMA * sigma_est;

            let old_len = current_matches.len();
            let mut keep_matches: Vec<(usize, usize)> = Vec::new();
            for &(match_idx, residual) in &residuals {
                if residual <= clip_threshold {
                    keep_matches.push(current_matches[match_idx]);
                }
            }

            if keep_matches.len() < old_len && keep_matches.len() >= 4 {
                debug!(
                    "  outer {}: MAD clip: {} → {} matches (σ={:.2e} rad, threshold={:.2e} rad)",
                    outer_iter,
                    old_len,
                    keep_matches.len(),
                    sigma_est,
                    clip_threshold,
                );
                current_matches = keep_matches;
            }
        }

        // ── Phase D: Re-associate (search for new inliers) ─────────────
        // Run every iteration (including the first): Phase A has already
        // converged the LS this pass, so the re-association is meaningful, and
        // detecting a stable match set here lets us break without burning an
        // extra confirming iteration. `n_rejected` keeps the existing
        // clip-driven behavior.
        {
            let cos_t = theta.cos();
            let sin_t = theta.sin();

            // Pixel radius for matching
            let radius_px = match_radius_rad as f64 / ps;

            // Adaptive radius from the MAD σ computed above (reused, not recomputed).
            let adaptive_radius_px = if let Some((_, sigma_est)) = mad_stats {
                (5.0 * sigma_est / ps).max(2.5).min(radius_px)
            } else {
                radius_px
            };

            // Current boresight in ICRS.
            let boresight = Vector3::from_array([
                crval_dec.cos() * crval_ra.cos(),
                crval_dec.cos() * crval_ra.sin(),
                crval_dec.sin(),
            ]);

            // (Re)query the catalog cone only when the cache is empty or the
            // boresight has drifted past the padding margin. Cache the star set
            // and its precomputed `StarRaDec` (atan2/asin done once, not per
            // outer iteration).
            let need_query = match &reassoc_cache {
                Some((qb, _, _)) => qb.dot(&boresight) < requery_cos,
                None => true,
            };
            if need_query {
                let idx = timed!(
                    buckets::WCS_REASSOC_QUERY,
                    star_catalog.query_indices_from_uvec(
                        Vector3::from_array([
                            boresight[0] as f32,
                            boresight[1] as f32,
                            boresight[2] as f32,
                        ]),
                        (search_radius + requery_margin) as f32,
                    )
                );
                #[cfg(feature = "profile")]
                {
                    profiling::count(buckets::WCS_REASSOC_CALL, 1);
                    profiling::count(buckets::WCS_REASSOC_STARS, idx.len() as u64);
                }
                let radec: Vec<StarRaDec> =
                    idx.iter().map(|&i| star_radec(&star_vectors[i])).collect();
                reassoc_cache = Some((boresight, idx, radec));
            }
            let (_, nearby_indices, nearby_radec) = reassoc_cache.as_ref().unwrap();

            // Project each cached catalog star to pixel coords via TAN + inverse
            // rotation, reusing the cached `StarRaDec`. Drop stars whose
            // predicted pixel lands farther than (max centroid radius + match
            // radius) from the optical center: by the triangle inequality such a
            // star cannot fall within `radius_px` of any centroid, so it could
            // never match — pruning it here shrinks the matching loop without
            // changing the result. (The cone query is padded ~1.5× the frame, so
            // a large fraction of cached stars project off-frame.)
            let prune_r = max_cent_dist_px + radius_px;
            let prune_r2 = prune_r * prune_r;
            let sin_dec0 = crval_dec.sin();
            let cos_dec0 = crval_dec.cos();
            timed!(buckets::WCS_REASSOC_PROJECT, {
                predicted.clear();
                for (k, &cat_idx) in nearby_indices.iter().enumerate() {
                    if let Some((xi, eta)) =
                        tan_project_pre(&nearby_radec[k], crval_ra, sin_dec0, cos_dec0)
                    {
                        let (pred_x, pred_y) = predict_pixel(xi, eta, cos_t, sin_t, inv_ps);
                        if pred_x * pred_x + pred_y * pred_y <= prune_r2 {
                            predicted.push((cat_idx, pred_x, pred_y));
                        }
                    }
                }
            });

            let new_matches: &[(usize, usize)] = timed!(
                buckets::WCS_REASSOC_MATCH,
                find_pixel_matches(
                    centroids_px,
                    max_match_centroids,
                    &predicted,
                    adaptive_radius_px,
                    &mut match_scratch,
                )
            );

            if new_matches.len() >= 4 {
                let mut sorted_new = new_matches.to_vec();
                sorted_new.sort();
                let mut sorted_cur = current_matches.clone();
                sorted_cur.sort();

                if sorted_new != sorted_cur {
                    debug!(
                        "  outer {}: re-associate: {} → {} matches (radius={:.1} px)",
                        outer_iter,
                        current_matches.len(),
                        new_matches.len(),
                        adaptive_radius_px,
                    );
                    current_matches = new_matches.to_vec();
                    continue;
                }
            }
        }

        // Converged: reaching here means re-association produced no change this
        // iteration (a change would have `continue`d above), so the match set is
        // stable. Break regardless of iteration index — Phase A already
        // converged the LS fit on this set.
        debug!("  outer {}: converged", outer_iter);
        break;
    }

    // ── Final MAD clip passes (clip-only, no re-association) ────────────
    for clip_pass in 0..3 {
        if current_matches.len() < 6 {
            break;
        }

        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let sin_dec0 = crval_dec.sin();
        let cos_dec0 = crval_dec.cos();
        let match_radec: Vec<StarRaDec> = current_matches
            .iter()
            .map(|&(_, cat_idx)| star_radec(&star_vectors[cat_idx]))
            .collect();

        let mut residuals: Vec<(usize, f64)> = Vec::new();
        for (match_idx, &(cent_idx, _)) in current_matches.iter().enumerate() {
            if let Some((xi_cat, eta_cat)) =
                tan_project_pre(&match_radec[match_idx], crval_ra, sin_dec0, cos_dec0)
            {
                let (px, py) = centroids_px[cent_idx];
                let (xi_pred, eta_pred) = predict_tanplane(px, py, cos_t, sin_t, ps);
                let dxi = xi_pred - xi_cat;
                let deta = eta_pred - eta_cat;
                residuals.push((match_idx, (dxi * dxi + deta * deta).sqrt()));
            }
        }

        if residuals.len() < 6 {
            break;
        }

        let (median, sigma_est) = residual_median_sigma(&residuals);
        let clip_threshold = median + CLIP_NSIGMA * sigma_est;

        let mut keep: Vec<(usize, usize)> = Vec::new();
        for &(match_idx, residual) in &residuals {
            if residual <= clip_threshold {
                keep.push(current_matches[match_idx]);
            }
        }

        let n_clipped = current_matches.len() - keep.len();
        if n_clipped == 0 || keep.len() < 4 {
            break;
        }

        debug!(
            "  final clip {}: {} → {} matches",
            clip_pass,
            current_matches.len(),
            keep.len(),
        );
        current_matches = keep;

        // Re-fit theta + CRVAL on cleaned set (one inner LS pass)
        {
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let sin_dec0 = crval_dec.sin();
            let cos_dec0 = crval_dec.cos();

            let mut ata = [[0.0f64; 3]; 3];
            let mut atb = [0.0f64; 3];
            for &(cent_idx, cat_idx) in &current_matches {
                let s = star_radec(&star_vectors[cat_idx]);
                if let Some((xi_cat, eta_cat)) = tan_project_pre(&s, crval_ra, sin_dec0, cos_dec0) {
                    let (px, py) = centroids_px[cent_idx];
                    let (xi_pred, eta_pred) = predict_tanplane(px, py, cos_t, sin_t, ps);
                    let r_xi = xi_cat - xi_pred;
                    let r_eta = eta_cat - eta_pred;
                    accumulate_normal_equations(
                        &mut ata, &mut atb, px, py, cos_t, sin_t, ps, r_xi, r_eta,
                    );
                }
            }
            if let Some(sol) = solve_3x3(&ata, &atb) {
                theta += sol[0];
                let (new_ra, new_dec) = inverse_tan_project(sol[1], sol[2], crval_ra, crval_dec);
                crval_ra = new_ra;
                crval_dec = new_dec;
            }
        }
    }

    // ── Compute final residual statistics ────────────────────────────────
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let sin_dec0 = crval_dec.sin();
    let cos_dec0 = crval_dec.cos();

    let mut final_residuals: Vec<f64> = Vec::with_capacity(current_matches.len());
    for &(cent_idx, cat_idx) in &current_matches {
        let s = star_radec(&star_vectors[cat_idx]);
        if let Some((xi_cat, eta_cat)) = tan_project_pre(&s, crval_ra, sin_dec0, cos_dec0) {
            let (px, py) = centroids_px[cent_idx];
            let (xi_pred, eta_pred) = predict_tanplane(px, py, cos_t, sin_t, ps);
            let dxi = xi_pred - xi_cat;
            let deta = eta_pred - eta_cat;
            final_residuals.push((dxi * dxi + deta * deta).sqrt());
        }
    }
    final_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let rmse = if final_residuals.is_empty() {
        0.0
    } else {
        (final_residuals.iter().map(|r| r * r).sum::<f64>() / final_residuals.len() as f64).sqrt()
    };
    let p90e = if final_residuals.is_empty() {
        0.0
    } else {
        final_residuals[(0.9 * (final_residuals.len() - 1) as f64) as usize]
    };
    let max_err = final_residuals.last().copied().unwrap_or(0.0);

    // Derive CD matrix from (theta, pixel_scale, parity)
    let cd = cd_from_theta(theta, ps, parity_flip);

    debug!(
        "WCS refine done: {} matches, θ={:.4}°, RMSE={:.2}\" p90={:.2}\" max={:.2}\"",
        current_matches.len(),
        theta.to_degrees(),
        rmse.to_degrees() * 3600.0,
        p90e.to_degrees() * 3600.0,
        max_err.to_degrees() * 3600.0,
    );

    WcsRefineResult {
        cd_matrix: cd,
        crval_rad: [crval_ra, crval_dec],
        theta_rad: theta,
        matches: current_matches,
        rmse_rad: rmse,
    }
}

// ── Derive rotation from WCS ────────────────────────────────────────────────

/// Derive a 3×3 ICRS→camera rotation matrix, FOV, and parity from a WCS CD matrix + CRVAL.
///
/// The tangent-plane basis vectors at `CRVAL = (α₀, δ₀)` in ICRS are:
/// - ξ direction (East): `(-sin α₀, cos α₀, 0)`
/// - η direction (North): `(-sin δ₀ cos α₀, -sin δ₀ sin α₀, cos δ₀)`
/// - boresight: `(cos δ₀ cos α₀, cos δ₀ sin α₀, sin δ₀)`
///
/// The CD matrix maps pixel `(Δx, Δy)` to tangent-plane `(ξ, η)`, so the
/// camera X direction in the tangent plane is proportional to `(CD11, CD21)`.
///
/// # Returns
/// `(rotation_matrix_f32, fov_rad_f32, parity_flip)`
pub fn wcs_to_rotation(
    cd: &[[f64; 2]; 2],
    crval_ra: f64,
    crval_dec: f64,
    image_width: u32,
) -> (Matrix3<f32>, f32, bool) {
    let sin_a = crval_ra.sin();
    let cos_a = crval_ra.cos();
    let sin_d = crval_dec.sin();
    let cos_d = crval_dec.cos();

    // Tangent-plane basis vectors in ICRS
    let e_xi = Vector3::from_array([-sin_a, cos_a, 0.0]);
    let e_eta = Vector3::from_array([-sin_d * cos_a, -sin_d * sin_a, cos_d]);
    let boresight = Vector3::from_array([cos_d * cos_a, cos_d * sin_a, sin_d]);

    // Camera axes in ICRS (unnormalized)
    // Camera +X pixel direction → (CD11, CD21) in tangent-plane
    let cam_x_icrs_raw = e_xi * cd[0][0] + e_eta * cd[1][0];
    // Camera +Y pixel direction → (CD12, CD22) in tangent-plane
    let cam_y_icrs_raw = e_xi * cd[0][1] + e_eta * cd[1][1];

    let cam_x_icrs = cam_x_icrs_raw.normalize();
    let cam_y_icrs = cam_y_icrs_raw.normalize();

    // Build rotation matrix: rows are camera axes expressed in ICRS
    // R maps ICRS → camera: camera_vec = R * icrs_vec
    let rot = Matrix3::new([
        [
            cam_x_icrs[0] as f32,
            cam_x_icrs[1] as f32,
            cam_x_icrs[2] as f32,
        ],
        [
            cam_y_icrs[0] as f32,
            cam_y_icrs[1] as f32,
            cam_y_icrs[2] as f32,
        ],
        [
            boresight[0] as f32,
            boresight[1] as f32,
            boresight[2] as f32,
        ],
    ]);

    // FOV from pixel scale in X direction.
    // ps_x = 1/f (true pinhole). Angular FOV = 2·atan(W/(2f)) = 2·atan(ps_x·W/2).
    let ps_x = cam_x_icrs_raw.norm(); // radians per pixel
    let fov = (2.0 * ((ps_x * image_width as f64) / 2.0).atan()) as f32;

    // Parity from determinant of CD
    let det_cd = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
    let parity_flip = det_cd < 0.0;

    (rot, fov, parity_flip)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tan_project_roundtrip() {
        let crval_ra = 1.2_f64;
        let crval_dec = 0.3_f64;

        let test_points = [(1.21, 0.31), (1.25, 0.25), (1.15, 0.35), (1.0, 0.0)];

        for &(ra, dec) in &test_points {
            let (xi, eta) = tan_project(ra, dec, crval_ra, crval_dec).unwrap();
            let (ra2, dec2) = inverse_tan_project(xi, eta, crval_ra, crval_dec);
            assert!(
                (ra - ra2).abs() < 1e-12 && (dec - dec2).abs() < 1e-12,
                "Roundtrip failed for ({}, {}): got ({}, {})",
                ra,
                dec,
                ra2,
                dec2,
            );
        }
    }

    #[test]
    fn test_tan_project_at_reference() {
        let crval_ra = 2.0;
        let crval_dec = -0.5;
        let (xi, eta) = tan_project(crval_ra, crval_dec, crval_ra, crval_dec).unwrap();
        assert!(xi.abs() < 1e-15 && eta.abs() < 1e-15);
    }

    #[test]
    fn test_tan_project_behind() {
        let crval_ra = 0.0;
        let crval_dec = 0.0;
        assert!(tan_project(std::f64::consts::PI, 0.0, crval_ra, crval_dec).is_none());
    }

    #[test]
    fn test_inverse_tan_project_at_origin() {
        let crval_ra = 1.5;
        let crval_dec = 0.7;
        let (ra, dec) = inverse_tan_project(0.0, 0.0, crval_ra, crval_dec);
        assert!((ra - crval_ra).abs() < 1e-15);
        assert!((dec - crval_dec).abs() < 1e-15);
    }

    #[test]
    fn test_solve_3x3_identity() {
        let a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b = [3.0, 5.0, 7.0];
        let x = solve_3x3(&a, &b).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-12);
        assert!((x[1] - 5.0).abs() < 1e-12);
        assert!((x[2] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_solve_3x3_known() {
        let a = [[2.0, 3.0, 1.0], [1.0, 1.0, 1.0], [1.0, 2.0, 3.0]];
        let b = [11.0, 6.0, 14.0];
        let x = solve_3x3(&a, &b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
        assert!((x[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_3x3_singular() {
        let a = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]];
        let b = [6.0, 12.0, 3.0];
        assert!(solve_3x3(&a, &b).is_none());
    }

    #[test]
    fn test_cd_inverse_roundtrip() {
        let cd = [[1.2e-5, -3.0e-6], [2.5e-6, 1.1e-5]];
        let inv = cd_inverse(&cd).unwrap();
        let i00 = cd[0][0] * inv[0][0] + cd[0][1] * inv[1][0];
        let i01 = cd[0][0] * inv[0][1] + cd[0][1] * inv[1][1];
        let i10 = cd[1][0] * inv[0][0] + cd[1][1] * inv[1][0];
        let i11 = cd[1][0] * inv[0][1] + cd[1][1] * inv[1][1];
        assert!((i00 - 1.0).abs() < 1e-12);
        assert!(i01.abs() < 1e-12);
        assert!(i10.abs() < 1e-12);
        assert!((i11 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cd_from_theta_no_parity() {
        let theta = 0.3_f64; // ~17°
        let ps = 1.7e-5;
        let cd = cd_from_theta(theta, ps, false);

        // det should be positive
        let det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
        assert!(det > 0.0);

        // Decompose should recover theta and scale
        let (t, sx, sy, parity) = decompose_cd(&cd);
        assert!(!parity);
        assert!((t - theta).abs() < 1e-12, "theta: {:.6} vs {:.6}", t, theta);
        assert!((sx - ps).abs() < 1e-18, "scale_x: {:.6e} vs {:.6e}", sx, ps);
        assert!((sy - ps).abs() < 1e-18, "scale_y: {:.6e} vs {:.6e}", sy, ps);
    }

    #[test]
    fn test_cd_from_theta_with_parity() {
        let theta = -0.5_f64;
        let ps = 2.0e-5;
        let cd = cd_from_theta(theta, ps, true);

        // det should be negative
        let det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
        assert!(det < 0.0);

        let (t, sx, sy, parity) = decompose_cd(&cd);
        assert!(parity);
        assert!((t - theta).abs() < 1e-12);
        assert!((sx - ps).abs() < 1e-18);
        assert!((sy - ps).abs() < 1e-18);
    }

    #[test]
    fn test_predict_tanplane_roundtrip() {
        let cos_t = 0.3_f64.cos();
        let sin_t = 0.3_f64.sin();
        let ps = 1.5e-5;
        let inv_ps = 1.0 / ps;

        let (px, py) = (100.0, -200.0);
        let (xi, eta) = predict_tanplane(px, py, cos_t, sin_t, ps);
        let (px2, py2) = predict_pixel(xi, eta, cos_t, sin_t, inv_ps);
        assert!((px - px2).abs() < 1e-10);
        assert!((py - py2).abs() < 1e-10);
    }

    #[test]
    fn test_wcs_to_rotation_simple() {
        // True pinhole: ps = 1/f where f = (W/2) / tan(fov/2).
        let crval_ra = std::f64::consts::FRAC_PI_2;
        let crval_dec = 0.0;
        let fov_deg = 10.0_f64;
        let image_width = 1000u32;
        let f = (image_width as f64 / 2.0) / (fov_deg.to_radians() / 2.0).tan();
        let ps = 1.0 / f;

        let cd = [[ps, 0.0], [0.0, ps]];
        let (rot, fov, parity) = wcs_to_rotation(&cd, crval_ra, crval_dec, image_width);

        assert!(!parity);
        assert!(
            (fov.to_degrees() - 10.0).abs() < 0.01,
            "FOV: {}",
            fov.to_degrees()
        );

        let bore_cam = rot * Vector3::from_array([0.0_f32, 1.0, 0.0]);
        assert!(bore_cam[2] > 0.99, "boresight z = {}", bore_cam[2]);
    }

    #[test]
    fn test_decompose_cd_identity_like() {
        let ps = 1.5e-5;
        // No rotation, no parity
        let cd = [[ps, 0.0], [0.0, ps]];
        let (theta, sx, sy, parity) = decompose_cd(&cd);
        assert!(!parity);
        assert!(theta.abs() < 1e-12);
        assert!((sx - ps).abs() < 1e-18);
        assert!((sy - ps).abs() < 1e-18);
    }
}
