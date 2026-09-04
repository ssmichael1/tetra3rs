//! WCS TAN-projection iterative refinement (constrained).
//!
//! After the initial 4-star pattern match provides a seed rotation via SVD (Wahba's problem),
//! this module refines the solution by fitting 3 parameters per image:
//! **rotation angle θ** and **tangent-plane offset (dξ₀, dη₀)**, with the pixel
//! scale locked by the caller — the LIS path locks it to the pattern-match
//! refined FOV (robust to a wrong focal-length estimate), while the tracking
//! path locks it to the CameraModel's 1/f.
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

use super::matching::{greedy_unique_matches, MatchScratch};
use super::solve::StarVectors;
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

    let dec = (cos_c * sin_dec0 + eta * sin_c * cos_dec0 / rho)
        .clamp(-1.0, 1.0)
        .asin();
    let ra = crval_ra + (xi * sin_c).atan2(rho * cos_dec0 * cos_c - eta * sin_dec0 * sin_c);
    (ra, dec)
}

// ── 2×2 matrix helpers ─────────────────────────────────────────────────────

/// Invert a 2×2 matrix. Returns `None` if singular (|det| < 1e-30).
///
/// Retained for tests; production code converts CD ↔ (θ, ps, parity)
/// analytically via `cd_from_theta` / `rotation_from_theta_crval`.
#[cfg(test)]
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
/// The CD matrix maps *observed* pixel offsets to tangent-plane coordinates.
/// θ is the fitted roll of the solve's working pixel frame — the frame in
/// which a detected parity flip has already negated x — so for a flipped
/// image the observed x must be negated before the rotation is applied:
/// ```text
/// CD = ps * R(θ)               (if parity_flip=false, det > 0)
/// CD = ps * R(θ) * diag(−1, 1)  (if parity_flip=true, det < 0)
/// ```
pub fn cd_from_theta(theta: f64, pixel_scale: f64, parity_flip: bool) -> [[f64; 2]; 2] {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let ps = pixel_scale;
    if parity_flip {
        [[-ps * cos_t, -ps * sin_t], [-ps * sin_t, ps * cos_t]]
    } else {
        [[ps * cos_t, -ps * sin_t], [ps * sin_t, ps * cos_t]]
    }
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
        if max_abs < 1e-30 || !max_abs.is_finite() {
            return None; // singular, or NaN/inf contaminated
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

    // NaN/inf anywhere in the system propagates here; never hand it back.
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
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

/// Tangent-plane basis at CRVAL, `[e_ξ, e_η, boresight]` in ICRS — the
/// camera rows of [`camera_rows_f64`] at θ = 0. Built once per fit pass
/// (CRVAL moves between passes) and shared by every star in the pass.
type TanBasis = [[f64; 3]; 3];

#[inline]
fn tan_basis(crval_ra: f64, crval_dec: f64) -> TanBasis {
    camera_rows_f64(0.0, crval_ra, crval_dec)
}

/// TAN projection of a catalog unit vector onto the tangent plane spanned by
/// `basis` (see [`tan_basis`]).
///
/// Equivalent to [`tan_project`] with no per-star transcendentals: for a unit
/// vector `v`, `v·boresight` is the gnomonic denominator and `(v·e_ξ, v·e_η)`
/// its numerators (Calabretta & Greisen 2002, §5.1.1, in Cartesian form).
/// This is the same math Phase-D re-association already uses; it replaces the
/// former per-star `atan2`/`asin`/`sin`/`cos` decode plus per-pass `cos(Δα)`/
/// `sin(Δα)`. Returns `None` for a star on or behind the tangent plane.
#[inline]
fn tan_project_vec(sv: &[f32; 3], basis: &TanBasis) -> Option<(f64, f64)> {
    let v = [sv[0] as f64, sv[1] as f64, sv[2] as f64];
    let dot = |r: &[f64; 3]| r[0] * v[0] + r[1] * v[1] + r[2] * v[2];
    let denom = dot(&basis[2]);
    if denom <= 1e-12 {
        return None;
    }
    Some((dot(&basis[0]) / denom, dot(&basis[1]) / denom))
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
///
/// Retained for tests; Phase-D re-association projects with the camera rows
/// from [`camera_rows_f64`] instead (identical math, no per-star trig).
#[cfg(test)]
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
    /// The pixel scale the fit was locked to (radians per pixel), echoed back
    /// so callers can derive the focal length / FOV without decomposing the
    /// CD matrix.
    pub pixel_scale: f64,
    /// Final matched pairs: `(centroid_local_idx, catalog_star_idx)`.
    pub matches: Vec<(usize, usize)>,
    /// RMSE of angular residuals in radians.
    pub rmse_rad: f64,
    /// Covariance of the fitted parameters `[θ, ξ₀, η₀]` (rad²): roll about
    /// the boresight and the tangent-plane offsets of the boresight, East and
    /// North at CRVAL. `σ²·(JᵀJ)⁻¹` with `σ² = Σ residual² / (2n − 3)`; the
    /// diagonal is `+∞` when the fit is unconstrained (fewer than 2 matches)
    /// or the normal matrix is singular.
    pub covariance: [[f64; 3]; 3],
}

/// Covariance of `[θ, ξ₀, η₀]` from the normal matrix `JᵀJ` of the
/// converged fit, the sum of squared residuals, and the number of matched
/// stars (two residual components each). See [`WcsRefineResult::covariance`].
fn covariance_from_normal(ata: &[[f64; 3]; 3], sum_r2: f64, n_matches: usize) -> [[f64; 3]; 3] {
    const UNCONSTRAINED: [[f64; 3]; 3] = [
        [f64::INFINITY, 0.0, 0.0],
        [0.0, f64::INFINITY, 0.0],
        [0.0, 0.0, f64::INFINITY],
    ];
    let dof = 2 * n_matches as i64 - 3;
    if dof < 1 {
        return UNCONSTRAINED;
    }
    let sigma2 = sum_r2 / dof as f64;
    // (JᵀJ)⁻¹ column by column.
    let mut inv = [[0.0f64; 3]; 3];
    for col in 0..3 {
        let mut e = [0.0f64; 3];
        e[col] = 1.0;
        let Some(x) = solve_3x3(ata, &e) else {
            return UNCONSTRAINED;
        };
        for row in 0..3 {
            inv[row][col] = x[row];
        }
    }
    // Symmetrize (the two triangles differ only by solver rounding).
    let mut cov = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            cov[i][j] = sigma2 * 0.5 * (inv[i][j] + inv[j][i]);
        }
    }
    cov
}

/// Normal matrix `JᵀJ` of the 3-parameter fit over `matches` at the given
/// `(θ, CRVAL)` (no right-hand side), and the number of stars that projected
/// validly.
fn normal_matrix(
    matches: &[(usize, usize)],
    star_vectors: StarVectors<'_>,
    centroids_px: &[(f64, f64)],
    theta: f64,
    crval_ra: f64,
    crval_dec: f64,
    ps: f64,
) -> ([[f64; 3]; 3], usize) {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let basis = tan_basis(crval_ra, crval_dec);
    let mut ata = [[0.0f64; 3]; 3];
    let mut atb = [0.0f64; 3];
    let mut n = 0usize;
    for &(cent_idx, cat_idx) in matches {
        if tan_project_vec(&star_vectors.get(cat_idx), &basis).is_none() {
            continue;
        }
        let (px, py) = centroids_px[cent_idx];
        accumulate_normal_equations(&mut ata, &mut atb, px, py, cos_t, sin_t, ps, 0.0, 0.0);
        n += 1;
    }
    (ata, n)
}

// ── Main refinement entry point ─────────────────────────────────────────────

/// Robust statistics of a residual list: the median residual and the
/// MAD-derived standard-deviation estimate. See [`crate::stats::median_mad_sigma`].
fn residual_median_sigma(residuals: &[(usize, f64)]) -> (f64, f64) {
    let mut res_vals: Vec<f64> = residuals.iter().map(|&(_, r)| r).collect();
    crate::stats::median_mad_sigma(&mut res_vals)
}

/// Sigma-clip factor for MAD-based outlier rejection (Phase C and the final
/// clean-up passes share this convention).
const CLIP_NSIGMA: f64 = 3.0;

/// MAD-based outlier rejection: keep the matches whose residual is within
/// `median + CLIP_NSIGMA·σ`.
///
/// Returns `Some(kept)` only when the clip actually removed something AND at
/// least 4 matches survive (below that the 3-DOF fit is unconstrained) — the
/// caller keeps its previous match set otherwise.
fn mad_clip_matches(
    residuals: &[(usize, f64)],
    matches: &[(usize, usize)],
    median: f64,
    sigma_est: f64,
) -> Option<Vec<(usize, usize)>> {
    let clip_threshold = median + CLIP_NSIGMA * sigma_est;
    let mut keep: Vec<(usize, usize)> = Vec::new();
    for &(match_idx, residual) in residuals {
        if residual <= clip_threshold {
            keep.push(matches[match_idx]);
        }
    }
    if keep.len() < matches.len() && keep.len() >= 4 {
        Some(keep)
    } else {
        None
    }
}

/// Tangent-plane residual magnitude for each match under the current
/// `(θ, CRVAL)` fit: distance between the catalog star's TAN projection and
/// the centroid's predicted position. Returns `(match_idx, residual_rad)`
/// pairs; matches whose star projects behind the tangent plane are skipped.
fn compute_residuals(
    matches: &[(usize, usize)],
    star_vectors: StarVectors<'_>,
    centroids_px: &[(f64, f64)],
    theta: f64,
    crval_ra: f64,
    crval_dec: f64,
    ps: f64,
) -> Vec<(usize, f64)> {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let basis = tan_basis(crval_ra, crval_dec);

    let mut residuals: Vec<(usize, f64)> = Vec::with_capacity(matches.len());
    for (match_idx, &(cent_idx, cat_idx)) in matches.iter().enumerate() {
        if let Some((xi_cat, eta_cat)) = tan_project_vec(&star_vectors.get(cat_idx), &basis) {
            let (px, py) = centroids_px[cent_idx];
            let (xi_pred, eta_pred) = predict_tanplane(px, py, cos_t, sin_t, ps);
            let dxi = xi_pred - xi_cat;
            let deta = eta_pred - eta_cat;
            residuals.push((match_idx, (dxi * dxi + deta * deta).sqrt()));
        }
    }
    residuals
}

/// One least-squares pass for `[δθ, dξ₀, dη₀]`: build the 3-parameter normal
/// equations over `matches` at the current `(θ, CRVAL)` and solve them.
///
/// Returns `None` if fewer than 3 matches project validly or the normal
/// equations are singular — callers skip the update / stop iterating.
fn ls_fit_once(
    matches: &[(usize, usize)],
    star_vectors: StarVectors<'_>,
    centroids_px: &[(f64, f64)],
    theta: f64,
    crval_ra: f64,
    crval_dec: f64,
    ps: f64,
) -> Option<[f64; 3]> {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    // CRVAL changes between passes; build its tangent basis once per pass.
    let basis = tan_basis(crval_ra, crval_dec);

    let mut ata = [[0.0f64; 3]; 3];
    let mut atb = [0.0f64; 3];
    let mut n_valid = 0u32;

    for &(cent_idx, cat_idx) in matches {
        let Some((xi_cat, eta_cat)) = tan_project_vec(&star_vectors.get(cat_idx), &basis) else {
            continue;
        };

        let (px, py) = centroids_px[cent_idx];
        let (xi_pred, eta_pred) = predict_tanplane(px, py, cos_t, sin_t, ps);
        accumulate_normal_equations(
            &mut ata,
            &mut atb,
            px,
            py,
            cos_t,
            sin_t,
            ps,
            xi_cat - xi_pred,
            eta_cat - eta_pred,
        );
        n_valid += 1;
    }

    if n_valid < 3 {
        return None;
    }
    solve_3x3(&ata, &atb)
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
/// * `star_vectors` — catalog star ICRS unit vectors (aberration-corrected on
///   access when the solve asked for it), indexed by catalog star index.
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
    star_vectors: StarVectors<'_>,
    star_catalog: &StarCatalog,
    pixel_scale: f64,
    parity_flip: bool,
    match_radius_rad: f32,
    max_match_centroids: usize,
    max_iterations: u32,
) -> WcsRefineResult {
    // ── Constants ────────────────────────────────────────────────────────
    const CONVERGENCE_RAD: f64 = 1e-12; // tangent-plane offset convergence

    let ps = pixel_scale;
    let inv_ps = 1.0 / ps; // focal_length_px

    // ── Step 0: Extract initial CRVAL and θ from SVD rotation ──────────
    // Boresight in ICRS = R^T * [0, 0, 1] = third row of R
    let bx = initial_rotation[(2, 0)] as f64;
    let by = initial_rotation[(2, 1)] as f64;
    let bz = initial_rotation[(2, 2)] as f64;
    let mut crval_ra = by.atan2(bx);
    // f32 rotation rows can round a hair past ±1; asin would return NaN.
    let mut crval_dec = bz.clamp(-1.0, 1.0).asin();

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

    // Phase-D re-association cache: the fit barely moves between outer
    // iterations, so we query the catalog cone once (padded by the margin) and
    // reuse the star set until the fit drifts past the margin. The cached set
    // is a superset of any single iteration's query, and the extra (annulus)
    // stars project well outside the image so they never enter the greedy
    // matcher — results are unchanged.
    //
    // After the fresh-query projection pass the cached list is also *pruned*:
    // a star whose prediction lands beyond `prune_r` plus the drift allowances
    // below cannot re-enter the matcher while the cache is valid (boresight
    // drift ≤ `requery_margin` shifts predictions by ≤ ~margin_px, a θ drift
    // within its own bound by ≤ margin_px again), so dropping it is
    // behavior-preserving and shrinks every subsequent projection pass.
    //
    // Cache key is therefore (boresight, θ): re-query when either drifts past
    // its margin. θ matters only because of pruning — a rotation δθ moves a
    // prediction at pixel radius r by r·δθ.
    let requery_margin = match_radius_rad as f64 * 2.0;
    let requery_cos = requery_margin.cos();
    let mut reassoc_cache: Option<(Vector3<f64>, f64, Vec<usize>)> = None;

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
        // ── Phase A: LS fit (δθ, dξ₀, dη₀) ──────────────────────────
        for inner_iter in 0..10 {
            if current_matches.len() < 3 {
                break;
            }
            #[cfg(feature = "profile")]
            profiling::count(buckets::WCS_INNER, 1);

            let Some(sol) = ls_fit_once(
                &current_matches,
                star_vectors,
                centroids_px,
                theta,
                crval_ra,
                crval_dec,
                ps,
            ) else {
                debug!("WCS refine: LS fit failed (too few valid or singular), aborting");
                break;
            };

            let [d_theta, dxi_0, deta_0] = sol;

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
        let residuals = compute_residuals(
            &current_matches,
            star_vectors,
            centroids_px,
            theta,
            crval_ra,
            crval_dec,
            ps,
        );

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
            if let Some(keep) = mad_clip_matches(&residuals, &current_matches, median, sigma_est) {
                debug!(
                    "  outer {}: MAD clip: {} → {} matches (σ={:.2e} rad, threshold={:.2e} rad)",
                    outer_iter,
                    current_matches.len(),
                    keep.len(),
                    sigma_est,
                    median + CLIP_NSIGMA * sigma_est,
                );
                current_matches = keep;
            }
        }

        // ── Phase D: Re-associate (search for new inliers) ─────────────
        // Run every iteration (including the first): Phase A has already
        // converged the LS this pass, so the re-association is meaningful, and
        // detecting a stable match set here lets us break without burning an
        // extra confirming iteration. `n_rejected` keeps the existing
        // clip-driven behavior.
        {
            // Pixel radius for matching
            let radius_px = match_radius_rad as f64 / ps;

            // Adaptive radius from the MAD σ computed above (reused, not recomputed).
            let adaptive_radius_px = if let Some((_, sigma_est)) = mad_stats {
                (5.0 * sigma_est / ps).max(2.5).min(radius_px)
            } else {
                radius_px
            };

            // Matching cut: a star predicted farther than (max centroid radius
            // + match radius) from the optical center cannot fall within
            // `radius_px` of any centroid (triangle inequality) — it could
            // never match, so it is not pushed to the matcher.
            let prune_r = max_cent_dist_px + radius_px;
            let prune_r2 = prune_r * prune_r;
            // Cache-prune cut: prediction drift while the cache stays valid is
            // bounded by ~margin_px per drift source (boresight and θ, each
            // re-queried past its own margin below), plus one extra margin_px
            // of slack for the gnomonic stretch of off-axis stars. A star
            // beyond `keep_r` on the fresh pass stays beyond `prune_r` until
            // the next re-query.
            let margin_px = requery_margin / ps;
            let keep_r = prune_r + 3.0 * margin_px;
            let keep_r2 = keep_r * keep_r;
            // θ re-query bound: δθ moves a prediction at radius r by r·δθ, so
            // budget one margin_px at the pruning cut radius.
            let theta_margin = margin_px / keep_r;

            // Current boresight in ICRS.
            let boresight = Vector3::from_array([
                crval_dec.cos() * crval_ra.cos(),
                crval_dec.cos() * crval_ra.sin(),
                crval_dec.sin(),
            ]);

            // (Re)query the catalog cone only when the cache is empty or the
            // fit has drifted past a margin.
            let need_query = match &reassoc_cache {
                Some((qb, qtheta, _)) => {
                    qb.dot(&boresight) < requery_cos || (theta - qtheta).abs() > theta_margin
                }
                None => true,
            };
            if need_query {
                // The cached query reads the catalog's precomputed unit
                // vectors instead of recomputing each candidate's with
                // `sin`/`cos` — same star set, ~4x cheaper.
                let idx = timed!(
                    buckets::WCS_REASSOC_QUERY,
                    star_catalog.query_indices_from_uvec_cached(
                        Vector3::from_array([
                            boresight[0] as f32,
                            boresight[1] as f32,
                            boresight[2] as f32,
                        ]),
                        (search_radius + requery_margin) as f32,
                        star_vectors.base(),
                    )
                );
                #[cfg(feature = "profile")]
                {
                    profiling::count(buckets::WCS_REASSOC_CALL, 1);
                    profiling::count(buckets::WCS_REASSOC_STARS, idx.len() as u64);
                }
                reassoc_cache = Some((boresight, theta, idx));
            }
            let (_, _, nearby_indices) = reassoc_cache.as_mut().unwrap();

            // Project each cached catalog star to pixel coords with the camera
            // rows built once from the current fit — identical math to the TAN
            // projection (`z` is the same denominator, same behind-plane cut)
            // with no per-star transcendentals. On the fresh-query pass, prune
            // the cached list to `keep_r` (see above).
            let [row_x, row_y, row_z] = camera_rows_f64(theta, crval_ra, crval_dec);
            timed!(buckets::WCS_REASSOC_PROJECT, {
                predicted.clear();
                let mut kept = 0usize;
                for k in 0..nearby_indices.len() {
                    let cat_idx = nearby_indices[k];
                    let sv = star_vectors.get(cat_idx);
                    let v = [sv[0] as f64, sv[1] as f64, sv[2] as f64];
                    let z = row_z[0] * v[0] + row_z[1] * v[1] + row_z[2] * v[2];
                    if z <= 1e-12 {
                        continue; // behind or on the tangent plane
                    }
                    let pred_x = inv_ps * (row_x[0] * v[0] + row_x[1] * v[1] + row_x[2] * v[2]) / z;
                    let pred_y = inv_ps * (row_y[0] * v[0] + row_y[1] * v[1] + row_y[2] * v[2]) / z;
                    let r2 = pred_x * pred_x + pred_y * pred_y;
                    if r2 <= prune_r2 {
                        predicted.push((cat_idx, pred_x, pred_y));
                    }
                    if need_query && r2 <= keep_r2 {
                        nearby_indices[kept] = cat_idx;
                        kept += 1;
                    }
                }
                if need_query {
                    nearby_indices.truncate(kept);
                }
            });

            let new_matches: &[(usize, usize)] = timed!(
                buckets::WCS_REASSOC_MATCH,
                greedy_unique_matches(
                    centroids_px,
                    max_match_centroids,
                    &predicted,
                    adaptive_radius_px * adaptive_radius_px,
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

        let residuals = compute_residuals(
            &current_matches,
            star_vectors,
            centroids_px,
            theta,
            crval_ra,
            crval_dec,
            ps,
        );

        if residuals.len() < 6 {
            break;
        }

        let (median, sigma_est) = residual_median_sigma(&residuals);
        let Some(keep) = mad_clip_matches(&residuals, &current_matches, median, sigma_est) else {
            // Nothing clipped, or too few survivors — the set is clean (or as
            // clean as it can get); stop.
            break;
        };

        debug!(
            "  final clip {}: {} → {} matches",
            clip_pass,
            current_matches.len(),
            keep.len(),
        );
        current_matches = keep;

        // Re-fit theta + CRVAL on the cleaned set (one LS pass)
        if let Some(sol) = ls_fit_once(
            &current_matches,
            star_vectors,
            centroids_px,
            theta,
            crval_ra,
            crval_dec,
            ps,
        ) {
            theta += sol[0];
            let (new_ra, new_dec) = inverse_tan_project(sol[1], sol[2], crval_ra, crval_dec);
            crval_ra = new_ra;
            crval_dec = new_dec;
        }
    }

    // ── Compute final residual statistics ────────────────────────────────
    // The RMSE feeds `WcsRefineResult::rmse_rad`; the p90/max order statistics
    // exist only for the debug log below, so the sort they need is skipped
    // entirely unless DEBUG logging is enabled (this runs once per solve on the
    // profiling-dominant path).
    let mut final_residuals: Vec<f64> = compute_residuals(
        &current_matches,
        star_vectors,
        centroids_px,
        theta,
        crval_ra,
        crval_dec,
        ps,
    )
    .into_iter()
    .map(|(_, r)| r)
    .collect();

    let sum_r2: f64 = final_residuals.iter().map(|r| r * r).sum();
    let rmse = if final_residuals.is_empty() {
        0.0
    } else {
        (sum_r2 / final_residuals.len() as f64).sqrt()
    };

    // Parameter covariance at the converged fit (see `WcsRefineResult`).
    let (ata, n_fit) = normal_matrix(
        &current_matches,
        star_vectors,
        centroids_px,
        theta,
        crval_ra,
        crval_dec,
        ps,
    );
    let covariance = covariance_from_normal(&ata, sum_r2, n_fit);

    // Derive CD matrix from (theta, pixel_scale, parity)
    let cd = cd_from_theta(theta, ps, parity_flip);

    if tracing::enabled!(tracing::Level::DEBUG) {
        final_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p90e = if final_residuals.is_empty() {
            0.0
        } else {
            final_residuals[(0.9 * (final_residuals.len() - 1) as f64) as usize]
        };
        let max_err = final_residuals.last().copied().unwrap_or(0.0);
        debug!(
            "WCS refine done: {} matches, θ={:.4}°, RMSE={:.2}\" p90={:.2}\" max={:.2}\"",
            current_matches.len(),
            theta.to_degrees(),
            rmse.to_degrees() * 3600.0,
            p90e.to_degrees() * 3600.0,
            max_err.to_degrees() * 3600.0,
        );
    }

    WcsRefineResult {
        cd_matrix: cd,
        crval_rad: [crval_ra, crval_dec],
        theta_rad: theta,
        pixel_scale: ps,
        matches: current_matches,
        rmse_rad: rmse,
        covariance,
    }
}

/// Build the ICRS→camera rotation matrix directly from the constrained-fit
/// parameters `(θ, CRVAL)`.
///
/// θ is the fitted roll of the solve's *working* pixel frame — the frame in
/// which a detected parity flip has already negated x. That frame is always
/// a proper right-handed frame (negating x undoes the mirror), so the same
/// formula applies regardless of parity and the result always has det +1:
/// `cam_x = cosθ·e_ξ + sinθ·e_η`, `cam_y = −sinθ·e_ξ + cosθ·e_η`. The mirror
/// itself is recorded separately in `parity_flip` (callers must negate
/// observed x before applying this rotation when it is set). Equivalent to
/// `wcs_to_rotation(&cd_from_theta(theta, ps, parity), …)` — the pixel scale
/// cancels in the normalization, so it is not needed.
pub fn rotation_from_theta_crval(theta: f64, crval_ra: f64, crval_dec: f64) -> Matrix3<f32> {
    let [cam_x, cam_y, boresight] = camera_rows_f64(theta, crval_ra, crval_dec);

    // Rows are camera axes expressed in ICRS: camera_vec = R * icrs_vec
    Matrix3::new([
        [cam_x[0] as f32, cam_x[1] as f32, cam_x[2] as f32],
        [cam_y[0] as f32, cam_y[1] as f32, cam_y[2] as f32],
        [
            boresight[0] as f32,
            boresight[1] as f32,
            boresight[2] as f32,
        ],
    ])
}

/// Camera axes (rows of the ICRS→camera rotation) in f64 for the constrained
/// fit `(θ, CRVAL)`: `[cam_x, cam_y, boresight]`.
///
/// Shared by [`rotation_from_theta_crval`] and Phase-D re-association, which
/// projects catalog stars with these rows directly: for a star unit vector v,
/// `z = v·boresight` equals the TAN-projection denominator, and
/// `(v·cam_x)/z, (v·cam_y)/z` equal `predict_pixel(tan_project(v))·ps` — the
/// same math without per-star transcendentals.
fn camera_rows_f64(theta: f64, crval_ra: f64, crval_dec: f64) -> [[f64; 3]; 3] {
    let sin_a = crval_ra.sin();
    let cos_a = crval_ra.cos();
    let sin_d = crval_dec.sin();
    let cos_d = crval_dec.cos();

    // Tangent-plane basis vectors in ICRS
    let e_xi = Vector3::<f64>::from_array([-sin_a, cos_a, 0.0]);
    let e_eta = Vector3::<f64>::from_array([-sin_d * cos_a, -sin_d * sin_a, cos_d]);
    let boresight = Vector3::<f64>::from_array([cos_d * cos_a, cos_d * sin_a, sin_d]);

    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let cam_x = (e_xi * cos_t + e_eta * sin_t).normalize();
    let cam_y = (e_xi * -sin_t + e_eta * cos_t).normalize();

    [
        [cam_x[0], cam_x[1], cam_x[2]],
        [cam_y[0], cam_y[1], cam_y[2]],
        [boresight[0], boresight[1], boresight[2]],
    ]
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
/// observed +X pixel direction in the tangent plane is proportional to
/// `(CD11, CD21)`.
///
/// When `det(CD) < 0` the image is mirrored (`parity_flip = true`). The
/// returned matrix is then the rotation of the *x-negated* (proper) frame —
/// the observed +X column is negated so the rows always form a right-handed
/// triad (det +1) — matching the convention of [`Solution`]'s
/// `qicrs2cam` / `parity_flip` pair: negate observed x before applying the
/// rotation.
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

    // Parity from determinant of CD
    let det_cd = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
    let parity_flip = det_cd < 0.0;

    // Camera axes in ICRS (unnormalized)
    // Observed +X pixel direction → (CD11, CD21) in tangent-plane. For a
    // mirrored image the working (proper) frame's +X is the negation.
    let cam_x_icrs_raw = if parity_flip {
        -(e_xi * cd[0][0] + e_eta * cd[1][0])
    } else {
        e_xi * cd[0][0] + e_eta * cd[1][0]
    };
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
    fn covariance_from_normal_scales_the_inverse() {
        // Diagonal normal matrix: cov = σ²·diag(1/a).
        let ata = [[4.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
        let n = 10; // dof = 17
        let sum_r2 = 17.0 * 1e-10; // σ² = 1e-10
        let cov = covariance_from_normal(&ata, sum_r2, n);
        assert!((cov[0][0] - 1e-10 / 4.0).abs() < 1e-22);
        assert!((cov[1][1] - 1e-10 / 2.0).abs() < 1e-22);
        assert!((cov[2][2] - 1e-10).abs() < 1e-22);
        assert_eq!(cov[0][1], 0.0);
        // Unconstrained: too few stars, or singular normal matrix.
        assert!(covariance_from_normal(&ata, 1.0, 1)[0][0].is_infinite());
        let singular = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(covariance_from_normal(&singular, 1.0, 10)[1][1].is_infinite());
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

        // det positive (proper rotation), and CD == ps·R(θ) element-wise.
        let det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
        assert!(det > 0.0);
        let (c, s) = (theta.cos(), theta.sin());
        let expected = [[ps * c, -ps * s], [ps * s, ps * c]];
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (cd[i][j] - expected[i][j]).abs() < 1e-18,
                    "CD[{i}][{j}]: {:.6e} vs {:.6e}",
                    cd[i][j],
                    expected[i][j]
                );
            }
        }
    }

    #[test]
    fn test_cd_from_theta_with_parity() {
        let theta = -0.5_f64;
        let ps = 2.0e-5;
        let cd = cd_from_theta(theta, ps, true);

        // det negative (mirror), and CD == ps·R(θ)·diag(−1, 1) element-wise.
        let det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
        assert!(det < 0.0);
        let (c, s) = (theta.cos(), theta.sin());
        let expected = [[-ps * c, -ps * s], [-ps * s, ps * c]];
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (cd[i][j] - expected[i][j]).abs() < 1e-18,
                    "CD[{i}][{j}]: {:.6e} vs {:.6e}",
                    cd[i][j],
                    expected[i][j]
                );
            }
        }
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

    /// Angle of the relative rotation between two rotation matrices, radians.
    fn rotation_angle_between(a: &Matrix3<f32>, b: &Matrix3<f32>) -> f64 {
        let rel = *a * b.transpose();
        let trace = (rel[(0, 0)] + rel[(1, 1)] + rel[(2, 2)]) as f64;
        ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos()
    }

    #[test]
    fn test_rotation_from_theta_crval_always_proper() {
        // Regression: the parity case used to return a reflection (det −1),
        // corrupting the quaternion and residuals of every parity-flipped
        // solve. The rotation describes the x-negated working frame, which is
        // proper, so det must be +1 for all inputs.
        for &theta in &[0.0_f64, 0.3, -0.5, 1.2, 3.0] {
            let rot = rotation_from_theta_crval(theta, 1.1, 0.4);
            assert!(
                (rot.det() - 1.0).abs() < 1e-5,
                "det = {} for theta = {}",
                rot.det(),
                theta
            );
        }
    }

    #[test]
    fn test_parity_conventions_consistent() {
        // End-to-end convention check for a mirrored image: the fit model
        // (predict_tanplane on x-negated pixels), the final rotation
        // (rotation_from_theta_crval), the exported CD matrix
        // (cd_from_theta with parity), and wcs_to_rotation must all agree.
        let crval_ra = 1.1_f64;
        let crval_dec = 0.4_f64;
        let theta = 0.3_f64; // roll of the x-negated working frame
        let ps = 10.0_f64.to_radians() / 1000.0; // ~10° over 1000 px

        let rot = rotation_from_theta_crval(theta, crval_ra, crval_dec);
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Stars synthesized at working-frame (x-negated) pixel coords.
        for &(px, py) in &[(300.0_f64, -200.0_f64), (-450.0, 100.0), (50.0, 425.0)] {
            // ICRS direction via the rotation: v = R^T · pixel_vector
            let norm = (px * px * ps * ps + py * py * ps * ps + 1.0).sqrt();
            let v_pix = Vector3::<f32>::from_array([
                (px * ps / norm) as f32,
                (py * ps / norm) as f32,
                (1.0 / norm) as f32,
            ]);
            let v_icrs = rot.transpose() * v_pix;
            let ra = (v_icrs[1] as f64).atan2(v_icrs[0] as f64);
            let dec = (v_icrs[2] as f64).asin();

            // Its TAN projection must equal the fit model's prediction.
            let (xi_cat, eta_cat) = tan_project(ra, dec, crval_ra, crval_dec).unwrap();
            let (xi_fit, eta_fit) = predict_tanplane(px, py, cos_t, sin_t, ps);
            // f32 unit vectors limit agreement to ~1e-7 rad (~0.02″)
            assert!(
                (xi_cat - xi_fit).abs() < 1e-6 && (eta_cat - eta_fit).abs() < 1e-6,
                "rotation/fit mismatch at ({px}, {py}): cat=({xi_cat:.3e}, {eta_cat:.3e}) fit=({xi_fit:.3e}, {eta_fit:.3e})"
            );

            // The CD matrix applies to OBSERVED pixels (x mirrored back).
            let (x_obs, y_obs) = (-px, py);
            let cd = cd_from_theta(theta, ps, true);
            let xi_cd = cd[0][0] * x_obs + cd[0][1] * y_obs;
            let eta_cd = cd[1][0] * x_obs + cd[1][1] * y_obs;
            assert!(
                (xi_cd - xi_fit).abs() < 1e-12 && (eta_cd - eta_fit).abs() < 1e-12,
                "CD/fit mismatch at ({px}, {py})"
            );
        }

        // wcs_to_rotation must recover the same proper rotation + parity flag
        // from the exported CD matrix.
        let cd = cd_from_theta(theta, ps, true);
        let (rot_back, _fov, parity) = wcs_to_rotation(&cd, crval_ra, crval_dec, 1000);
        assert!(parity, "det(CD) < 0 must report parity_flip");
        assert!((rot_back.det() - 1.0).abs() < 1e-5);
        let ang = rotation_angle_between(&rot, &rot_back);
        assert!(
            ang < 1e-6,
            "wcs_to_rotation disagrees with rotation_from_theta_crval by {ang:.2e} rad"
        );
    }
}
