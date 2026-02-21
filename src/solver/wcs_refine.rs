//! WCS TAN-projection iterative refinement.
//!
//! After the initial 4-star pattern match provides a seed rotation via SVD (Wahba's problem),
//! this module refines the solution by fitting a WCS CD matrix + CRVAL using linear
//! least-squares in the gnomonic tangent plane. This replaces the previous iterative-SVD
//! + golden-section FOV tuning approach.
//!
//! The CD matrix (2×2) maps pixel offsets from the optical center (CRPIX) to tangent-plane
//! coordinates (radians) at the reference point (CRVAL). It naturally captures pixel scale,
//! rotation, parity, and any pixel-axis skew in a single linear transform.
//!
//! ## Algorithm
//!
//! 1. Extract initial CRVAL (RA, Dec of boresight) from the SVD rotation matrix.
//! 2. Iteratively:
//!    a. TAN-project matched catalog stars at current CRVAL → (ξ, η) in radians.
//!    b. Solve two independent 3-parameter linear systems for
//!       `[CD11, CD12, dξ₀]` and `[CD21, CD22, dη₀]`.
//!    c. Update CRVAL from `(dξ₀, dη₀)` via inverse TAN projection.
//!    d. MAD-based outlier rejection.
//!    e. Re-associate: project catalog stars to pixel space via CD⁻¹, match to centroids.
//!    f. Converge when offsets vanish, no outliers rejected, and match set is stable.

use nalgebra::{Matrix3, Vector3};
use tracing::debug;

use crate::starcatalog::StarCatalog;

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

// ── 3×3 linear solve ────────────────────────────────────────────────────────

/// Solve a 3×3 linear system `Ax = b` via Gaussian elimination with partial pivoting.
///
/// The normal equations `(AᵀA)x = Aᵀb` for our 3-parameter LS problem are always 3×3,
/// so this avoids pulling in a general linear algebra solver.
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

/// Greedy 1-to-1 matching between centroid pixel positions and predicted catalog positions.
///
/// Returns `Vec<(centroid_idx, catalog_star_idx)>` of unique matches
/// within `radius_px` pixels.
fn find_pixel_matches(
    centroid_pixels: &[(f64, f64)],
    max_centroids: usize,
    predicted: &[(usize, f64, f64)], // (catalog_star_idx, pred_x, pred_y)
    radius_px: f64,
) -> Vec<(usize, usize)> {
    let radius_sq = radius_px * radius_px;
    let n_cent = centroid_pixels.len().min(max_centroids);

    // Collect all candidate pairs within radius
    let mut candidates: Vec<(f64, usize, usize)> = Vec::new(); // (dist_sq, cent_idx, cat_idx)
    for (cent_idx, &(cx, cy)) in centroid_pixels[..n_cent].iter().enumerate() {
        for &(cat_idx, px, py) in predicted {
            let dx = cx - px;
            let dy = cy - py;
            let d2 = dx * dx + dy * dy;
            if d2 <= radius_sq {
                candidates.push((d2, cent_idx, cat_idx));
            }
        }
    }

    // Sort by distance (closest first)
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Greedy unique 1-to-1 assignment
    let mut used_cent = vec![false; n_cent];
    let mut used_cat = std::collections::HashSet::new();
    let mut matches = Vec::new();

    for &(_, cent_idx, cat_idx) in &candidates {
        if !used_cent[cent_idx] && !used_cat.contains(&cat_idx) {
            used_cent[cent_idx] = true;
            used_cat.insert(cat_idx);
            matches.push((cent_idx, cat_idx));
        }
    }

    matches
}

// ── WCS refinement result ───────────────────────────────────────────────────

/// Result of the WCS TAN-projection iterative refinement.
pub struct WcsRefineResult {
    /// CD matrix: `[[CD11, CD12], [CD21, CD22]]` in tangent-plane radians per pixel.
    pub cd_matrix: [[f64; 2]; 2],
    /// Reference point `[RA, Dec]` in radians.
    pub crval_rad: [f64; 2],
    /// Final matched pairs: `(centroid_local_idx, catalog_star_idx)`.
    pub matches: Vec<(usize, usize)>,
    /// RMSE of angular residuals in radians.
    pub rmse_rad: f64,
    /// 90th-percentile angular residual in radians.
    pub p90e_rad: f64,
    /// Maximum angular residual in radians.
    pub max_err_rad: f64,
}

// ── Main refinement entry point ─────────────────────────────────────────────

/// Iterative WCS TAN-projection refinement.
///
/// Starting from an initial rotation matrix (from the SVD pattern match) and an initial
/// match set (from verification), refines the WCS solution by fitting a CD matrix + CRVAL
/// via linear least-squares in the tangent plane, with MAD-based outlier rejection and
/// catalog re-association.
///
/// # Arguments
///
/// * `initial_rotation` — 3×3 ICRS→camera rotation from the initial SVD solve.
/// * `initial_matches` — initial matched pairs `(centroid_local_idx, catalog_star_idx)`.
/// * `centroids_px` — pixel coordinates of centroids after undistortion and CRPIX
///   subtraction, with parity already applied. Indexed by local_idx (brightness-sorted).
/// * `star_vectors` — catalog star ICRS unit vectors, indexed by catalog star index.
/// * `star_catalog` — spatial index for cone queries.
/// * `match_radius_rad` — initial match radius in radians (from `config.match_radius * fov`).
/// * `max_match_centroids` — maximum number of centroids to consider for matching.
/// * `max_iterations` — maximum outer-loop iterations.
///
/// # Returns
///
/// A [`WcsRefineResult`] with the refined CD matrix, CRVAL, match set, and residual stats.
#[allow(clippy::too_many_arguments)]
pub fn wcs_refine(
    initial_rotation: &Matrix3<f32>,
    initial_matches: &[(usize, usize)],
    centroids_px: &[(f64, f64)],
    star_vectors: &[[f32; 3]],
    star_catalog: &StarCatalog,
    match_radius_rad: f32,
    max_match_centroids: usize,
    max_iterations: u32,
) -> WcsRefineResult {
    // ── Constants ────────────────────────────────────────────────────────
    const MAD_SCALE: f64 = 1.4826; // MAD → σ for Gaussian
    const CLIP_NSIGMA: f64 = 3.0;
    const CONVERGENCE_RAD: f64 = 1e-12; // tangent-plane offset convergence

    // ── Step 0: Extract initial CRVAL from SVD rotation ─────────────────
    // Boresight in ICRS = R^T * [0, 0, 1] = third row of R (since R^T_i2 = R_2i)
    let bx = initial_rotation[(2, 0)] as f64;
    let by = initial_rotation[(2, 1)] as f64;
    let bz = initial_rotation[(2, 2)] as f64;
    let mut crval_ra = by.atan2(bx);
    let mut crval_dec = bz.asin();

    debug!(
        "WCS refine: initial CRVAL = ({:.4}°, {:.4}°), {} matches, {} centroids",
        crval_ra.to_degrees(),
        crval_dec.to_degrees(),
        initial_matches.len(),
        centroids_px.len(),
    );

    // ── Working state ───────────────────────────────────────────────────
    let mut current_matches: Vec<(usize, usize)> = initial_matches.to_vec();
    let mut cd = [[0.0f64; 2]; 2]; // will be set on first iteration

    // ── Outer refinement loop ───────────────────────────────────────────
    for outer_iter in 0..max_iterations {
        // ── Phase A: LS fit CD + offset ─────────────────────────────────
        // Iterate CRVAL/CD convergence (inner loop — typically 2-4 iterations)
        for inner_iter in 0..10 {
            if current_matches.len() < 3 {
                break;
            }

            // TAN-project all matched catalog stars at current CRVAL
            // and build design matrix A = [x_i, y_i, 1]
            // with observation vectors xi_vec, eta_vec
            let mut ata = [[0.0f64; 3]; 3];
            let mut atb_xi = [0.0f64; 3];
            let mut atb_eta = [0.0f64; 3];
            let mut n_valid = 0u32;

            for &(cent_idx, cat_idx) in &current_matches {
                let sv = &star_vectors[cat_idx];
                let star_ra = (sv[1] as f64).atan2(sv[0] as f64);
                let star_dec = (sv[2] as f64).asin();

                let Some((xi, eta)) = tan_project(star_ra, star_dec, crval_ra, crval_dec) else {
                    continue;
                };

                let (px, py) = centroids_px[cent_idx];
                let row = [px, py, 1.0];

                // Accumulate A^T A and A^T b (shared design matrix for both xi and eta)
                for i in 0..3 {
                    for j in 0..3 {
                        ata[i][j] += row[i] * row[j];
                    }
                    atb_xi[i] += row[i] * xi;
                    atb_eta[i] += row[i] * eta;
                }
                n_valid += 1;
            }

            if n_valid < 3 {
                break;
            }

            // Solve the two 3×3 systems
            let Some(sol_xi) = solve_3x3(&ata, &atb_xi) else {
                debug!("WCS refine: singular normal equations (xi), aborting");
                break;
            };
            let Some(sol_eta) = solve_3x3(&ata, &atb_eta) else {
                debug!("WCS refine: singular normal equations (eta), aborting");
                break;
            };

            // Extract CD matrix and tangent-plane offset
            cd = [
                [sol_xi[0], sol_xi[1]],
                [sol_eta[0], sol_eta[1]],
            ];
            let dxi_0 = sol_xi[2];
            let deta_0 = sol_eta[2];

            // Update CRVAL from tangent-plane offset
            let (new_ra, new_dec) = inverse_tan_project(dxi_0, deta_0, crval_ra, crval_dec);
            crval_ra = new_ra;
            crval_dec = new_dec;

            debug!(
                "  inner {}: CD=[{:.6e}, {:.6e}; {:.6e}, {:.6e}], offset=({:.3e}, {:.3e}) rad",
                inner_iter, cd[0][0], cd[0][1], cd[1][0], cd[1][1], dxi_0, deta_0,
            );

            // Check convergence
            if dxi_0.abs() + deta_0.abs() < CONVERGENCE_RAD {
                break;
            }
        }

        // ── Phase B: Compute residuals ──────────────────────────────────
        let mut residuals: Vec<(usize, f64)> = Vec::with_capacity(current_matches.len());
        for (match_idx, &(cent_idx, cat_idx)) in current_matches.iter().enumerate() {
            let sv = &star_vectors[cat_idx];
            let star_ra = (sv[1] as f64).atan2(sv[0] as f64);
            let star_dec = (sv[2] as f64).asin();

            if let Some((xi_cat, eta_cat)) = tan_project(star_ra, star_dec, crval_ra, crval_dec) {
                let (px, py) = centroids_px[cent_idx];
                let xi_pred = cd[0][0] * px + cd[0][1] * py;
                let eta_pred = cd[1][0] * px + cd[1][1] * py;
                let dxi = xi_pred - xi_cat;
                let deta = eta_pred - eta_cat;
                let residual = (dxi * dxi + deta * deta).sqrt(); // radians
                residuals.push((match_idx, residual));
            }
        }

        // ── Phase C: MAD-based outlier rejection ────────────────────────
        let mut n_rejected = 0usize;

        if residuals.len() >= 6 {
            let mut res_vals: Vec<f64> = residuals.iter().map(|&(_, r)| r).collect();
            res_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = res_vals[res_vals.len() / 2];
            let mut abs_devs: Vec<f64> = res_vals.iter().map(|r| (r - median).abs()).collect();
            abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mad = abs_devs[abs_devs.len() / 2];
            let sigma_est = MAD_SCALE * mad;
            let clip_threshold = median + CLIP_NSIGMA * sigma_est;

            let old_len = current_matches.len();
            let mut keep_matches: Vec<(usize, usize)> = Vec::new();
            for &(match_idx, residual) in &residuals {
                if residual <= clip_threshold {
                    keep_matches.push(current_matches[match_idx]);
                }
            }
            n_rejected = old_len - keep_matches.len();

            if n_rejected > 0 && keep_matches.len() >= 4 {
                debug!(
                    "  outer {}: MAD clip: {} → {} matches (σ={:.2e} rad, threshold={:.2e} rad)",
                    outer_iter,
                    old_len,
                    keep_matches.len(),
                    sigma_est,
                    clip_threshold,
                );
                current_matches = keep_matches;
            } else {
                n_rejected = 0; // not enough inliers or nothing to clip
            }
        }

        // ── Phase D: Re-associate (search for new inliers) ─────────────
        // Only on iterations after the first (let the first iteration establish CD)
        if outer_iter > 0 || n_rejected > 0 {
            if let Some(cd_inv) = cd_inverse(&cd) {
                // Pixel scale (approx) for converting match radius
                let ps_x = (cd[0][0] * cd[0][0] + cd[1][0] * cd[1][0]).sqrt();
                let radius_px = match_radius_rad as f64 / ps_x;

                // Adaptive radius from MAD if available
                let adaptive_radius_px = if residuals.len() >= 6 {
                    let mut res_vals: Vec<f64> = residuals.iter().map(|&(_, r)| r).collect();
                    res_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mad = {
                        let median = res_vals[res_vals.len() / 2];
                        let mut ad: Vec<f64> = res_vals.iter().map(|r| (r - median).abs()).collect();
                        ad.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        ad[ad.len() / 2]
                    };
                    let sigma_est = MAD_SCALE * mad;
                    let adaptive_rad = (5.0 * sigma_est / ps_x).max(2.5).min(radius_px);
                    adaptive_rad
                } else {
                    radius_px
                };

                // Query catalog stars near boresight
                let boresight = Vector3::new(
                    crval_dec.cos() * crval_ra.cos(),
                    crval_dec.cos() * crval_ra.sin(),
                    crval_dec.sin(),
                );
                // Search cone: use maximum centroid distance from center (in pixels)
                // converted to radians, with 1.5× margin for edge effects.
                let max_cent_dist_px = centroids_px
                    .iter()
                    .map(|(x, y)| (x * x + y * y).sqrt())
                    .fold(0.0f64, f64::max);
                let search_radius =
                    (ps_x * max_cent_dist_px * 1.5).max(match_radius_rad as f64 * 2.0);
                let nearby_indices = star_catalog.query_indices_from_uvec(
                    Vector3::new(boresight.x as f32, boresight.y as f32, boresight.z as f32),
                    search_radius as f32,
                );

                // Project each catalog star to pixel coords via TAN + CD_inv
                let mut predicted: Vec<(usize, f64, f64)> = Vec::new();
                for &cat_idx in &nearby_indices {
                    let sv = &star_vectors[cat_idx];
                    let sra = (sv[1] as f64).atan2(sv[0] as f64);
                    let sdec = (sv[2] as f64).asin();
                    if let Some((xi, eta)) = tan_project(sra, sdec, crval_ra, crval_dec) {
                        let pred_x = cd_inv[0][0] * xi + cd_inv[0][1] * eta;
                        let pred_y = cd_inv[1][0] * xi + cd_inv[1][1] * eta;
                        predicted.push((cat_idx, pred_x, pred_y));
                    }
                }

                let new_matches = find_pixel_matches(
                    centroids_px,
                    max_match_centroids,
                    &predicted,
                    adaptive_radius_px,
                );

                if new_matches.len() >= 4 {
                    let mut sorted_new = new_matches.clone();
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
                        current_matches = new_matches;
                        // Do another refinement pass with the new match set
                        continue;
                    }
                }
            }
        }

        // Converged: either no outliers rejected, or re-association returned
        // the same match set (clipped matches were genuine outliers, no replacements)
        if outer_iter > 0 {
            debug!("  outer {}: converged", outer_iter);
            break;
        }
    }

    // ── Final MAD clip passes (clip-only, no re-association) ────────────
    for clip_pass in 0..3 {
        if current_matches.len() < 6 {
            break;
        }

        let mut residuals: Vec<(usize, f64)> = Vec::new();
        for (match_idx, &(cent_idx, cat_idx)) in current_matches.iter().enumerate() {
            let sv = &star_vectors[cat_idx];
            let sra = (sv[1] as f64).atan2(sv[0] as f64);
            let sdec = (sv[2] as f64).asin();
            if let Some((xi_cat, eta_cat)) = tan_project(sra, sdec, crval_ra, crval_dec) {
                let (px, py) = centroids_px[cent_idx];
                let xi_pred = cd[0][0] * px + cd[0][1] * py;
                let eta_pred = cd[1][0] * px + cd[1][1] * py;
                let dxi = xi_pred - xi_cat;
                let deta = eta_pred - eta_cat;
                residuals.push((match_idx, (dxi * dxi + deta * deta).sqrt()));
            }
        }

        if residuals.len() < 6 {
            break;
        }

        let mut res_vals: Vec<f64> = residuals.iter().map(|&(_, r)| r).collect();
        res_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = res_vals[res_vals.len() / 2];
        let mut abs_devs: Vec<f64> = res_vals.iter().map(|r| (r - median).abs()).collect();
        abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = abs_devs[abs_devs.len() / 2];
        let sigma_est = MAD_SCALE * mad;
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

        // Re-fit CD on cleaned set (one inner LS pass)
        let mut ata = [[0.0f64; 3]; 3];
        let mut atb_xi = [0.0f64; 3];
        let mut atb_eta = [0.0f64; 3];
        for &(cent_idx, cat_idx) in &current_matches {
            let sv = &star_vectors[cat_idx];
            let sra = (sv[1] as f64).atan2(sv[0] as f64);
            let sdec = (sv[2] as f64).asin();
            if let Some((xi, eta)) = tan_project(sra, sdec, crval_ra, crval_dec) {
                let (px, py) = centroids_px[cent_idx];
                let row = [px, py, 1.0];
                for i in 0..3 {
                    for j in 0..3 {
                        ata[i][j] += row[i] * row[j];
                    }
                    atb_xi[i] += row[i] * xi;
                    atb_eta[i] += row[i] * eta;
                }
            }
        }
        if let (Some(sx), Some(se)) = (solve_3x3(&ata, &atb_xi), solve_3x3(&ata, &atb_eta)) {
            cd = [[sx[0], sx[1]], [se[0], se[1]]];
            let (new_ra, new_dec) = inverse_tan_project(sx[2], se[2], crval_ra, crval_dec);
            crval_ra = new_ra;
            crval_dec = new_dec;
        }
    }

    // ── Compute final residual statistics ────────────────────────────────
    let mut final_residuals: Vec<f64> = Vec::with_capacity(current_matches.len());
    for &(cent_idx, cat_idx) in &current_matches {
        let sv = &star_vectors[cat_idx];
        let sra = (sv[1] as f64).atan2(sv[0] as f64);
        let sdec = (sv[2] as f64).asin();
        if let Some((xi_cat, eta_cat)) = tan_project(sra, sdec, crval_ra, crval_dec) {
            let (px, py) = centroids_px[cent_idx];
            let xi_pred = cd[0][0] * px + cd[0][1] * py;
            let eta_pred = cd[1][0] * px + cd[1][1] * py;
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

    debug!(
        "WCS refine done: {} matches, RMSE={:.2}\" p90={:.2}\" max={:.2}\"",
        current_matches.len(),
        rmse.to_degrees() * 3600.0,
        p90e.to_degrees() * 3600.0,
        max_err.to_degrees() * 3600.0,
    );

    WcsRefineResult {
        cd_matrix: cd,
        crval_rad: [crval_ra, crval_dec],
        matches: current_matches,
        rmse_rad: rmse,
        p90e_rad: p90e,
        max_err_rad: max_err,
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
    let e_xi = Vector3::new(-sin_a, cos_a, 0.0);
    let e_eta = Vector3::new(-sin_d * cos_a, -sin_d * sin_a, cos_d);
    let boresight = Vector3::new(cos_d * cos_a, cos_d * sin_a, sin_d);

    // Camera axes in ICRS (unnormalized)
    // Camera +X pixel direction → (CD11, CD21) in tangent-plane
    let cam_x_icrs_raw = e_xi * cd[0][0] + e_eta * cd[1][0];
    // Camera +Y pixel direction → (CD12, CD22) in tangent-plane
    let cam_y_icrs_raw = e_xi * cd[0][1] + e_eta * cd[1][1];

    let cam_x_icrs = cam_x_icrs_raw.normalize();
    let cam_y_icrs = cam_y_icrs_raw.normalize();

    // Build rotation matrix: rows are camera axes expressed in ICRS
    // R maps ICRS → camera: camera_vec = R * icrs_vec
    let rot = Matrix3::new(
        cam_x_icrs.x as f32,
        cam_x_icrs.y as f32,
        cam_x_icrs.z as f32,
        cam_y_icrs.x as f32,
        cam_y_icrs.y as f32,
        cam_y_icrs.z as f32,
        boresight.x as f32,
        boresight.y as f32,
        boresight.z as f32,
    );

    // FOV from pixel scale in X direction
    let ps_x = cam_x_icrs_raw.norm(); // radians per pixel
    let fov = (ps_x * image_width as f64) as f32;

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
        // Test at several positions
        let crval_ra = 1.2_f64; // ~69°
        let crval_dec = 0.3_f64; // ~17°

        let test_points = [
            (1.21, 0.31),  // near reference
            (1.25, 0.25),  // offset
            (1.15, 0.35),  // other side
            (1.0, 0.0),    // further away
        ];

        for &(ra, dec) in &test_points {
            let (xi, eta) = tan_project(ra, dec, crval_ra, crval_dec).unwrap();
            let (ra2, dec2) = inverse_tan_project(xi, eta, crval_ra, crval_dec);
            assert!(
                (ra - ra2).abs() < 1e-12 && (dec - dec2).abs() < 1e-12,
                "Roundtrip failed for ({}, {}): got ({}, {})",
                ra, dec, ra2, dec2,
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
        // Point opposite to reference should fail
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
        // 2x + 3y + z = 11
        // x + y + z = 6
        // x + 2y + 3z = 14
        // Solution: x=1, y=2, z=3
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
        // CD * CD_inv should be identity
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
    fn test_wcs_to_rotation_simple() {
        // A simple camera pointed at RA=90°, Dec=0° with 10° FOV, 1000px wide
        let crval_ra = std::f64::consts::FRAC_PI_2; // 90°
        let crval_dec = 0.0;
        let fov_deg = 10.0_f64;
        let image_width = 1000u32;
        let ps = fov_deg.to_radians() / image_width as f64; // rad/px

        // Simple unrotated camera: CD = [[ps, 0], [0, ps]]
        let cd = [[ps, 0.0], [0.0, ps]];
        let (rot, fov, parity) = wcs_to_rotation(&cd, crval_ra, crval_dec, image_width);

        assert!(!parity);
        assert!((fov.to_degrees() - 10.0).abs() < 0.01, "FOV: {}", fov.to_degrees());

        // Boresight should be at RA=90°, Dec=0° → ICRS (0, 1, 0)
        // R * [0, 1, 0] should give [0, 0, 1] (camera boresight)
        let bore_cam = rot * Vector3::new(0.0_f32, 1.0, 0.0);
        assert!(bore_cam.z > 0.99, "boresight z = {}", bore_cam.z);
    }
}
