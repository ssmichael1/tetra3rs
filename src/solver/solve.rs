//! Plate solving: given image centroids and an approximate FOV, find the
//! camera pointing direction as a quaternion.
//!
//! Closely follows tetra3's `solve_from_centroids()`:
//! 1. Convert centroids to camera-frame unit vectors.
//! 2. Apply cluster-buster thinning.
//! 3. For each 4-centroid combination (brightest first):
//!    a. Compute edge ratios, look up matching catalog patterns.
//!    b. For each match, estimate rotation via SVD (Wahba problem).
//!    c. Verify by projecting catalog stars and counting matches.
//!    d. Accept if false-positive probability is below threshold.

use std::time::Instant;

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use tracing::debug;

use crate::Centroid;

use super::combinations::BreadthFirstCombinations;
use super::pattern::{
    compute_edge_ratios, compute_pattern_key, compute_pattern_key_hash, compute_sorted_edge_angles,
    get_table_indices, hash_to_index, NUM_EDGES, NUM_EDGE_RATIOS, PATTERN_SIZE,
};
use super::{SolveConfig, SolveResult, SolveStatus, SolverDatabase};

/// Angle between two unit vectors using the cross-product / dot-product trick.
///
/// Returns `atan2(|a × b|, a · b)` which is numerically stable for all angles,
/// unlike `arccos(dot)` (loses precision near 0/π) or `2*arcsin(‖a-b‖/2)`
/// (cancellation when a ≈ b in f32).
#[inline]
fn angle_between(a: &Vector3<f32>, b: &Vector3<f32>) -> f32 {
    let cross = a.cross(b);
    let sin_theta = cross.norm();
    let cos_theta = a.dot(b);
    sin_theta.atan2(cos_theta)
}

// ── Solve entry point ───────────────────────────────────────────────────────

impl SolverDatabase {
    /// Solve for the camera pointing direction given image centroids.
    ///
    /// Centroids should have the `mass` field populated for brightness sorting.
    /// Centroid (x, y) are in pixel coordinates with (0, 0) at the image center.
    /// +X points right, +Y points down in the image.
    ///
    /// The `SolveConfig` must specify `fov_estimate_rad` (horizontal FOV in radians)
    /// and `image_width` / `image_height` (in pixels) so the solver can compute the
    /// pixel scale.
    ///
    /// If `fov_max_error_rad` is set, the solver sweeps FOV values across the range
    /// `[fov_estimate - fov_max_error, fov_estimate + fov_max_error]`, trying the
    /// exact estimate first, then spiraling outward. This makes the solver robust
    /// to uncertain FOV estimates.
    ///
    /// Returns a `SolveResult` with the ICRS→camera quaternion on success.
    pub fn solve_from_centroids(
        &self,
        centroids: &[Centroid],
        config: &SolveConfig,
    ) -> SolveResult {
        let t0 = Instant::now();

        // ── Undistort centroids once (pixel-space, FOV-independent) ──
        let undistorted: Vec<Centroid>;
        let working_centroids: &[Centroid] = match &config.distortion {
            Some(d) => {
                undistorted = centroids
                    .iter()
                    .map(|c| {
                        let (ux, uy) = d.undistort(c.x as f64, c.y as f64);
                        Centroid {
                            x: ux as f32,
                            y: uy as f32,
                            mass: c.mass,
                            cov: c.cov,
                        }
                    })
                    .collect();
                &undistorted
            }
            None => centroids,
        };

        // Build FOV sweep: exact estimate first, then spiral outward
        let fov_values = build_fov_sweep(
            config.fov_estimate_rad,
            config.fov_max_error_rad,
            config.match_radius,
        );

        debug!(
            "FOV sweep: {} values from {:.2}° to {:.2}°",
            fov_values.len(),
            fov_values
                .iter()
                .cloned()
                .reduce(f32::min)
                .unwrap_or(0.0)
                .to_degrees(),
            fov_values
                .iter()
                .cloned()
                .reduce(f32::max)
                .unwrap_or(0.0)
                .to_degrees(),
        );

        let mut last_status = SolveStatus::NoMatch;

        for &fov_try in &fov_values {
            // Check timeout
            if let Some(t) = config.solve_timeout_ms {
                if elapsed_ms(t0) > t as f32 {
                    return SolveResult::failure(SolveStatus::Timeout, elapsed_ms(t0));
                }
            }

            debug!("Trying FOV = {:.3}°", fov_try.to_degrees());
            let result = self.solve_at_fov(working_centroids, config, fov_try, t0);
            match result.status {
                SolveStatus::MatchFound => return result,
                SolveStatus::TooFew => return result,
                s => last_status = s,
            }
        }

        SolveResult::failure(last_status, elapsed_ms(t0))
    }

    /// Attempt a solve at a specific FOV value.
    fn solve_at_fov(
        &self,
        centroids: &[Centroid],
        config: &SolveConfig,
        fov_estimate: f32,
        t0: Instant,
    ) -> SolveResult {
        let pixel_scale = if config.image_width > 0 {
            fov_estimate / config.image_width as f32
        } else {
            0.0
        };

        // Sort centroids by brightness (highest mass = brightest first).
        // Centroids without mass are placed last.
        let mut sorted_indices: Vec<usize> = (0..centroids.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            let ma = centroids[a].mass.unwrap_or(f32::MIN);
            let mb = centroids[b].mass.unwrap_or(f32::MIN);
            mb.partial_cmp(&ma).unwrap_or(std::cmp::Ordering::Equal)
        });

        let num_centroids = sorted_indices.len();

        if num_centroids < PATTERN_SIZE {
            return SolveResult::failure(SolveStatus::TooFew, elapsed_ms(t0));
        }

        // ── Compute unit vectors in camera frame ──
        // Centroid (x, y) in pixels → scale to radians → uvec = normalize(x_rad, y_rad, 1)
        // Note: distortion correction (if any) was already applied in solve_from_centroids.
        let centroid_vectors: Vec<[f32; 3]> = sorted_indices
            .iter()
            .map(|&i| {
                let x = centroids[i].x * pixel_scale;
                let y = centroids[i].y * pixel_scale;
                let z = 1.0f32;
                let norm = (x * x + y * y + z * z).sqrt();
                [x / norm, y / norm, z / norm]
            })
            .collect();

        // Lazily-created x-flipped copy for parity-flipped images.
        // Built on first use, cached for subsequent pattern attempts.
        let mut flipped_vectors: Option<Vec<[f32; 3]>> = None;

        // ── Cluster-buster thinning ──
        // Apply the same separation constraint as database generation to avoid
        // wasting pattern attempts on dense clusters.
        let verification_stars = self.props.verification_stars_per_fov;
        let separation = separation_for_density(fov_estimate, verification_stars);
        let cos_sep = separation.cos();

        let mut keep_for_patterns = vec![false; num_centroids];
        for i in 0..num_centroids {
            let vi = &centroid_vectors[i];
            let mut occupied = false;
            for j in 0..i {
                if keep_for_patterns[j] {
                    let vj = &centroid_vectors[j];
                    let dot = vi[0] * vj[0] + vi[1] * vj[1] + vi[2] * vj[2];
                    if dot > cos_sep {
                        occupied = true;
                        break;
                    }
                }
            }
            if !occupied {
                keep_for_patterns[i] = true;
            }
        }

        let pattern_centroid_inds: Vec<usize> = (0..num_centroids)
            .filter(|&i| keep_for_patterns[i])
            .collect();
        let num_pattern_centroids = pattern_centroid_inds.len();

        debug!(
            "Centroids: {} total, {} for patterns after cluster busting",
            num_centroids, num_pattern_centroids
        );

        if num_pattern_centroids < PATTERN_SIZE {
            return SolveResult::failure(SolveStatus::TooFew, elapsed_ms(t0));
        }

        // Trim match centroids to verification limit
        let match_centroid_count = num_centroids.min(verification_stars as usize);

        // ── Solver parameters ──
        let p_bins = self.props.pattern_bins;
        let p_max_err = config
            .match_max_error
            .unwrap_or(self.props.pattern_max_error)
            .max(self.props.pattern_max_error);
        let match_threshold = config.match_threshold / self.props.num_patterns as f64;
        let timeout_ms = config.solve_timeout_ms;

        debug!(
            "Checking up to C({},{}) = {} image patterns",
            num_pattern_centroids,
            PATTERN_SIZE,
            n_choose_k(num_pattern_centroids, PATTERN_SIZE)
        );

        // ── Main solve loop ──
        let mut status = SolveStatus::NoMatch;

        for image_pattern_local in
            BreadthFirstCombinations::new(&pattern_centroid_inds, PATTERN_SIZE)
        {
            // Check timeout
            if let Some(t) = timeout_ms {
                if elapsed_ms(t0) > t as f32 {
                    debug!("Timeout after {:.1}ms", elapsed_ms(t0));
                    status = SolveStatus::Timeout;
                    break;
                }
            }

            // Get image pattern vectors
            let image_vecs: [[f32; 3]; 4] = [
                centroid_vectors[image_pattern_local[0]],
                centroid_vectors[image_pattern_local[1]],
                centroid_vectors[image_pattern_local[2]],
                centroid_vectors[image_pattern_local[3]],
            ];

            // Compute edge angles and ratios
            let edge_angles = compute_sorted_edge_angles(&image_vecs);
            let image_largest_edge = edge_angles[NUM_EDGES - 1];
            let image_ratios = compute_edge_ratios(&edge_angles);

            // Broadened range for pattern key lookup
            let ratio_min: [f32; NUM_EDGE_RATIOS] =
                std::array::from_fn(|i| image_ratios[i] - p_max_err);
            let ratio_max: [f32; NUM_EDGE_RATIOS] =
                std::array::from_fn(|i| image_ratios[i] + p_max_err);

            let image_key = compute_pattern_key(&image_ratios, p_bins);

            // Compute the range of pattern keys to search
            let key_min: [u32; NUM_EDGE_RATIOS] =
                std::array::from_fn(|i| (ratio_min[i] * p_bins as f32).max(0.0) as u32);
            let key_max: [u32; NUM_EDGE_RATIOS] =
                std::array::from_fn(|i| (ratio_max[i] * p_bins as f32).min(p_bins as f32) as u32);

            // Build list of candidate pattern keys, sorted by distance from image_key
            let mut pattern_key_list: Vec<(u32, [u32; NUM_EDGE_RATIOS])> = Vec::new();
            enumerate_key_range(&key_min, &key_max, &image_key, &mut pattern_key_list);
            pattern_key_list.sort_unstable_by_key(|&(dist, _)| dist);

            // Try each candidate pattern key
            for &(_, ref pkey) in &pattern_key_list {
                let pkey_hash = compute_pattern_key_hash(pkey, p_bins);
                let hidx = hash_to_index(pkey_hash, self.pattern_catalog.len() as u64);

                // Walk the hash chain
                let table_indices = get_table_indices(hidx, &self.pattern_catalog);
                if table_indices.is_empty() {
                    continue;
                }

                // Pre-filter by 16-bit key hash
                let key_hash16 = (pkey_hash & 0xFFFF) as u16;

                for &tidx in &table_indices {
                    if self.pattern_key_hashes[tidx] != key_hash16 {
                        continue;
                    }

                    // FOV consistency check: the catalog pattern's largest edge
                    // should be close to the image pattern's largest edge.
                    let cat_largest = self.pattern_largest_edge[tidx];
                    if let Some(fov_err) = config.fov_max_error_rad {
                        // Implied FOV from this match
                        let implied_fov = cat_largest / image_largest_edge * fov_estimate;
                        if (implied_fov - fov_estimate).abs() > fov_err {
                            continue;
                        }
                    }

                    // Full edge-ratio comparison
                    let cat_pat = self.pattern_catalog[tidx];
                    let cat_vecs: [[f32; 3]; 4] = [
                        self.star_vectors[cat_pat[0] as usize],
                        self.star_vectors[cat_pat[1] as usize],
                        self.star_vectors[cat_pat[2] as usize],
                        self.star_vectors[cat_pat[3] as usize],
                    ];
                    let cat_edges = compute_sorted_edge_angles(&cat_vecs);
                    let cat_largest_edge = cat_edges[NUM_EDGES - 1];
                    let cat_ratios = compute_edge_ratios(&cat_edges);

                    // Check all edge ratios are within tolerance
                    let ratios_ok = (0..NUM_EDGE_RATIOS)
                        .all(|i| cat_ratios[i] > ratio_min[i] && cat_ratios[i] < ratio_max[i]);
                    if !ratios_ok {
                        continue;
                    }

                    // ── Estimate rotation via SVD ──

                    // Refine FOV estimate from this match
                    let mut fov = cat_largest_edge / image_largest_edge * fov_estimate;

                    // Sort image pattern by centroid distance (canonical ordering)
                    let mut img_order: [usize; 4] = [0, 1, 2, 3];
                    sort_by_centroid_distance_inline(&mut img_order, &image_vecs);

                    // Catalog pattern is already pre-sorted during database generation.
                    // Build matched vector pairs.
                    let matched_img: [[f32; 3]; 4] =
                        std::array::from_fn(|i| image_vecs[img_order[i]]);
                    let matched_cat: [[f32; 3]; 4] = std::array::from_fn(|i| cat_vecs[i]);

                    // SVD rotation: finds R such that camera_vec ≈ R * icrs_vec
                    let mut rotation_matrix = find_rotation_matrix(&matched_img, &matched_cat);

                    // Determine parity from the rotation determinant.
                    // centroid_vectors is never mutated; when parity is needed we use
                    // a lazily-created x-flipped copy for verification matching.
                    let parity_flip;
                    let working_vectors: &[[f32; 3]];
                    if rotation_matrix.determinant() < 0.0 {
                        // Wrong parity (e.g. FITS image with CDELT1 < 0).
                        parity_flip = true;
                        // Recompute rotation with flipped image pattern vectors
                        let matched_img_flip: [[f32; 3]; 4] = std::array::from_fn(|i| {
                            let orig = image_vecs[img_order[i]];
                            [-orig[0], orig[1], orig[2]]
                        });
                        rotation_matrix = find_rotation_matrix(&matched_img_flip, &matched_cat);
                        if rotation_matrix.determinant() < 0.0 {
                            continue; // still a reflection — skip
                        }
                        // Lazily create flipped centroid vectors for matching
                        let fv = flipped_vectors.get_or_insert_with(|| {
                            centroid_vectors
                                .iter()
                                .map(|v| [-v[0], v[1], v[2]])
                                .collect()
                        });
                        working_vectors = fv;
                    } else {
                        parity_flip = false;
                        working_vectors = &centroid_vectors;
                    }

                    // ── Verify by matching nearby catalog stars ──

                    let fov_diagonal = fov * 1.42; // sqrt(2) ≈ 1.42 for square FOV
                    let match_radius_rad = config.match_radius * fov;

                    // Find catalog stars within the diagonal FOV
                    let image_center_icrs =
                        rotation_matrix.transpose() * Vector3::new(0.0, 0.0, 1.0);
                    let nearby_inds = self
                        .star_catalog
                        .query_indices_from_uvec(image_center_icrs, fov_diagonal / 2.0);

                    // Project catalog stars to camera frame
                    let mut nearby_cam_positions: Vec<(usize, f32, f32)> = Vec::new();
                    for &cat_idx in &nearby_inds {
                        let sv = &self.star_vectors[cat_idx];
                        let icrs_v = Vector3::new(sv[0], sv[1], sv[2]);
                        let cam_v = rotation_matrix * icrs_v;
                        // Only keep stars in front of the camera (z > 0)
                        if cam_v.z > 0.0 {
                            let cx = cam_v.x / cam_v.z; // radians from boresight
                            let cy = cam_v.y / cam_v.z;
                            nearby_cam_positions.push((cat_idx, cx, cy));
                        }
                    }
                    // Limit to 2x the number of image centroids (like tetra3)
                    nearby_cam_positions.truncate(2 * match_centroid_count);
                    let num_nearby = nearby_cam_positions.len();

                    // Match image centroids to projected catalog stars
                    let mut current_matches = find_centroid_matches(
                        &working_vectors[..match_centroid_count],
                        &nearby_cam_positions,
                        match_radius_rad,
                    );
                    let mut current_num_matches = current_matches.len();

                    // ── Compute false-positive probability ──
                    let prob_single = num_nearby as f64 * (config.match_radius as f64).powi(2);
                    let prob_mismatch = binomial_cdf(
                        (match_centroid_count as i64 - (current_num_matches as i64 - 2)).max(0)
                            as u32,
                        match_centroid_count as u32,
                        1.0 - prob_single.min(1.0),
                    );

                    if prob_mismatch >= match_threshold {
                        continue;
                    }

                    debug!(
                        "MATCH: {} matches, prob={:.2e}, fov={:.3}°",
                        current_num_matches,
                        prob_mismatch * self.props.num_patterns as f64,
                        fov.to_degrees()
                    );

                    // ── Recompute centroid vectors at pattern-implied FOV ──
                    // The initial centroid_vectors used fov_estimate (the sweep
                    // value), which may differ from the pattern-implied `fov`.
                    // Recomputing here makes all downstream refinement independent
                    // of the user's fov_estimate.
                    let refined_ps = fov / config.image_width.max(1) as f32;
                    let parity_sign_ref: f32 = if parity_flip { -1.0 } else { 1.0 };
                    let refined_centroid_vectors: Vec<[f32; 3]> = sorted_indices
                        .iter()
                        .map(|&i| {
                            let x = parity_sign_ref * centroids[i].x * refined_ps;
                            let y = centroids[i].y * refined_ps;
                            let z = 1.0f32;
                            let norm = (x * x + y * y + z * z).sqrt();
                            [x / norm, y / norm, z / norm]
                        })
                        .collect();
                    let working_vectors: &[[f32; 3]] = &refined_centroid_vectors;

                    // Re-match with corrected vectors
                    {
                        let center_icrs = rotation_matrix.transpose() * Vector3::new(0.0, 0.0, 1.0);
                        let nearby = self
                            .star_catalog
                            .query_indices_from_uvec(center_icrs, fov_diagonal / 2.0);
                        let mut cam_positions: Vec<(usize, f32, f32)> = Vec::new();
                        for &cat_idx in &nearby {
                            let sv = &self.star_vectors[cat_idx];
                            let icrs_v = Vector3::new(sv[0], sv[1], sv[2]);
                            let cam_v = rotation_matrix * icrs_v;
                            if cam_v.z > 0.0 {
                                let cx = cam_v.x / cam_v.z;
                                let cy = cam_v.y / cam_v.z;
                                cam_positions.push((cat_idx, cx, cy));
                            }
                        }
                        cam_positions.truncate(2 * match_centroid_count);

                        let fresh = find_centroid_matches(
                            &working_vectors[..match_centroid_count],
                            &cam_positions,
                            match_radius_rad,
                        );
                        if fresh.len() >= 4 {
                            debug!(
                                "Re-match at pattern FOV {:.3}°: {} -> {} matches",
                                fov.to_degrees(),
                                current_num_matches,
                                fresh.len(),
                            );
                            current_matches = fresh;
                            current_num_matches = current_matches.len();
                        }
                    }

                    // ── Iterative refinement ──
                    // Each pass: SVD on current inliers → re-project → re-match.
                    // Terminates early only when the match *set* is unchanged across
                    // an iteration (not just the count), so wrong associations that
                    // are being swapped out for correct ones keep driving convergence.
                    let refine_iters = config.refine_iterations.max(1);
                    let mut current_rotation = rotation_matrix;

                    for iteration in 0..refine_iters {
                        // Build matched vector pairs from current matches
                        let mut iter_img: Vec<[f32; 3]> = Vec::with_capacity(current_num_matches);
                        let mut iter_cat: Vec<[f32; 3]> = Vec::with_capacity(current_num_matches);
                        for &(cent_local_idx, cat_idx) in &current_matches {
                            iter_img.push(working_vectors[cent_local_idx]);
                            iter_cat.push(self.star_vectors[cat_idx]);
                        }

                        // Refine rotation via SVD
                        current_rotation = find_rotation_matrix_dyn(&iter_img, &iter_cat);

                        // On the last iteration, skip re-matching
                        if iteration + 1 >= refine_iters {
                            break;
                        }

                        // Re-project catalog stars using refined rotation
                        let center_icrs =
                            current_rotation.transpose() * Vector3::new(0.0, 0.0, 1.0);
                        let nearby = self
                            .star_catalog
                            .query_indices_from_uvec(center_icrs, fov_diagonal / 2.0);

                        let mut cam_positions: Vec<(usize, f32, f32)> = Vec::new();
                        for &cat_idx in &nearby {
                            let sv = &self.star_vectors[cat_idx];
                            let icrs_v = Vector3::new(sv[0], sv[1], sv[2]);
                            let cam_v = current_rotation * icrs_v;
                            if cam_v.z > 0.0 {
                                let cx = cam_v.x / cam_v.z;
                                let cy = cam_v.y / cam_v.z;
                                cam_positions.push((cat_idx, cx, cy));
                            }
                        }
                        cam_positions.truncate(2 * match_centroid_count);

                        let new_matches = find_centroid_matches(
                            &working_vectors[..match_centroid_count],
                            &cam_positions,
                            match_radius_rad,
                        );
                        let new_num_matches = new_matches.len();

                        // Stop only when the match set is identical (not just same size).
                        // Sort both for set-equality comparison.
                        let mut sorted_new = new_matches.clone();
                        sorted_new.sort_unstable();
                        let mut sorted_cur = current_matches.clone();
                        sorted_cur.sort_unstable();
                        let set_unchanged = sorted_new == sorted_cur;

                        debug!(
                            "Refinement iter {}: {} -> {} inliers{}",
                            iteration + 1,
                            current_num_matches,
                            new_num_matches,
                            if set_unchanged { " (converged)" } else { "" },
                        );

                        current_matches = new_matches;
                        current_num_matches = new_num_matches;

                        if set_unchanged {
                            break;
                        }
                    }

                    // ── Iterative MAD clip + re-associate with adaptive radius ──
                    // Each iteration:
                    //   1. Compute per-match residuals
                    //   2. MAD-based outlier rejection (robust to contamination)
                    //   3. Re-fit SVD on inliers
                    //   4. Compute adaptive match radius from inlier RMSE
                    //   5. Re-project catalog + re-match ALL centroids with tight radius
                    //   6. SVD on the re-matched set
                    // Stops when the match set converges (no clip AND no set change).
                    {
                        const MAD_SCALE: f32 = 1.4826; // MAD-to-σ for Gaussian
                        const CLIP_NSIGMA: f32 = 3.0;
                        const MAX_CLIP_ITERS: usize = 10;
                        // Adaptive radius is clamped to [floor, original_radius].
                        // Floor ≈ 5 arcsec in radians — prevents over-tightening.
                        const RADIUS_FLOOR: f32 = 2.5e-5;

                        let mut adaptive_radius: f32;

                        for clip_iter in 0..MAX_CLIP_ITERS {
                            if current_matches.len() < 4 {
                                break;
                            }

                            // ── 1. Compute residuals ──
                            let mut residuals: Vec<(usize, f32)> =
                                Vec::with_capacity(current_matches.len());
                            for (i, &(ci, cat_idx)) in current_matches.iter().enumerate() {
                                let img_v = Vector3::new(
                                    working_vectors[ci][0],
                                    working_vectors[ci][1],
                                    working_vectors[ci][2],
                                );
                                let cat_v = Vector3::new(
                                    self.star_vectors[cat_idx][0],
                                    self.star_vectors[cat_idx][1],
                                    self.star_vectors[cat_idx][2],
                                );
                                let img_icrs = current_rotation.transpose() * img_v;
                                residuals.push((i, angle_between(&img_icrs, &cat_v)));
                            }

                            // ── 2. MAD clip ──
                            let mut sorted_res: Vec<f32> =
                                residuals.iter().map(|&(_, r)| r).collect();
                            sorted_res.sort_by(|a, b| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let median = sorted_res[sorted_res.len() / 2];

                            let mut abs_devs: Vec<f32> =
                                sorted_res.iter().map(|r| (r - median).abs()).collect();
                            abs_devs.sort_by(|a, b| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let mad = abs_devs[abs_devs.len() / 2];

                            let sigma_est = MAD_SCALE * mad;
                            // Floor: don't clip below ~0.5 arcsec
                            let clip_threshold = median + CLIP_NSIGMA * sigma_est.max(2.5e-6);

                            let inliers: Vec<(usize, usize)> = residuals
                                .iter()
                                .filter(|&&(_, r)| r <= clip_threshold)
                                .map(|&(i, _)| current_matches[i])
                                .collect();

                            let n_rejected = current_matches.len() - inliers.len();

                            // ── 3. Re-fit SVD on cleaned inliers (if any were clipped) ──
                            if n_rejected > 0 && inliers.len() >= 4 {
                                debug!(
                                    "MAD clip iter {}: {} -> {} inliers (median {:.2e}, σ_est {:.2e}, thresh {:.2e} rad)",
                                    clip_iter + 1,
                                    current_matches.len(),
                                    inliers.len(),
                                    median,
                                    sigma_est,
                                    clip_threshold,
                                );

                                let clean_img: Vec<[f32; 3]> =
                                    inliers.iter().map(|&(ci, _)| working_vectors[ci]).collect();
                                let clean_cat: Vec<[f32; 3]> = inliers
                                    .iter()
                                    .map(|&(_, ci)| self.star_vectors[ci])
                                    .collect();
                                current_rotation = find_rotation_matrix_dyn(&clean_img, &clean_cat);
                                current_matches = inliers;
                                current_num_matches = current_matches.len();
                            }

                            // ── 4. Adaptive match radius from inlier statistics ──
                            // Use 5× the MAD-estimated σ as the re-match radius,
                            // clamped between RADIUS_FLOOR and the original radius.
                            // This dramatically reduces wrong associations in dense
                            // fields once the rotation is well-determined.
                            adaptive_radius = (5.0 * MAD_SCALE * mad)
                                .max(RADIUS_FLOOR)
                                .min(match_radius_rad);

                            debug!(
                                "Clip iter {}: adaptive match radius {:.2e} rad ({:.1} arcsec)",
                                clip_iter + 1,
                                adaptive_radius,
                                adaptive_radius.to_degrees() * 3600.0,
                            );

                            // ── 5. Re-project + re-match ALL centroids ──
                            let center_icrs =
                                current_rotation.transpose() * Vector3::new(0.0, 0.0, 1.0);
                            let nearby = self
                                .star_catalog
                                .query_indices_from_uvec(center_icrs, fov_diagonal / 2.0);
                            let mut cam_positions: Vec<(usize, f32, f32)> = Vec::new();
                            for &cat_idx in &nearby {
                                let sv = &self.star_vectors[cat_idx];
                                let icrs_v = Vector3::new(sv[0], sv[1], sv[2]);
                                let cam_v = current_rotation * icrs_v;
                                if cam_v.z > 0.0 {
                                    let cx = cam_v.x / cam_v.z;
                                    let cy = cam_v.y / cam_v.z;
                                    cam_positions.push((cat_idx, cx, cy));
                                }
                            }
                            cam_positions.truncate(2 * match_centroid_count);

                            let fresh = find_centroid_matches(
                                &working_vectors[..match_centroid_count],
                                &cam_positions,
                                adaptive_radius,
                            );

                            // Check for set convergence
                            let mut sorted_fresh = fresh.clone();
                            sorted_fresh.sort_unstable();
                            let mut sorted_cur = current_matches.clone();
                            sorted_cur.sort_unstable();
                            let set_unchanged = sorted_fresh == sorted_cur;

                            if fresh.len() >= 4 {
                                if fresh.len() != current_num_matches || !set_unchanged {
                                    debug!(
                                        "Clip iter {} re-match: {} -> {} matches (radius {:.1} arcsec)",
                                        clip_iter + 1,
                                        current_num_matches,
                                        fresh.len(),
                                        adaptive_radius.to_degrees() * 3600.0,
                                    );
                                }

                                current_matches = fresh;
                                current_num_matches = current_matches.len();

                                // ── 6. SVD on re-matched set ──
                                let rematched_img: Vec<[f32; 3]> = current_matches
                                    .iter()
                                    .map(|&(ci, _)| working_vectors[ci])
                                    .collect();
                                let rematched_cat: Vec<[f32; 3]> = current_matches
                                    .iter()
                                    .map(|&(_, ci)| self.star_vectors[ci])
                                    .collect();
                                current_rotation =
                                    find_rotation_matrix_dyn(&rematched_img, &rematched_cat);
                            }

                            // Converged: nothing clipped AND match set is stable
                            if n_rejected == 0 && set_unchanged {
                                debug!(
                                    "Clip+re-match converged at iter {} with {} matches (radius {:.1} arcsec)",
                                    clip_iter + 1,
                                    current_num_matches,
                                    adaptive_radius.to_degrees() * 3600.0,
                                );
                                break;
                            }
                        }
                    }

                    // ── FOV (pixel-scale) tuning ──
                    // The initial FOV estimate from the pattern match may be slightly
                    // off, creating a systematic radial error pattern.  We do a 1-D
                    // golden-section search over pixel_scale to minimize the RMSE of
                    // the current inlier set.  This is cheap (just recompute unit
                    // vectors + SVD for ~20 evaluations) and dramatically reduces
                    // systematic residuals.
                    {
                        let parity_sign: f32 = if parity_flip { -1.0 } else { 1.0 };

                        // Objective: given a pixel_scale, recompute unit vectors for
                        // matched centroids, SVD rotation, and return sum-of-squared
                        // angular residuals.
                        let eval_fov = |ps: f32| -> f32 {
                            let img_vecs: Vec<[f32; 3]> = current_matches
                                .iter()
                                .map(|&(ci, _)| {
                                    let orig_idx = sorted_indices[ci];
                                    let x = parity_sign * centroids[orig_idx].x * ps;
                                    let y = centroids[orig_idx].y * ps;
                                    let z = 1.0f32;
                                    let norm = (x * x + y * y + z * z).sqrt();
                                    [x / norm, y / norm, z / norm]
                                })
                                .collect();
                            let cat_vecs: Vec<[f32; 3]> = current_matches
                                .iter()
                                .map(|&(_, cat_idx)| self.star_vectors[cat_idx])
                                .collect();
                            let rot = find_rotation_matrix_dyn(&img_vecs, &cat_vecs);

                            // Sum of squared angular residuals
                            let mut sse = 0.0f32;
                            for (iv, cv) in img_vecs.iter().zip(cat_vecs.iter()) {
                                let img_v = Vector3::new(iv[0], iv[1], iv[2]);
                                let cat_v = Vector3::new(cv[0], cv[1], cv[2]);
                                let img_icrs = rot.transpose() * img_v;
                                let ang = angle_between(&img_icrs, &cat_v);
                                sse += ang * ang;
                            }
                            sse
                        };

                        // Golden-section search over pixel_scale
                        // Search ±2% around the *pattern-implied* FOV (not the
                        // sweep estimate), so the result is independent of what
                        // the user passed as fov_estimate.
                        let ps_center = fov / config.image_width.max(1) as f32;
                        let mut a = ps_center * 0.98;
                        let mut b = ps_center * 1.02;
                        let golden = 0.6180339887f32;
                        let tol = ps_center * 1e-6;

                        let mut x1 = b - golden * (b - a);
                        let mut x2 = a + golden * (b - a);
                        let mut f1 = eval_fov(x1);
                        let mut f2 = eval_fov(x2);

                        for _ in 0..30 {
                            if (b - a) < tol {
                                break;
                            }
                            if f1 < f2 {
                                b = x2;
                                x2 = x1;
                                f2 = f1;
                                x1 = b - golden * (b - a);
                                f1 = eval_fov(x1);
                            } else {
                                a = x1;
                                x1 = x2;
                                f1 = f2;
                                x2 = a + golden * (b - a);
                                f2 = eval_fov(x2);
                            }
                        }

                        let best_ps = (a + b) / 2.0;
                        let tuned_fov = best_ps * config.image_width.max(1) as f32;

                        debug!(
                            "FOV tuning: {:.4}° -> {:.4}° (Δ = {:.2} arcsec)",
                            fov.to_degrees(),
                            tuned_fov.to_degrees(),
                            (tuned_fov - fov).abs().to_degrees() * 3600.0,
                        );

                        // Recompute working vectors and rotation at the tuned pixel scale
                        let tuned_img: Vec<[f32; 3]> = current_matches
                            .iter()
                            .map(|&(ci, _)| {
                                let orig_idx = sorted_indices[ci];
                                let x = parity_sign * centroids[orig_idx].x * best_ps;
                                let y = centroids[orig_idx].y * best_ps;
                                let z = 1.0f32;
                                let norm = (x * x + y * y + z * z).sqrt();
                                [x / norm, y / norm, z / norm]
                            })
                            .collect();
                        let tuned_cat: Vec<[f32; 3]> = current_matches
                            .iter()
                            .map(|&(_, cat_idx)| self.star_vectors[cat_idx])
                            .collect();
                        current_rotation = find_rotation_matrix_dyn(&tuned_img, &tuned_cat);

                        fov = tuned_fov;

                        // Rebuild ALL centroid vectors at tuned scale for the final clip
                        // (we need working_vectors to reflect the tuned FOV)
                    }

                    // Recompute all centroid vectors at the tuned pixel scale
                    let tuned_pixel_scale = fov / config.image_width.max(1) as f32;
                    let parity_sign_f: f32 = if parity_flip { -1.0 } else { 1.0 };
                    let tuned_centroid_vectors: Vec<[f32; 3]> = sorted_indices
                        .iter()
                        .map(|&i| {
                            let x = parity_sign_f * centroids[i].x * tuned_pixel_scale;
                            let y = centroids[i].y * tuned_pixel_scale;
                            let z = 1.0f32;
                            let norm = (x * x + y * y + z * z).sqrt();
                            [x / norm, y / norm, z / norm]
                        })
                        .collect();
                    let working_vectors: &[[f32; 3]] = &tuned_centroid_vectors;

                    // ── Post-FOV-tuning re-match + MAD clip ──
                    // The pixel scale changed during FOV tuning, so the old matches
                    // may include marginal stars that no longer fit, and new stars
                    // may now be within radius. Re-project, re-match, then do one
                    // MAD clip cycle to clean up before the final clip.
                    {
                        let fov_diag = fov * 1.42;
                        let center_icrs =
                            current_rotation.transpose() * Vector3::new(0.0, 0.0, 1.0);
                        let nearby = self
                            .star_catalog
                            .query_indices_from_uvec(center_icrs, fov_diag / 2.0);
                        let mut cam_positions: Vec<(usize, f32, f32)> = Vec::new();
                        for &cat_idx in &nearby {
                            let sv = &self.star_vectors[cat_idx];
                            let icrs_v = Vector3::new(sv[0], sv[1], sv[2]);
                            let cam_v = current_rotation * icrs_v;
                            if cam_v.z > 0.0 {
                                let cx = cam_v.x / cam_v.z;
                                let cy = cam_v.y / cam_v.z;
                                cam_positions.push((cat_idx, cx, cy));
                            }
                        }
                        cam_positions.truncate(2 * match_centroid_count);

                        let fresh = find_centroid_matches(
                            &working_vectors[..match_centroid_count],
                            &cam_positions,
                            match_radius_rad,
                        );

                        if fresh.len() >= 4 {
                            debug!(
                                "Post-FOV-tuning re-match: {} -> {} matches",
                                current_num_matches,
                                fresh.len(),
                            );
                            current_matches = fresh;
                            current_num_matches = current_matches.len();

                            // SVD on re-matched set
                            let rm_img: Vec<[f32; 3]> = current_matches
                                .iter()
                                .map(|&(ci, _)| working_vectors[ci])
                                .collect();
                            let rm_cat: Vec<[f32; 3]> = current_matches
                                .iter()
                                .map(|&(_, ci)| self.star_vectors[ci])
                                .collect();
                            current_rotation = find_rotation_matrix_dyn(&rm_img, &rm_cat);

                            // One MAD clip cycle on the fresh matches
                            if current_matches.len() >= 6 {
                                const MAD_SCALE_POST: f32 = 1.4826;
                                const CLIP_NSIGMA_POST: f32 = 3.0;

                                let mut res_post: Vec<(usize, f32)> =
                                    Vec::with_capacity(current_matches.len());
                                for (i, &(ci, cat_idx)) in current_matches.iter().enumerate() {
                                    let img_v = Vector3::new(
                                        working_vectors[ci][0],
                                        working_vectors[ci][1],
                                        working_vectors[ci][2],
                                    );
                                    let cat_v = Vector3::new(
                                        self.star_vectors[cat_idx][0],
                                        self.star_vectors[cat_idx][1],
                                        self.star_vectors[cat_idx][2],
                                    );
                                    let img_icrs = current_rotation.transpose() * img_v;
                                    res_post.push((i, angle_between(&img_icrs, &cat_v)));
                                }

                                let mut sr: Vec<f32> = res_post.iter().map(|&(_, r)| r).collect();
                                sr.sort_by(|a, b| {
                                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                                });
                                let median = sr[sr.len() / 2];
                                let mut ad: Vec<f32> =
                                    sr.iter().map(|r| (r - median).abs()).collect();
                                ad.sort_by(|a, b| {
                                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                                });
                                let mad = ad[ad.len() / 2];
                                let sigma_est = MAD_SCALE_POST * mad;
                                let thresh = median + CLIP_NSIGMA_POST * sigma_est.max(2.5e-6);

                                let clean: Vec<(usize, usize)> = res_post
                                    .iter()
                                    .filter(|&&(_, r)| r <= thresh)
                                    .map(|&(i, _)| current_matches[i])
                                    .collect();

                                let n_rej = current_matches.len() - clean.len();
                                if n_rej > 0 && clean.len() >= 4 {
                                    debug!(
                                        "Post-FOV-tuning MAD clip: {} -> {} (rejected {})",
                                        current_matches.len(),
                                        clean.len(),
                                        n_rej,
                                    );
                                    let ci_img: Vec<[f32; 3]> =
                                        clean.iter().map(|&(ci, _)| working_vectors[ci]).collect();
                                    let ci_cat: Vec<[f32; 3]> = clean
                                        .iter()
                                        .map(|&(_, ci)| self.star_vectors[ci])
                                        .collect();
                                    current_rotation = find_rotation_matrix_dyn(&ci_img, &ci_cat);
                                    current_matches = clean;
                                    current_num_matches = current_matches.len();
                                }
                            }
                        }
                    }

                    // ── Final outlier-only clip (no re-association, multi-pass) ──
                    // After the iterative MAD clip + re-associate loop has converged,
                    // do up to 3 MAD clip passes on the final match set, re-fitting
                    // SVD on the survivors ONLY — no re-matching step, so no new
                    // wrong associations can creep in.  Multiple passes handle
                    // cascading outliers where one bad star inflates the MAD estimate
                    // enough to mask another.
                    {
                        const MAD_SCALE_FINAL: f32 = 1.4826;
                        const CLIP_NSIGMA_FINAL: f32 = 3.0;
                        const MAX_FINAL_CLIP_PASSES: usize = 3;

                        for final_pass in 0..MAX_FINAL_CLIP_PASSES {
                            if current_matches.len() < 6 {
                                break;
                            }

                            // Compute residuals
                            let mut residuals_final: Vec<(usize, f32)> =
                                Vec::with_capacity(current_matches.len());
                            for (i, &(ci, cat_idx)) in current_matches.iter().enumerate() {
                                let img_v = Vector3::new(
                                    working_vectors[ci][0],
                                    working_vectors[ci][1],
                                    working_vectors[ci][2],
                                );
                                let cat_v = Vector3::new(
                                    self.star_vectors[cat_idx][0],
                                    self.star_vectors[cat_idx][1],
                                    self.star_vectors[cat_idx][2],
                                );
                                let img_icrs = current_rotation.transpose() * img_v;
                                residuals_final.push((i, angle_between(&img_icrs, &cat_v)));
                            }

                            // MAD clip
                            let mut sorted_res: Vec<f32> =
                                residuals_final.iter().map(|&(_, r)| r).collect();
                            sorted_res.sort_by(|a, b| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let median = sorted_res[sorted_res.len() / 2];
                            let mut abs_devs: Vec<f32> =
                                sorted_res.iter().map(|r| (r - median).abs()).collect();
                            abs_devs.sort_by(|a, b| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let mad = abs_devs[abs_devs.len() / 2];
                            let sigma_est = MAD_SCALE_FINAL * mad;
                            let clip_threshold = median + CLIP_NSIGMA_FINAL * sigma_est.max(2.5e-6);

                            let clean: Vec<(usize, usize)> = residuals_final
                                .iter()
                                .filter(|&&(_, r)| r <= clip_threshold)
                                .map(|&(i, _)| current_matches[i])
                                .collect();

                            let n_final_rejected = current_matches.len() - clean.len();
                            if n_final_rejected > 0 && clean.len() >= 4 {
                                debug!(
                                    "Final clean-up clip pass {}: {} -> {} matches (rejected {})",
                                    final_pass + 1,
                                    current_matches.len(),
                                    clean.len(),
                                    n_final_rejected,
                                );
                                // Re-fit SVD on clean set only — no re-matching
                                let clean_img: Vec<[f32; 3]> =
                                    clean.iter().map(|&(ci, _)| working_vectors[ci]).collect();
                                let clean_cat: Vec<[f32; 3]> =
                                    clean.iter().map(|&(_, ci)| self.star_vectors[ci]).collect();
                                current_rotation = find_rotation_matrix_dyn(&clean_img, &clean_cat);
                                current_matches = clean;
                                current_num_matches = current_matches.len();
                            } else {
                                // Nothing to clip — converged
                                break;
                            }
                        }
                    }

                    // ── Build final output vectors from current_matches ──
                    let mut all_img_vecs: Vec<[f32; 3]> = Vec::with_capacity(current_num_matches);
                    let mut all_cat_vecs: Vec<[f32; 3]> = Vec::with_capacity(current_num_matches);
                    let mut matched_cat_ids: Vec<u64> = Vec::with_capacity(current_num_matches);
                    let mut matched_cent_inds: Vec<usize> = Vec::with_capacity(current_num_matches);

                    for &(cent_local_idx, cat_idx) in &current_matches {
                        all_img_vecs.push(working_vectors[cent_local_idx]);
                        all_cat_vecs.push(self.star_vectors[cat_idx]);
                        matched_cat_ids.push(self.star_catalog_ids[cat_idx]);
                        matched_cent_inds.push(sorted_indices[cent_local_idx]);
                    }

                    // ── Compute residuals ──
                    let mut residuals: Vec<f32> = Vec::with_capacity(current_num_matches);
                    for i in 0..current_num_matches {
                        let img_v = Vector3::new(
                            all_img_vecs[i][0],
                            all_img_vecs[i][1],
                            all_img_vecs[i][2],
                        );
                        let cat_v = Vector3::new(
                            all_cat_vecs[i][0],
                            all_cat_vecs[i][1],
                            all_cat_vecs[i][2],
                        );
                        // Rotate image vector to ICRS and compare
                        let img_in_icrs = current_rotation.transpose() * img_v;
                        residuals.push(angle_between(&img_in_icrs, &cat_v));
                    }
                    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                    let rmse = if residuals.is_empty() {
                        0.0
                    } else {
                        (residuals.iter().map(|r| r * r).sum::<f32>() / residuals.len() as f32)
                            .sqrt()
                    };
                    let p90e = if residuals.is_empty() {
                        0.0
                    } else {
                        residuals[(0.9 * (residuals.len() - 1) as f32) as usize]
                    };
                    let max_err = residuals.last().copied().unwrap_or(0.0);

                    // Convert to quaternion
                    let rot3 = Rotation3::from_matrix_unchecked(current_rotation);
                    let quat = UnitQuaternion::from_rotation_matrix(&rot3);

                    return SolveResult {
                        qicrs2cam: Some(quat),
                        fov_rad: Some(fov),
                        num_matches: Some(current_num_matches as u32),
                        rmse_rad: Some(rmse),
                        p90e_rad: Some(p90e),
                        max_err_rad: Some(max_err),
                        prob: Some(prob_mismatch * self.props.num_patterns as f64),
                        solve_time_ms: elapsed_ms(t0),
                        status: SolveStatus::MatchFound,
                        parity_flip,
                        matched_catalog_ids: matched_cat_ids,
                        matched_centroid_indices: matched_cent_inds,
                    };
                }
            }
        }

        SolveResult::failure(status, elapsed_ms(t0))
    }
}

// ── Helper functions ────────────────────────────────────────────────────────

/// Build FOV values to try: exact estimate first, then spiraling outward.
///
/// Step size is chosen so that the verification match_radius can tolerate the
/// worst-case scale error at the midpoint between steps.
fn build_fov_sweep(fov_estimate: f32, fov_max_error: Option<f32>, match_radius: f32) -> Vec<f32> {
    let mut values = vec![fov_estimate];

    if let Some(max_error) = fov_max_error {
        if max_error > 0.0 {
            // Step = 2 * match_radius * fov_estimate.
            // At the midpoint between steps, a star at the FOV edge has position
            // error ≈ (step/2)/(fov) * (fov/2) = step/4. With step = 2*mr*fov,
            // that's mr*fov/2, well within the match_radius_rad = mr*fov.
            let step = (2.0 * match_radius * fov_estimate).max(0.001_f32.to_radians());
            let mut offset = step;
            while offset <= max_error {
                values.push(fov_estimate + offset);
                if fov_estimate - offset > 0.0 {
                    values.push(fov_estimate - offset);
                }
                offset += step;
            }
        }
    }

    values
}

fn elapsed_ms(t0: Instant) -> f32 {
    t0.elapsed().as_secs_f32() * 1000.0
}

fn separation_for_density(fov_rad: f32, stars_per_fov: u32) -> f32 {
    (fov_rad / 2.0) * (std::f32::consts::PI / stars_per_fov as f32).sqrt()
}

fn n_choose_k(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Enumerate all pattern keys in the given range, tagged with distance² from center.
fn enumerate_key_range(
    key_min: &[u32; NUM_EDGE_RATIOS],
    key_max: &[u32; NUM_EDGE_RATIOS],
    center: &[u32; NUM_EDGE_RATIOS],
    out: &mut Vec<(u32, [u32; NUM_EDGE_RATIOS])>,
) {
    // Recursive Cartesian product over the 5 dimensions.
    let mut current = [0u32; NUM_EDGE_RATIOS];
    enumerate_key_range_recursive(key_min, key_max, center, 0, &mut current, out);
}

fn enumerate_key_range_recursive(
    key_min: &[u32; NUM_EDGE_RATIOS],
    key_max: &[u32; NUM_EDGE_RATIOS],
    center: &[u32; NUM_EDGE_RATIOS],
    dim: usize,
    current: &mut [u32; NUM_EDGE_RATIOS],
    out: &mut Vec<(u32, [u32; NUM_EDGE_RATIOS])>,
) {
    if dim == NUM_EDGE_RATIOS {
        let dist_sq: u32 = (0..NUM_EDGE_RATIOS)
            .map(|i| {
                let d = current[i] as i32 - center[i] as i32;
                (d * d) as u32
            })
            .sum();
        out.push((dist_sq, *current));
        return;
    }
    for v in key_min[dim]..=key_max[dim] {
        current[dim] = v;
        enumerate_key_range_recursive(key_min, key_max, center, dim + 1, current, out);
    }
}

/// Sort 4 indices by their vectors' distance from the pattern centroid.
fn sort_by_centroid_distance_inline(order: &mut [usize; 4], vectors: &[[f32; 3]; 4]) {
    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    let mut cz = 0.0f32;
    for v in vectors.iter() {
        cx += v[0];
        cy += v[1];
        cz += v[2];
    }
    cx /= 4.0;
    cy /= 4.0;
    cz /= 4.0;

    order.sort_by(|&a, &b| {
        let va = &vectors[a];
        let vb = &vectors[b];
        let da = (va[0] - cx).powi(2) + (va[1] - cy).powi(2) + (va[2] - cz).powi(2);
        let db = (vb[0] - cx).powi(2) + (vb[1] - cy).powi(2) + (vb[2] - cz).powi(2);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Compute the least-squares rotation matrix from image vectors to catalog vectors.
///
/// Uses SVD of the cross-covariance matrix H = Σ(img_i ⊗ cat_i).
/// The resulting R satisfies: camera_vec ≈ R * icrs_vec.
///
/// The SVD is computed in f64 for precision, then the result is converted back to f32.
fn find_rotation_matrix<const N: usize>(
    image_vectors: &[[f32; 3]; N],
    catalog_vectors: &[[f32; 3]; N],
) -> Matrix3<f32> {
    let mut h = nalgebra::Matrix3::<f64>::zeros();
    for i in 0..N {
        let img = nalgebra::Vector3::<f64>::new(
            image_vectors[i][0] as f64,
            image_vectors[i][1] as f64,
            image_vectors[i][2] as f64,
        );
        let cat = nalgebra::Vector3::<f64>::new(
            catalog_vectors[i][0] as f64,
            catalog_vectors[i][1] as f64,
            catalog_vectors[i][2] as f64,
        );
        h += img * cat.transpose();
    }

    let svd = h.svd(true, true);
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();
    let r64 = u * v_t;
    r64.cast::<f32>()
}

/// Also support dynamically-sized slices for the refinement step.
fn find_rotation_matrix_dyn(
    image_vectors: &[[f32; 3]],
    catalog_vectors: &[[f32; 3]],
) -> Matrix3<f32> {
    let mut h = nalgebra::Matrix3::<f64>::zeros();
    for (img, cat) in image_vectors.iter().zip(catalog_vectors.iter()) {
        let img_v = nalgebra::Vector3::<f64>::new(img[0] as f64, img[1] as f64, img[2] as f64);
        let cat_v = nalgebra::Vector3::<f64>::new(cat[0] as f64, cat[1] as f64, cat[2] as f64);
        h += img_v * cat_v.transpose();
    }
    let svd = h.svd(true, true);
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();
    let r64 = u * v_t;
    r64.cast::<f32>()
}

/// Find unique 1-to-1 matches between image centroids and projected catalog positions.
///
/// Returns Vec<(centroid_local_idx, catalog_star_idx)>.
fn find_centroid_matches(
    centroid_vectors: &[[f32; 3]],
    catalog_positions: &[(usize, f32, f32)], // (star_idx, cam_x, cam_y) in radians
    match_radius: f32,
) -> Vec<(usize, usize)> {
    // For each centroid, project to camera-plane angular coordinates
    let centroid_xy: Vec<(f32, f32)> = centroid_vectors
        .iter()
        .map(|v| {
            if v[2] > 0.0 {
                (v[0] / v[2], v[1] / v[2])
            } else {
                (f32::MAX, f32::MAX)
            }
        })
        .collect();

    let r2 = match_radius * match_radius;

    // Compute pairwise distances and find pairs within radius
    let mut candidates: Vec<(f32, usize, usize)> = Vec::new(); // (dist², cent_idx, cat_pos_idx)
    for (ci, &(cx, cy)) in centroid_xy.iter().enumerate() {
        for (pi, &(_cat_idx, px, py)) in catalog_positions.iter().enumerate() {
            let dx = cx - px;
            let dy = cy - py;
            let d2 = dx * dx + dy * dy;
            if d2 < r2 {
                candidates.push((d2, ci, pi));
            }
        }
    }

    // Sort by distance (best matches first)
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Greedy unique 1-to-1 matching
    let mut used_centroids = vec![false; centroid_vectors.len()];
    let mut used_catalog = vec![false; catalog_positions.len()];
    let mut matches = Vec::new();

    for &(_, ci, pi) in &candidates {
        if !used_centroids[ci] && !used_catalog[pi] {
            used_centroids[ci] = true;
            used_catalog[pi] = true;
            matches.push((ci, catalog_positions[pi].0));
        }
    }

    matches
}

// ── Binomial CDF (no external dependency) ───────────────────────────────────

/// Compute the binomial CDF: P(X <= k) where X ~ Binomial(n, p).
/// Uses iterative computation for numerical stability at typical sizes (n < 500).
fn binomial_cdf(k: u32, n: u32, p: f64) -> f64 {
    if k >= n {
        return 1.0;
    }
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return if k >= n { 1.0 } else { 0.0 };
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
