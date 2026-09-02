//! Lost-in-space hypothesis generation: the 4-star pattern search.
//!
//! [`PatternSearch`] owns everything between "brightness-sorted centroids"
//! and "candidate attitude": the FOV sweep, cluster-buster thinning, the
//! breadth-first combination enumerator, pattern-key range enumeration and
//! hash-table probing, the edge-ratio check, the 4-star Wahba SVD, parity
//! detection, and the measured-FOV rebuild of the verification vectors. It
//! knows nothing about verification or refinement: each candidate is handed
//! to a caller-supplied acceptance callback as a [`Hypothesis`], and the
//! first accepted solution ends the search.
//!
//! The tracking path is the other hypothesis source (`track.rs`); both feed
//! the same verification (`verify.rs`) and refinement (`solve.rs`) stages.

use super::clock::Instant;

use numeris::Matrix3;
use tracing::debug;

use crate::Centroid;

use super::combinations::BreadthFirstCombinations;
use super::database::separation_for_density;
use super::pattern::{
    compute_edge_ratios, compute_pattern_key, compute_pattern_key_hash, compute_sorted_edge_angles,
    hash_to_index, sort_pattern_by_centroid_distance, NUM_EDGES, NUM_EDGE_RATIOS, PATTERN_SIZE,
};
use super::preprocess::{centroid_unit_vectors, unit_vector_from_pixels, CentroidVectors};
use super::solve::{elapsed_ms, failure, wahba_rotation};
use super::verify::diagonal_factor;
use super::{
    pixel_scale_from_fov, Solution, SolveConfig, SolveResult, SolveStatus, SolverDatabase,
};

#[cfg(feature = "profile")]
use crate::solver::profiling::{self, buckets};

/// A candidate attitude produced by a hypothesis source, ready for
/// verification.
pub(super) struct Hypothesis<'a> {
    /// ICRS→camera rotation (proper; parity already folded in — see
    /// `vectors.parity_flip`).
    pub rotation: Matrix3<f32>,
    /// FOV (radians) measured from the matched pattern — the scale the
    /// refinement should lock to.
    pub fov: f32,
    /// Verification vectors for this candidate: brightness-ordered, parity
    /// applied, built either at the swept scale or (when the measured FOV
    /// differs meaningfully) rebuilt at `fov`'s scale — see `pixel_scale`.
    pub vectors: CentroidVectors<'a>,
    /// Number of *tested* centroids (index < `match_centroid_count`) used to
    /// form this hypothesis. They match by construction and are excluded
    /// from the verification statistic.
    pub hypothesis_stars: usize,
}

/// The lost-in-space pattern search over one centroid set.
pub(super) struct PatternSearch<'a> {
    db: &'a SolverDatabase,
    config: &'a SolveConfig,
    /// Preprocessed centroids (CRPIX-subtracted, undistorted, in pixels).
    centroids: &'a [Centroid],
    /// Brightness-sorted centroid index order.
    sorted_indices: &'a [usize],
    /// (Possibly aberration-corrected) catalog unit vectors.
    star_vectors: &'a [[f32; 3]],
    /// Number of brightest centroids the verification tests.
    match_centroid_count: usize,
    t0: Instant,
    /// FOV values to try: exact estimate first, then spiraling outward.
    fov_values: Vec<f32>,
    /// Image patterns enumerated so far, summed over the FOV sweep; the
    /// `max_patterns_checked` budget is applied to this total.
    patterns_checked: u64,
}

impl<'a> PatternSearch<'a> {
    /// Set up a search. The caller guarantees more than `PATTERN_SIZE`
    /// centroids (fewer can never pass verification).
    pub(super) fn new(
        db: &'a SolverDatabase,
        centroids: &'a [Centroid],
        sorted_indices: &'a [usize],
        config: &'a SolveConfig,
        star_vectors: &'a [[f32; 3]],
        match_centroid_count: usize,
        t0: Instant,
    ) -> Self {
        let fov_values = build_fov_sweep(
            config.fov_estimate_rad(),
            config.fov_max_error_rad,
            db.props.pattern_max_error,
            diagonal_factor(config),
        );
        Self {
            db,
            config,
            centroids,
            sorted_indices,
            star_vectors,
            match_centroid_count,
            t0,
            fov_values,
            patterns_checked: 0,
        }
    }

    /// Run the search: sweep the FOV values, enumerate patterns, and hand
    /// every ratio-passing candidate to `on_hypothesis`. Returns the first
    /// solution the callback accepts, or the failure status of the
    /// exhausted search (`Timeout` if a budget tripped first).
    pub(super) fn run(
        &mut self,
        on_hypothesis: &mut dyn FnMut(&Hypothesis<'_>) -> Option<Solution>,
    ) -> SolveResult {
        let t0 = self.t0;
        let config = self.config;

        debug!(
            "FOV sweep: {} values from {:.2}° to {:.2}°",
            self.fov_values.len(),
            self.fov_values
                .iter()
                .cloned()
                .reduce(f32::min)
                .unwrap_or(0.0)
                .to_degrees(),
            self.fov_values
                .iter()
                .cloned()
                .reduce(f32::max)
                .unwrap_or(0.0)
                .to_degrees(),
        );

        let mut last_status = SolveStatus::NoMatch;

        for i in 0..self.fov_values.len() {
            let fov_try = self.fov_values[i];
            // Check search budgets (wall-clock and pattern count)
            if let Some(t) = config.solve_timeout_ms {
                if elapsed_ms(t0) > t as f32 {
                    return failure(SolveStatus::Timeout, t0);
                }
            }
            if let Some(max) = config.max_patterns_checked {
                if self.patterns_checked >= max {
                    return failure(SolveStatus::Timeout, t0);
                }
            }

            debug!("Trying FOV = {:.3}°", fov_try.to_degrees());
            match self.pass_at_fov(fov_try, on_hypothesis) {
                Ok(solution) => return Ok(solution),
                // TooFew here means cluster-buster thinning left fewer than 4
                // pattern centroids. The thinning separation scales with the
                // FOV being tried, so a different FOV in the sweep may still
                // succeed — keep going.
                Err(status) => last_status = status,
            }
        }

        failure(last_status, t0)
    }

    /// One pass of the pattern search at a specific FOV value.
    fn pass_at_fov(
        &mut self,
        fov_estimate: f32,
        on_hypothesis: &mut dyn FnMut(&Hypothesis<'_>) -> Option<Solution>,
    ) -> Result<Solution, SolveStatus> {
        let config = self.config;
        let centroids = self.centroids;
        let sorted_indices = self.sorted_indices;
        let star_vectors = self.star_vectors;
        let t0 = self.t0;

        #[cfg(feature = "profile")]
        profiling::count(buckets::FOV_PASS, 1);

        // True pinhole pixel scale (rad/px): ps = 1/f where f = (W/2) / tan(fov/2).
        // Derived from the sweep's FOV value, not the camera model's focal
        // length — each sweep iteration tries a different scale.
        let pixel_scale = if config.image_width() > 0 && fov_estimate > 0.0 {
            pixel_scale_from_fov(config.image_width(), fov_estimate as f64) as f32
        } else {
            0.0
        };

        let num_centroids = sorted_indices.len();

        // ── Compute unit vectors in camera frame ──
        // Centroid (x, y) in pixels → scale to radians → uvec = normalize(x_rad, y_rad, 1)
        // Note: distortion correction (if any) was already applied by `preprocess`.
        let centroid_vectors = centroid_unit_vectors(centroids, sorted_indices, pixel_scale, 1.0);

        // Lazily-created x-flipped copy for parity-flipped images.
        // Built on first use, cached for subsequent pattern attempts.
        let mut flipped_vectors: Option<Vec<[f32; 3]>> = None;

        // Scratch buffer for centroid vectors rebuilt at a candidate's
        // measured FOV (see the rebuild step in the candidate loop). Reused
        // across candidates to avoid per-candidate allocation.
        let mut rebuilt_vectors: Vec<[f32; 3]> = Vec::new();

        // ── Cluster-buster thinning ──
        // Apply the same separation constraint as database generation to avoid
        // wasting pattern attempts on dense clusters.
        let verification_stars = self.db.props.verification_stars_per_fov;
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

        let mut pattern_centroid_inds: Vec<usize> = (0..num_centroids)
            .filter(|&i| keep_for_patterns[i])
            .collect();
        let num_after_thinning = pattern_centroid_inds.len();
        // Only the brightest `pattern_checking_stars` survivors form
        // patterns (see `SolveConfig::pattern_checking_stars`): the table
        // holds patterns among each field's brightest well-separated stars,
        // so bright quads are the ones that can hit, and the cap bounds a
        // no-match search at C(N, 4) per FOV value. `pattern_centroid_inds`
        // is ascending in brightness rank, so truncation keeps the brightest.
        pattern_centroid_inds.truncate(config.pattern_checking_stars as usize);
        let num_pattern_centroids = pattern_centroid_inds.len();

        debug!(
            "Centroids: {} total, {} for patterns after cluster busting, {} after the brightness cap",
            num_centroids, num_after_thinning, num_pattern_centroids
        );

        if num_pattern_centroids < PATTERN_SIZE {
            return Err(SolveStatus::TooFew);
        }

        // ── Solver parameters ──
        let p_bins = self.db.props.pattern_bins;
        // A tolerance below the database's quantization error cannot work
        // (patterns were binned at pattern_max_error), so floor it there.
        let p_max_err = match config.match_max_error {
            Some(user_err) if user_err < self.db.props.pattern_max_error => {
                debug!(
                    "match_max_error {:.2e} below database pattern_max_error {:.2e}; using the latter",
                    user_err, self.db.props.pattern_max_error
                );
                self.db.props.pattern_max_error
            }
            Some(user_err) => user_err,
            None => self.db.props.pattern_max_error,
        };
        // Ceiling on the tolerance. The candidate-key search enumerates a 5-D
        // Cartesian product of ~(2·err·bins + 1)^5 tuples per star combination;
        // with no cap a large match_max_error (e.g. 0.1 at 250 bins ≈ 345M
        // tuples, ~8 GB) exhausts memory. Bound the per-dimension bin span, but
        // never below the database's own quantization error (the floor above).
        const MAX_KEY_SPAN_BINS: f32 = 16.0;
        let err_ceiling =
            (MAX_KEY_SPAN_BINS / (2.0 * p_bins as f32)).max(self.db.props.pattern_max_error);
        let p_max_err = if p_max_err > err_ceiling {
            debug!(
                "match_max_error {:.2e} exceeds enumeration ceiling {:.2e} ({} bins); clamping",
                p_max_err, err_ceiling, p_bins
            );
            err_ceiling
        } else {
            p_max_err
        };
        let timeout_ms = config.solve_timeout_ms;
        let max_patterns = config.max_patterns_checked;

        // Guard against a corrupt or placeholder database. An empty table
        // makes the hash-probe arithmetic below divide by zero; a non-empty
        // table claiming zero generated patterns is inconsistent.
        let table_len = self.db.pattern_catalog.len() as u64;
        if table_len == 0 || self.db.props.num_patterns == 0 {
            return Err(SolveStatus::NoMatch);
        }

        debug!(
            "Checking up to C({},{}) = {} image patterns",
            num_pattern_centroids,
            PATTERN_SIZE,
            n_choose_k(num_pattern_centroids, PATTERN_SIZE)
        );

        // ── Main solve loop ──
        let mut status = SolveStatus::NoMatch;
        let mut pattern_key_list: Vec<(u32, [u32; NUM_EDGE_RATIOS])> = Vec::new();

        for image_pattern_local in
            BreadthFirstCombinations::<PATTERN_SIZE>::new(&pattern_centroid_inds)
        {
            // Check search budgets (wall-clock and pattern count)
            if let Some(t) = timeout_ms {
                if elapsed_ms(t0) > t as f32 {
                    debug!(
                        "Timeout after {:.1}ms ({} patterns checked)",
                        elapsed_ms(t0),
                        self.patterns_checked
                    );
                    status = SolveStatus::Timeout;
                    break;
                }
            }
            if let Some(max) = max_patterns {
                if self.patterns_checked >= max {
                    debug!(
                        "Pattern budget exhausted: {} patterns checked in {:.1}ms",
                        self.patterns_checked,
                        elapsed_ms(t0)
                    );
                    status = SolveStatus::Timeout;
                    break;
                }
            }
            self.patterns_checked += 1;

            // Get image pattern vectors
            let image_vecs: [[f32; 3]; 4] = [
                centroid_vectors[image_pattern_local[0]],
                centroid_vectors[image_pattern_local[1]],
                centroid_vectors[image_pattern_local[2]],
                centroid_vectors[image_pattern_local[3]],
            ];

            #[cfg(feature = "profile")]
            profiling::count(buckets::COMBOS, 1);

            // Compute edge angles and ratios
            // (image-side edges: this is exactly what an N×N precomputed
            // pairwise-angle matrix would replace with table lookups.)
            let (edge_angles, image_ratios) = timed!(buckets::IMAGE_EDGES, {
                let ea = compute_sorted_edge_angles(&image_vecs);
                let ir = compute_edge_ratios(&ea);
                (ea, ir)
            });
            let image_largest_edge = edge_angles[NUM_EDGES - 1];

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
            pattern_key_list.clear();
            timed!(buckets::KEY_ENUM, {
                enumerate_key_range(&key_min, &key_max, &image_key, &mut pattern_key_list);
                pattern_key_list.sort_unstable_by_key(|&(dist, _)| dist);
            });

            // Try each candidate pattern key
            for (_, pkey) in &pattern_key_list {
                let pkey_hash = compute_pattern_key_hash(pkey, p_bins);
                let hidx = hash_to_index(pkey_hash, table_len);

                // Pre-filter by 16-bit key hash
                let key_hash16 = (pkey_hash & 0xFFFF) as u16;

                // Walk the hash chain inline (quadratic probing). Generator
                // tables keep load ≤ 0.5 on a prime size, so an empty slot is
                // always reached; the `table_len` cap only bounds the walk on a
                // corrupt/over-full table (which would otherwise loop forever).
                for c in 0u64..table_len {
                    let tidx = ((hidx.wrapping_add(c.wrapping_mul(c))) % table_len) as usize;
                    let entry = self.db.pattern_catalog.get(tidx);
                    if entry.is_empty() {
                        break; // end of chain
                    }
                    if entry.key_hash != key_hash16 {
                        continue;
                    }

                    #[cfg(feature = "profile")]
                    profiling::count(buckets::CANDIDATES, 1);

                    // FOV consistency check: the catalog pattern's largest edge
                    // should be close to the image pattern's largest edge.
                    let cat_largest = entry.largest_edge;
                    if let Some(fov_err) = config.fov_max_error_rad {
                        // Implied FOV from this match
                        let implied_fov = cat_largest / image_largest_edge * fov_estimate;
                        if (implied_fov - fov_estimate).abs() > fov_err {
                            continue;
                        }
                    }

                    // Full edge-ratio comparison
                    let cat_pat = entry.star_indices;
                    let cat_vecs: [[f32; 3]; 4] = [
                        star_vectors[cat_pat[0] as usize],
                        star_vectors[cat_pat[1] as usize],
                        star_vectors[cat_pat[2] as usize],
                        star_vectors[cat_pat[3] as usize],
                    ];
                    // Catalog-side edges: the analogue of `image_edges`, but
                    // computed per surviving candidate and NOT precomputable
                    // per-image (depends on which catalog pattern matched).
                    let (cat_edges, cat_ratios) = timed!(buckets::CAT_EDGES, {
                        let ce = compute_sorted_edge_angles(&cat_vecs);
                        let cr = compute_edge_ratios(&ce);
                        (ce, cr)
                    });
                    let cat_largest_edge = cat_edges[NUM_EDGES - 1];

                    // Check all edge ratios are within tolerance
                    let ratios_ok = (0..NUM_EDGE_RATIOS)
                        .all(|i| cat_ratios[i] > ratio_min[i] && cat_ratios[i] < ratio_max[i]);
                    if !ratios_ok {
                        continue;
                    }

                    // ── Estimate rotation via SVD ──

                    // Refine FOV estimate from this match
                    let fov = cat_largest_edge / image_largest_edge * fov_estimate;
                    // With `fov_max_error_rad: None` (the default) nothing above
                    // bounds this ratio, and a tight image quad matching a wide
                    // catalog pattern can imply a FOV past π — where tan(fov/2)
                    // flips sign and the verification geometry (pixel scale,
                    // density region) turns to nonsense. Such a candidate can
                    // only waste refinement work or weaken the statistical
                    // gate, never be right; skip it.
                    if !(fov.is_finite() && fov > 0.0 && fov < std::f32::consts::PI) {
                        continue;
                    }

                    // ── Rebuild vectors at the measured scale when the sweep
                    // value is meaningfully off ──
                    // Edge-ratio keys cancel a pixel-scale error to first
                    // order, but the SVD attitude and verification residuals
                    // do not: vectors built at a wrong scale stretch the field
                    // radially by ε·θ, which overwhelms the verification match
                    // radius (match_radius·fov) long before the ratio
                    // tolerance is threatened. Rebuilding at this candidate's
                    // measured FOV makes a single sweep pass robust to any
                    // FOV-estimate error within `fov_max_error` (this is what
                    // lets `build_fov_sweep` use a coarse, pattern-tolerance-
                    // derived step).
                    //
                    // Rebuilding also *registers* accepted solutions better —
                    // even at sub-percent mismatches the measured scale beats
                    // the swept one for the match set handed to refinement
                    // (measured on the TESS multi-sector calibration: pass-1
                    // fits are ~2× tighter with the rebuild than without).
                    // Skipped only when the mismatch is negligible — at
                    // threshold, the residual at the half-diagonal (~0.7·fov)
                    // is ≲ 0.2 of the match radius — so a well-estimated
                    // field pays nothing.
                    let scale_mismatch = (fov / fov_estimate - 1.0).abs();
                    let rebuild = scale_mismatch > 0.25 * config.match_radius;
                    let (pat_vecs, ps_meas): ([[f32; 3]; 4], f32) = if rebuild {
                        let ps = pixel_scale_from_fov(config.image_width(), fov as f64) as f32;
                        (
                            std::array::from_fn(|k| {
                                unit_vector_from_pixels(
                                    &centroids[sorted_indices[image_pattern_local[k]]],
                                    ps,
                                    1.0,
                                )
                            }),
                            ps,
                        )
                    } else {
                        (image_vecs, pixel_scale)
                    };

                    // Sort image pattern by centroid distance (canonical ordering)
                    let mut img_order: [usize; 4] = [0, 1, 2, 3];
                    sort_pattern_by_centroid_distance(&mut img_order, |i| pat_vecs[i]);

                    // Catalog pattern is already pre-sorted during database generation.
                    // Build matched vector pairs.
                    let matched_img: [[f32; 3]; 4] =
                        std::array::from_fn(|i| pat_vecs[img_order[i]]);
                    let matched_cat: [[f32; 3]; 4] = std::array::from_fn(|i| cat_vecs[i]);

                    #[cfg(feature = "profile")]
                    profiling::count(buckets::RATIO_PASS, 1);

                    // SVD rotation: finds R such that camera_vec ≈ R * icrs_vec.
                    // A degenerate cross-covariance (e.g. duplicate or collinear
                    // centroids) fails the SVD — skip the candidate, don't panic.
                    let Some(mut rotation_matrix) = timed!(
                        buckets::SVD,
                        wahba_rotation(matched_img.iter().zip(matched_cat.iter()))
                    ) else {
                        continue;
                    };

                    // Determine parity from the rotation determinant.
                    // centroid_vectors is never mutated; when parity is needed we use
                    // a lazily-created x-flipped copy for verification matching.
                    let parity_flip = if rotation_matrix.det() < 0.0 {
                        // Wrong parity (e.g. FITS image with CDELT1 < 0).
                        // Derive the parity-flipped rotation WITHOUT a second SVD.
                        //
                        // Flipping the x-component of every image vector is
                        // img' = D·img with D = diag(-1, 1, 1). `find_rotation_matrix`
                        // builds H = Σ imgᵢ · catᵢᵀ, decomposes H = U·S·Vᵀ, and
                        // returns R = U·Vᵀ. With flipped image vectors
                        //   H' = Σ (D·imgᵢ)·catᵢᵀ = D·H = (D·U)·S·Vᵀ,
                        // a valid SVD because D is orthogonal. Hence
                        //   R' = (D·U)·Vᵀ = D·(U·Vᵀ) = D·R,
                        // i.e. R' is just R with its first ROW negated. Since
                        // R = U·Vᵀ is invariant to the per-singular-vector sign
                        // freedom of the decomposition, this is mathematically
                        // exact and reproduces the second SVD bit-for-bit (up to
                        // f32 rounding of a single extra negation).
                        //
                        // det(R') = det(D)·det(R) = −det(R) > 0 here, so R' is
                        // always a proper rotation; the old "still a reflection →
                        // skip" branch can never trigger and is therefore dropped.
                        rotation_matrix[(0, 0)] = -rotation_matrix[(0, 0)];
                        rotation_matrix[(0, 1)] = -rotation_matrix[(0, 1)];
                        rotation_matrix[(0, 2)] = -rotation_matrix[(0, 2)];
                        true
                    } else {
                        false
                    };
                    let working_vectors: &[[f32; 3]] = if rebuild {
                        // Rebuild the full verification set at the measured
                        // scale (parity applied directly via the x-sign).
                        let sign = if parity_flip { -1.0f32 } else { 1.0 };
                        rebuilt_vectors.clear();
                        rebuilt_vectors.extend(
                            sorted_indices
                                .iter()
                                .map(|&i| unit_vector_from_pixels(&centroids[i], ps_meas, sign)),
                        );
                        &rebuilt_vectors
                    } else if parity_flip {
                        // Lazily create flipped centroid vectors for matching
                        flipped_vectors.get_or_insert_with(|| {
                            centroid_vectors
                                .iter()
                                .map(|v| [-v[0], v[1], v[2]])
                                .collect()
                        })
                    } else {
                        &centroid_vectors
                    };

                    // ── Hand the hypothesis to the acceptance stage ──
                    // The pattern stars that fall inside the tested
                    // (brightest `match_centroid_count`) set are
                    // hypothesis-forming, not evidence — the verification
                    // excludes them from its statistic.
                    let hypothesis_stars = image_pattern_local
                        .iter()
                        .filter(|&&i| i < self.match_centroid_count)
                        .count();
                    let hypothesis = Hypothesis {
                        rotation: rotation_matrix,
                        fov,
                        vectors: CentroidVectors {
                            pixel_scale: ps_meas,
                            parity_flip,
                            data: working_vectors,
                        },
                        hypothesis_stars,
                    };
                    if let Some(solution) = on_hypothesis(&hypothesis) {
                        return Ok(solution);
                    }
                }
            }
        }

        Err(status)
    }
}

// ── FOV sweep ───────────────────────────────────────────────────────────────

/// Build FOV values to try: exact estimate first, then spiraling outward.
///
/// The swept value only enters the pattern search through the pixel scale
/// used to build centroid unit vectors, and pattern keys are *edge ratios*,
/// so a scale error cancels to first order. Everything downstream
/// self-corrects: the FOV-consistency filter compares the implied FOV (nearly
/// independent of the swept value) against the full `fov_max_error`, and the
/// SVD/verification/refinement all run on vectors rebuilt at the FOV
/// *measured* from each matched pattern (`cat_largest/image_largest ·
/// fov_try` — see the rebuild step in the candidate loop).
///
/// What remains is the second-order tangent-plane nonlinearity: a relative
/// scale error ε perturbs edge ratios by ≈ (θ_hd²/3)·ε, where θ_hd is the
/// half-diagonal angle. The step is sized so that at the midpoint between
/// sweep values this drift stays within the database's ratio quantization
/// tolerance:
///
///   step ≈ pattern_max_error / (θ_hd²/3) · fov_estimate
///
/// (midpoint drift ≤ pattern_max_error/2, leaving the other half of the
/// search tolerance for centroid noise). At 10° FOV with the default 0.003
/// tolerance this gives step ≈ 0.6·fov — a single sweep value covers ±2° —
/// while at very wide FOV (θ_hd large) the sweep stays fine enough to keep
/// quantization drift bounded.
fn build_fov_sweep(
    fov_estimate: f32,
    fov_max_error: Option<f32>,
    pattern_max_error: f32,
    diag_factor: f32,
) -> Vec<f32> {
    let mut values = vec![fov_estimate];

    if let Some(max_error) = fov_max_error {
        if max_error > 0.0 {
            // FOV values at or beyond π are geometrically meaningless, so an
            // infinite/huge max_error (which would otherwise stall the
            // `offset += step` accumulation below into an unbounded loop)
            // clamps to the widest sweep that can matter.
            let max_error = max_error.min(std::f32::consts::PI);
            // Half-diagonal angle at the estimated FOV (fov is width-referenced).
            let theta_hd = ((fov_estimate as f64 / 2.0).tan() * diag_factor as f64).atan();
            // Second-order ratio-drift coefficient; guarded so a degenerate
            // (near-zero) FOV yields one huge step rather than a division
            // blowup.
            let curvature = (theta_hd * theta_hd / 3.0).max(1e-12);
            let step_rel = pattern_max_error as f64 / curvature;
            let step = ((step_rel * fov_estimate as f64) as f32).max(0.001_f32.to_radians());
            let mut offset = step;
            // Also stop once the upward branch reaches π: every candidate at
            // such a FOV is rejected later anyway, after a full pattern search.
            while offset <= max_error && fov_estimate + offset < std::f32::consts::PI {
                values.push(fov_estimate + offset);
                if fov_estimate - offset > 0.0 {
                    values.push(fov_estimate - offset);
                }
                let next = offset + step;
                if next <= offset {
                    // f32 saturation: adding `step` no longer changes `offset`.
                    break;
                }
                offset = next;
            }
        }
    }

    values
}

fn n_choose_k(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let mut result = 1usize;
    for i in 0..k {
        // Saturate rather than overflow-panic for very large centroid counts;
        // this value only feeds a debug log, so a saturated estimate is fine.
        result = result.saturating_mul(n - i) / (i + 1);
    }
    result
}

/// Enumerate pattern keys in the given range, tagged with distance² from center.
///
/// Only *monotone non-decreasing* tuples (`key[i] ≤ key[i+1]`) are emitted:
/// catalog keys are quantized from ascending edge ratios
/// (`compute_sorted_edge_angles` → `compute_edge_ratios` →
/// `compute_pattern_key`, a monotone per-dimension quantization), so every
/// stored key satisfies the invariant and a non-monotone probe can never hit.
/// Skipping them is behavior-preserving and cuts a large share of the hash
/// probes whenever adjacent range dimensions overlap.
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
    // Monotone pruning: this dimension can never be below the previous one in
    // any stored key (see `enumerate_key_range`), so start the scan there.
    let lo = if dim > 0 {
        key_min[dim].max(current[dim - 1])
    } else {
        key_min[dim]
    };
    for v in lo..=key_max[dim] {
        current[dim] = v;
        enumerate_key_range_recursive(key_min, key_max, center, dim + 1, current, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate_key_range_monotone_pruning() {
        // The enumeration must emit exactly the monotone non-decreasing
        // tuples of the full Cartesian product (catalog keys are quantized
        // from ascending edge ratios, so non-monotone probes can never hit),
        // each tagged with its distance² from the center key.
        let cases: [([u32; NUM_EDGE_RATIOS], [u32; NUM_EDGE_RATIOS]); 3] = [
            // Wide overlapping spans — pruning removes most tuples.
            ([2, 2, 2, 2, 2], [5, 5, 5, 5, 5]),
            // Disjoint ascending spans — nothing to prune.
            ([0, 2, 4, 6, 8], [1, 3, 5, 7, 9]),
            // Partially overlapping adjacent spans.
            ([1, 2, 2, 4, 4], [3, 4, 5, 5, 6]),
        ];
        for (key_min, key_max) in cases {
            let center: [u32; NUM_EDGE_RATIOS] =
                std::array::from_fn(|i| (key_min[i] + key_max[i]) / 2);

            let mut got: Vec<(u32, [u32; NUM_EDGE_RATIOS])> = Vec::new();
            enumerate_key_range(&key_min, &key_max, &center, &mut got);

            // Brute-force reference: full product, filtered to monotone.
            let mut expected: Vec<(u32, [u32; NUM_EDGE_RATIOS])> = Vec::new();
            for a in key_min[0]..=key_max[0] {
                for b in key_min[1]..=key_max[1] {
                    for c in key_min[2]..=key_max[2] {
                        for d in key_min[3]..=key_max[3] {
                            for e in key_min[4]..=key_max[4] {
                                let k = [a, b, c, d, e];
                                if k.windows(2).all(|w| w[0] <= w[1]) {
                                    let dist_sq: u32 = (0..NUM_EDGE_RATIOS)
                                        .map(|i| {
                                            let dd = k[i] as i32 - center[i] as i32;
                                            (dd * dd) as u32
                                        })
                                        .sum();
                                    expected.push((dist_sq, k));
                                }
                            }
                        }
                    }
                }
            }
            assert_eq!(got, expected, "min={key_min:?} max={key_max:?}");
        }
    }

    #[test]
    fn test_build_fov_sweep_terminates_on_extreme_max_error() {
        // Regression: with `offset += step` stalling below f32 precision, an
        // infinite/huge max_error used to loop (and push) forever. Values at
        // or beyond π are meaningless, so the sweep must clamp and terminate.
        let fov = 10.0_f32.to_radians();
        for max_error in [f32::INFINITY, f32::MAX, 1e30] {
            let values = build_fov_sweep(fov, Some(max_error), 0.003, 1.2);
            assert!(!values.is_empty());
            assert!(
                values.iter().all(|&v| v > 0.0 && v < std::f32::consts::PI),
                "sweep emitted a FOV outside (0, π)"
            );
            assert!(
                values.len() < 100_000,
                "sweep exploded to {} values for max_error {max_error}",
                values.len()
            );
        }
        // NaN and negative skip the sweep entirely, leaving just the estimate.
        assert_eq!(build_fov_sweep(fov, Some(f32::NAN), 0.003, 1.2).len(), 1);
        assert_eq!(build_fov_sweep(fov, Some(-1.0), 0.003, 1.2).len(), 1);
    }
}
