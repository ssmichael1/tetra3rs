//! Database generation: builds the pattern hash table from a star catalog.
//!
//! Closely follows tetra3's `generate_database()` algorithm:
//! 1. Load stars, apply magnitude cut, sort by brightness.
//! 2. Build spatial index for fast cone queries.
//! 3. For each FOV scale, distribute lattice fields over the sky.
//! 4. In each field, generate 4-star patterns (brightest first) and hash them.

use std::collections::HashSet;

use tracing::info;

use crate::catalogs::hipparcos::{load_hipparcos_catalog_from_file, HipparcosStar};
use crate::star::star_from_hipparcos;
use crate::{Star, StarCatalog};

use super::combinations::BreadthFirstCombinations;
use super::pattern::{
    self, compute_edge_ratios, compute_pattern_key, compute_pattern_key_hash,
    compute_sorted_edge_angles, distance_from_angle, hash_to_index, insert_pattern, next_prime,
    sort_u32_pattern_by_centroid_distance, PATTERN_SIZE,
};
use super::{DatabaseProperties, GenerateDatabaseConfig, SolverDatabase};

// ── Sky geometry utilities ──────────────────────────────────────────────────

/// Approximate number of FOV-sized fields needed to tile the full sky.
fn num_fields_for_sky(fov_rad: f32) -> usize {
    // Solid angle of a cone with half-angle fov/2: 2π(1 − cos(fov/2))
    // Full sky: 4π steradians
    let half_fov = fov_rad / 2.0;
    let cone_solid_angle = 2.0 * std::f32::consts::PI * (1.0 - half_fov.cos());
    if cone_solid_angle <= 0.0 {
        return 1;
    }
    let n = (4.0 * std::f32::consts::PI / cone_solid_angle).ceil() as usize;
    n.max(1)
}

/// Minimum angular separation between stars for a given FOV and star density.
/// This is the "cluster buster" that prevents dense star clusters from
/// dominating the pattern budget.
fn separation_for_density(fov_rad: f32, stars_per_fov: u32) -> f32 {
    // Area of a FOV circle ≈ π(fov/2)². With N uniformly distributed stars,
    // average spacing ≈ (fov/2) * sqrt(π/N).
    (fov_rad / 2.0) * (std::f32::consts::PI / stars_per_fov as f32).sqrt()
}

/// Generate N approximately-uniform points on the unit sphere using the
/// Fibonacci sphere lattice (golden spiral).
fn fibonacci_sphere_lattice(n: usize) -> Vec<[f32; 3]> {
    let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        // z uniformly spaced from ~+1 to ~-1
        let z = 1.0 - (2.0 * i as f64 + 1.0) / n as f64;
        let r = (1.0 - z * z).sqrt();
        let theta = 2.0 * std::f64::consts::PI * i as f64 / golden_ratio;
        let x = r * theta.cos();
        let y = r * theta.sin();
        points.push([x as f32, y as f32, z as f32]);
    }
    points
}

// ── Database generation ─────────────────────────────────────────────────────

impl SolverDatabase {
    /// Generate a solver database from a Hipparcos catalog file.
    ///
    /// This is the main entry point for building a new database. It closely
    /// follows tetra3's `generate_database()`.
    pub fn generate_from_hipparcos(
        catalog_path: &str,
        config: &GenerateDatabaseConfig,
    ) -> anyhow::Result<Self> {
        info!("Loading Hipparcos catalog from {}", catalog_path);
        let hip_stars = load_hipparcos_catalog_from_file(catalog_path)?;
        info!("Loaded {} raw Hipparcos entries", hip_stars.len());

        Self::generate_from_stars(&hip_stars, config)
    }

    /// Generate a solver database from pre-loaded Hipparcos stars.
    pub fn generate_from_stars(
        hip_stars: &[HipparcosStar],
        config: &GenerateDatabaseConfig,
    ) -> anyhow::Result<Self> {
        let max_fov = config.max_fov_deg.to_radians();
        let min_fov = config
            .min_fov_deg
            .map(|d| d.to_radians())
            .unwrap_or(max_fov);

        let pattern_bins = (0.25 / config.pattern_max_error).round() as u32;
        info!("Pattern bins: {}, max_error: {}", pattern_bins, config.pattern_max_error);

        let epoch_pm_year = config.epoch_proper_motion_year;
        info!("Proper motion epoch: {:?}", epoch_pm_year);

        // Convert to Star, filtering out entries that fail conversion
        let mut stars: Vec<Star> = hip_stars
            .iter()
            .map(|h| star_from_hipparcos(h, epoch_pm_year))
            .collect();

        // Sort by brightness (ascending magnitude = brightest first)
        stars.sort_by(|a, b| a.mag.partial_cmp(&b.mag).unwrap_or(std::cmp::Ordering::Equal));

        // Determine magnitude cutoff
        let star_max_magnitude = config.star_max_magnitude.unwrap_or_else(|| {
            compute_magnitude_cutoff(&stars, min_fov, config.verification_stars_per_fov)
        });

        // Apply magnitude cut
        let num_before = stars.len();
        stars.retain(|s| s.mag <= star_max_magnitude);
        info!(
            "Kept {} of {} stars brighter than magnitude {:.1}",
            stars.len(),
            num_before,
            star_max_magnitude
        );

        let num_stars = stars.len();

        // Precompute unit vectors
        let star_vectors: Vec<[f32; 3]> = stars.iter().map(|s| {
            let v = s.uvec();
            [v.x, v.y, v.z]
        }).collect();

        // Save catalog IDs before building the spatial index
        let star_catalog_ids: Vec<u64> = stars.iter().map(|s| s.id).collect();

        // Build spatial catalog (stars are already brightness-sorted)
        let star_catalog = StarCatalog::new(config.catalog_nside, stars);
        info!("Built star catalog with nside={}", config.catalog_nside);

        // ── Determine FOV scales for pattern generation ──
        let fov_ratio = max_fov / min_fov;
        let fov_divisions = if fov_ratio < config.multiscale_step.sqrt() {
            1
        } else {
            let log_ratio = fov_ratio.ln() / config.multiscale_step.ln();
            log_ratio.ceil() as usize + 1
        };

        let pattern_fovs: Vec<f32> = if fov_divisions <= 1 {
            vec![max_fov]
        } else {
            (0..fov_divisions)
                .map(|i| {
                    let t = i as f32 / (fov_divisions - 1) as f32;
                    (min_fov.ln() + t * (max_fov.ln() - min_fov.ln())).exp()
                })
                .collect()
        };
        info!(
            "Generating patterns at {} FOV scales: {:?} deg",
            pattern_fovs.len(),
            pattern_fovs.iter().map(|f| f.to_degrees()).collect::<Vec<_>>()
        );

        // ── Generate patterns across all FOV scales ──
        let mut pattern_set: HashSet<[u32; PATTERN_SIZE]> = HashSet::new();

        // Process FOVs from largest to smallest (reversed like tetra3)
        for &pattern_fov in pattern_fovs.iter().rev() {
            let pattern_stars_separation = if fov_divisions <= 1 {
                separation_for_density(min_fov, config.verification_stars_per_fov)
            } else {
                separation_for_density(pattern_fov, config.verification_stars_per_fov)
            };
            let _pattern_stars_dist = distance_from_angle(pattern_stars_separation);

            info!(
                "FOV {:.2}°: cluster-buster separation {:.3}°",
                pattern_fov.to_degrees(),
                pattern_stars_separation.to_degrees()
            );

            // ── Cluster buster: select well-separated pattern stars ──
            let mut keep_for_patterns = vec![false; num_stars];
            for star_ind in 0..num_stars {
                // Check if any already-kept star is too close
                let dir = nalgebra::Vector3::new(
                    star_vectors[star_ind][0],
                    star_vectors[star_ind][1],
                    star_vectors[star_ind][2],
                );
                let nearby = star_catalog.query_indices_from_uvec(dir, pattern_stars_separation);
                let occupied = nearby.iter().any(|&idx| keep_for_patterns[idx]);
                if !occupied {
                    keep_for_patterns[star_ind] = true;
                }
            }

            let pattern_star_indices: Vec<usize> = (0..num_stars)
                .filter(|&i| keep_for_patterns[i])
                .collect();
            info!("Pattern stars at this FOV: {}", pattern_star_indices.len());

            // ── Distribute lattice fields and generate patterns ──
            let fov_angle = pattern_fov / 2.0;
            let _fov_dist = distance_from_angle(fov_angle);
            let n_fields = num_fields_for_sky(pattern_fov)
                * config.lattice_field_oversampling as usize;

            let lattice_points = fibonacci_sphere_lattice(n_fields);
            let mut total_added = 0usize;

            for center in &lattice_points {
                // Find pattern stars within this lattice field
                let center_v = nalgebra::Vector3::new(center[0], center[1], center[2]);
                let field_stars_all = star_catalog.query_indices_from_uvec(center_v, fov_angle);

                // Keep only pattern-eligible stars, in brightness order
                let field_pattern_stars: Vec<usize> = field_stars_all
                    .into_iter()
                    .filter(|&idx| keep_for_patterns[idx])
                    .collect();
                // These are already in brightness order since star_catalog indices
                // correspond to the brightness-sorted star array, and query returns
                // sorted indices.

                if field_pattern_stars.len() < PATTERN_SIZE {
                    continue;
                }

                // Generate 4-star combinations, brightest first
                let mut patterns_this_field = 0u32;
                for combo in BreadthFirstCombinations::new(&field_pattern_stars, PATTERN_SIZE) {
                    let mut pat = [
                        combo[0] as u32,
                        combo[1] as u32,
                        combo[2] as u32,
                        combo[3] as u32,
                    ];
                    pat.sort_unstable(); // canonical ordering for dedup
                    let is_new = pattern_set.insert(pat);
                    if is_new {
                        total_added += 1;
                        if pattern_set.len() % 100_000 == 0 {
                            info!("Generated {} patterns so far...", pattern_set.len());
                        }
                    }
                    patterns_this_field += 1;
                    if patterns_this_field >= config.patterns_per_lattice_field as u32 {
                        break;
                    }
                }
            }
            info!("Added {} new patterns at this FOV ({} total)", total_added, pattern_set.len());
        }

        let pattern_list: Vec<[u32; PATTERN_SIZE]> = pattern_set.into_iter().collect();
        info!("Total unique patterns: {}", pattern_list.len());

        // ── Build hash table ──
        // Use quadratic probing. Table size = next_prime(2 * num_patterns).
        let catalog_length = next_prime(2 * pattern_list.len() as u64) as usize;
        info!(
            "Hash table size: {} (load factor {:.2})",
            catalog_length,
            pattern_list.len() as f64 / catalog_length as f64
        );

        let mut pattern_catalog = vec![[0u32; PATTERN_SIZE]; catalog_length];
        let mut pattern_largest_edge_table = vec![0.0f32; catalog_length];
        let mut pattern_key_hashes_table = vec![0u16; catalog_length];

        for pat in &pattern_list {
            // Get the 4 star vectors
            let vectors: [[f32; 3]; 4] = [
                star_vectors[pat[0] as usize],
                star_vectors[pat[1] as usize],
                star_vectors[pat[2] as usize],
                star_vectors[pat[3] as usize],
            ];

            // Compute edge angles, ratios, and pattern key
            let edge_angles = compute_sorted_edge_angles(&vectors);
            let largest_angle = edge_angles[pattern::NUM_EDGES - 1];
            let edge_ratios = compute_edge_ratios(&edge_angles);
            let pkey = compute_pattern_key(&edge_ratios, pattern_bins);
            let pkey_hash = compute_pattern_key_hash(&pkey, pattern_bins);
            let hidx = hash_to_index(pkey_hash, catalog_length as u64);

            // Sort pattern by centroid distance for canonical ordering
            let mut sorted_pat = *pat;
            sort_u32_pattern_by_centroid_distance(&mut sorted_pat, &star_vectors);

            // Insert into hash table
            let slot = insert_pattern(sorted_pat, hidx, &mut pattern_catalog);
            pattern_largest_edge_table[slot] = largest_angle;
            pattern_key_hashes_table[slot] = (pkey_hash & 0xFFFF) as u16;
        }

        info!("Database generation complete.");
        info!(
            "Star table: {} stars ({} bytes)",
            num_stars,
            num_stars * std::mem::size_of::<Star>()
        );
        info!(
            "Pattern catalog: {} slots ({} bytes)",
            catalog_length,
            catalog_length * PATTERN_SIZE * 4
        );

        let props = DatabaseProperties {
            pattern_bins,
            pattern_max_error: config.pattern_max_error,
            max_fov_rad: max_fov,
            min_fov_rad: min_fov,
            star_max_magnitude,
            num_patterns: pattern_list.len() as u32,
            epoch_equinox: 2000, // Hipparcos uses ICRS ≈ J2000
            epoch_proper_motion_year: epoch_pm_year.unwrap_or(1991.25) as f32,
            verification_stars_per_fov: config.verification_stars_per_fov,
            lattice_field_oversampling: config.lattice_field_oversampling,
            patterns_per_lattice_field: config.patterns_per_lattice_field,
        };

        Ok(SolverDatabase {
            star_catalog,
            star_vectors,
            star_catalog_ids,
            pattern_catalog,
            pattern_largest_edge: pattern_largest_edge_table,
            pattern_key_hashes: pattern_key_hashes_table,
            props,
        })
    }
}

// ── Magnitude cutoff computation ────────────────────────────────────────────

/// Automatically compute the magnitude cutoff based on required star density.
/// Follows tetra3's approach: histogram star magnitudes, find the cutoff
/// that gives enough stars to fill verification_stars_per_fov in each FOV.
fn compute_magnitude_cutoff(stars: &[Star], min_fov: f32, verification_stars_per_fov: u32) -> f32 {
    if stars.is_empty() {
        return 10.0;
    }

    let num_fovs = num_fields_for_sky(min_fov);
    // Total stars needed across the sky, with tetra3's empirical fudge factor
    let total_stars_needed = (num_fovs as f64 * verification_stars_per_fov as f64 * 0.7) as usize;

    if total_stars_needed >= stars.len() {
        // Need all stars in the catalog
        return stars.last().unwrap().mag;
    }

    // Stars are already sorted by magnitude (brightest first).
    // The cutoff is the magnitude of the N-th star.
    stars[total_stars_needed.min(stars.len() - 1)].mag
}

// ── Serialization ───────────────────────────────────────────────────────────

impl SolverDatabase {
    /// Serialize the database to bytes using rkyv.
    pub fn to_rkyv_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .expect("rkyv serialization failed")
            .to_vec()
    }

    /// Save the database to a file using rkyv.
    pub fn save_to_file(&self, path: &str) -> anyhow::Result<()> {
        let bytes = self.to_rkyv_bytes();
        std::fs::write(path, &bytes)?;
        info!("Saved database to {} ({} bytes)", path, bytes.len());
        Ok(())
    }

    /// Load a database from an rkyv file.
    pub fn load_from_file(path: &str) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let db = rkyv::from_bytes::<Self, rkyv::rancor::Error>(&bytes)
            .map_err(|e| anyhow::anyhow!("rkyv deserialization failed: {}", e))?;
        info!(
            "Loaded database: {} stars, {} patterns",
            db.star_catalog.len(),
            db.props.num_patterns
        );
        Ok(db)
    }
}
