//! Database generation: builds the pattern hash table from a star catalog.
//!
//! Closely follows tetra3's `generate_database()` algorithm:
//! 1. Load stars, apply magnitude cut, sort by brightness.
//! 2. Build spatial index for fast cone queries.
//! 3. For each FOV scale, distribute lattice fields over the sky.
//! 4. In each field, generate 4-star patterns (brightest first) and hash them.

use std::collections::HashSet;

use log::{info, warn};

use std::sync::Arc;

use serde::Deserialize;
#[cfg(test)]
use serde::Serialize;

use crate::{Star, StarCatalog};

use super::combinations::BreadthFirstCombinations;
use super::pattern::{
    self, compute_edge_ratios, compute_pattern_key, compute_pattern_key_hash,
    compute_sorted_edge_angles, hash_to_index, next_prime, sort_pattern_by_centroid_distance,
    PATTERN_SIZE,
};
use super::pattern_catalog::PackedStore;
use super::pattern_wire;
use super::{DatabaseProperties, GenerateDatabaseConfig, PatternEntry, SolverDatabase};

// ── Sky geometry utilities ──────────────────────────────────────────────────

/// Hard ceiling on lattice points per FOV scale during generation. The count
/// scales as `oversampling / fov²`, so a tiny-but-valid `min_fov_deg` (e.g.
/// 0.1° at the default oversampling of 100 → ~5×10⁸ points ≈ 6 GB) would
/// otherwise exhaust memory with no diagnostic. 10⁸ points (1.2 GB) is far
/// beyond any published tetra3-style database.
const MAX_LATTICE_POINTS: usize = 100_000_000;

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
/// dominating the pattern budget. The solver applies the same constraint to
/// image centroids so its thinning mirrors database generation.
pub(super) fn separation_for_density(fov_rad: f32, stars_per_fov: u32) -> f32 {
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
    /// This is the main entry point for building a new database from Hipparcos.
    /// It closely follows tetra3's `generate_database()`.
    #[cfg(feature = "hipparcos")]
    pub fn generate_from_hipparcos(
        catalog_path: &str,
        config: &GenerateDatabaseConfig,
    ) -> crate::Result<Self> {
        use crate::catalogs::hipparcos::load_hipparcos_catalog_from_file;
        use crate::star::star_from_hipparcos;

        info!("Loading Hipparcos catalog from {}", catalog_path);
        let hip_stars = load_hipparcos_catalog_from_file(catalog_path)?;
        info!("Loaded {} raw Hipparcos entries", hip_stars.len());

        let stars: Vec<Star> = hip_stars
            .iter()
            .map(|h| star_from_hipparcos(h, config.epoch_proper_motion_year))
            .collect();

        let default_pm_year = 1991.25; // Hipparcos reference epoch
        Self::generate_from_star_list(stars, config, default_pm_year)
    }

    /// Generate a solver database from a Gaia binary catalog file.
    ///
    /// Expects a `.bin` file in the compact GDR3 format from the `gaia-catalog`
    /// package (header magic `GDR3`).
    ///
    /// Negative source_ids indicate Hipparcos gap-fill stars from the merged
    /// catalog.
    pub fn generate_from_gaia(
        catalog_path: &str,
        config: &GenerateDatabaseConfig,
    ) -> crate::Result<Self> {
        use crate::catalogs::gaia::load_gaia_binary;
        use crate::star::star_from_gaia;

        info!("Loading Gaia catalog from {}", catalog_path);
        let gaia_stars = load_gaia_binary(catalog_path)?;
        info!("Loaded {} Gaia entries", gaia_stars.len());

        let stars: Vec<Star> = gaia_stars
            .iter()
            .map(|g| star_from_gaia(g, config.epoch_proper_motion_year))
            .collect();

        let default_pm_year = 2016.0; // Gaia DR3 reference epoch
        Self::generate_from_star_list(stars, config, default_pm_year)
    }

    /// Core database generation from a pre-converted list of generic stars.
    ///
    /// `default_pm_year` is used as the proper motion epoch when
    /// `config.epoch_proper_motion_year` is `None`.
    pub fn generate_from_star_list(
        mut stars: Vec<Star>,
        config: &GenerateDatabaseConfig,
        default_pm_year: f64,
    ) -> crate::Result<Self> {
        config.validate()?;
        let max_fov = config.max_fov_deg.to_radians();
        let min_fov = config
            .min_fov_deg
            .map(|d| d.to_radians())
            .unwrap_or(max_fov);

        let pattern_bins = (0.25 / config.pattern_max_error).round() as u32;
        info!(
            "Pattern bins: {}, max_error: {}",
            pattern_bins, config.pattern_max_error
        );

        let epoch_pm_year = config.epoch_proper_motion_year;
        info!("Proper motion epoch: {:?}", epoch_pm_year);

        // Sort by brightness (ascending magnitude = brightest first)
        stars.sort_by(|a, b| {
            a.mag
                .partial_cmp(&b.mag)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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
        let star_vectors: Vec<[f32; 3]> = stars
            .iter()
            .map(|s| {
                let v = s.uvec();
                [v[0], v[1], v[2]]
            })
            .collect();

        // Save catalog IDs before building the spatial index
        let star_catalog_ids: Vec<i64> = stars.iter().map(|s| s.id).collect();

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
            pattern_fovs
                .iter()
                .map(|f| f.to_degrees())
                .collect::<Vec<_>>()
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

            info!(
                "FOV {:.2}°: cluster-buster separation {:.3}°",
                pattern_fov.to_degrees(),
                pattern_stars_separation.to_degrees()
            );

            // ── Cluster buster: select well-separated pattern stars ──
            let mut keep_for_patterns = vec![false; num_stars];
            for star_ind in 0..num_stars {
                // Check if any already-kept star is too close
                let dir = numeris::Vector3::from_array([
                    star_vectors[star_ind][0],
                    star_vectors[star_ind][1],
                    star_vectors[star_ind][2],
                ]);
                let nearby = star_catalog.query_indices_from_uvec_cached(
                    dir,
                    pattern_stars_separation,
                    &star_vectors,
                );
                let occupied = nearby.iter().any(|&idx| keep_for_patterns[idx]);
                if !occupied {
                    keep_for_patterns[star_ind] = true;
                }
            }

            let pattern_star_indices: Vec<usize> =
                (0..num_stars).filter(|&i| keep_for_patterns[i]).collect();
            info!("Pattern stars at this FOV: {}", pattern_star_indices.len());

            // ── Distribute lattice fields and generate patterns ──
            let fov_angle = pattern_fov / 2.0;
            let n_fields = num_fields_for_sky(pattern_fov)
                .saturating_mul(config.lattice_field_oversampling as usize);
            if n_fields > MAX_LATTICE_POINTS {
                return Err(crate::Error::InvalidInput(format!(
                    "lattice would need {n_fields} field centers at FOV {:.3}° \
                     (limit {MAX_LATTICE_POINTS}); raise min_fov_deg or lower \
                     lattice_field_oversampling",
                    pattern_fov.to_degrees()
                )));
            }

            let lattice_points = fibonacci_sphere_lattice(n_fields);
            let mut total_added = 0usize;

            for center in &lattice_points {
                // Find pattern stars within this lattice field
                let center_v = numeris::Vector3::from_array([center[0], center[1], center[2]]);
                let field_stars_all =
                    star_catalog.query_indices_from_uvec_cached(center_v, fov_angle, &star_vectors);

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
                for combo in BreadthFirstCombinations::<PATTERN_SIZE>::new(&field_pattern_stars) {
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
                        if pattern_set.len().is_multiple_of(100_000) {
                            info!("Generated {} patterns so far...", pattern_set.len());
                        }
                    }
                    patterns_this_field += 1;
                    if patterns_this_field >= config.patterns_per_lattice_field {
                        break;
                    }
                }
            }
            info!(
                "Added {} new patterns at this FOV ({} total)",
                total_added,
                pattern_set.len()
            );
        }

        // Sort so the table layout is a pure function of the input: HashSet
        // iteration order varies per process, which made hash-chain order —
        // and therefore the candidate order / `Solution.prob` divisor of a
        // solve — differ between databases generated from identical inputs.
        let mut pattern_list: Vec<[u32; PATTERN_SIZE]> = pattern_set.into_iter().collect();
        pattern_list.sort_unstable();
        info!("Total unique patterns: {}", pattern_list.len());

        // ── Build hash table ──
        // Use quadratic probing. Table size = next_prime(2 * num_patterns).
        let catalog_length = next_prime(2 * pattern_list.len() as u64) as usize;
        info!(
            "Hash table size: {} (load factor {:.2})",
            catalog_length,
            pattern_list.len() as f64 / catalog_length as f64
        );

        let items = pattern_list.iter().map(|pat| {
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
            sort_pattern_by_centroid_distance(&mut sorted_pat, |i| star_vectors[i as usize]);

            let entry = PatternEntry::new(sorted_pat, largest_angle, (pkey_hash & 0xFFFF) as u16);
            (hidx, entry)
        });
        let pattern_catalog = super::PatternCatalog::build(catalog_length, items);

        info!("Database generation complete.");
        info!(
            "Star table: {} stars ({} bytes)",
            num_stars,
            num_stars * std::mem::size_of::<Star>()
        );
        info!(
            "Pattern catalog: {} slots ({} bytes)",
            catalog_length,
            pattern_catalog.heap_bytes()
        );

        let props = DatabaseProperties {
            pattern_bins,
            pattern_max_error: config.pattern_max_error,
            max_fov_rad: max_fov,
            min_fov_rad: min_fov,
            star_max_magnitude,
            num_patterns: pattern_list.len() as u32,
            epoch_equinox: 2000, // ICRS ≈ J2000
            epoch_proper_motion_year: epoch_pm_year.unwrap_or(default_pm_year) as f32,
            verification_stars_per_fov: config.verification_stars_per_fov,
            lattice_field_oversampling: config.lattice_field_oversampling,
            patterns_per_lattice_field: config.patterns_per_lattice_field,
        };

        Ok(SolverDatabase {
            star_catalog,
            star_vectors,
            star_catalog_ids,
            pattern_catalog,
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

/// Magic prefix of a serialized database (`to_bytes` / `save_to_file`).
const DB_MAGIC: &[u8; 4] = b"T3DB";
/// Current serialized-database format version, written after the magic as
/// a little-endian `u16`. Bump when the postcard payload's layout changes so
/// an old crate fails a new file with a clear message instead of an opaque
/// decode error or a silently mis-decoded table; keep reading the previous
/// version if a frozen mirror of its layout (like [`SolverDatabaseV1`]) is
/// cheap.
///
/// - 1: every field derived; the pattern table a postcard `Vec<PatternEntry>`.
///   Written by 0.13 and earlier (pre-0.13 files carry no header at all).
///   Deprecated in 0.14: loads with a warning.
/// - 2: the pattern table as an occupancy bitmap + packed entries
///   (`pattern_wire`); the other fields unchanged.
const DB_FORMAT_VERSION: u16 = 2;

/// Format version 1 layout, frozen. **Deprecated** (loading one logs a
/// warning): remove it at the next format change. Its pattern table decodes
/// through serde's per-element path, so a v1 load is slower than v2 —
/// regenerate or re-save large databases to upgrade them. The other fields are the current
/// types: if any of those change, bump the version again and drop this
/// rather than adapt it.
#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
struct SolverDatabaseV1 {
    star_catalog: StarCatalog,
    star_vectors: Vec<[f32; 3]>,
    star_catalog_ids: Vec<i64>,
    pattern_catalog: PatternCatalogV1,
    props: DatabaseProperties,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
struct PatternCatalogV1 {
    entries: Vec<PatternEntry>,
}

impl From<SolverDatabaseV1> for SolverDatabase {
    fn from(v1: SolverDatabaseV1) -> Self {
        Self {
            star_catalog: v1.star_catalog,
            star_vectors: v1.star_vectors,
            star_catalog_ids: v1.star_catalog_ids,
            pattern_catalog: super::PatternCatalog::from_dense(&v1.pattern_catalog.entries),
            props: v1.props,
        }
    }
}

/// Format version 2 layout with the pattern table's sections borrowed from
/// the input, so [`SolverDatabase::from_vec`] can keep the input buffer as
/// the table's backing store instead of copying the packed section. Must
/// match the derived `Serialize` of [`SolverDatabase`] field for field.
#[derive(Deserialize)]
struct SolverDatabaseV2<'a> {
    star_catalog: StarCatalog,
    star_vectors: Vec<[f32; 3]>,
    star_catalog_ids: Vec<i64>,
    #[serde(borrow)]
    pattern_catalog: PatternCatalogV2<'a>,
    props: DatabaseProperties,
}

/// `(n_slots, occupancy bitmap, packed entries)` — see `pattern_wire`.
#[derive(Deserialize)]
struct PatternCatalogV2<'a>(u64, &'a [u8], &'a [u8]);

impl SolverDatabaseV2<'_> {
    /// `owner`: the buffer the input was borrowed from, to share as the
    /// packed section's storage; `None` copies the section.
    fn into_db(self, owner: Option<&Arc<Vec<u8>>>) -> crate::Result<SolverDatabase> {
        let PatternCatalogV2(n_slots, bitmap, packed) = self.pattern_catalog;
        let store = || match owner {
            Some(buf) => PackedStore::Shared {
                buf: Arc::clone(buf),
                // `packed` is a subslice of `buf` (both come from the same
                // decode), so the offset is in range.
                start: packed.as_ptr() as usize - buf.as_ptr() as usize,
                len: packed.len(),
            },
            None => PackedStore::Owned(packed.to_vec()),
        };
        let pattern_catalog = pattern_wire::unpack(n_slots, bitmap, packed, store)
            .map_err(crate::Error::InvalidInput)?;
        Ok(SolverDatabase {
            star_catalog: self.star_catalog,
            star_vectors: self.star_vectors,
            star_catalog_ids: self.star_catalog_ids,
            pattern_catalog,
            props: self.props,
        })
    }
}

impl SolverDatabase {
    /// Serialize the database: a 6-byte header (`"T3DB"` + format version)
    /// followed by the postcard payload. See [`Self::from_bytes`].
    pub fn to_bytes(&self) -> crate::Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(6 + 64);
        bytes.extend_from_slice(DB_MAGIC);
        bytes.extend_from_slice(&DB_FORMAT_VERSION.to_le_bytes());
        postcard::to_extend(self, bytes).map_err(Into::into)
    }

    /// Decode a database produced by [`Self::to_bytes`] — the current format
    /// or version 1 (0.13 and earlier; also the bare pre-header payload,
    /// detected by the missing magic) — and check its invariants with
    /// [`Self::validate`]. Version 1 is deprecated: it still loads, with a
    /// logged warning, until the next format change; re-save to upgrade.
    ///
    /// Fails with [`crate::Error::InvalidInput`] on an unsupported format
    /// version, and with the decode error on a truncated or corrupt payload.
    pub fn from_bytes(bytes: &[u8]) -> crate::Result<Self> {
        Self::decode(bytes, None)
    }

    /// [`Self::from_bytes`], taking ownership of the buffer: a current-format
    /// database keeps it as the pattern table's storage rather than copying
    /// the table out of it (the table is ~90% of a large file), halving peak
    /// memory and skipping a multi-GB copy. The whole buffer stays alive with
    /// the database.
    pub fn from_vec(bytes: Vec<u8>) -> crate::Result<Self> {
        let buf = Arc::new(bytes);
        Self::decode(&buf, Some(&buf))
    }

    fn decode(bytes: &[u8], owner: Option<&Arc<Vec<u8>>>) -> crate::Result<Self> {
        let (version, payload) = match bytes.strip_prefix(DB_MAGIC) {
            Some(rest) => {
                let (ver, payload) = rest.split_at_checked(2).ok_or_else(|| {
                    crate::Error::InvalidInput("database header truncated after the magic".into())
                })?;
                (u16::from_le_bytes([ver[0], ver[1]]), payload)
            }
            // Legacy (pre-header) file: the whole buffer is a v1 payload. A
            // legacy payload starting with the magic bytes would need
            // nside = 0x54 followed by n_lat = 0x33, which `validate()`
            // rejects (n_lat must be 3·nside), so misdetection cannot load.
            None => (1, bytes),
        };
        let db = match version {
            1 => postcard::from_bytes::<SolverDatabaseV1>(payload)?.into(),
            2 => postcard::from_bytes::<SolverDatabaseV2>(payload)?.into_db(owner)?,
            _ => {
                return Err(crate::Error::InvalidInput(format!(
                    "unsupported database format version {version} \
                     (this crate reads versions 1 and {DB_FORMAT_VERSION})"
                )));
            }
        };
        db.validate()?;
        if version == 1 {
            warn!(
                "Loaded a format-1 solver database (written by tetra3 0.13 or earlier). \
                 Format 1 is deprecated and stops loading at the next format change; \
                 re-save it with save_to_file to upgrade (format 2 also loads ~4x faster)."
            );
        }
        Ok(db)
    }

    /// Save the database to a file using postcard.
    pub fn save_to_file(&self, path: &str) -> crate::Result<()> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, &bytes)?;
        info!("Saved database to {} ({} bytes)", path, bytes.len());
        Ok(())
    }

    /// Load a database file written by [`Self::save_to_file`] (or a legacy
    /// pre-header file). See [`Self::from_bytes`].
    ///
    /// The decoded database is checked with [`Self::validate`], so a corrupt,
    /// truncated, or tampered file fails here with a descriptive
    /// [`crate::Error::InvalidInput`] instead of panicking mid-solve.
    pub fn load_from_file(path: &str) -> crate::Result<Self> {
        let db = Self::from_vec(std::fs::read(path)?)?;
        info!(
            "Loaded database: {} stars, {} patterns",
            db.star_catalog.len(),
            db.props.num_patterns
        );
        Ok(db)
    }

    /// Check the cross-field invariants the solver relies on but postcard
    /// deserialization cannot: every structure decodes independently, so a
    /// file that parses cleanly can still carry pattern entries indexing past
    /// the star table, a spatial index inconsistent with its stars, or
    /// properties that blow up the key-enumeration bounds. Each of those is a
    /// deferred panic (or memory blow-up) on the first solve.
    ///
    /// Called by [`Self::load_from_file`]; call it yourself after decoding
    /// database bytes from any other untrusted source (e.g. pickle).
    pub fn validate(&self) -> crate::Result<()> {
        use crate::Error::InvalidInput;
        self.star_catalog.validate()?;

        let n_stars = self.star_catalog.len();
        if self.star_vectors.len() != n_stars || self.star_catalog_ids.len() != n_stars {
            return Err(InvalidInput(format!(
                "SolverDatabase: star_vectors ({}) / star_catalog_ids ({}) must both match \
                 the star catalog ({n_stars} stars)",
                self.star_vectors.len(),
                self.star_catalog_ids.len()
            )));
        }

        let p = &self.props;
        if !(p.pattern_max_error.is_finite()
            && p.pattern_max_error > 0.0
            && p.pattern_max_error <= 0.25)
        {
            return Err(InvalidInput(format!(
                "SolverDatabase: props.pattern_max_error must be in (0, 0.25], got {}",
                p.pattern_max_error
            )));
        }
        // Generation always derives bins from the error tolerance; a file
        // claiming a different (huge) bin count inflates the solver's 5-D
        // candidate-key enumeration without bound.
        let expected_bins = (0.25 / p.pattern_max_error).round() as u32;
        if p.pattern_bins != expected_bins {
            return Err(InvalidInput(format!(
                "SolverDatabase: props.pattern_bins ({}) inconsistent with \
                 pattern_max_error {} (expected {expected_bins})",
                p.pattern_bins, p.pattern_max_error
            )));
        }
        let fov_ok = p.max_fov_rad.is_finite()
            && p.max_fov_rad > 0.0
            && p.max_fov_rad < std::f32::consts::PI
            && p.min_fov_rad.is_finite()
            && p.min_fov_rad > 0.0
            && p.min_fov_rad <= p.max_fov_rad;
        if !fov_ok {
            return Err(InvalidInput(format!(
                "SolverDatabase: props FOV range [{}, {}] rad must be finite, positive, \
                 ordered, and below π",
                p.min_fov_rad, p.max_fov_rad
            )));
        }
        if p.verification_stars_per_fov == 0 {
            return Err(InvalidInput(
                "SolverDatabase: props.verification_stars_per_fov must be >= 1".into(),
            ));
        }
        if p.num_patterns as usize > self.pattern_catalog.len() {
            return Err(InvalidInput(format!(
                "SolverDatabase: props.num_patterns ({}) exceeds the pattern table size ({})",
                p.num_patterns,
                self.pattern_catalog.len()
            )));
        }

        // Every occupied entry is indexed straight into star_vectors during
        // hash probing — before any filter can reject it — so the largest
        // index in the packed section must be in range.
        if self
            .pattern_catalog
            .max_star_index()
            .is_some_and(|max| max as usize >= n_stars)
        {
            return Err(InvalidInput(format!(
                "SolverDatabase: pattern entry references star index past the \
                 {n_stars}-star table"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod header_tests {
    use super::*;
    use crate::solver::PatternCatalog;

    /// A small but `validate()`-clean database.
    fn tiny_db() -> SolverDatabase {
        let stars: Vec<Star> = (0..5)
            .map(|i| Star {
                id: 100 + i as i64,
                ra_rad: 0.3 * i as f32,
                dec_rad: 0.1 * i as f32 - 0.2,
                mag: 3.0 + i as f32,
            })
            .collect();
        let star_vectors = stars
            .iter()
            .map(|s| {
                let v = s.uvec();
                [v[0], v[1], v[2]]
            })
            .collect();
        let star_catalog_ids = stars.iter().map(|s| s.id).collect();
        let star_catalog = StarCatalog::new(1, stars);
        let mut dense = vec![PatternEntry::EMPTY; 101];
        dense[7] = PatternEntry::new([0, 1, 2, 3], 0.05, 0xabcd);
        dense[100] = PatternEntry::new([1, 2, 3, 4], 0.06, 0x1234);
        let pattern_catalog = PatternCatalog::from_dense(&dense);
        SolverDatabase {
            star_catalog,
            star_vectors,
            star_catalog_ids,
            pattern_catalog,
            props: DatabaseProperties {
                pattern_bins: 250,
                pattern_max_error: 0.001,
                max_fov_rad: 0.2,
                min_fov_rad: 0.05,
                star_max_magnitude: 8.0,
                num_patterns: 2,
                epoch_equinox: 2000,
                epoch_proper_motion_year: 2026.0,
                verification_stars_per_fov: 20,
                lattice_field_oversampling: 100,
                patterns_per_lattice_field: 50,
            },
        }
    }

    fn assert_same(a: &SolverDatabase, b: &SolverDatabase) {
        assert_eq!(a.star_catalog.stars(), b.star_catalog.stars());
        assert_eq!(a.star_vectors, b.star_vectors);
        assert_eq!(a.star_catalog_ids, b.star_catalog_ids);
        assert_eq!(a.pattern_catalog, b.pattern_catalog);
        assert_eq!(a.props, b.props);
    }

    #[test]
    fn current_format_roundtrips() {
        let db = tiny_db();
        let bytes = db.to_bytes().unwrap();
        assert_eq!(&bytes[..4], DB_MAGIC);
        assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), 2);
        assert_same(&SolverDatabase::from_bytes(&bytes).unwrap(), &db);
    }

    #[test]
    fn from_vec_shares_the_buffer() {
        let db = tiny_db();
        let bytes = db.to_bytes().unwrap();
        let copied = SolverDatabase::from_bytes(&bytes).unwrap();
        let shared = SolverDatabase::from_vec(bytes).unwrap();
        assert!(!copied.pattern_catalog.shares_buffer());
        assert!(shared.pattern_catalog.shares_buffer());
        assert_same(&shared, &db);
        assert_same(&copied, &db);
    }

    /// Records warnings with the thread that logged them (tests run in
    /// parallel; each checks only its own thread's records).
    struct WarnCapture(std::sync::Mutex<Vec<(std::thread::ThreadId, String)>>);

    impl log::Log for WarnCapture {
        fn enabled(&self, m: &log::Metadata) -> bool {
            m.level() <= log::Level::Warn
        }
        fn log(&self, r: &log::Record) {
            if self.enabled(r.metadata()) {
                let rec = (std::thread::current().id(), r.args().to_string());
                self.0.lock().unwrap().push(rec);
            }
        }
        fn flush(&self) {}
    }

    static WARNINGS: WarnCapture = WarnCapture(std::sync::Mutex::new(Vec::new()));

    fn format_1_warnings() -> usize {
        let me = std::thread::current().id();
        let recs = WARNINGS.0.lock().unwrap();
        recs.iter()
            .filter(|(t, m)| *t == me && m.contains("format-1"))
            .count()
    }

    #[test]
    fn version_1_load_warns_and_version_2_does_not() {
        // The only logger set in this test binary.
        log::set_logger(&WARNINGS).unwrap();
        log::set_max_level(log::LevelFilter::Warn);

        let db = tiny_db();
        SolverDatabase::from_bytes(&db.to_bytes().unwrap()).unwrap();
        assert_eq!(format_1_warnings(), 0);

        let v1 = SolverDatabaseV1 {
            star_catalog: db.star_catalog.clone(),
            star_vectors: db.star_vectors.clone(),
            star_catalog_ids: db.star_catalog_ids.clone(),
            pattern_catalog: PatternCatalogV1 {
                entries: db.pattern_catalog.to_dense(),
            },
            props: db.props.clone(),
        };
        SolverDatabase::from_bytes(&postcard::to_allocvec(&v1).unwrap()).unwrap();
        assert_eq!(format_1_warnings(), 1);
    }

    #[test]
    fn version_1_and_legacy_files_still_load() {
        let db = tiny_db();
        let v1 = SolverDatabaseV1 {
            star_catalog: db.star_catalog.clone(),
            star_vectors: db.star_vectors.clone(),
            star_catalog_ids: db.star_catalog_ids.clone(),
            pattern_catalog: PatternCatalogV1 {
                entries: db.pattern_catalog.to_dense(),
            },
            props: db.props.clone(),
        };
        let payload = postcard::to_allocvec(&v1).unwrap();

        let mut with_header = DB_MAGIC.to_vec();
        with_header.extend_from_slice(&1u16.to_le_bytes());
        with_header.extend_from_slice(&payload);
        assert_same(&SolverDatabase::from_bytes(&with_header).unwrap(), &db);

        // Pre-0.13: no header, bare v1 payload.
        assert_same(&SolverDatabase::from_bytes(&payload).unwrap(), &db);

        // A v1 payload labelled v2 (or vice versa) is an error, not a panic.
        with_header[4..6].copy_from_slice(&2u16.to_le_bytes());
        assert!(SolverDatabase::from_bytes(&with_header).is_err());
        let mut v2_as_v1 = db.to_bytes().unwrap();
        v2_as_v1[4..6].copy_from_slice(&1u16.to_le_bytes());
        assert!(SolverDatabase::from_bytes(&v2_as_v1).is_err());
    }

    #[test]
    fn validate_rejects_star_index_past_the_table() {
        let mut db = tiny_db();
        assert!(db.validate().is_ok());
        let mut dense = db.pattern_catalog.to_dense();
        dense[50] = PatternEntry::new([0, 0, 0, 5], 0.05, 1);
        db.pattern_catalog = PatternCatalog::from_dense(&dense);
        let err = db.validate().unwrap_err().to_string();
        assert!(err.contains("past the 5-star table"), "{err}");
    }

    #[test]
    fn rejects_unsupported_version_and_truncated_header() {
        let mut bytes = DB_MAGIC.to_vec();
        bytes.extend_from_slice(&(DB_FORMAT_VERSION + 1).to_le_bytes());
        bytes.extend_from_slice(&[0u8; 32]);
        let err = SolverDatabase::from_bytes(&bytes).unwrap_err().to_string();
        assert!(err.contains("unsupported database format version"), "{err}");

        let err = SolverDatabase::from_bytes(&DB_MAGIC[..])
            .unwrap_err()
            .to_string();
        assert!(err.contains("truncated"), "{err}");
        let err = SolverDatabase::from_bytes(&bytes[..5])
            .unwrap_err()
            .to_string();
        assert!(err.contains("truncated"), "{err}");
    }

    #[test]
    fn garbage_payload_is_an_error_not_a_panic() {
        let mut bytes = DB_MAGIC.to_vec();
        bytes.extend_from_slice(&DB_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&[0xFF; 40]);
        assert!(SolverDatabase::from_bytes(&bytes).is_err());
        // Legacy path (no magic): also an error, never a panic.
        assert!(SolverDatabase::from_bytes(&[0xFF; 40]).is_err());
        assert!(SolverDatabase::from_bytes(&[]).is_err());
    }
}
