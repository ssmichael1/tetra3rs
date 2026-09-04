//! Spatial star catalog optimized for fast cone (angular-radius) searches.
//!
//! `StarCatalog` stores stars in a custom HEALPix-style spherical binning:
//! latitude is partitioned into `3 * nside` bins in `z = sin(dec)`, and
//! longitude into `4 * nside` bins in right ascension, for a total of
//! `12 * nside^2` cells. Each cell maps to a compact slice of star indices.
//!
//! Query flow:
//! 1. Compute candidate cells intersecting the cone around a pointing direction.
//! 2. Scan only stars in those cells.
//! 3. Apply exact angular filtering using a dot-product threshold.
//!
//! This keeps search time close to local star density instead of full-catalog size.

use std::f32::consts::{PI, TAU};

use numeris::Vector3;
use serde::{Deserialize, Serialize};

use crate::Star;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarCatalog {
    // Index internals are private so the spatial-index invariant (derived cell
    // layout must match `stars`) can't be broken from outside. Access stars via
    // `stars()` / `len()`. Field order is the postcard wire layout — keep it.
    nside: u32,
    n_lat: u32,
    n_lon: u32,
    stars: Vec<Star>,
    cell_offsets: Vec<u32>,
    star_indices: Vec<u32>,
}

/// Largest `nside` accepted by [`StarCatalog::new`] (and enforced by
/// [`StarCatalog::validate`]). The index allocates `12 * nside²` cells, so an
/// unbounded value is a memory-exhaustion vector (`nside = 10_000` ≈ 29 GB of
/// bin headers) and overflows the `u32` cell arithmetic near `nside ≈ 19_000`.
/// 1024 (12.6M cells) is far beyond any practical star-density need — the
/// default is 16.
pub const MAX_NSIDE: u32 = 1024;

impl StarCatalog {
    /// Build a catalog and spatial index from owned stars.
    ///
    /// `nside` controls resolution. The number of sky cells is `12 * nside^2`.
    ///
    /// # Panics
    ///
    /// Panics unless `1 <= nside <= MAX_NSIDE`. Config-driven callers
    /// ([`crate::GenerateDatabaseConfig`]) reject out-of-range values with an
    /// error before reaching this constructor.
    pub fn new(nside: u32, stars: Vec<Star>) -> Self {
        assert!(
            nside > 0 && nside <= MAX_NSIDE,
            "nside must be in [1, {MAX_NSIDE}], got {nside}"
        );
        let n_lat = 3 * nside;
        let n_lon = 4 * nside;
        let n_cells = (n_lat as usize) * (n_lon as usize);

        let mut bins: Vec<Vec<u32>> = vec![Vec::new(); n_cells];
        for (star_idx, star) in stars.iter().enumerate() {
            let cell = Self::cell_for_radec(n_lat, n_lon, star.ra_rad, star.dec_rad);
            bins[cell as usize].push(star_idx as u32);
        }

        let mut cell_offsets = Vec::with_capacity(n_cells + 1);
        let mut star_indices = Vec::with_capacity(stars.len());
        cell_offsets.push(0);
        for cell_bin in bins {
            star_indices.extend(cell_bin);
            cell_offsets.push(star_indices.len() as u32);
        }

        Self {
            nside,
            n_lat,
            n_lon,
            stars,
            cell_offsets,
            star_indices,
        }
    }

    /// Return the index resolution parameter.
    pub fn nside(&self) -> u32 {
        self.nside
    }

    /// Return the total number of stars in the catalog.
    pub fn len(&self) -> usize {
        self.stars.len()
    }

    /// Return `true` when the catalog contains no stars.
    pub fn is_empty(&self) -> bool {
        self.stars.is_empty()
    }

    /// Return all catalog stars as an immutable slice.
    pub fn stars(&self) -> &[Star] {
        &self.stars
    }

    /// Check the spatial-index invariants that [`Self::new`] establishes by
    /// construction but that serde deserialization can bypass (the fields are
    /// private, yet `#[derive(Deserialize)]` writes them directly).
    ///
    /// Every cone query indexes `cell_offsets`, `star_indices`, and `stars`
    /// with values derived from these fields, so a catalog decoded from a
    /// corrupt or tampered file must pass this check before use — otherwise
    /// the first query panics out-of-bounds. Called by
    /// [`crate::SolverDatabase::load_from_file`].
    pub fn validate(&self) -> crate::Result<()> {
        use crate::Error::InvalidInput;
        if self.nside == 0 || self.nside > MAX_NSIDE {
            return Err(InvalidInput(format!(
                "StarCatalog: nside must be in [1, {MAX_NSIDE}], got {}",
                self.nside
            )));
        }
        if self.n_lat != 3 * self.nside || self.n_lon != 4 * self.nside {
            return Err(InvalidInput(format!(
                "StarCatalog: n_lat/n_lon ({}/{}) inconsistent with nside {}",
                self.n_lat, self.n_lon, self.nside
            )));
        }
        let n_cells = (self.n_lat as usize) * (self.n_lon as usize);
        if self.cell_offsets.len() != n_cells + 1 {
            return Err(InvalidInput(format!(
                "StarCatalog: cell_offsets has {} entries, expected {} (12*nside²+1)",
                self.cell_offsets.len(),
                n_cells + 1
            )));
        }
        if self.cell_offsets[0] != 0
            || self.cell_offsets.windows(2).any(|w| w[0] > w[1])
            || self.cell_offsets[n_cells] as usize != self.star_indices.len()
        {
            return Err(InvalidInput(
                "StarCatalog: cell_offsets must rise monotonically from 0 to star_indices.len()"
                    .into(),
            ));
        }
        let n_stars = self.stars.len();
        if self.star_indices.len() != n_stars {
            return Err(InvalidInput(format!(
                "StarCatalog: star_indices has {} entries for {} stars",
                self.star_indices.len(),
                n_stars
            )));
        }
        if self.star_indices.iter().any(|&i| i as usize >= n_stars) {
            return Err(InvalidInput(
                "StarCatalog: star_indices contains an index past the star table".into(),
            ));
        }
        Ok(())
    }

    /// Query stars within an angular radius of a pointing direction.
    ///
    /// Input coordinates are in radians (`ra_rad`, `dec_rad`, `radius_rad`).
    /// Returns indices into the internal star storage.
    pub fn query_indices(&self, ra_rad: f32, dec_rad: f32, radius_rad: f32) -> Vec<usize> {
        let dir = radec_to_uvec(ra_rad, dec_rad);
        self.query_indices_from_uvec(dir, radius_rad)
    }

    /// Query stars within an angular radius of a pointing direction, returning
    /// references to matching stars. Test-only convenience over
    /// [`query_indices`](Self::query_indices), which the solver uses directly.
    #[cfg(test)]
    pub fn query_stars(&self, ra_rad: f32, dec_rad: f32, radius_rad: f32) -> Vec<&Star> {
        self.query_indices(ra_rad, dec_rad, radius_rad)
            .into_iter()
            .map(|idx| &self.stars[idx])
            .collect()
    }

    /// Query stars around a (possibly non-unit) direction vector.
    ///
    /// `dir` is normalized internally; `radius_rad` is clamped to `[0, π]`.
    /// Returns indices into the internal star storage. Each candidate star's
    /// unit vector is recomputed from its `(ra, dec)` via trigonometry; for hot
    /// paths that already hold precomputed unit vectors, prefer the internal
    /// `query_indices_from_uvec_cached`.
    pub fn query_indices_from_uvec(&self, dir: Vector3<f32>, radius_rad: f32) -> Vec<usize> {
        self.query_impl(dir, radius_rad, None)
    }

    /// Same as [`query_indices_from_uvec`](Self::query_indices_from_uvec) but
    /// reads each candidate star's unit vector from `unit_vectors` instead of
    /// recomputing it via `sin`/`cos`.
    ///
    /// `unit_vectors` MUST be index-aligned with [`stars`](Self::stars) (i.e.
    /// `unit_vectors[i]` is the unit vector of `stars()[i]`). When that holds,
    /// the result is identical to `query_indices_from_uvec`, just without the
    /// per-star trigonometry — a meaningful saving when this is called once per
    /// catalog star (database generation) or per solve candidate.
    pub(crate) fn query_indices_from_uvec_cached(
        &self,
        dir: Vector3<f32>,
        radius_rad: f32,
        unit_vectors: &[[f32; 3]],
    ) -> Vec<usize> {
        debug_assert_eq!(
            unit_vectors.len(),
            self.stars.len(),
            "unit_vectors cache must be index-aligned with stars"
        );
        self.query_impl(dir, radius_rad, Some(unit_vectors))
    }

    fn query_impl(
        &self,
        dir: Vector3<f32>,
        radius_rad: f32,
        cache: Option<&[[f32; 3]]>,
    ) -> Vec<usize> {
        if self.is_empty() {
            return Vec::new();
        }
        let radius = radius_rad.clamp(0.0, PI);
        let dir = normalize_or_fallback(dir);
        let cos_radius = radius.cos();

        let z_step = 2.0 / self.n_lat as f32;
        let lon_step = TAU / self.n_lon as f32;

        let z_center = dir[2].clamp(-1.0, 1.0);
        let z_min = (z_center - radius.sin()).max(-1.0);
        let z_max = (z_center + radius.sin()).min(1.0);

        let mut out = Vec::new();
        for lat_bin in Self::z_bin_range(self.n_lat, z_min, z_max) {
            // Bound the RA half-span with the smallest |cos dec| anywhere in
            // this latitude bin — its edge nearest a pole — not the bin
            // center. Stars sit anywhere in the bin, and near the poles the
            // span a star at the polar edge needs is many times the span at
            // the center (a bin touching a pole needs every RA bin). Using
            // the center silently dropped most stars within a few degrees of
            // either pole.
            let z_lo = -1.0 + lat_bin as f32 * z_step;
            let z_hi = z_lo + z_step;
            let z_edge = z_lo.abs().max(z_hi.abs()).min(1.0);
            let cos_dec = (1.0 - z_edge * z_edge).max(0.0).sqrt().max(1e-6);

            let mut lon_half_span = (radius / cos_dec).min(PI);
            lon_half_span += lon_step;

            let mut phi = dir[1].atan2(dir[0]);
            if phi < 0.0 {
                phi += TAU;
            }

            let lon_min = phi - lon_half_span;
            let lon_max = phi + lon_half_span;

            if lon_max - lon_min >= TAU {
                for lon_bin in 0..self.n_lon {
                    self.collect_cell_matches(lat_bin, lon_bin, dir, cos_radius, cache, &mut out);
                }
                continue;
            }

            self.for_each_wrapped_lon_bin(lon_min, lon_max, |lon_bin| {
                self.collect_cell_matches(lat_bin, lon_bin, dir, cos_radius, cache, &mut out);
            });
        }

        out.sort_unstable();
        out.dedup();
        out
    }

    fn collect_cell_matches(
        &self,
        lat_bin: u32,
        lon_bin: u32,
        dir: Vector3<f32>,
        cos_radius: f32,
        cache: Option<&[[f32; 3]]>,
        out: &mut Vec<usize>,
    ) {
        let cell = (lat_bin * self.n_lon + lon_bin) as usize;
        let start = self.cell_offsets[cell] as usize;
        let end = self.cell_offsets[cell + 1] as usize;

        for flat_idx in start..end {
            let star_idx = self.star_indices[flat_idx] as usize;
            let star_dir = match cache {
                Some(vectors) => Vector3::from_array(vectors[star_idx]),
                None => self.stars[star_idx].uvec(),
            };
            if dir.dot(&star_dir) >= cos_radius {
                out.push(star_idx);
            }
        }
    }

    fn for_each_wrapped_lon_bin<F>(&self, lon_min: f32, lon_max: f32, mut f: F)
    where
        F: FnMut(u32),
    {
        let start = wrap_angle(lon_min);
        let end = wrap_angle(lon_max);

        let start_bin = Self::phi_to_lon_bin(self.n_lon, start);
        let end_bin = Self::phi_to_lon_bin(self.n_lon, end);

        if start_bin <= end_bin {
            for lon_bin in start_bin..=end_bin {
                f(lon_bin);
            }
            return;
        }

        for lon_bin in start_bin..self.n_lon {
            f(lon_bin);
        }
        for lon_bin in 0..=end_bin {
            f(lon_bin);
        }
    }

    fn z_bin_range(n_lat: u32, z_min: f32, z_max: f32) -> std::ops::RangeInclusive<u32> {
        let start = Self::z_to_lat_bin(n_lat, z_min);
        let end = Self::z_to_lat_bin(n_lat, z_max);
        start..=end
    }

    fn cell_for_radec(n_lat: u32, n_lon: u32, ra_rad: f32, dec_rad: f32) -> u32 {
        let mut phi = wrap_angle(ra_rad);
        if phi >= TAU {
            phi = 0.0;
        }
        let z = dec_rad.sin().clamp(-1.0, 1.0);
        let lat_bin = Self::z_to_lat_bin(n_lat, z);
        let lon_bin = Self::phi_to_lon_bin(n_lon, phi);
        lat_bin * n_lon + lon_bin
    }

    fn z_to_lat_bin(n_lat: u32, z: f32) -> u32 {
        let u = ((z.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0);
        let mut idx = (u * n_lat as f32).floor() as u32;
        if idx >= n_lat {
            idx = n_lat - 1;
        }
        idx
    }

    fn phi_to_lon_bin(n_lon: u32, phi: f32) -> u32 {
        let u = (phi / TAU).clamp(0.0, 1.0 - f32::EPSILON);
        let mut idx = (u * n_lon as f32).floor() as u32;
        if idx >= n_lon {
            idx = n_lon - 1;
        }
        idx
    }
}

fn wrap_angle(theta_rad: f32) -> f32 {
    theta_rad.rem_euclid(TAU)
}

/// ICRS unit vector from (RA, Dec) in radians. Also used by [`Star::uvec`].
pub(crate) fn radec_to_uvec(ra_rad: f32, dec_rad: f32) -> Vector3<f32> {
    let (sin_ra, cos_ra) = ra_rad.sin_cos();
    let (sin_dec, cos_dec) = dec_rad.sin_cos();
    Vector3::from_array([cos_dec * cos_ra, cos_dec * sin_ra, sin_dec])
}

fn normalize_or_fallback(v: Vector3<f32>) -> Vector3<f32> {
    let n = v.norm();
    if n > 0.0 {
        v / n
    } else {
        Vector3::from_array([1.0, 0.0, 0.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deg2rad(d: f32) -> f32 {
        d.to_radians()
    }

    #[test]
    #[cfg(feature = "hipparcos")]
    fn check_cone_query_from_hipparcos() {
        let hipfile = "data/hip2.dat";
        let data = std::fs::read_to_string(hipfile).expect("Failed to read Hipparcos catalog file");
        let hip_stars = crate::catalogs::hipparcos::load_hipparcos_catalog(&data);
        let stars: Vec<Star> = hip_stars
            .iter()
            .map(|hip_star| crate::star::star_from_hipparcos(hip_star, None))
            .collect();
        let catalog = StarCatalog::new(16, stars);

        // Polaris is HIP 11767 at RA=37.95°, Dec=89.26°, V=1.97
        let hits = catalog.query_stars(deg2rad(37.95), deg2rad(89.26), deg2rad(0.5));
        assert!(!hits.is_empty());
        let polaris = hits
            .iter()
            .find(|s| s.id == 11767)
            .expect("Polaris not found");
        assert!((polaris.mag - 1.97).abs() < 0.1);

        // Pick a RA, and DEC, and search for stars with 1 degree.
        // Compare with results of manual search
        // over full vector of stars
        let ra = deg2rad(120.0);
        let dec = deg2rad(30.0);
        let radius = deg2rad(1.0);
        let hits = catalog.query_stars(ra, dec, radius);
        let expected: Vec<i64> = catalog
            .stars
            .iter()
            .filter(|s| {
                let star_dir = s.uvec();
                let query_dir = radec_to_uvec(ra, dec);
                let cos_angle = query_dir.dot(&star_dir);
                let angle = cos_angle.acos();
                angle <= radius
            })
            .map(|s| s.id)
            .collect();
        let mut hit_ids: Vec<i64> = hits.iter().map(|s| s.id).collect();
        hit_ids.sort_unstable();
        assert_eq!(hit_ids, expected);
    }

    /// Field-for-field mirror of `StarCatalog`'s postcard wire layout, used to
    /// craft catalogs that violate the private-field invariants — exactly what
    /// a corrupt or tampered database file can produce through serde.
    #[derive(serde::Serialize)]
    struct RawCatalog {
        nside: u32,
        n_lat: u32,
        n_lon: u32,
        stars: Vec<Star>,
        cell_offsets: Vec<u32>,
        star_indices: Vec<u32>,
    }

    #[test]
    fn validate_accepts_constructed_and_rejects_tampered() {
        let stars = vec![Star {
            id: 1,
            ra_rad: 0.5,
            dec_rad: 0.2,
            mag: 3.0,
        }];
        let catalog = StarCatalog::new(2, stars.clone());
        assert!(catalog.validate().is_ok());

        // Round-trip through postcard stays valid.
        let bytes = postcard::to_allocvec(&catalog).unwrap();
        let decoded: StarCatalog = postcard::from_bytes(&bytes).unwrap();
        assert!(decoded.validate().is_ok());

        let decode = |raw: &RawCatalog| -> StarCatalog {
            postcard::from_bytes(&postcard::to_allocvec(raw).unwrap()).unwrap()
        };

        // Each of these decodes cleanly but would panic (OOB index or u32
        // underflow) inside the first cone query without validate().
        let n_cells = 6 * 8; // nside 2
        let cases = [
            RawCatalog {
                nside: 0, // n_lat-1 underflow in z_to_lat_bin
                n_lat: 0,
                n_lon: 0,
                stars: stars.clone(),
                cell_offsets: vec![0],
                star_indices: vec![0],
            },
            RawCatalog {
                nside: 2,
                n_lat: 6,
                n_lon: 8,
                stars: stars.clone(),
                cell_offsets: vec![0; 3], // too short for 48 cells
                star_indices: vec![0],
            },
            RawCatalog {
                nside: 2,
                n_lat: 6,
                n_lon: 8,
                stars: stars.clone(),
                cell_offsets: vec![7; n_cells + 1], // offsets past star_indices
                star_indices: vec![0],
            },
            RawCatalog {
                nside: 2,
                n_lat: 6,
                n_lon: 8,
                stars: stars.clone(),
                cell_offsets: {
                    let mut o = vec![0; n_cells + 1];
                    o[n_cells] = 1;
                    o
                },
                star_indices: vec![99], // index past the 1-star table
            },
        ];
        for (i, raw) in cases.iter().enumerate() {
            assert!(decode(raw).validate().is_err(), "case {i} passed validate");
        }
    }

    #[test]
    fn cone_query_finds_nearby_stars() {
        let stars = vec![
            Star {
                id: 1,
                ra_rad: deg2rad(0.0),
                dec_rad: deg2rad(0.0),
                mag: 2.0,
            },
            Star {
                id: 2,
                ra_rad: deg2rad(2.0),
                dec_rad: deg2rad(1.0),
                mag: 2.5,
            },
            Star {
                id: 3,
                ra_rad: deg2rad(40.0),
                dec_rad: deg2rad(-10.0),
                mag: 5.0,
            },
        ];

        let index = StarCatalog::new(8, stars);
        let hits = index.query_stars(deg2rad(0.5), deg2rad(0.25), deg2rad(3.0));
        let mut ids: Vec<i64> = hits.iter().map(|s| s.id).collect();
        ids.sort_unstable();

        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn cone_query_handles_ra_wraparound() {
        let stars = vec![
            Star {
                id: 10,
                ra_rad: deg2rad(359.0),
                dec_rad: deg2rad(0.0),
                mag: 3.0,
            },
            Star {
                id: 11,
                ra_rad: deg2rad(1.0),
                dec_rad: deg2rad(0.0),
                mag: 3.0,
            },
            Star {
                id: 12,
                ra_rad: deg2rad(180.0),
                dec_rad: deg2rad(0.0),
                mag: 3.0,
            },
        ];

        let index = StarCatalog::new(8, stars);
        let hits = index.query_stars(deg2rad(0.0), deg2rad(0.0), deg2rad(3.0));
        let mut ids: Vec<i64> = hits.iter().map(|s| s.id).collect();
        ids.sort_unstable();

        assert_eq!(ids, vec![10, 11]);
    }

    #[test]
    fn query_from_uvec_matches_radec_query() {
        let stars = vec![
            Star {
                id: 20,
                ra_rad: deg2rad(120.0),
                dec_rad: deg2rad(30.0),
                mag: 2.0,
            },
            Star {
                id: 21,
                ra_rad: deg2rad(124.0),
                dec_rad: deg2rad(30.5),
                mag: 2.1,
            },
        ];

        let index = StarCatalog::new(4, stars);
        let by_radec = index.query_indices(deg2rad(122.0), deg2rad(30.0), deg2rad(3.0));
        let by_uvec = index
            .query_indices_from_uvec(radec_to_uvec(deg2rad(122.0), deg2rad(30.0)), deg2rad(3.0));

        assert_eq!(by_radec, by_uvec);
    }
    /// Regression for the polar RA-span bug: every latitude bin's RA span
    /// must be sized for its polar edge, not its center, or queries within
    /// a few degrees of either pole miss most of their stars. Compares the
    /// indexed query against a brute-force dot-product scan on a uniform
    /// synthetic sky at declinations from the equator to 89.5°.
    #[test]
    fn cone_query_matches_brute_force_near_poles() {
        // Deterministic xorshift64* — keeps the test dependency-free.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut unit = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u32 << 24) as f32
        };
        let stars: Vec<Star> = (0..50_000)
            .map(|i| {
                let z = 2.0 * unit() - 1.0;
                Star {
                    id: i,
                    ra_rad: unit() * TAU,
                    dec_rad: z.asin(),
                    mag: 5.0,
                }
            })
            .collect();
        let catalog = StarCatalog::new(16, stars);

        for dec_deg in [0.0f32, 45.0, 80.0, 85.0, 88.0, 89.5, -89.5] {
            for radius_deg in [1.0f32, 3.0, 7.0] {
                for ra_deg in [10.0f32, 100.0, 200.0, 300.0] {
                    let dir = radec_to_uvec(deg2rad(ra_deg), deg2rad(dec_deg));
                    let radius = deg2rad(radius_deg);
                    let got = catalog.query_indices_from_uvec(dir, radius);
                    let cos_r = radius.cos();
                    let expected: Vec<usize> = catalog
                        .stars
                        .iter()
                        .enumerate()
                        .filter(|(_, s)| dir.dot(&s.uvec()) >= cos_r)
                        .map(|(i, _)| i)
                        .collect();
                    assert_eq!(
                        got, expected,
                        "cone query at dec {dec_deg}°, ra {ra_deg}°, radius {radius_deg}°"
                    );
                }
            }
        }
    }
}
