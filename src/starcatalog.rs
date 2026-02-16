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

use nalgebra::Vector3;
use rkyv::{Archive, Deserialize, Serialize};

use crate::Star;

#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct StarCatalog {
    pub nside: u32,
    pub n_lat: u32,
    pub n_lon: u32,
    pub stars: Vec<Star>,
    pub cell_offsets: Vec<u32>,
    pub star_indices: Vec<u32>,
}

impl StarCatalog {
    /// Build a catalog and spatial index from owned stars.
    ///
    /// `nside` controls resolution and must be greater than zero.
    /// The number of sky cells is `12 * nside^2`.
    pub fn new(nside: u32, stars: Vec<Star>) -> Self {
        assert!(nside > 0, "nside must be > 0");
        let n_lat = 3 * nside;
        let n_lon = 4 * nside;
        let n_cells = (n_lat * n_lon) as usize;

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

    /// Build a catalog by cloning stars from a slice.
    pub fn from_slice(nside: u32, stars: &[Star]) -> Self {
        Self::new(nside, stars.to_vec())
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

    /// Query stars within an angular radius of a pointing direction.
    ///
    /// Input coordinates are in radians (`ra_rad`, `dec_rad`, `radius_rad`).
    /// Returns indices into the internal star storage.
    pub fn query_indices(&self, ra_rad: f32, dec_rad: f32, radius_rad: f32) -> Vec<usize> {
        let dir = radec_to_uvec(ra_rad, dec_rad);
        self.query_indices_from_uvec(dir, radius_rad)
    }

    /// Query stars within an angular radius of a pointing direction.
    ///
    /// Input coordinates are in radians (`ra_rad`, `dec_rad`, `radius_rad`).
    /// Returns references to matching stars.
    pub fn query_stars(&self, ra_rad: f32, dec_rad: f32, radius_rad: f32) -> Vec<&Star> {
        self.query_indices(ra_rad, dec_rad, radius_rad)
            .into_iter()
            .map(|idx| &self.stars[idx])
            .collect()
    }

    /// Query stars around a (possibly non-unit) direction vector.
    ///
    /// `dir` is normalized internally; `radius_rad` is clamped to `[0, π]`.
    /// Returns indices into the internal star storage.
    pub fn query_indices_from_uvec(&self, dir: Vector3<f32>, radius_rad: f32) -> Vec<usize> {
        if self.is_empty() {
            return Vec::new();
        }
        let radius = radius_rad.clamp(0.0, PI);
        let dir = normalize_or_fallback(dir);
        let cos_radius = radius.cos();

        let z_step = 2.0 / self.n_lat as f32;
        let lon_step = TAU / self.n_lon as f32;

        let z_center = dir.z.clamp(-1.0, 1.0);
        let z_min = (z_center - radius.sin()).max(-1.0);
        let z_max = (z_center + radius.sin()).min(1.0);

        let mut out = Vec::new();
        for lat_bin in Self::z_bin_range(self.n_lat, z_min, z_max) {
            let zc = -1.0 + (lat_bin as f32 + 0.5) * z_step;
            let dec_center = zc.clamp(-1.0, 1.0).asin();
            let cos_dec = dec_center.cos().abs().max(1e-6);

            let mut lon_half_span = (radius / cos_dec).min(PI);
            lon_half_span += lon_step;

            let mut phi = dir.y.atan2(dir.x);
            if phi < 0.0 {
                phi += TAU;
            }

            let lon_min = phi - lon_half_span;
            let lon_max = phi + lon_half_span;

            if lon_max - lon_min >= TAU {
                for lon_bin in 0..self.n_lon {
                    self.collect_cell_matches(lat_bin, lon_bin, dir, cos_radius, &mut out);
                }
                continue;
            }

            self.for_each_wrapped_lon_bin(lon_min, lon_max, |lon_bin| {
                self.collect_cell_matches(lat_bin, lon_bin, dir, cos_radius, &mut out);
            });
        }

        out.sort_unstable();
        out.dedup();
        out
    }

    /// Query stars around a (possibly non-unit) direction vector.
    ///
    /// `dir` is normalized internally; `radius_rad` is clamped to `[0, π]`.
    /// Returns references to matching stars.
    pub fn query_stars_from_uvec(&self, dir: Vector3<f32>, radius_rad: f32) -> Vec<&Star> {
        self.query_indices_from_uvec(dir, radius_rad)
            .into_iter()
            .map(|idx| &self.stars[idx])
            .collect()
    }

    fn collect_cell_matches(
        &self,
        lat_bin: u32,
        lon_bin: u32,
        dir: Vector3<f32>,
        cos_radius: f32,
        out: &mut Vec<usize>,
    ) {
        let cell = (lat_bin * self.n_lon + lon_bin) as usize;
        let start = self.cell_offsets[cell] as usize;
        let end = self.cell_offsets[cell + 1] as usize;

        for flat_idx in start..end {
            let star_idx = self.star_indices[flat_idx] as usize;
            let star = &self.stars[star_idx];
            let star_dir = star.uvec();
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

fn radec_to_uvec(ra_rad: f32, dec_rad: f32) -> Vector3<f32> {
    let (sin_ra, cos_ra) = ra_rad.sin_cos();
    let (sin_dec, cos_dec) = dec_rad.sin_cos();
    Vector3::new(cos_dec * cos_ra, cos_dec * sin_ra, sin_dec)
}

fn normalize_or_fallback(v: Vector3<f32>) -> Vector3<f32> {
    let n = v.norm();
    if n > 0.0 {
        v / n
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deg2rad(d: f32) -> f32 {
        d.to_radians()
    }

    #[test]
    fn check_cone_query_from_hipparcos() {
        let hipfile = "data/hip2.dat";
        let data = std::fs::read_to_string(hipfile).expect("Failed to read Hipparcos catalog file");
        let hip_stars = crate::catalogs::hipparcos::load_hipparcos_catalog(&data);
        let stars: Vec<Star> = hip_stars
            .iter()
            .map(|hip_star| crate::star_from_hipparcos(hip_star, None))
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
        let expected: Vec<u64> = catalog
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
        let mut hit_ids: Vec<u64> = hits.iter().map(|s| s.id).collect();
        hit_ids.sort_unstable();
        assert_eq!(hit_ids, expected);
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
        let mut ids: Vec<u64> = hits.iter().map(|s| s.id).collect();
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
        let mut ids: Vec<u64> = hits.iter().map(|s| s.id).collect();
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
}
