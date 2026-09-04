//! Shared synthetic-sky helpers for the integration tests, the golden dump
//! and the profiler example (which includes this file by path).
//!
//! Everything here is brute force on purpose — field membership is decided
//! by a dot product against every star, never by the spatial index — so a
//! test of the index (or of the solver) never depends on the code under
//! test to build its inputs.
#![allow(dead_code)]

use numeris::{Matrix3, Vector3};
use tetra3::{Centroid, GenerateDatabaseConfig, Star};

/// Minimal deterministic xorshift64* RNG (dependency-free, identical
/// streams across the consumers so recorded outputs stay comparable).
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform f32 in [0, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Uniform f64 in [0, 1).
    pub fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal deviate (Box–Muller).
    pub fn gauss(&mut self) -> f32 {
        let u1 = self.unit_f64().max(1e-12);
        let u2 = self.unit_f64();
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }

    /// Uniformly distributed unit vector.
    pub fn direction(&mut self) -> Vector3<f32> {
        let z = 2.0 * self.unit() - 1.0;
        let phi = self.unit() * std::f32::consts::TAU;
        let r = (1.0 - z * z).max(0.0).sqrt();
        Vector3::from_array([r * phi.cos(), r * phi.sin(), z]).normalize()
    }
}

/// Rotation matrix (ICRS → camera) for a boresight at `ra`/`dec` (radians)
/// and a roll about it. Camera +Z is the boresight; without roll, camera +X
/// points toward decreasing RA (celestial north × boresight).
pub fn rotation_from_ra_dec_roll(ra: f32, dec: f32, roll: f32) -> Matrix3<f32> {
    let boresight = Vector3::from_array([dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()]);
    let cam_z = boresight.normalize();
    let north = Vector3::from_array([0.0, 0.0, 1.0]);
    let raw_x = north.cross(&cam_z);
    let cam_x_noroll = if raw_x.norm() > 1e-6 {
        raw_x.normalize()
    } else {
        Vector3::from_array([1.0, 0.0, 0.0])
            .cross(&cam_z)
            .normalize()
    };
    let cam_y_noroll = cam_z.cross(&cam_x_noroll);
    let cam_x = cam_x_noroll * roll.cos() + cam_y_noroll * roll.sin();
    let cam_y = -cam_x_noroll * roll.sin() + cam_y_noroll * roll.cos();
    Matrix3::new([
        [cam_x[0], cam_x[1], cam_x[2]],
        [cam_y[0], cam_y[1], cam_y[2]],
        [cam_z[0], cam_z[1], cam_z[2]],
    ])
}

/// Pinhole pixel scale (rad/px) for a horizontal FOV and image width.
pub fn pixel_scale(fov_rad: f32, image_width: u32) -> f32 {
    1.0 / ((image_width as f32 / 2.0) / (fov_rad / 2.0).tan())
}

/// Project every catalog star that lands inside a square frame of
/// half-width `half_fov` (tangent-plane radians) into pixel centroids, by
/// brute force over `star_vectors`. `mags` supplies the brightness
/// (`mass = 10 − mag`); `noise_px` adds Gaussian centroid noise.
pub fn project_field(
    star_vectors: &[[f32; 3]],
    mags: &[f32],
    rot: &Matrix3<f32>,
    half_fov: f32,
    pixel_scale: f32,
    noise_px: f32,
    rng: &mut Rng,
) -> Vec<Centroid> {
    let mut centroids = Vec::new();
    for (i, sv) in star_vectors.iter().enumerate() {
        let v = *rot * Vector3::from_array([sv[0], sv[1], sv[2]]);
        if v[2] <= 0.01 {
            continue;
        }
        let (cx, cy) = (v[0] / v[2], v[1] / v[2]);
        if cx.abs() < half_fov && cy.abs() < half_fov {
            let (nx, ny) = if noise_px > 0.0 {
                (noise_px * rng.gauss(), noise_px * rng.gauss())
            } else {
                (0.0, 0.0)
            };
            centroids.push(Centroid {
                x: cx / pixel_scale + nx,
                y: cy / pixel_scale + ny,
                mass: Some(10.0 - mags[i]),
                cov: None,
            });
        }
    }
    centroids
}

/// `n` stars uniformly distributed over the sphere (ids 0..n, magnitude 5).
pub fn uniform_sky(n: usize, seed: u64) -> Vec<Star> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| {
            let z = 2.0 * rng.unit() - 1.0;
            Star {
                id: i as i64,
                ra_rad: rng.unit() * std::f32::consts::TAU,
                dec_rad: z.asin(),
                mag: 5.0,
            }
        })
        .collect()
}

/// The 10°-FOV database configuration shared by the profiler example, the
/// golden dump and the statistical tests (built from `data/gaia_merged.bin`
/// in well under a second).
pub fn profiler_db_config() -> GenerateDatabaseConfig {
    GenerateDatabaseConfig {
        max_fov_deg: 12.0,
        min_fov_deg: None,
        star_max_magnitude: Some(7.0),
        pattern_max_error: 0.003,
        lattice_field_oversampling: 50,
        patterns_per_lattice_field: 100,
        verification_stars_per_fov: 40,
        multiscale_step: 1.5,
        epoch_proper_motion_year: Some(2025.0),
        catalog_nside: 8,
    }
}
