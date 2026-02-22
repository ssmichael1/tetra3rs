//! Stellar aberration utilities.
//!
//! Provides an approximate Earth barycentric velocity for first-order
//! stellar aberration correction.

/// Mean orbital speed of Earth in km/s.
const V_ORB_KM_S: f64 = 29.7859;

/// Obliquity of the ecliptic (J2000), in radians.
const OBLIQUITY_RAD: f64 = 23.4393 * (std::f64::consts::PI / 180.0);

/// Sun's mean ecliptic longitude at J2000.0, in degrees.
const L0_DEG: f64 = 280.460;

/// Daily rate of the Sun's mean ecliptic longitude, in degrees/day.
const L_RATE_DEG: f64 = 0.9856474;

/// Approximate Earth barycentric velocity in km/s (ICRS equatorial frame).
///
/// Uses a circular-orbit approximation of Earth's motion around the Sun.
/// Accuracy is ~0.5 km/s (~1.7%) — limited by the neglected orbital
/// eccentricity (e ≈ 0.017). This is sufficient for stellar aberration
/// correction, where the effect itself is ~20″ and the eccentricity term
/// contributes only ~0.3″.
///
/// The Sun's barycentric motion (~12 m/s from Jupiter) is also neglected.
///
/// # Arguments
///
/// * `days_since_j2000` — Days since the J2000.0 epoch (2000 January 1,
///   12:00 TT). TT ≈ UTC + 69 s; the difference is negligible for this
///   approximation.
///
/// # Returns
///
/// `[vx, vy, vz]` in km/s, ICRS equatorial frame. Pass directly to
/// [`SolveConfig::observer_velocity_km_s`](crate::SolveConfig::observer_velocity_km_s).
///
/// # Example
///
/// ```
/// use tetra3::earth_barycentric_velocity;
///
/// // 2025 July 10 ≈ J2000 + 9321 days
/// let v = earth_barycentric_velocity(9321.0);
/// let speed = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
/// assert!((speed - 29.8).abs() < 0.5);
/// ```
pub fn earth_barycentric_velocity(days_since_j2000: f64) -> [f64; 3] {
    // Sun's mean ecliptic longitude
    let lambda_deg = L0_DEG + L_RATE_DEG * days_since_j2000;
    let lambda = lambda_deg.to_radians();

    let sin_l = lambda.sin();
    let cos_l = lambda.cos();
    let cos_e = OBLIQUITY_RAD.cos();
    let sin_e = OBLIQUITY_RAD.sin();

    // Earth velocity in ecliptic: [v_orb * sin(λ), -v_orb * cos(λ), 0]
    // Rotate ecliptic → equatorial (ICRS)
    [
        V_ORB_KM_S * sin_l,
        -V_ORB_KM_S * cos_l * cos_e,
        -V_ORB_KM_S * cos_l * sin_e,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speed_is_approximately_30_km_s() {
        // Check at several dates throughout a year
        for d in [0.0, 91.0, 182.0, 273.0, 365.25] {
            let v = earth_barycentric_velocity(d);
            let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!(
                (speed - V_ORB_KM_S).abs() < 0.01,
                "day {d}: speed {speed:.4} km/s, expected ~{V_ORB_KM_S}"
            );
        }
    }

    #[test]
    fn test_vernal_equinox_direction() {
        // At vernal equinox (λ_Sun ≈ 0°), Earth velocity should be
        // predominantly in -Y equatorial (toward RA ≈ 18h)
        // λ_Sun = 0 when d ≈ (360 - 280.46) / 0.9856 ≈ 80.7 days ≈ March 21
        let d = (360.0 - L0_DEG) / L_RATE_DEG;
        let v = earth_barycentric_velocity(d);

        // vx should be near zero (sin(0) = 0)
        assert!(v[0].abs() < 1.0, "vx at equinox: {}", v[0]);
        // vy should be large negative
        assert!(v[1] < -25.0, "vy at equinox should be << 0: {}", v[1]);
    }

    #[test]
    fn test_half_year_reversal() {
        // Velocity should roughly reverse after half a year
        let v1 = earth_barycentric_velocity(0.0);
        let v2 = earth_barycentric_velocity(365.25 / 2.0);
        // Dot product should be negative (roughly anti-parallel)
        let dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
        assert!(
            dot < -800.0, // -v_orb² ≈ -887
            "half-year dot product should be strongly negative: {dot}"
        );
    }
}
