//! Radial distortion model: r_distorted = r × (1 + k1·r² + k2·r⁴ + k3·r⁶).
//!
//! Classic Brown-Conrady radial distortion with up to 3 coefficients.
//! All coordinates are in pixels relative to the optical center.

/// Radial distortion with up to 3 coefficients.
///
/// The forward model maps ideal radius `r` to distorted radius:
///
/// ```text
/// r_d = r × (1 + k1·r² + k2·r⁴ + k3·r⁶)
/// ```
///
/// Undistortion (inverse) is computed via Newton-Raphson iteration.
#[derive(Debug, Clone)]
pub struct RadialDistortion {
    /// First radial coefficient (barrel < 0, pincushion > 0).
    pub k1: f64,
    /// Second radial coefficient.
    pub k2: f64,
    /// Third radial coefficient.
    pub k3: f64,
}

impl RadialDistortion {
    /// Create a new radial distortion model with the given coefficients.
    ///
    /// Set unused coefficients to 0.0. For example, `RadialDistortion::new(-1e-8, 0.0, 0.0)`
    /// for a simple barrel distortion.
    pub fn new(k1: f64, k2: f64, k3: f64) -> Self {
        Self { k1, k2, k3 }
    }

    /// Forward distortion: ideal → distorted.
    ///
    /// Given ideal (pinhole) pixel coordinates `(x, y)`, returns the
    /// distorted coordinates `(x_d, y_d)` where the star actually appears.
    pub fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r2 * r4;
        let scale = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        (x * scale, y * scale)
    }

    /// Inverse distortion: distorted → ideal (undistort).
    ///
    /// Given observed (distorted) pixel coordinates, returns the ideal
    /// (pinhole) coordinates. Uses Newton-Raphson iteration.
    pub fn undistort(&self, x_d: f64, y_d: f64) -> (f64, f64) {
        let r_d = (x_d * x_d + y_d * y_d).sqrt();
        if r_d < 1e-12 {
            return (x_d, y_d);
        }

        // Newton-Raphson to find r such that r × (1 + k1·r² + k2·r⁴ + k3·r⁶) = r_d
        let mut r = r_d; // initial guess: no distortion
        for _ in 0..20 {
            let r2 = r * r;
            let r4 = r2 * r2;
            let r6 = r2 * r4;

            // f(r) = r × (1 + k1·r² + k2·r⁴ + k3·r⁶) - r_d
            let f = r * (1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6) - r_d;
            // f'(r) = 1 + 3·k1·r² + 5·k2·r⁴ + 7·k3·r⁶
            let df = 1.0 + 3.0 * self.k1 * r2 + 5.0 * self.k2 * r4 + 7.0 * self.k3 * r6;

            let delta = f / df;
            r -= delta;

            if delta.abs() < 1e-12 {
                break;
            }
        }

        let scale = r / r_d;
        (x_d * scale, y_d * scale)
    }

    /// Returns `true` if all coefficients are zero (no distortion).
    pub fn is_zero(&self) -> bool {
        self.k1 == 0.0 && self.k2 == 0.0 && self.k3 == 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_radial() {
        let d = RadialDistortion::new(-7e-9, 2e-15, 0.0);
        // Test at various radii
        for &(x, y) in &[
            (100.0, 200.0),
            (500.0, 300.0),
            (0.0, 1000.0),
            (1024.0, 512.0),
        ] {
            let (xd, yd) = d.distort(x, y);
            let (xu, yu) = d.undistort(xd, yd);
            assert!(
                (xu - x).abs() < 1e-6 && (yu - y).abs() < 1e-6,
                "Roundtrip failed for ({}, {}): got ({}, {})",
                x,
                y,
                xu,
                yu
            );
        }
    }

    #[test]
    fn test_zero_distortion() {
        let d = RadialDistortion::new(0.0, 0.0, 0.0);
        let (xu, yu) = d.undistort(100.0, 200.0);
        assert!((xu - 100.0).abs() < 1e-12);
        assert!((yu - 200.0).abs() < 1e-12);
    }

    #[test]
    fn test_origin() {
        let d = RadialDistortion::new(-1e-6, 1e-12, 0.0);
        let (xu, yu) = d.undistort(0.0, 0.0);
        assert!(xu.abs() < 1e-12);
        assert!(yu.abs() < 1e-12);
    }
}
