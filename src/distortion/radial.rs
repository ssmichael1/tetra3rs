//! Brown-Conrady radial+tangential distortion model.
//!
//! ```text
//! x_d = x · (1 + k1·r² + k2·r⁴ + k3·r⁶)  +  2·p1·x·y + p2·(r² + 2·x²)
//! y_d = y · (1 + k1·r² + k2·r⁴ + k3·r⁶)  +  p1·(r² + 2·y²) + 2·p2·x·y
//! ```
//!
//! All coordinates are in pixels relative to the optical center (the
//! camera model's `crpix` is subtracted before this model is applied).
//! Setting `p1 = p2 = 0` reduces to pure radial Brown-Conrady, which is the
//! historical default and what [`RadialDistortion::new`] constructs.

/// Brown-Conrady radial+tangential distortion.
///
/// The forward model is the standard OpenCV / photogrammetry distortion:
/// up to 3 radial coefficients (`k1, k2, k3`) plus 2 tangential / decentering
/// coefficients (`p1, p2`). With `p1 = p2 = 0` this is pure radial.
///
/// Undistortion (inverse) is computed via 2D Newton iteration on the forward
/// model — see [`Self::undistort`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RadialDistortion {
    /// First radial coefficient (barrel < 0, pincushion > 0).
    pub k1: f64,
    /// Second radial coefficient.
    pub k2: f64,
    /// Third radial coefficient.
    pub k3: f64,
    /// First tangential / decentering coefficient.
    #[serde(default)]
    pub p1: f64,
    /// Second tangential / decentering coefficient.
    #[serde(default)]
    pub p2: f64,
}

impl RadialDistortion {
    /// Create a pure-radial distortion model (`p1 = p2 = 0`).
    ///
    /// Set unused radial coefficients to 0.0. For example,
    /// `RadialDistortion::new(-1e-8, 0.0, 0.0)` for simple barrel distortion.
    pub fn new(k1: f64, k2: f64, k3: f64) -> Self {
        Self {
            k1,
            k2,
            k3,
            p1: 0.0,
            p2: 0.0,
        }
    }

    /// Create a full Brown-Conrady model with both radial and tangential
    /// coefficients.
    pub fn with_tangential(k1: f64, k2: f64, k3: f64, p1: f64, p2: f64) -> Self {
        Self {
            k1,
            k2,
            k3,
            p1,
            p2,
        }
    }

    /// Forward distortion: ideal → distorted.
    ///
    /// Given ideal (pinhole) pixel coordinates `(x, y)`, returns the
    /// distorted coordinates `(x_d, y_d)` where the star actually appears.
    pub fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r2 * r4;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        let dx_t = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let dy_t = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        (x * radial + dx_t, y * radial + dy_t)
    }

    /// Inverse distortion: distorted → ideal (undistort).
    ///
    /// Given observed (distorted) pixel coordinates, returns the ideal
    /// (pinhole) coordinates. Uses 2D Newton iteration on the forward model.
    pub fn undistort(&self, x_d: f64, y_d: f64) -> (f64, f64) {
        // Initial guess: assume no distortion.
        let mut x = x_d;
        let mut y = y_d;
        for _ in 0..20 {
            // Forward distort the current ideal estimate.
            let r2 = x * x + y * y;
            let r4 = r2 * r2;
            let r6 = r2 * r4;
            let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
            let radial_prime = self.k1 + 2.0 * self.k2 * r2 + 3.0 * self.k3 * r4;
            let dx_t = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
            let dy_t = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
            let fx = x * radial + dx_t;
            let fy = y * radial + dy_t;

            // Residual (forward(x, y) − x_d).
            let rx = fx - x_d;
            let ry = fy - y_d;
            if rx * rx + ry * ry < 1e-20 {
                break;
            }

            // Jacobian of forward distort wrt (x, y).
            // d/dx [x·radial] = radial + x · radial_prime · 2x = radial + 2·x²·radial_prime
            // d/dy [x·radial] = x · radial_prime · 2y = 2·x·y·radial_prime
            // d/dx [2·p1·x·y + p2·(r²+2x²)] = 2·p1·y + p2·(2x + 4x) = 2·p1·y + 6·p2·x
            // d/dy [2·p1·x·y + p2·(r²+2x²)] = 2·p1·x + p2·2y = 2·p1·x + 2·p2·y
            // d/dx [p1·(r²+2y²) + 2·p2·x·y] = p1·2x + 2·p2·y = 2·p1·x + 2·p2·y
            // d/dy [p1·(r²+2y²) + 2·p2·x·y] = p1·(2y + 4y) + 2·p2·x = 6·p1·y + 2·p2·x
            let j11 = radial + 2.0 * x * x * radial_prime + 2.0 * self.p1 * y + 6.0 * self.p2 * x;
            let j12 = 2.0 * x * y * radial_prime + 2.0 * self.p1 * x + 2.0 * self.p2 * y;
            let j21 = 2.0 * x * y * radial_prime + 2.0 * self.p1 * x + 2.0 * self.p2 * y;
            let j22 = radial + 2.0 * y * y * radial_prime + 6.0 * self.p1 * y + 2.0 * self.p2 * x;

            let det = j11 * j22 - j12 * j21;
            if det.abs() < 1e-15 {
                break;
            }
            let inv_det = 1.0 / det;
            // Newton step: (x, y) -= J⁻¹ · r
            let dx_step = inv_det * (j22 * rx - j12 * ry);
            let dy_step = inv_det * (-j21 * rx + j11 * ry);
            x -= dx_step;
            y -= dy_step;

            if dx_step.abs() + dy_step.abs() < 1e-12 {
                break;
            }
        }
        (x, y)
    }

    /// Returns `true` if all coefficients are zero (no distortion).
    pub fn is_zero(&self) -> bool {
        self.k1 == 0.0
            && self.k2 == 0.0
            && self.k3 == 0.0
            && self.p1 == 0.0
            && self.p2 == 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_radial_only() {
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
    fn test_roundtrip_full_brown_conrady() {
        // Realistic-magnitude Brown-Conrady with tangential terms.
        let d = RadialDistortion::with_tangential(-7e-9, 2e-15, 0.0, 5e-7, -3e-7);
        for &(x, y) in &[
            (100.0, 200.0),
            (500.0, 300.0),
            (0.0, 1000.0),
            (1024.0, 512.0),
            (-800.0, -700.0),
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
