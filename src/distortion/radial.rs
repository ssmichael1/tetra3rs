//! Brown-Conrady radial+tangential distortion model.
//!
//! ```text
//! x_d = x · (1 + k1·r² + k2·r⁴ + k3·r⁶)  +  2·p1·x·y + p2·(r² + 2·x²)
//! y_d = y · (1 + k1·r² + k2·r⁴ + k3·r⁶)  +  p1·(r² + 2·y²) + 2·p2·x·y
//! ```
//!
//! All coordinates are in pixels relative to the distortion model's own
//! `center` (the optical axis, `[0, 0]` by default) — the model shifts in
//! and out of that frame internally, so callers pass coordinates in the
//! image-center-origin frame. Setting `p1 = p2 = 0` reduces to pure radial
//! Brown-Conrady, which is the historical default and what
//! [`RadialDistortion::new`] constructs.
//!
//! # References
//!
//! - **Conrady, A. E.** (1919). "Decentred Lens-Systems."
//!   *Monthly Notices of the Royal Astronomical Society*, 79(5): 384-390.
//!   — The original derivation of the tangential / decentering distortion
//!   form. <https://doi.org/10.1093/mnras/79.5.384>
//! - **Brown, D. C.** (1966). "Decentering Distortion of Lenses."
//!   *Photogrammetric Engineering*, 32(3): 444-462. — Modernized the
//!   Conrady formulation; gave the radial-plus-tangential form used today.
//! - **Brown, D. C.** (1971). "Close-Range Camera Calibration."
//!   *Photogrammetric Engineering*, 37(8): 855-866. — Camera calibration
//!   procedure; basis for the OpenCV / photogrammetry conventions.
//! - **Zhang, Z.** (2000). "A Flexible New Technique for Camera Calibration."
//!   *IEEE TPAMI*, 22(11): 1330-1334. — Multi-image planar-target
//!   calibration that became the standard method (and the model
//!   implemented by OpenCV's `calibrateCamera`).
//!   <https://doi.org/10.1109/34.888718>
//! - **OpenCV documentation** for the equivalent
//!   `(k1, k2, k3, p1, p2)` formulation:
//!   <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>

/// Brown-Conrady radial+tangential distortion.
///
/// The forward model is the standard OpenCV / photogrammetry distortion:
/// up to 3 radial coefficients (`k1, k2, k3`) plus 2 tangential / decentering
/// coefficients (`p1, p2`). With `p1 = p2 = 0` this is pure radial.
///
/// Undistortion (inverse) is computed via 2D Newton iteration on the forward
/// model — see [`Self::undistort`].
///
/// # References
///
/// - Conrady, A. E. (1919). *MNRAS* 79: 384.
/// - Brown, D. C. (1966). *Photogrammetric Engineering* 32: 444.
/// - Zhang, Z. (2000). *IEEE TPAMI* 22(11): 1330.
/// - See the [`distortion` module docs](crate::distortion) for full citations.
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
    /// Distortion center (optical axis) in pixels, in the same
    /// image-center-origin frame as the model's input coordinates.
    /// `[0, 0]` (the default) centers the distortion on the geometric image
    /// center. Distinct from the camera model's `crpix` (the projection
    /// origin): on mosaic cameras the optical axis — where radial
    /// distortion is physically centered — can be far off the detector,
    /// e.g. near a CCD corner on TESS, while the projection origin must
    /// stay near the image center for the solver's geometry.
    #[serde(default)]
    pub center: [f64; 2],
}

impl RadialDistortion {
    /// Create a pure-radial distortion model (`p1 = p2 = 0`), centered on
    /// the geometric image center.
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
            center: [0.0, 0.0],
        }
    }

    /// Create a full Brown-Conrady model with both radial and tangential
    /// coefficients, centered on the geometric image center.
    pub fn with_tangential(k1: f64, k2: f64, k3: f64, p1: f64, p2: f64) -> Self {
        Self {
            k1,
            k2,
            k3,
            p1,
            p2,
            center: [0.0, 0.0],
        }
    }

    /// Create a full Brown-Conrady model centered on the given optical-axis
    /// position (pixels, image-center-origin frame).
    pub fn with_center(cx: f64, cy: f64, k1: f64, k2: f64, k3: f64, p1: f64, p2: f64) -> Self {
        Self {
            k1,
            k2,
            k3,
            p1,
            p2,
            center: [cx, cy],
        }
    }

    /// Check that every coefficient and the center are finite. Call after
    /// loading a model from an untrusted source (saved file, pickle bytes) —
    /// a NaN coefficient silently poisons every undistorted coordinate.
    pub fn validate(&self) -> crate::Result<()> {
        let finite = [self.k1, self.k2, self.k3, self.p1, self.p2]
            .iter()
            .all(|c| c.is_finite())
            && self.center.iter().all(|c| c.is_finite());
        if !finite {
            return Err(crate::Error::InvalidInput(
                "RadialDistortion: coefficients and center must be finite".into(),
            ));
        }
        Ok(())
    }

    /// Forward distortion: ideal → distorted.
    ///
    /// Given ideal (pinhole) pixel coordinates `(x, y)`, returns the
    /// distorted coordinates `(x_d, y_d)` where the star actually appears.
    pub fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let (x, y) = (x - self.center[0], y - self.center[1]);
        let (xd, yd) = self.distort_centered(x, y);
        (xd + self.center[0], yd + self.center[1])
    }

    /// Forward distortion in optical-axis-centered coordinates.
    fn distort_centered(&self, x: f64, y: f64) -> (f64, f64) {
        let e = brown_conrady_forward(self.k1, self.k2, self.k3, self.p1, self.p2, x, y);
        (e.fx, e.fy)
    }

    /// Inverse distortion: distorted → ideal (undistort).
    ///
    /// Given observed (distorted) pixel coordinates, returns the ideal
    /// (pinhole) coordinates. Uses 2D Newton iteration on the forward model.
    ///
    /// Returns `(NaN, NaN)` when the iteration does not converge — the point
    /// lies outside the region where the model is invertible (e.g. beyond the
    /// radius at which strong barrel distortion folds back on itself), so no
    /// ideal position maps to it. The solver drops non-finite centroids.
    pub fn undistort(&self, x_d: f64, y_d: f64) -> (f64, f64) {
        let (x_d, y_d) = (x_d - self.center[0], y_d - self.center[1]);
        // Initial guess: assume no distortion.
        let mut x = x_d;
        let mut y = y_d;
        // One more evaluation than Newton steps, so the last step's result
        // is checked too.
        for _ in 0..=20 {
            // Forward distort the current ideal estimate (with Jacobian).
            let e = brown_conrady_forward(self.k1, self.k2, self.k3, self.p1, self.p2, x, y);

            // Residual (forward(x, y) − x_d).
            let rx = e.fx - x_d;
            let ry = e.fy - y_d;
            if rx * rx + ry * ry < UNDISTORT_TOL_PX * UNDISTORT_TOL_PX {
                return (x + self.center[0], y + self.center[1]);
            }

            let (j11, j12, j21, j22) = (e.j11, e.j12, e.j12, e.j22);
            let det = j11 * j22 - j12 * j21;
            if !det.is_finite() || det.abs() < 1e-15 {
                break;
            }
            let inv_det = 1.0 / det;
            // Newton step: (x, y) -= J⁻¹ · r
            x -= inv_det * (j22 * rx - j12 * ry);
            y -= inv_det * (-j21 * rx + j11 * ry);
        }
        (f64::NAN, f64::NAN)
    }
}

/// Residual (pixels) at which the inverse-distortion Newton iteration stops.
/// Far below any centroiding error, and far above the f64 rounding of the
/// forward model at sensor-scale coordinates, so a convergent iteration
/// always reaches it.
const UNDISTORT_TOL_PX: f64 = 1e-10;

/// One evaluation of the Brown-Conrady forward model at optical-axis-centered
/// coordinates: the distorted position, the 2×2 Jacobian of the forward map
/// (symmetric mixed term, so `j21 == j12`), and the radius powers.
///
/// This is the **single source** of the model formulas — used by
/// [`RadialDistortion::distort`]/[`RadialDistortion::undistort`] and by the
/// intrinsics LM fit in `distortion::fit`, so a sign or coefficient fix lands
/// everywhere at once.
pub(crate) struct BrownConradyEval {
    pub fx: f64,
    pub fy: f64,
    pub j11: f64,
    pub j12: f64,
    pub j22: f64,
    pub r2: f64,
    pub r4: f64,
    pub r6: f64,
}

/// Evaluate the Brown-Conrady forward model (see [`BrownConradyEval`]).
///
/// ```text
///     r² = x² + y²,  radial = 1 + k1·r² + k2·r⁴ + k3·r⁶
///     fx = x·radial + 2·p1·x·y + p2·(r² + 2x²)
///     fy = y·radial + p1·(r² + 2y²) + 2·p2·x·y
/// ```
///
/// Jacobian derivation (radial_prime = d(radial)/d(r²)):
/// - d/dx [x·radial] = radial + 2·x²·radial_prime; d/dy = 2·x·y·radial_prime
/// - d/dx [2·p1·x·y + p2·(r²+2x²)] = 2·p1·y + 6·p2·x; d/dy = 2·p1·x + 2·p2·y
/// - d/dy [p1·(r²+2y²) + 2·p2·x·y] = 6·p1·y + 2·p2·x
#[inline]
pub(crate) fn brown_conrady_forward(
    k1: f64,
    k2: f64,
    k3: f64,
    p1: f64,
    p2: f64,
    x: f64,
    y: f64,
) -> BrownConradyEval {
    let r2 = x * x + y * y;
    let r4 = r2 * r2;
    let r6 = r2 * r4;
    let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    let radial_prime = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;
    let dx_t = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    let dy_t = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    BrownConradyEval {
        fx: x * radial + dx_t,
        fy: y * radial + dy_t,
        j11: radial + 2.0 * x * x * radial_prime + 2.0 * p1 * y + 6.0 * p2 * x,
        j12: 2.0 * x * y * radial_prime + 2.0 * p1 * x + 2.0 * p2 * y,
        j22: radial + 2.0 * y * y * radial_prime + 6.0 * p1 * y + 2.0 * p2 * x,
        r2,
        r4,
        r6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_off_center() {
        // Mosaic-style: distortion centered near a detector corner.
        let d = RadialDistortion::with_center(-1100.0, -1080.0, -5e-9, 1e-15, 0.0, 1e-7, -2e-7);
        for &(x, y) in &[(0.0, 0.0), (-900.0, -1000.0), (1024.0, 1024.0)] {
            let (xd, yd) = d.distort(x, y);
            let (xu, yu) = d.undistort(xd, yd);
            assert!(
                (xu - x).abs() < 1e-6 && (yu - y).abs() < 1e-6,
                "roundtrip failed at ({x}, {y}): got ({xu}, {yu})",
            );
        }
        // Distortion must vanish at the model's own center.
        let (xd, yd) = d.distort(-1100.0, -1080.0);
        assert!((xd + 1100.0).abs() < 1e-9 && (yd + 1080.0).abs() < 1e-9);
    }

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
    fn test_undistort_outside_invertible_region_is_nan() {
        // Strong barrel distortion folds over at ideal r ≈ 1291 px, where the
        // distorted radius peaks at ≈ 861 px: no ideal point maps beyond it.
        // Newton used to return whatever iterate it ended on (the wrong
        // quadrant for this corner).
        let d = RadialDistortion::new(-2e-7, 0.0, 0.0);
        for &(xd, yd) in &[(1024.0, 1024.0), (900.0, 0.0)] {
            let (xu, yu) = d.undistort(xd, yd);
            assert!(xu.is_nan() && yu.is_nan(), "({xd}, {yd}) → ({xu}, {yu})");
        }
        // Inside the fold the inverse is still exact.
        for &(x, y) in &[(100.0, 200.0), (600.0, -300.0), (850.0, 850.0)] {
            let (xd, yd) = d.distort(x, y);
            let (xu, yu) = d.undistort(xd, yd);
            assert!((xu - x).abs() < 1e-6 && (yu - y).abs() < 1e-6);
        }
        // Non-finite input stays non-finite rather than panicking.
        let (xu, yu) = d.undistort(f64::NAN, 0.0);
        assert!(xu.is_nan() && yu.is_nan());
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
