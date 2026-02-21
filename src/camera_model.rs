//! Camera intrinsics model: focal length, optical center, parity, and distortion.
//!
//! `CameraModel` cleanly separates per-camera intrinsics from per-image extrinsics
//! (pointing + roll). It provides the mapping between pixel coordinates and
//! tangent-plane coordinates used by the solver.
//!
//! # Coordinate conventions
//!
//! - **Pixel coordinates**: origin at image center, +X right, +Y down (same as centroids).
//! - **Tangent-plane coordinates** `(ξ, η)`: small-angle radians on the gnomonic projection
//!   plane, with ξ = East and η = North when parity is not flipped.
//!
//! # Pipeline
//!
//! ```text
//! pixel → subtract crpix → undistort → apply parity → divide by f → tangent plane
//! tangent plane → multiply by f → apply parity → distort → add crpix → pixel
//! ```

use crate::distortion::Distortion;

/// Camera intrinsics model.
///
/// Encapsulates focal length, optical center offset, parity, and lens distortion
/// into a single struct that maps between pixel and tangent-plane coordinates.
#[derive(Debug, Clone)]
pub struct CameraModel {
    /// Focal length in pixels: `f = (width/2) / tan(fov/2)`.
    pub focal_length_px: f64,
    /// Optical center offset from the geometric image center, in pixels `[x, y]`.
    /// For most cameras this is `[0.0, 0.0]`.
    pub crpix: [f64; 2],
    /// Whether the image x-axis is flipped (e.g. FITS images with CDELT1 < 0).
    pub parity_flip: bool,
    /// Lens distortion model (applied in pixel space, after crpix subtraction).
    pub distortion: Distortion,
}

impl CameraModel {
    /// Create a camera model from a horizontal field of view and image width.
    ///
    /// Sets crpix to `[0, 0]`, no parity flip, no distortion.
    pub fn from_fov(fov_rad: f64, image_width: u32) -> Self {
        let f = (image_width as f64 / 2.0) / (fov_rad / 2.0).tan();
        Self {
            focal_length_px: f,
            crpix: [0.0, 0.0],
            parity_flip: false,
            distortion: Distortion::None,
        }
    }

    /// Pixel scale in radians per pixel (approximate, at image center).
    ///
    /// This is `1 / focal_length_px` and equals the tangent-plane scale.
    pub fn pixel_scale(&self) -> f64 {
        1.0 / self.focal_length_px
    }

    /// Derive the horizontal field of view in radians for a given image width.
    pub fn fov_rad(&self, image_width: u32) -> f64 {
        2.0 * ((image_width as f64 / 2.0) / self.focal_length_px).atan()
    }

    /// Convert pixel coordinates to tangent-plane coordinates.
    ///
    /// Pipeline: subtract crpix → undistort → apply parity → divide by focal length.
    pub fn pixel_to_tanplane(&self, px: f64, py: f64) -> (f64, f64) {
        // 1. Subtract optical center offset
        let x = px - self.crpix[0];
        let y = py - self.crpix[1];

        // 2. Undistort (distorted pixel → ideal pixel)
        let (ux, uy) = self.distortion.undistort(x, y);

        // 3. Apply parity flip
        let ux = if self.parity_flip { -ux } else { ux };

        // 4. Divide by focal length → tangent-plane radians
        (ux / self.focal_length_px, uy / self.focal_length_px)
    }

    /// Convert tangent-plane coordinates to pixel coordinates.
    ///
    /// Pipeline: multiply by focal length → apply parity → distort → add crpix.
    pub fn tanplane_to_pixel(&self, xi: f64, eta: f64) -> (f64, f64) {
        // 1. Multiply by focal length
        let x = xi * self.focal_length_px;
        let y = eta * self.focal_length_px;

        // 2. Apply parity flip (inverse = same operation)
        let x = if self.parity_flip { -x } else { x };

        // 3. Distort (ideal pixel → distorted pixel)
        let (dx, dy) = self.distortion.distort(x, y);

        // 4. Add optical center offset
        (dx + self.crpix[0], dy + self.crpix[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_fov_and_recovery() {
        let fov_deg = 10.0_f64;
        let fov_rad = fov_deg.to_radians();
        let width = 2048u32;

        let cam = CameraModel::from_fov(fov_rad, width);

        // Recover FOV
        let recovered_fov = cam.fov_rad(width);
        assert!(
            (recovered_fov - fov_rad).abs() < 1e-12,
            "FOV recovery: expected {:.6}, got {:.6}",
            fov_rad,
            recovered_fov,
        );

        // Pixel scale should be approximately fov / width for small angles
        let ps = cam.pixel_scale();
        let expected_ps = fov_rad / width as f64;
        assert!(
            (ps - expected_ps).abs() / expected_ps < 0.01,
            "pixel scale: expected {:.6e}, got {:.6e}",
            expected_ps,
            ps,
        );
    }

    #[test]
    fn test_roundtrip_no_distortion() {
        let cam = CameraModel::from_fov(10.0_f64.to_radians(), 1024);

        let test_points = [
            (0.0, 0.0),
            (100.0, 200.0),
            (-300.0, 150.0),
            (512.0, -400.0),
        ];

        for &(px, py) in &test_points {
            let (xi, eta) = cam.pixel_to_tanplane(px, py);
            let (px2, py2) = cam.tanplane_to_pixel(xi, eta);
            assert!(
                (px - px2).abs() < 1e-10 && (py - py2).abs() < 1e-10,
                "Roundtrip failed for ({}, {}): got ({}, {})",
                px,
                py,
                px2,
                py2,
            );
        }
    }

    #[test]
    fn test_roundtrip_with_crpix() {
        let cam = CameraModel {
            focal_length_px: 5000.0,
            crpix: [10.0, -5.0],
            parity_flip: false,
            distortion: Distortion::None,
        };

        let test_points = [(0.0, 0.0), (100.0, 200.0), (-50.0, 75.0)];

        for &(px, py) in &test_points {
            let (xi, eta) = cam.pixel_to_tanplane(px, py);
            let (px2, py2) = cam.tanplane_to_pixel(xi, eta);
            assert!(
                (px - px2).abs() < 1e-10 && (py - py2).abs() < 1e-10,
                "Roundtrip with crpix failed for ({}, {}): got ({}, {})",
                px,
                py,
                px2,
                py2,
            );
        }

        // Center pixel should map to crpix offset in tanplane
        let (xi0, eta0) = cam.pixel_to_tanplane(10.0, -5.0);
        assert!(
            xi0.abs() < 1e-12 && eta0.abs() < 1e-12,
            "Optical center should map to tanplane origin: ({}, {})",
            xi0,
            eta0,
        );
    }

    #[test]
    fn test_parity_flip() {
        let cam_normal = CameraModel {
            focal_length_px: 5000.0,
            crpix: [0.0, 0.0],
            parity_flip: false,
            distortion: Distortion::None,
        };
        let cam_flipped = CameraModel {
            focal_length_px: 5000.0,
            crpix: [0.0, 0.0],
            parity_flip: true,
            distortion: Distortion::None,
        };

        let (xi_n, eta_n) = cam_normal.pixel_to_tanplane(100.0, 200.0);
        let (xi_f, eta_f) = cam_flipped.pixel_to_tanplane(100.0, 200.0);

        // Parity flip negates xi, preserves eta
        assert!(
            (xi_n + xi_f).abs() < 1e-12,
            "Parity should negate xi: normal={}, flipped={}",
            xi_n,
            xi_f,
        );
        assert!(
            (eta_n - eta_f).abs() < 1e-12,
            "Parity should preserve eta: normal={}, flipped={}",
            eta_n,
            eta_f,
        );

        // Roundtrip with parity
        let (px, py) = cam_flipped.tanplane_to_pixel(xi_f, eta_f);
        assert!(
            (px - 100.0).abs() < 1e-10 && (py - 200.0).abs() < 1e-10,
            "Parity roundtrip failed: got ({}, {})",
            px,
            py,
        );
    }

    #[test]
    fn test_center_pixel_maps_to_origin() {
        let cam = CameraModel::from_fov(15.0_f64.to_radians(), 2048);
        let (xi, eta) = cam.pixel_to_tanplane(0.0, 0.0);
        assert!(xi.abs() < 1e-15 && eta.abs() < 1e-15);
    }

    #[test]
    fn test_pixel_to_tanplane_known_values() {
        // For a 10° FOV, 1000px wide camera:
        // A pixel at (500, 0) = right edge should map to xi ≈ tan(5°) ≈ 0.0875 rad
        let fov_rad = 10.0_f64.to_radians();
        let cam = CameraModel::from_fov(fov_rad, 1000);

        let (xi, _eta) = cam.pixel_to_tanplane(500.0, 0.0);
        let expected_xi = (5.0_f64).to_radians().tan();
        assert!(
            (xi - expected_xi).abs() < 1e-10,
            "Edge pixel xi: expected {:.6}, got {:.6}",
            expected_xi,
            xi,
        );
    }
}
