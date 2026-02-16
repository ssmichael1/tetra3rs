//! Define a centroid (center of mass) representing
//! a star detection in an unresolved image
//! Centroids are the output of the star extraction process and are used as input to the star matching process.
//!

use crate::Matrix2;
use crate::Vector3;

#[derive(Debug, Clone, PartialEq)]
pub struct Centroid {
    pub x: f32,               // Centroid position in radians along columns (image x-axis)
    pub y: f32,               // Centroid position in radians along rows (image y-axis)
    pub mass: Option<f32>, // Optional "brightness" value that can be used for filtering, but the exact meaning is image-dependent.
    pub cov: Option<Matrix2>, // Optional covariance matrix representing the uncertainty in the centroid position.
}

impl Centroid {
    /// Unit vector pointing to the centroid's position in camera coordinates.
    /// Assumes the camera's optical axis is aligned with the +Z axis, +X points to the right in the image, and +Y points down in the image.
    pub fn uvec(&self) -> Vector3 {
        let x = self.x;
        let y = self.y;
        // For small angles, we can approximate the unit vector as:
        // uvec ≈ (x, y, 1) normalized
        let z = 1.0;
        let norm = (x * x + y * y + z * z).sqrt();
        Vector3::new(x / norm, y / norm, z / norm)
    }
}
