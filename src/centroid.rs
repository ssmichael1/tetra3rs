//! Define a centroid (center of mass) representing
//! a star detection in an unresolved image
//! Centroids are the output of the star extraction process and are used as input to the star matching process.
//!

use crate::Matrix2;
use crate::Vector3;

#[derive(Debug, Clone, PartialEq)]
pub struct Centroid {
    /// Centroid position in pixels along columns (image x-axis).
    /// Origin is at the image center, so values can be positive or negative.
    /// +X points right in the image.
    pub x: f32,
    /// Centroid position in pixels along rows (image y-axis).
    /// Origin is at the image center, so values can be positive or negative.
    /// +Y points down in the image.
    pub y: f32,
    /// Optional "brightness" value used for sorting (brighter = higher).
    /// The exact meaning is image-dependent.
    pub mass: Option<f32>,
    /// Optional covariance matrix representing the uncertainty in the centroid position.
    pub cov: Option<Matrix2>,
}

impl Centroid {
    /// Unit vector pointing to the centroid's position in camera coordinates.
    ///
    /// Converts pixel coordinates to angular coordinates using the given pixel scale
    /// (radians per pixel), then computes a unit vector assuming the camera's optical
    /// axis is aligned with +Z, +X right, +Y down.
    pub fn uvec(&self, pixel_scale: f32) -> Vector3 {
        let x = self.x * pixel_scale;
        let y = self.y * pixel_scale;
        // For small angles: uvec ≈ (x, y, 1) normalized
        let z = 1.0;
        let norm = (x * x + y * y + z * z).sqrt();
        Vector3::new(x / norm, y / norm, z / norm)
    }
}
