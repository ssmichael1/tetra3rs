//! Lens distortion models for correcting optical distortion in star images.
//!
//! Distortion is applied by **undistorting** observed centroid pixel coordinates
//! before they are converted to unit vectors for pattern matching and solving.
//! This means the distortion correction is applied once, up front, and benefits
//! the entire solver pipeline (pattern hashing, verification, and refinement).
//!
//! # Supported models
//!
//! - [`Distortion::Radial`] — classic radial distortion with up to 3 coefficients (k1, k2, k3)
//! - [`Distortion::Polynomial`] — SIP-like polynomial distortion with arbitrary cross-terms
//!
//! # Usage
//!
//! Distortion models can be:
//!
//! 1. **Provided** to the solver via [`SolveConfig::distortion`](crate::SolveConfig) if the
//!    distortion is already known (e.g. from a calibration procedure or FITS header).
//!
//! 2. **Fitted** from solve results using [`fit_radial_distortion`] or
//!    [`fit_polynomial_distortion`]. These functions perform iterative robust
//!    fitting with sigma-clipping to reject mismatched stars.

pub mod calibrate;
pub mod fit;
pub mod polynomial;
pub mod radial;

pub use calibrate::{calibrate_camera, CalibrateConfig, CalibrateResult};
pub use polynomial::PolynomialDistortion;
pub use radial::RadialDistortion;

/// Lens distortion model.
///
/// An enum-based distortion model that supports radial and polynomial
/// distortion correction. All coordinates are in pixels relative to
/// the optical center (typically the image center).
#[derive(Debug, Clone)]
pub enum Distortion {
    /// No distortion correction.
    None,
    /// Radial distortion: r_distorted = r × (1 + k1·r² + k2·r⁴ + k3·r⁶).
    Radial(RadialDistortion),
    /// SIP-like polynomial distortion with independent x,y correction terms.
    Polynomial(PolynomialDistortion),
}

impl Distortion {
    /// Convert observed (distorted) pixel coordinates to ideal (pinhole) coordinates.
    ///
    /// This is the primary operation used by the solver to correct centroids
    /// before computing unit vectors.
    pub fn undistort(&self, x: f64, y: f64) -> (f64, f64) {
        match self {
            Distortion::None => (x, y),
            Distortion::Radial(r) => r.undistort(x, y),
            Distortion::Polynomial(p) => p.undistort(x, y),
        }
    }

    /// Convert ideal (pinhole) pixel coordinates to observed (distorted) coordinates.
    ///
    /// This is the forward model: given where a star *should* appear in a perfect
    /// pinhole camera, compute where it actually appears due to lens distortion.
    pub fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        match self {
            Distortion::None => (x, y),
            Distortion::Radial(r) => r.distort(x, y),
            Distortion::Polynomial(p) => p.distort(x, y),
        }
    }

    /// Returns `true` if this is `Distortion::None`.
    pub fn is_none(&self) -> bool {
        matches!(self, Distortion::None)
    }
}

impl Default for Distortion {
    fn default() -> Self {
        Distortion::None
    }
}
