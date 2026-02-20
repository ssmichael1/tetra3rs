//! Polynomial (SIP-like) distortion model.
//!
//! Models arbitrary 2D distortion using polynomial correction terms:
//!
//! ```text
//! x_distorted = x + Σ A_pq · x^p · y^q     (2 ≤ p+q ≤ order)
//! y_distorted = y + Σ B_pq · x^p · y^q
//! ```
//!
//! The inverse (undistortion) uses a separately fitted polynomial:
//!
//! ```text
//! x_ideal = x_obs + Σ AP_pq · x_obs^p · y_obs^q
//! y_ideal = y_obs + Σ BP_pq · x_obs^p · y_obs^q
//! ```
//!
//! Unlike the radial model, this captures tangential distortion, decentering,
//! and other effects that aren't radially symmetric — critical for cameras
//! like TESS where each CCD is offset from the optical axis.
//!
//! Coordinates are in pixels relative to the image center. The coefficients
//! are stored normalized: internally the (x, y) inputs are divided by a
//! `scale` factor (typically half the image width) before evaluating the
//! polynomial, so the coefficients stay in a numerically well-conditioned range.

/// SIP-like polynomial distortion model.
///
/// Forward: ideal → distorted. Inverse: distorted → ideal.
/// Both directions are stored as explicit polynomials (no iterative inversion needed).
#[derive(Debug, Clone)]
pub struct PolynomialDistortion {
    /// Polynomial order (2..=6 typically).
    pub order: u32,
    /// Normalization scale: coordinates are divided by this before evaluation.
    /// Typically image_width / 2.
    pub scale: f64,
    /// Forward A coefficients (x correction, ideal → distorted) in normalized coords.
    /// Stored as a flat vector; use `coeff_index(p, q)` to access.
    pub a_coeffs: Vec<f64>,
    /// Forward B coefficients (y correction, ideal → distorted) in normalized coords.
    pub b_coeffs: Vec<f64>,
    /// Inverse AP coefficients (x correction, distorted → ideal) in normalized coords.
    pub ap_coeffs: Vec<f64>,
    /// Inverse BP coefficients (y correction, distorted → ideal) in normalized coords.
    pub bp_coeffs: Vec<f64>,
}

impl PolynomialDistortion {
    /// Create a new polynomial distortion model.
    ///
    /// All coefficient vectors must have exactly `num_coeffs(order)` elements.
    pub fn new(
        order: u32,
        scale: f64,
        a_coeffs: Vec<f64>,
        b_coeffs: Vec<f64>,
        ap_coeffs: Vec<f64>,
        bp_coeffs: Vec<f64>,
    ) -> Self {
        let n = num_coeffs(order);
        assert_eq!(a_coeffs.len(), n, "a_coeffs length mismatch");
        assert_eq!(b_coeffs.len(), n, "b_coeffs length mismatch");
        assert_eq!(ap_coeffs.len(), n, "ap_coeffs length mismatch");
        assert_eq!(bp_coeffs.len(), n, "bp_coeffs length mismatch");
        Self {
            order,
            scale,
            a_coeffs,
            b_coeffs,
            ap_coeffs,
            bp_coeffs,
        }
    }

    /// Create a zero (identity) polynomial distortion model.
    pub fn zero(order: u32, scale: f64) -> Self {
        let n = num_coeffs(order);
        Self {
            order,
            scale,
            a_coeffs: vec![0.0; n],
            b_coeffs: vec![0.0; n],
            ap_coeffs: vec![0.0; n],
            bp_coeffs: vec![0.0; n],
        }
    }

    /// Forward distortion: ideal → distorted (pixel coords, relative to image center).
    pub fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let u = x / self.scale;
        let v = y / self.scale;
        let dx = eval_poly(&self.a_coeffs, self.order, u, v);
        let dy = eval_poly(&self.b_coeffs, self.order, u, v);
        (x + dx * self.scale, y + dy * self.scale)
    }

    /// Inverse distortion: distorted → ideal (pixel coords, relative to image center).
    pub fn undistort(&self, x_d: f64, y_d: f64) -> (f64, f64) {
        let u = x_d / self.scale;
        let v = y_d / self.scale;
        let dx = eval_poly(&self.ap_coeffs, self.order, u, v);
        let dy = eval_poly(&self.bp_coeffs, self.order, u, v);
        (x_d + dx * self.scale, y_d + dy * self.scale)
    }

    /// Returns `true` if all coefficients are zero.
    pub fn is_zero(&self) -> bool {
        self.a_coeffs.iter().all(|&c| c == 0.0)
            && self.b_coeffs.iter().all(|&c| c == 0.0)
            && self.ap_coeffs.iter().all(|&c| c == 0.0)
            && self.bp_coeffs.iter().all(|&c| c == 0.0)
    }
}

// ── Polynomial term helpers ─────────────────────────────────────────────────

/// Number of polynomial coefficients for the given order.
///
/// Terms are (p, q) with 2 ≤ p+q ≤ order:
///   order 2: 3 terms  (2,0),(1,1),(0,2)
///   order 3: 7 terms  + (3,0),(2,1),(1,2),(0,3)
///   order 4: 12 terms + (4,0),(3,1),(2,2),(1,3),(0,4)
///   etc.
pub fn num_coeffs(order: u32) -> usize {
    let mut count = 0;
    for s in 2..=order {
        count += (s + 1) as usize; // s+1 terms for each sum p+q=s
    }
    count
}

/// Map (p, q) with 2 ≤ p+q ≤ order to a flat index.
///
/// Terms are enumerated in order of increasing sum, then decreasing p:
///   sum=2: (2,0)=0, (1,1)=1, (0,2)=2
///   sum=3: (3,0)=3, (2,1)=4, (1,2)=5, (0,3)=6
///   sum=4: (4,0)=7, (3,1)=8, (2,2)=9, (1,3)=10, (0,4)=11
pub fn coeff_index(p: u32, q: u32) -> usize {
    let s = p + q;
    assert!(s >= 2, "p+q must be >= 2");
    // Base offset: number of terms for sums 2..(s-1)
    let mut base = 0usize;
    for ss in 2..s {
        base += (ss + 1) as usize;
    }
    // Within sum=s, terms are ordered by decreasing p: (s,0), (s-1,1), ..., (0,s)
    base + (s - p) as usize
}

/// Enumerate all (p, q) pairs for the given order.
pub fn term_pairs(order: u32) -> Vec<(u32, u32)> {
    let mut pairs = Vec::with_capacity(num_coeffs(order));
    for s in 2..=order {
        for p in (0..=s).rev() {
            let q = s - p;
            pairs.push((p, q));
        }
    }
    pairs
}

/// Evaluate a polynomial correction: Σ c_i · x^p_i · y^q_i
/// `coeffs` is a flat vector indexed by `coeff_index(p, q)`.
fn eval_poly(coeffs: &[f64], order: u32, x: f64, y: f64) -> f64 {
    let mut result = 0.0;
    let mut idx = 0;
    for s in 2..=order {
        for p in (0..=s).rev() {
            let q = s - p;
            result += coeffs[idx] * x.powi(p as i32) * y.powi(q as i32);
            idx += 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_coeffs() {
        assert_eq!(num_coeffs(2), 3);
        assert_eq!(num_coeffs(3), 7);
        assert_eq!(num_coeffs(4), 12);
        assert_eq!(num_coeffs(5), 18);
    }

    #[test]
    fn test_coeff_index() {
        // sum=2
        assert_eq!(coeff_index(2, 0), 0);
        assert_eq!(coeff_index(1, 1), 1);
        assert_eq!(coeff_index(0, 2), 2);
        // sum=3
        assert_eq!(coeff_index(3, 0), 3);
        assert_eq!(coeff_index(2, 1), 4);
        assert_eq!(coeff_index(1, 2), 5);
        assert_eq!(coeff_index(0, 3), 6);
        // sum=4
        assert_eq!(coeff_index(4, 0), 7);
        assert_eq!(coeff_index(3, 1), 8);
        assert_eq!(coeff_index(2, 2), 9);
        assert_eq!(coeff_index(1, 3), 10);
        assert_eq!(coeff_index(0, 4), 11);
    }

    #[test]
    fn test_term_pairs() {
        let pairs = term_pairs(3);
        assert_eq!(
            pairs,
            vec![(2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
        );
    }

    #[test]
    fn test_zero_distortion_roundtrip() {
        let d = PolynomialDistortion::zero(4, 1024.0);
        let (xu, yu) = d.undistort(100.0, -200.0);
        assert!((xu - 100.0).abs() < 1e-12);
        assert!((yu + 200.0).abs() < 1e-12);
        let (xd, yd) = d.distort(100.0, -200.0);
        assert!((xd - 100.0).abs() < 1e-12);
        assert!((yd + 200.0).abs() < 1e-12);
    }

    #[test]
    fn test_distort_undistort_basic() {
        // Create a simple distortion: only x²·y term (index for (2,1) = 4 at order 3)
        let n = num_coeffs(4);
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        a[coeff_index(2, 0)] = 0.01; // x² → dx
        b[coeff_index(0, 2)] = -0.005; // y² → dy

        // Compute inverse from a grid (simple test)
        let d = PolynomialDistortion::new(
            4,
            1024.0,
            a,
            b,
            vec![0.0; n], // ap (inverse) not set here
            vec![0.0; n], // bp
        );

        // Forward: distort(0, 0) = (0, 0)
        let (xd, yd) = d.distort(0.0, 0.0);
        assert!(xd.abs() < 1e-12);
        assert!(yd.abs() < 1e-12);

        // Forward at (512, 512): u=0.5, v=0.5
        // dx = 0.01 * 0.5² = 0.0025 (normalized), * 1024 = 2.56 px
        let (xd, _yd) = d.distort(512.0, 512.0);
        assert!((xd - 512.0 - 2.56).abs() < 1e-10);
    }
}
