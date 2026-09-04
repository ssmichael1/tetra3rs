//! Centroid preprocessing shared by every solve mode: drop non-finite
//! inputs, subtract the optical center, undistort, order by brightness, and
//! build camera-frame unit vectors at a given pixel scale.

use tracing::debug;

use crate::camera_model::CameraModel;
use crate::Centroid;

/// Centroids ready for the solver: CRPIX-subtracted, undistorted, with
/// non-finite entries removed. Positions are still in *pixels* (the pixel
/// scale is applied later, per hypothesis source), and parity has not been
/// applied.
pub(super) struct Preprocessed {
    /// Working centroids, in input order minus the dropped entries.
    pub centroids: Vec<Centroid>,
    /// `orig_indices[i]` is the index in the caller's input slice that
    /// working centroid `i` came from. Dropping non-finite centroids compacts
    /// the list, so this map is needed to report
    /// `Solution.matched_centroid_indices` back in the caller's frame.
    pub orig_indices: Vec<usize>,
}

/// Subtract CRPIX and undistort every finite centroid (pixel-space,
/// FOV-independent). Non-finite inputs (NaN/inf) would quantize to bogus
/// pattern keys and degrade the solve to a silent NoMatch, so they are
/// dropped here (and any that undistort to non-finite) rather than fed
/// downstream.
pub(super) fn preprocess(centroids: &[Centroid], cam: &CameraModel) -> Preprocessed {
    let n_input = centroids.len();
    let mut preprocessed: Vec<Centroid> = Vec::with_capacity(n_input);
    let mut orig_indices: Vec<usize> = Vec::with_capacity(n_input);
    for (idx, c) in centroids.iter().enumerate() {
        if !(c.x.is_finite() && c.y.is_finite()) {
            continue;
        }
        // Subtract optical center offset
        let cx = c.x as f64 - cam.crpix[0];
        let cy = c.y as f64 - cam.crpix[1];
        // Undistort (distorted observed → ideal pinhole)
        let (ux, uy) = cam.distortion.undistort(cx, cy);
        let (ux, uy) = (ux as f32, uy as f32);
        if !(ux.is_finite() && uy.is_finite()) {
            continue;
        }
        preprocessed.push(Centroid {
            x: ux,
            y: uy,
            // A non-finite mass would feed NaN into the brightness sort's
            // comparator (not a total order — std may panic); treat as unknown.
            mass: c.mass.filter(|m| m.is_finite()),
            cov: c.cov,
        });
        orig_indices.push(idx);
    }
    if preprocessed.len() < n_input {
        debug!(
            "Dropped {} non-finite centroid(s) before solve",
            n_input - preprocessed.len()
        );
    }
    Preprocessed {
        centroids: preprocessed,
        orig_indices,
    }
}

/// Per-centroid position uncertainty in pixels, in brightness order:
/// `sqrt(mean of the covariance diagonal)` when [`Centroid::cov`] is
/// present and finite, else `0.0` (unknown — the verification then uses
/// its stage σ alone). Isotropic on purpose: the verification's position
/// likelihood is radial.
pub(super) fn centroid_sigma_px(centroids: &[Centroid], sorted_indices: &[usize]) -> Vec<f32> {
    sorted_indices
        .iter()
        .map(|&i| match centroids[i].cov {
            Some(c) => {
                let v = 0.5 * (c[(0, 0)] + c[(1, 1)]);
                if v.is_finite() && v > 0.0 {
                    v.sqrt()
                } else {
                    0.0
                }
            }
            None => 0.0,
        })
        .collect()
}

/// Brightness-ordered camera-frame unit vectors together with the geometry
/// they were built at.
///
/// Every verification consumer takes one of these instead of a bare slice so
/// the pixel scale the vectors encode travels with them: a stage that needs
/// vectors at a *different* scale (e.g. the post-refinement re-verify, which
/// runs at the refined scale) can see the mismatch and rebuild instead of
/// silently testing at the wrong scale.
#[derive(Clone, Copy)]
pub(super) struct CentroidVectors<'a> {
    /// Pixel scale (rad/px) the vectors were built at.
    pub pixel_scale: f32,
    /// Whether the x-axis was negated (parity applied).
    pub parity_flip: bool,
    /// The unit vectors, in brightness order (`sorted_indices` order).
    pub data: &'a [[f32; 3]],
}

/// Brightness-sorted centroid index order: highest mass (brightest) first,
/// centroids without mass last. Shared by the LIS and tracking front-ends.
pub(super) fn sort_indices_by_brightness(centroids: &[Centroid]) -> Vec<usize> {
    let mut sorted_indices: Vec<usize> = (0..centroids.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        let ma = centroids[a].mass.unwrap_or(f32::MIN);
        let mb = centroids[b].mass.unwrap_or(f32::MIN);
        mb.partial_cmp(&ma).unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted_indices
}

/// Camera-frame unit vectors for brightness-ordered centroids:
/// `normalize(parity·x·ps, y·ps, 1)`. The LIS path passes `parity_sign = 1.0`
/// (it detects parity later from the rotation determinant); tracking applies
/// the camera model's parity up front.
pub(super) fn centroid_unit_vectors(
    centroids: &[Centroid],
    sorted_indices: &[usize],
    pixel_scale: f32,
    parity_sign: f32,
) -> Vec<[f32; 3]> {
    sorted_indices
        .iter()
        .map(|&i| unit_vector_from_pixels(&centroids[i], pixel_scale, parity_sign))
        .collect()
}

/// Unit vector in the camera frame for a single centroid at the given pixel
/// scale (rad/px), with optional x-negation for parity-flipped images.
#[inline]
pub(super) fn unit_vector_from_pixels(
    centroid: &Centroid,
    pixel_scale: f32,
    parity_sign: f32,
) -> [f32; 3] {
    let x = parity_sign * centroid.x * pixel_scale;
    let y = centroid.y * pixel_scale;
    let z = 1.0f32;
    let norm = (x * x + y * y + z * z).sqrt();
    [x / norm, y / norm, z / norm]
}
