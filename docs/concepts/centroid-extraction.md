# Centroid Extraction

tetra3rs includes a complete centroid extraction pipeline that detects stars in an image and computes sub-pixel positions with uncertainty estimates. The pipeline is designed to be robust across a range of conditions — from low-SNR ground-based images to crowded spacecraft fields.

## Pipeline Overview

```
Raw image
  │
  ├─ 1. Local background subtraction (block median + bilinear interpolation)
  │
  ├─ 2. Global noise estimation (lower-half sigma clipping)
  │
  ├─ 3. Thresholding (σ above background)
  │
  ├─ 4. Connected component labeling (union-find)
  │
  ├─ 5. Blob filtering (size, elongation)
  │
  ├─ 6. Centroid computation (intensity-weighted moments + quadratic refinement)
  │
  └─ Output: sorted list of Centroid objects
```

## 1. Local Background Subtraction

Large-scale intensity variations (nebulosity, Milky Way emission, vignetting, illumination gradients) must be removed before detection. The algorithm:

1. Divides the image into `block_size × block_size` tiles (configurable via `local_bg_block_size`, default 64)
2. Computes the **median** pixel value within each tile (ignoring zeros and non-finite values)
3. Reconstructs a smooth background surface by **bilinear interpolation** between tile centers
4. Subtracts the interpolated surface, clamping negative values to zero

The median is chosen over the mean because it is robust to contamination from bright stars within each tile.

**Tuning**: Smaller blocks follow finer spatial structure but risk subtracting actual stars. Typical values are 16–128 pixels, or roughly 1–3% of image width. Set `local_bg_block_size=None` to skip local background subtraction entirely.

## 2. Global Noise Estimation

After local background subtraction, the residual noise level is estimated robustly using only the uncontaminated portion of the pixel distribution:

1. Compute the **global median** of all pixels — this is the background level
2. Select only pixels **at or below the median** (the "lower half")
3. Apply **iterative sigma clipping** (5 iterations, 3σ factor) to reject remaining outliers
4. The standard deviation of the clipped lower-half pixels is the noise estimate σ

The detection threshold is then:

$$
\text{threshold} = \mu_{\text{bg}} + \sigma_{\text{threshold}} \times \sigma
$$

**Why the lower half?** Stars and nebulosity only contaminate pixels *above* the background. By using only the lower half of the distribution and assuming symmetry, we get a clean estimate of the Gaussian noise floor without any astrophysical contamination.

## 3. Connected Component Labeling

Pixels above the threshold are grouped into blobs using a **two-pass union-find** algorithm with 8-connectivity (diagonal neighbors included):

**Pass 1** — Scan left-to-right, top-to-bottom. For each above-threshold pixel, check already-labeled neighbors (left, above, and diagonals). If neighbors exist, assign the minimum label and union all neighbors. Otherwise, assign a new label.

**Pass 2** — Resolve all labels to their root through path compression, then remap to sequential IDs.

This produces a label map where each connected group of bright pixels shares a unique integer label.

## 4. Blob Filtering

Blobs are filtered before centroid computation:

- **Minimum pixels** (`min_pixels`, default 3) — Rejects hot pixels and single-pixel noise spikes
- **Maximum pixels** (`max_pixels`, default 10,000) — Rejects very extended objects, large nebulae, or noise blobs from detector artifacts
- **Elongation** (`max_elongation`, default 3.0) — Computed from the covariance eigenvalue ratio $\sqrt{\lambda_{\max} / \lambda_{\min}}$. Rejects cosmic ray hits, satellite trails, and diffraction spikes. Set to `None` to disable.

## 5. Centroid Computation

Each surviving blob is centroided through several substeps:

### Per-Blob Local Background

For each blob, a refined local background is computed from a pixel annulus:

1. Expand the blob's bounding box by a 5-pixel margin
2. Collect all non-blob pixels within this annulus
3. Take the **median** as the per-blob local background

This corrects for residual gradients not removed by the block-based subtraction in step 1.

### Intensity-Weighted Moments

The centroid position is computed as the intensity-weighted center of mass:

$$
\bar{x} = \frac{\sum_i I_i \, x_i}{\sum_i I_i}, \qquad
\bar{y} = \frac{\sum_i I_i \, y_i}{\sum_i I_i}
$$

where $I_i = \max(\text{pixel}_i - \text{local\_bg}, \; 0)$ is the background-subtracted intensity at pixel $i$.

### Covariance Matrix

The intensity-weighted second moments give a 2×2 covariance matrix:

$$
\mathbf{C} = \begin{pmatrix}
\sigma_{xx} & \sigma_{xy} \\
\sigma_{xy} & \sigma_{yy}
\end{pmatrix}
$$

where $\sigma_{xx} = \frac{\sum I_i (x_i - \bar{x})^2}{\sum I_i}$ and similarly for the other terms.

The eigenvalues and eigenvectors of $\mathbf{C}$ define the principal axes of the PSF — an uncertainty ellipse that characterizes the shape and orientation of each detection. This is stored in `Centroid.cov` and can be used for weighted matching or quality assessment.

### Quadratic Sub-Pixel Refinement

For blobs with ≥5 pixels and a peak at least 1 pixel from the image edge, a quadratic surface is fit to the 3×3 neighborhood around the peak pixel:

$$
f(x, y) = a + bx + cy + dx^2 + ey^2 + fxy
$$

The sub-pixel offset is found by setting $\nabla f = 0$ and solving the resulting 2×2 linear system. The refinement is accepted only if:

- The offset is within ±0.5 pixels of the peak
- The quadratic position agrees with the center-of-mass within 0.5 pixels

Otherwise, the center-of-mass position is kept. This fallback ensures robustness for blended or asymmetric sources where the quadratic model breaks down.

## 6. Output

Centroids are converted to **image-center origin** coordinates:

$$
x_{\text{out}} = x_{\text{px}} - \frac{W}{2}, \qquad
y_{\text{out}} = y_{\text{px}} - \frac{H}{2}
$$

where $W$ and $H$ are the image dimensions. The coordinate convention is +X right, +Y down — the same as the camera frame used by the solver. See [Coordinate Conventions](coordinates.md) for details.

The final list is **sorted by brightness** (integrated intensity above background, descending). This ordering is important for the solver, which tries patterns from the brightest stars first.

Optionally, the list can be truncated to `max_centroids` entries.

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma_threshold` | 5.0 | Detection threshold in σ above background |
| `min_pixels` | 3 | Minimum blob size (pixels) |
| `max_pixels` | 10,000 | Maximum blob size (pixels) |
| `max_centroids` | None | Maximum centroids to return (None = all) |
| `local_bg_block_size` | 64 | Tile size for local background (None = skip) |
| `max_elongation` | 3.0 | Maximum elongation ratio (None = disabled) |

!!! tip "TESS example"
    For TESS Full Frame Images with their wide-field defocused optics, typical settings are `sigma_threshold=300`, `min_pixels=4`, `local_bg_block_size=16`, `max_elongation=6.0`. The high sigma threshold rejects the dense background, and the relaxed elongation allows for the broad, slightly asymmetric TESS PSF.
