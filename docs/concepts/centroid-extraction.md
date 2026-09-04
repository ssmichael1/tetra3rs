# Centroid Extraction

tetra3rs includes a complete centroid extraction pipeline that detects stars in an image and computes sub-pixel positions with uncertainty estimates. The pipeline is designed to be robust across a range of conditions — from low-SNR ground-based images to crowded spacecraft fields.

!!! tip "Two extractors"
    This page describes the default **connected-component pipeline**
    (`extract_centroids` / `extract_centroids_from_raw`), which prioritizes
    fidelity. For latency-critical use (e.g. frame-rate tracking on embedded
    hardware) there is also a **fast single-pass extractor**
    (`extract_centroids_fast`), faster at lower fidelity — see
    [Fast single-pass extraction](#fast-single-pass-extraction-extract_centroids_fast)
    below.

## Pipeline Overview

```
Raw image
  │
  ├─ 1. Local background subtraction (block median + bilinear interpolation)
  │
  ├─ 2. Global noise estimation (lower-half sigma clipping)
  │
  ├─ 3. Gaussian matched filter (PSF-tuned σ, on by default)
  │
  ├─ 4. Thresholding (σ above background, filter-compensated)
  │
  ├─ 5. Connected-region detection (run-length union-find)
  │
  ├─ 6. Blob filtering (size, elongation, sharpness, border, deblend)
  │
  ├─ 7. Centroid computation (intensity-weighted moments + quadratic refinement)
  │
  └─ Output: sorted list of Centroid objects
```

## 1. Local Background Subtraction

Large-scale intensity variations (nebulosity, Milky Way emission, vignetting, illumination gradients) must be removed before detection. The algorithm:

1. Divides the image into `block_size × block_size` tiles (configurable via `local_bg_block_size`, default 64)
2. Computes the **median** pixel value within each tile from a phase-staggered stride subsample (non-finite samples excluded; zeros and negatives are kept — they are legitimate background on dark-subtracted frames)
3. Reconstructs a smooth background surface by **bilinear interpolation** between tile centers, linearly extrapolating beyond the outermost centers
4. Subtracts the interpolated surface, clamping negative values to zero in the measurement image (the matched filter, when enabled, convolves the *unclamped* residual — see below)

The median is chosen over the mean because it is robust to contamination from bright stars within each tile.

**Tuning**: Smaller blocks follow finer spatial structure but risk subtracting actual stars. Typical values are 16–128 pixels, or roughly 1–3% of image width. Set `local_bg_block_size=None` to skip local background subtraction entirely.

## 2. Global Noise Estimation

After local background subtraction, the residual noise level is estimated robustly using only the uncontaminated portion of the pixel distribution:

1. Compute the **global median** of all pixels — this is the background level
2. Select only pixels **at or below the median** (the "lower half")
3. Compute the RMS of these pixels **about the median** — for Gaussian noise the lower half's second moment about the median equals the full variance, so this mirrors directly into the true Gaussian σ
4. Apply **iterative sigma clipping** (5 iterations, 3σ factor) to the lower tail to reject remaining outliers (e.g. dead or strongly negative pixels), re-estimating about the median each pass

The detection threshold is then:

$$
\text{threshold} = \mu_{\text{bg}} + \sigma_{\text{threshold}} \times \sigma
$$

**Why the lower half?** Stars and nebulosity only contaminate pixels *above* the background. By using only the lower half of the distribution and assuming symmetry, we get a clean estimate of the Gaussian noise floor without any astrophysical contamination.

!!! warning "Changed in 0.9: `sigma_threshold` scale"
    Earlier versions computed the lower-half RMS about the lower half's *own
    mean*, which under-measures Gaussian noise by ~40% — a configured
    `sigma_threshold=5.0` was really a ~3σ cut. The estimator now returns true
    Gaussian sigmas (matching `extract_centroids_fast`, which was always
    correct). To keep your previous effective detection depth, multiply old
    configured values by ≈0.6 (e.g. `5.0` → `3.0`); leave them unchanged to
    get the threshold you had nominally been asking for.

## 3. Gaussian Matched Filter

Enabled by default (`matched_filter_sigma=1.5`), the *unclamped* bg-subtracted residual is convolved with a separable Gaussian (kernel truncated at 3σ) before thresholding. The filtered image is used **only** to form the detection mask — centroid positions and intensities are still measured on the unfiltered residual, so photometry is unaffected. Set `matched_filter_sigma=None` to threshold the unfiltered residual instead.

A matched filter concentrates a star's PSF into fewer effective pixels, raising the per-pixel signal-to-noise of point sources relative to single-pixel noise spikes: for a σ≈1.5 px PSF the peak-SNR gain is ~2× (≈0.75 mag more detection depth at the same false-positive rate). The gain is largest for faint stars in noisy or dense images, where many spurious blobs would otherwise survive the threshold and contaminate the brightness-sorted top-N list.

**Tuning**: σ within a factor of ~2 of the true PSF width recovers nearly all the SNR — the filter has a broad optimum. The detection threshold is automatically scaled by the kernel's noise-suppression factor, so `sigma_threshold` always means "sigmas of the noise actually present in the thresholded image" whether the filter is on or off — **no retuning is needed when toggling it**. `ExtractionResult.threshold` reports the threshold actually applied.

**Cost**: the separable 1D blur (+ one full-image float buffer) is now one of the larger stages of the much-faster pipeline — roughly 60% extra over the unfiltered path on a sparse 4 Mpx frame single-threaded (~8 ms extra), a much smaller fraction on dense fields where per-blob work dominates, and about half that with the `parallel` feature. Disable it (`None`) only if that overhead matters more than detection depth in your application.

## 4. Connected-Region Detection

Pixels above the threshold are grouped into blobs by a **run-length union-find** sweep with 8-connectivity (diagonal neighbors included): a single raster pass turns the threshold predicate into horizontal runs of lit pixels, merges runs that touch (including diagonally) runs on the previous row through a union-find structure, and hands each downstream stage its blob as a list of runs. No per-pixel label map is materialized. Both extraction paths — this one and `extract_centroids_fast` — share this detection core.

## 5. Blob Filtering

Blobs are filtered before centroid computation:

- **Minimum pixels** (`min_pixels`, default 3) — Rejects hot pixels and single-pixel noise spikes
- **Maximum pixels** (`max_pixels`, default 10,000) — Rejects very extended objects, large nebulae, or noise blobs from detector artifacts
- **Elongation** (`max_elongation`, default 3.0) — Computed from the covariance eigenvalue ratio $\sqrt{\lambda_{\max} / \lambda_{\min}}$. Rejects cosmic ray hits, satellite trails, and diffraction spikes. Set to `None` to disable.
- **Sharpness** (`max_sharpness`, default 0.9) — DAOFIND-style hot-pixel / cosmic-ray gate: rejects blobs whose peak sharpness $(\text{peak} - \text{mean of 8 neighbors})/\text{peak}$ exceeds the limit, measured on the *unfiltered* background-subtracted image (so a matched-filter-smeared hot pixel is still caught). The default passes any system whose PSF spans multiple pixels — a critically sampled PSF scores ~0.5, a strongly undersampled one ~0.85, a hot pixel ~1.0. Set to `None` for severely undersampled data (PSF FWHM below ~1.5 px, e.g. resampled survey cutouts), where real stars are geometrically indistinguishable from hot pixels.
- **Border margin** (`border_margin`, default 0 = off) — Drops blobs whose bounding box comes within the margin of an image edge. A star cut off by the frame boundary has a truncated PSF, biasing its center of mass toward the interior — a plausible but wrong position.
- **Deblending** (`deblend`, default `"off"`) — With `"reject"`, drops blobs containing more than one distinct intensity peak. A blended star pair otherwise produces one centroid at the flux-weighted midpoint — a wrong position the pattern hash will consume. Saturated blobs are exempt (plateau noise fakes maxima on a genuinely single star).

## 6. Centroid Computation

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

When all nine background-subtracted samples are positive, the fit is performed on **log intensity**: a Gaussian PSF is exactly quadratic in $\ln(v)$, so the log fit removes the linear fit's S-curve bias (~0.05–0.1 px at quarter-pixel peak phases). Blobs with a non-positive sample in the window (faint stars whose wings dip below the local background) keep the linear-intensity fit.

The sub-pixel offset is found by setting $\nabla f = 0$ and solving the resulting 2×2 linear system. The refinement is accepted only if:

- The offset is within ±0.5 pixels of the peak
- The quadratic position agrees with the center-of-mass within 0.5 pixels

Otherwise, the center-of-mass position is kept. This fallback ensures robustness for blended or asymmetric sources where the quadratic model breaks down. Blobs whose peak reaches `saturation_level` (opt-in, default off) skip the refinement entirely and keep the center-of-mass position — a flat-topped or bloomed profile has no meaningful maximum.

## 7. Output

Centroids are converted to **geometric-image-center origin** coordinates
(pixel centers at integer indices, so the center is `(W−1)/2`, `(H−1)/2` —
matching FITS / astropy / OpenCV; see [Coordinate Conventions](coordinates.md)):

$$
x_{\text{out}} = x_{\text{px}} - \frac{W-1}{2}, \qquad
y_{\text{out}} = y_{\text{px}} - \frac{H-1}{2}
$$

where $W$ and $H$ are the image dimensions. The coordinate convention is +X right, +Y down — the same as the camera frame used by the solver. See [Coordinate Conventions](coordinates.md) for details.

The final list is **sorted by brightness** (integrated intensity above background, descending). This ordering is important for the solver, which tries patterns from the brightest stars first.

Optionally, the list can be truncated to `max_centroids` entries.

## Performance

On a 2048×2048 TESS frame the connected-component path runs in **~26 ms**
single-threaded (matched filter on); the fast path below in ~15–20 ms. Block
medians are computed from a stride subsample of each tile, and background
subtraction, noise statistics, and the filter input are produced by one fused
pass over the image — there is no materialized full-image background buffer.

### Multi-threading (`parallel` feature)

The Rust crate's `parallel` feature multi-threads the extraction hot paths via
[rayon](https://docs.rs/rayon). It parallelizes the local-background stage
(step 1) — independent block medians and the fused per-row residual pass — and
enables numeris's parallel `imageproc` routines, so the matched-filter Gaussian
blur (step 3) runs threaded too. Results are bit-identical to the
single-threaded build.

Measured on an 8-core machine: **~1.9×** on a sparse 2-megapixel field and
**~1.45×** on a dense TESS frame (~37k blobs). The dense case scales less
because the per-blob centroid loop (step 6) — a small fraction of total time on
typical fields — is left sequential, as is the run-merging detection sweep
(step 5), which is inherently sequential.

Build with `cargo build --release --features image,parallel`.

## Reusing buffers across frames (`CentroidExtractor`)

`extract_centroids` allocates its full-image working buffers — the residual
images, the matched filter's output and the detection bit mask, about 48 MB
at 2048² — fresh on every call. The allocation itself is cheap, but the first
touch of each page is not: roughly 0.3 ms per 2048² frame serially, 0.5 ms
with the `parallel` feature. In a frame loop, `CentroidExtractor` keeps those
buffers between calls (resizing only when the frame size changes) and gives
bit-identical results:

```python
extractor = tetra3rs.CentroidExtractor()
for frame in frames:
    result = extractor.extract(frame, sigma_threshold=5.0, max_centroids=100)
```

`extract()` takes exactly the arguments of `extract_centroids`. Use one
extractor per thread. (Rust: `CentroidExtractor::extract_from_raw` /
`extract_from_image`.)

## Fast single-pass extraction (`extract_centroids_fast`)

The pipeline above prioritizes fidelity — local-background interpolation, an
optional matched filter, full connected-component analysis, per-blob annulus
backgrounds, and shape/elongation gating. For latency-critical applications
(frame-rate tracking, embedded star trackers) tetra3rs offers an alternative
that reads each pixel **once**: `extract_centroids_fast`
(`FastCentroidConfig` in Rust). It is an *additive* alternative — the
connected-component path stays the default and the right choice for
calibration and faint-star work.

### Algorithm

1. **Coarse background pre-pass.** A subsampled grid (~1/64 of the pixels)
   gives a `bg_grid`-spaced background grid and a global noise σ. Cheap, and
   still tracks large-scale gradients (vignetting, Milky Way).
2. **Single raster sweep.** Each pixel is thresholded against the bilinearly
   interpolated background (`> bg + sigma_threshold·σ`). Lit pixels are grouped
   into connected regions on the fly via **run-length + union-find**, with
   intensity-weighted moments (including second moments, so `Centroid.cov` is
   populated like the main path) accumulated inline. One center-of-mass is
   emitted per region, with the same 3×3 quadratic peak refinement as the main
   pipeline (gated to agree with the CoM).

No convolution and no second pass, so the cost is dominated by a single memory
read rather than compute.

### Speed and accuracy

On 2048×2048 TESS frames, single-threaded: **~15–20 ms vs ~26 ms** for the
connected-component path, with equal end-to-end solve accuracy and
~**0.1 px** centroid agreement on bright stars. (The gap has narrowed as the
connected-component path got faster; the fast path's remaining edge is that
it reads each pixel once and never convolves.)

The trade-offs are deliberate:

- **No matched filter**, so faint-star sensitivity is lower — sized for the
  brightest stars a tracker locks onto, not deep detection.
- **Center-of-mass is threshold-clipped**: sub-pixel accuracy is ~0.1 px on
  bright stars, degrading for faint ones. Use the connected-component path
  for calibration / tight astrometry.
- A **global noise σ** is used (adequate when read/shot noise is roughly
  uniform even where the background *level* varies).

It returns the same `ExtractionResult` (center-origin coordinates, brightest
first), so it is a drop-in for `solve_from_centroids`.

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma_threshold` | 5.0 | Detection threshold in noise σ above the local background |
| `bg_grid` | 64 | Coarse background-grid block size (pixels) |
| `min_pixels` | 2 | Minimum pixels in a region (rejects hot pixels) |
| `max_pixels` | 10,000 | Maximum pixels in a region (rejects satellite trails / bloomed regions, which would otherwise be the *brightest* centroid handed to the solver) |
| `max_elongation` | None | Maximum elongation ratio (opt-in — moment-based elongation is noisy for few-pixel regions) |
| `max_sharpness` | 0.9 | Hot-pixel / cosmic-ray gate (see [blob filtering](#5-blob-filtering); None = disabled) |
| `saturation_level` | None | Peak level at which sub-pixel refinement is skipped (CoM kept) |
| `border_margin` | 0 | Drop regions within this margin of an image edge (0 = off) |
| `max_centroids` | None | Maximum centroids to return, brightest first (None = all) |

The coarse background grid already absorbs structure larger than a block
(~64 px) before these filters see it, so `max_pixels` matters mainly for sharp
bloomed regions and `max_elongation` for trails.

```python
result = tetra3rs.extract_centroids_fast(image, sigma_threshold=5.0, max_centroids=30)
solution = db.solve_from_centroids(result.centroids, camera_model=cam)
```

## Configuration Reference (connected-component pipeline)

These parameters configure the default `extract_centroids` pipeline (the fast
path's knobs are listed [above](#configuration)).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma_threshold` | 5.0 | Detection threshold in (true Gaussian) σ above background |
| `min_pixels` | 3 | Minimum blob size (pixels) |
| `max_pixels` | 10,000 | Maximum blob size (pixels) |
| `max_centroids` | None | Maximum centroids to return (None = all) |
| `local_bg_block_size` | 64 | Tile size for local background (None = skip) |
| `max_elongation` | 3.0 | Maximum elongation ratio (None = disabled) |
| `matched_filter_sigma` | 1.5 | Gaussian matched filter σ in pixels (None = disabled) |
| `max_sharpness` | 0.9 | Hot-pixel / cosmic-ray peak-sharpness gate (None = disabled) |
| `saturation_level` | None | Sensor saturation level; saturated peaks keep the CoM position |
| `deblend` | `"off"` | `"reject"` drops blobs with multiple distinct intensity peaks |
| `border_margin` | 0 | Drop blobs within this margin of an image edge (0 = off) |

!!! tip "TESS example"
    For TESS Full Frame Images with their wide-field defocused optics, typical settings are `sigma_threshold=180`, `min_pixels=4`, `local_bg_block_size=16`, `max_elongation=6.0`. The high sigma threshold rejects the dense background, and the relaxed elongation allows for the broad, slightly asymmetric TESS PSF. (Before 0.9 the equivalent setting was `sigma_threshold=300` — see the noise-estimator note in [step 2](#2-global-noise-estimation).)
