# Centroid Extraction

Before an image can be solved, its stars have to be found and located. `extract_centroids` (Rust: `extract_centroids_from_raw` / `extract_centroids_from_image`, behind the `image` feature) takes a grayscale image and returns a list of star centroids — sub-pixel position, flux, position uncertainty and shape — sorted brightest first and ready for `solve_from_centroids`.

There is one pipeline and one parameter that really matters, `psf_sigma`: how wide a star is on your detector.

## Pipeline Overview

The pipeline **detects on a smoothed copy of the frame and measures on the raw pixels**:

```
Raw image
  │
  ├─ 1. Detection image (optional 2×2 binning + Gaussian smoothing = matched filter)
  │
  ├─ 2. Background level and noise σ, per block, measured on the detection image
  │
  ├─ 3. Threshold → connected regions (run-length union-find)
  │
  ├─ 4. Peaks: every local maximum in a region is a candidate star
  │
  ├─ 5. Measurement on the raw pixels (iterative Gaussian-windowed centroid)
  │
  └─ Output: Centroid list, brightest first
```

Smoothing is the right thing for *finding* stars — it gathers a star's light into one peak and averages the noise down. It is the wrong thing for *locating* them, so positions are never read off the smoothed image: each detection only seeds a measurement made on the original pixels.

## 1. Detection Image

The frame is convolved with a small separable Gaussian whose width matches the PSF. This is a **matched filter**: for a star of Gaussian width `psf_sigma` in uncorrelated noise, it is the linear filter that maximizes the star's signal-to-noise ratio at its peak. A faint star that is spread over a dozen pixels, none of them individually significant, becomes one clearly significant peak.

Optionally the frame is first **binned 2×2**. Binning is itself a (box) smoothing, so the Gaussian that follows is narrowed to compensate and the two together still match the PSF. See [Detection binning](#detection-binning) for the trade-off.

Non-finite pixels (NaN, ±∞) are treated as background.

## 2. Background and Noise

The detection image is divided into blocks of `bg_block_size` pixels (default 64). Each block contributes two numbers:

- a **background level** — the block's median, which a few bright stars inside the block do not move. The level is bilinearly interpolated between block centers (and linearly extrapolated to the image edges), so vignetting, stray light, sky gradients and the Milky Way are followed smoothly.
- a **noise σ** — measured from the residuals against that interpolated surface, using only the pixels at or below the residual median. Stars and nebulosity only ever add light, so the lower half of the distribution is the uncontaminated half; its RMS about the median is the Gaussian noise σ.

Two things about this are worth knowing:

**The statistics are taken on the detection image, not the raw one.** The threshold is applied to the smoothed image, so that is where the noise has to be known. Measuring it there — rather than measuring the raw noise and predicting what smoothing does to it — keeps the threshold correct when the pixel noise is *correlated*, as it is in resampled, debayered or compressed images.

**The noise is per block.** A frame with a bright sky gradient or strong vignetting has more noise in one corner than another; each block is thresholded against its own σ.

**Tuning**: `bg_block_size` should be much larger than a star and smaller than the background structure you want removed — roughly 1–3% of the image width. Smaller blocks follow finer structure but are noisier and can be pulled up by a bright star; in crowded fields a larger block (128) is steadier.

## 3. Threshold and Regions

A pixel of the detection image is "lit" when it exceeds

$$
\text{level} + \texttt{sigma\_threshold} \times \sigma
$$

with the level and σ of step 2. Lit pixels are grouped into connected regions (8-connectivity) by a single run-length union-find sweep.

### What `sigma_threshold` means

Because the threshold sits on the matched-filtered image, `sigma_threshold` is a cut on a **star's** signal-to-noise ratio — the significance of the whole star, all its pixels combined — not on any single pixel's. `sigma_threshold=5` keeps stars detected at 5σ or better.

This makes the number comparable across cameras: 5σ is 5σ whether the star covers two pixels or twenty. It also means that a threshold tuned for an extractor that cuts on individual raw pixels does not carry over — the equivalent value here is smaller (for the TESS frames in the tutorials, 180 became 50).

For plate solving you rarely want the faintest stars. The solver works from the brightest few dozen, and in a crowded field a high threshold (tens of σ) is a simple way to keep only stars that stand clear of their neighbours.

`ExtractionResult.threshold` reports the threshold at a typical (median) block, in detection-image units. `ExtractionResult.background_sigma` reports the noise converted back to **raw-pixel** units; the conversion assumes uncorrelated pixel noise, so on resampled or compressed frames read it as an equivalent white-noise level.

Regions larger than `max_pixels` (default 10,000) are dropped whole. These are satellite trails, horizon glow and stray-light blobs — and since results are sorted by brightness, such an object would otherwise be the *first* centroid handed to the solver.

## 4. Peaks

A region only says which pixels belong together. The stars are its **local maxima**: every pixel of the detection image that is brighter than its eight neighbours is a candidate.

- The **brightest peak** of a region is always measured.
- A **fainter peak** in the same region has to show that it is a separate star rather than noise or structure on a bright neighbour's wing. It must settle on its own position (not slide into the neighbour), lie at least `2 × psf_sigma` from every star already accepted, and — in a local fit of a Gaussian on a constant pedestal — have an amplitude that is significant against the noise and at least half the pedestal it sits on. On a bright star's wing the pedestal is the wing, which is what stops wings from spawning false companions.

So close pairs are separated automatically; there is no deblending switch. The limit is resolution: two stars closer than a few `psf_sigma` have a single maximum in the smoothed image and come out as one centroid, near the brighter star but pulled toward its companion.

## 5. Measurement

Each candidate is measured on the **raw pixels** with an iterative Gaussian-windowed centroid. Starting from the peak, the position is repeatedly moved to the centroid of the background-subtracted pixels weighted by a Gaussian window centered on the current estimate:

$$
\mathbf{x} \leftarrow \mathbf{x} + g \, \frac{\sum_i w_i \, d_i \, (\mathbf{x}_i - \mathbf{x})}{\sum_i w_i \, d_i},
\qquad
w_i = \exp\!\left(-\frac{|\mathbf{x}_i - \mathbf{x}|^2}{2\sigma_w^2}\right)
$$

where $d_i$ is pixel $i$ minus the local background and $g \ge 1$ is a step gain that makes up for the window pulling each step short. The fixed point is where the image's correlation with a Gaussian is greatest — the maximum of the matched filter, evaluated in the continuum instead of on the pixel grid. For a Gaussian PSF in background-limited noise that is the maximum-likelihood position.

Compared with a center of mass over the above-threshold pixels, this has no footprint to depend on: the window is the same for a faint star and a bright one, the noisy outer pixels are weighted down rather than cut off, and an error in the background level cancels by symmetry.

The window width is `psf_sigma`, but never less than 1 px — a narrower window cannot be sampled on the pixel grid without the result wobbling with the star's sub-pixel position.

A star whose window does not fit inside the frame (closer than about `3 × psf_sigma` to an edge) is dropped, because a truncated window pulls the position toward the interior. `border_margin` widens that exclusion zone if you want one.

### What a centroid carries

| Field | Meaning |
|---|---|
| `x`, `y` | Position in pixels, origin at the image center, +X right, +Y down (see [Coordinate Conventions](coordinates.md)) |
| `brightness` (Rust: `mass`) | Flux above background, from the same Gaussian weighting. The list is sorted on it |
| `cov` | 2×2 covariance of the **position**, px² — how well the star is located |
| `shape` | 2×2 intensity second moments, px² — how big and how round the star is |

`cov` and `shape` are easy to confuse and differ by orders of magnitude:

- **`shape`** describes the star. For a round star its diagonal is about `psf_sigma²` — 2.25 px² for a 1.5 px PSF — whatever the brightness. Its eigenvectors and eigenvalues give the orientation and axis lengths of an elongated detection. The solver does not use it.
- **`cov`** describes the measurement. It is the variance of `x` and `y` implied by the measured background noise, and it shrinks as the star gets brighter: from around 0.2 px² for a star near a 5σ threshold down to 10⁻⁴ px² and below for a bright one (a hundredth of a pixel). The solver folds it into each star's match tolerance. It accounts for background noise only — not the star's own photon noise, and not pixel-response or PSF-model errors, which set the floor for very bright stars.

To draw a star's outline, use `shape`. To weight a fit or judge a residual, use `cov`.

### Saturated and very broad objects

A saturated star has a flat top: there is no peak for the window to settle on. The same is true of anything far broader than the window (wider than about twice `psf_sigma`), or a measurement that does not converge. In those cases the region is measured by the plain **center of mass** of its pixels and yields a single centroid.

This happens automatically from the measured width. Setting `saturation_level` (the raw pixel value at which the sensor clips) additionally sends *mildly* saturated stars down the same path — ones whose clipped core is still narrow enough to look like a star.

## Filters

| Filter | Removes | How |
|---|---|---|
| `reject_hot_pixels` (default on) | Hot pixels, cosmic-ray hits | Measured width much narrower than the PSF |
| `max_elongation` (default 3.0) | Streaks, trails, unresolved blends, bleed columns | Major/minor axis ratio of the measured shape |
| `max_pixels` (default 10,000) | Satellite trails, glow, stray light | Region size |
| `border_margin` (default 0) | Stars near the edge | Distance of the centroid from the frame edge |

### Hot pixels

A hot pixel is one pixel wide; a star is as wide as the PSF. With `reject_hot_pixels=True`, a detection whose measured width is under about 0.4 px (or `0.4 × psf_sigma`, if that is larger) is discarded.

The test needs stars to be measurably wider than a pixel. For **severely undersampled** data — PSF FWHM below about 1.2 px, e.g. survey cutouts resampled to a coarse scale — a real star is geometrically indistinguishable from a hot pixel and the filter would remove stars. Turn it off there (`reject_hot_pixels=False`).

### Elongation, and CCDs that bloom

`max_elongation` rejects detections whose long axis exceeds the short one by more than the given ratio: satellite streaks too small for `max_pixels`, cosmic-ray tracks, and pairs too close to split.

On a CCD without anti-blooming, a saturated star spills charge along its column, and the result is exactly such an elongated blob. Rejecting the long ones is correct — their center of mass can be pixels away from the star. But saturated stars are also the *brightest* stars, which are the ones the solver builds its patterns from. If solves fail for lack of bright stars on such a sensor, raise `max_elongation` to 5–6: that keeps stars with a short bleed, whose center of mass is still good, and continues to drop the long ones. `None` disables the gate.

## Choosing `psf_sigma`

`psf_sigma` is the Gaussian σ of a star image in pixels: FWHM / 2.355. It sets both the detection kernel and the measurement window.

It does not need to be exact. A value within a factor of about 1.5 of the true width keeps the position error within about 30% of the best achievable, so one setting per camera is enough, and modest focus or temperature drift does not call for retuning.

To find it for a new camera, extract a typical frame and read back what was measured:

```python
result = tetra3rs.extract_centroids(image)   # psf_sigma=1.5
print(result.psf_sigma_estimate)             # e.g. 0.8
```

`psf_sigma_estimate` is the median width of the well-exposed, unsaturated stars in the frame (`None` if there are fewer than five). It is the width of a star *as sampled by the detector*, so it includes the pixel's own footprint: optics that deliver σ = 0.75 px measure about 0.80 px. That sampled width is what is in the image, and is the value to pass. Use the estimate as follows:

- **Close to what you passed** — nothing to do.
- **Clearly different** — pass the estimate as `psf_sigma` and extract again.
- **About twice what you passed** — the estimate cannot see much beyond twice the window width, so it has hit its ceiling and the true value may be larger still. Double `psf_sigma` and look again.

The valid range is (0, 10] pixels.

## Detection Binning

`detection_binning` chooses the resolution of the detection image:

| Value | Detection image |
|---|---|
| `1` | Full resolution |
| `2` | Frame binned 2×2 |
| `None` (default) | 2 when `psf_sigma ≥ 1.25`, otherwise 1 |

Binning makes every full-image pass four times cheaper, and the extraction as a whole about twice as fast. What it costs:

- **Accuracy: nothing.** Positions are always measured on the unbinned pixels; binning only affects which stars are found.
- **Depth: about a tenth** of the detection signal-to-noise, for a PSF of σ ≥ 1.25 px. This only matters for stars right at the threshold.

For a narrower PSF a 2×2 bin is wider than the star, and would cost real depth and merge close pairs — which is why the default switches to full resolution below 1.25 px. Force `detection_binning=1` when you want the deepest detection regardless of speed.

## Output

Centroids use **geometric-image-center origin** coordinates (pixel centers at integer indices, so the center is `(W−1)/2`, `(H−1)/2` — matching FITS / astropy / OpenCV):

$$
x_{\text{out}} = x_{\text{px}} - \frac{W-1}{2}, \qquad
y_{\text{out}} = y_{\text{px}} - \frac{H-1}{2}
$$

with +X right and +Y down, the camera frame the solver uses. See [Coordinate Conventions](coordinates.md).

The list is **sorted by brightness**, brightest first, which is the order the solver tries patterns in. `max_centroids` truncates it.

`ExtractionResult` also reports `background_mean`, `background_sigma`, `threshold`, `num_blobs_raw` (the number of regions above threshold, before any filter — one region can hold several stars) and `psf_sigma_estimate`.

## Accuracy and Performance

On synthetic 2048×2048 frames with about 2100 stars and a PSF of σ = 1.5 px, the position error stays within about 10% of the read-noise Cramér–Rao bound — $\sqrt{2}\,\sigma_{\text{psf}} / \text{SNR}$ per axis — at all brightnesses.

Timings for the same frames on an 8-core Apple M-series machine:

| Detection | All cores | One core |
|---|---|---|
| Automatic (binned 2×2) | 1.3 ms | 3.3 ms |
| Full resolution (`detection_binning=1`) | 2.5 ms | 7.3 ms |

### Multi-threading

The full-image passes and the per-region measurement are multi-threaded with [rayon](https://docs.rs/rayon). Results are bit-identical for every thread count, including one.

**Python.** The wheels are built with threading on, and `extract_centroids` and `CentroidExtractor.extract` take a keyword-only `threads` argument:

| `threads` | Behavior |
|---|---|
| `None` (default) | Every available core, or `RAYON_NUM_THREADS` if that is set. Frames under about 512×512 run on one thread, because waking the workers costs more than it saves. |
| `1` | Single-threaded. |
| `n` | Exactly `n` threads. |

```python
result = tetra3rs.extract_centroids(image)             # all cores
result = tetra3rs.extract_centroids(image, threads=1)  # one core
```

Set `threads=1` when you already run one process or thread per core (a `multiprocessing` pool, a frame loop per camera), or to leave cores free for other work.

The worker threads live in pools owned by the extension, never in a process-global pool, which makes extraction **safe after `fork()`**: threads do not survive a fork, and a child that inherited a pool would wait on it forever. A forked child (`multiprocessing`'s `fork` start method, `os.fork`) notices the pools belong to its parent and starts its own. Database loading and unpickling, which decode and validate the pattern table on all cores, use the same pools and are fork-safe the same way.

**Rust.** Threading is the crate's `parallel` feature (`cargo build --release --features image,parallel`); without it the same code runs sequentially. Parallel regions use the rayon pool they are called in — the global one by default, sized by `RAYON_NUM_THREADS` — so to choose a thread count for a call, run it inside your own pool with `ThreadPool::install`.

### Reusing buffers across frames (`CentroidExtractor`)

`extract_centroids` allocates its full-image working buffers fresh on every call. The allocation itself is cheap, but the first touch of each page is not. In a frame loop, `CentroidExtractor` keeps those buffers between calls (resizing only when the frame size changes) and gives bit-identical results:

```python
extractor = tetra3rs.CentroidExtractor()
for frame in frames:
    result = extractor.extract(frame, psf_sigma=1.5, max_centroids=100)
```

`extract()` takes exactly the arguments of `extract_centroids`. Use one extractor per thread. (Rust: `CentroidExtractor::extract_from_raw` / `extract_from_image`.)

## Configuration Reference

Python keyword arguments of `extract_centroids` / `CentroidExtractor.extract`; the fields of `CentroidExtractionConfig` in Rust have the same names. In Python everything after `sigma_threshold` is keyword-only.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma_threshold` | 5.0 | Detection threshold: a star's matched-filter signal-to-noise ratio |
| `psf_sigma` | 1.5 | Expected PSF width, Gaussian σ in pixels (FWHM / 2.355); range (0, 10] |
| `bg_block_size` | 64 | Block size of the background / noise grid, pixels |
| `max_centroids` | None | Maximum centroids to return, brightest first (None = all) |
| `saturation_level` | None | Raw pixel value at which the sensor saturates; such stars are measured by center of mass |
| `border_margin` | 0 | Drop centroids within this many pixels of an image edge |
| `max_pixels` | 10,000 | Regions larger than this are dropped whole |
| `max_elongation` | 3.0 | Maximum major/minor axis ratio (None = disabled) |
| `reject_hot_pixels` | True | Reject detections much narrower than the PSF |
| `detection_binning` | None | 1 = full resolution, 2 = binned 2×2, None = automatic |
| `threads` (Python only) | None | Worker threads (None = all cores) |

!!! tip "Example: TESS full-frame images"
    TESS is undersampled (PSF σ ≈ 0.8 px), its fields are crowded, and its brightest stars saturate and bleed along columns:

    ```python
    result = tetra3rs.extract_centroids(
        image,
        psf_sigma=0.8,          # narrow PSF
        sigma_threshold=50.0,   # only stars standing clear of the crowd
        bg_block_size=128,      # steadier background in a crowded field
        max_elongation=6.0,     # keep bright stars with a short bleed
    )
    ```

!!! tip "Example: severely undersampled survey cutouts"
    For 10° SkyView / DSS cutouts, where a star is smaller than a pixel:

    ```python
    result = tetra3rs.extract_centroids(
        image,
        psf_sigma=0.8,
        sigma_threshold=6.0,
        reject_hot_pixels=False,   # a star looks like a hot pixel here
        max_centroids=200,
    )
    ```

!!! warning "Changed: one extraction pipeline"
    This pipeline replaces both the earlier connected-component extractor and `extract_centroids_fast`, which no longer exists. If you are updating code:

    - `min_pixels`, `matched_filter_sigma`, `max_sharpness` and `deblend` are gone. Use `psf_sigma` in place of `matched_filter_sigma`, `reject_hot_pixels` in place of `max_sharpness`; close pairs are always separated.
    - `local_bg_block_size` is now `bg_block_size` and cannot be `None`.
    - `sigma_threshold` is a cut on the star's matched-filter signal-to-noise ratio, so values tuned for the old extractor need retuning (they will be too high; see [What `sigma_threshold` means](#what-sigma_threshold-means)).
    - `Centroid.cov` is now the covariance of the **position**; the intensity moments it used to hold are in `Centroid.shape`.
    - `ExtractionResult.background_sigma` is the raw-pixel noise and `threshold` is on the smoothed detection image.
    - Pickles of `Centroid`, `ExtractionResult` and `SolveResult` from earlier versions do not load.
