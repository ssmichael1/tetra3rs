# Changelog

Only recent releases are listed. Older entries are in this file's git history (`git show vX.Y.Z:CHANGELOG.md`). Full detail for each change lives in the linked PR.

## Unreleased

### Changed

- The lost-in-space solve is split into explicit stages — `preprocess` → `PatternSearch` (hypothesis source: FOV sweep, hash lookup, SVD, parity) → `verify_attitude` → `accept_lis_candidate` (pre-gate, refine, re-verify, correction) — with the verification vectors carrying their pixel scale (`CentroidVectors`) so a stage needing a different scale rebuilds instead of silently testing at the wrong one; tracking is the second hypothesis source into the same verify/refine tail. Crate-private; results bit-identical on 1500 synthetic fields across plain, spurious, FOV-sweep, parity, hinted and aberration scenarios. ([#60](https://github.com/ssmichael1/tetra3rs/pull/60))
- WCS refinement projects catalog stars with the tangent-plane basis at CRVAL (dot products, as Phase-D re-association already did) instead of decoding each matched star to RA/Dec and evaluating per-star trig every pass; `wcs_refine` 16.9 → 12.2 µs per solve on the profiler field (mean solve 50 → 41 µs), results unchanged within f64 rounding. ([#59](https://github.com/ssmichael1/tetra3rs/pull/59))

### Fixed

- `StarCatalog` cone queries sized each latitude bin's RA span from the bin center, missing most stars within a few degrees of either pole (dec 88°: ~65% missed; dec 89.5°: ~85%) — verification, tracking, WCS re-association, and database generation all share the query, so polar fields solved with half the matches or not at all. The span is now bounded at the bin's polar edge; brute-force regression test added. Regenerate databases built before this fix. ([#59](https://github.com/ssmichael1/tetra3rs/pull/59))
- The post-refinement acceptance re-verify reused verification vectors built at the swept FOV whenever the measured-FOV rebuild was skipped, so on wide frames (≳3000 px) true edge matches fell outside the tightened 2.5 px radius floor and weakened the acceptance statistic; the re-verify now always uses vectors at the refined scale. ([#59](https://github.com/ssmichael1/tetra3rs/pull/59))

## 0.12.0 - 2026-08-30

### Changed

- `extract_centroids_fast` thresholds each row into a packed bit mask (per-column background plan precomputed once, `p > thr[c]` vectorized) and reads runs off the bits with `trailing_zeros`/`trailing_ones` instead of calling a per-pixel closure: ~6× faster on tracker-like frames (2048²: 7.3 → 1.2 ms), ~2.6× on dense survey frames; with `parallel` the mask is built row-parallel (2048²: 0.53 ms, ~13×). Output is bit-identical. ([#53](https://github.com/ssmichael1/tetra3rs/pull/53))
- `BackgroundGrid::build` streams each block row across its blocks (one task per block row under `parallel`) instead of a strided gather per block, and the brightness sort orders `(mass, index)` keys with an unstable sort; both bit-identical. ([#53](https://github.com/ssmichael1/tetra3rs/pull/53))
- The CCL centroider (`extract_centroids_from_raw` / `_from_image`) evaluates the background surface per row from the grid's column plan instead of per pixel, packs `filtered > threshold` into a bit mask and sweeps runs off the words (the annulus "not in a blob" test is a bit test on the same mask), and under `parallel` labels the mask in 64-row bands and runs the per-region annulus / moment / refine stage one task per region (`par::map_indices_init`) with a branch-free annulus gather; the matched-filter blur goes through numeris 0.5.19's `gaussian_blur_into` (no full-image intermediate). Bit-identical output; 2048² default config: 38 → 21 ms serial, 28 → 8 ms parallel on 8 cores. ([#54](https://github.com/ssmichael1/tetra3rs/pull/54))
- `to_grayscale_f32` reads `ImageLuma8` / `ImageLumaA8` buffers directly instead of cloning through `to_luma8()`. ([#54](https://github.com/ssmichael1/tetra3rs/pull/54))
- Python: `extract_centroids` / `extract_centroids_fast` convert C-contiguous numpy images straight from the flat slice (memcpy for f32), saving ~2.4 ms per 2048² frame; non-contiguous views keep the strided path, output unchanged ([#52](https://github.com/ssmichael1/tetra3rs/pull/52))

## 0.11.0 - 2026-08-30

### Added

- Browser wasm support (`wasm32-unknown-unknown`): the solver clock goes through `solver::clock::Instant` — `std::time::Instant` everywhere with a working clock, `web_time::Instant` (`performance.now()`) on `wasm32-unknown-unknown` where `std`'s `Instant::now()` aborts; WASI/Emscripten keep `std`; `profile`'s `timed!` uses the same clock; CI lints the wasm32 target. Based on #46 by @trams ([#46](https://github.com/ssmichael1/tetra3rs/pull/46), [#48](https://github.com/ssmichael1/tetra3rs/pull/48))
- `SolveConfig::max_patterns_checked` (Python `max_patterns_checked=`): a search budget in image patterns, summed over the FOV sweep, checked alongside `solve_timeout_ms` — whichever trips first ends the search with `SolveStatus::Timeout`; deterministic across machines and finite on clockless targets. Default 10 M (the 5 s timeout normally trips first natively); `None` = unbounded; `Some(0)` fails `validate()` ([#49](https://github.com/ssmichael1/tetra3rs/pull/49))

### Changed

- Distortion-fit sigma-clip estimates σ over all points (MAD) so the inlier set converges instead of shrinking onto bright stars; the fixed 5 px stage-2 recovery is removed (it re-admitted mismatches). TESS model residuals improve ~2×; inlier counts drop ~17%. ([#51](https://github.com/ssmichael1/tetra3rs/pull/51))
- Single- and multi-image calibration share one pooled fitter (`fit::fit_pooled`); the legacy `fit_polynomial_distortion` / `fit_radial_distortion` path is removed (crate-private; results unchanged). ([#51](https://github.com/ssmichael1/tetra3rs/pull/51))
- The CCL and fast centroiders share the region extent/border gate and the sharpness/refine/assembly tail (bit-identical output). ([#51](https://github.com/ssmichael1/tetra3rs/pull/51))

### Fixed

- Sigma-clip in polynomial/radial distortion fits used `k·σ` without the median offset on non-negative residual magnitudes, rejecting ~14% of good points per pass; now `median + k·σ` (matches `wcs_refine`), and a mask with fewer inliers than parameters is never committed. ([#51](https://github.com/ssmichael1/tetra3rs/pull/51))
- CCL centroid extraction turned a single `+inf`/`NaN` pixel into a `(NaN, NaN)` centroid with infinite mass ranked brightest; non-finite pixels are now background on every path. ([#51](https://github.com/ssmichael1/tetra3rs/pull/51))
- `calibrate_camera` fed non-finite centroids at matched indices through the WCS and pooled fits and returned `Ok` with an all-NaN model; such centroids are skipped, `solve_3x3` bails on NaN/inf, and the fitted `CameraModel` is validated before returning. ([#51](https://github.com/ssmichael1/tetra3rs/pull/51))
- `Some(NaN)` centroid mass reached the brightness sort's comparator (not a total order — std may panic); now treated as `None`. ([#51](https://github.com/ssmichael1/tetra3rs/pull/51))
- FOV sweep emitted values above π when `fov_max_error_rad` was large, each costing a full pattern search that could only be rejected later. ([#51](https://github.com/ssmichael1/tetra3rs/pull/51))
- `asin` arguments at the pole are clamped in `wcs_refine` (f32 rotation rows can round past ±1). ([#51](https://github.com/ssmichael1/tetra3rs/pull/51))
- Python: `calibrate_camera([res], [cents], ...)` (one-element list form) raised `TypeError`; `calibrate_camera` with zero image dimensions returned a model that could not be unpickled — both now behave like the single/solve forms. ([#51](https://github.com/ssmichael1/tetra3rs/pull/51))
- Docs: `CentroidExtractionConfig::max_elongation` default is `Some(3.0)`, not `None`. ([#51](https://github.com/ssmichael1/tetra3rs/pull/51))

## 0.10.0 - 2026-08-21

Robustness sweep: validate at every trust boundary (file load, pickle, public constructors) so corrupt data and degenerate arguments fail with a descriptive error instead of a deferred panic, hang, or silently-wrong result. No solver-algorithm changes; outputs on valid inputs are unchanged ([#47](https://github.com/ssmichael1/tetra3rs/pull/47)).

### Changed

- **Breaking (Rust):** `SolverDatabase::to_bytes` returns `crate::Result<Vec<u8>>`; new `SolveStatus::InvalidConfig` variant (appended last, wire values unchanged) returned by `solve_from_centroids` after the new public `SolveConfig::validate()` rejects a degenerate config (placeholder camera model, non-finite match/FOV/velocity params); Python `SolveFailure.status` gains `'invalid_config'` ([#47](https://github.com/ssmichael1/tetra3rs/pull/47))
- **Breaking:** `CameraModel::from_fov` and `StarCatalog::new` validate in release builds (documented panics; Python `ValueError`), as do zero image dimensions, degenerate `CameraModel` args, and non-finite distortion coefficients ([#47](https://github.com/ssmichael1/tetra3rs/pull/47))
- `calibrate_camera` argument errors (mismatched lengths, polynomial order outside `[2, 6]`) are `Err(InvalidInput)` instead of panics ([#47](https://github.com/ssmichael1/tetra3rs/pull/47))
- Generation limits: `catalog_nside ≤ 1024`, plausible `epoch_proper_motion_year`, lattice capped at 10⁸ points, `matched_filter_sigma ≤ 64` — errors instead of OOM ([#47](https://github.com/ssmichael1/tetra3rs/pull/47))
- Python releases the GIL during solve, generate, calibrate, load, and extraction ([#47](https://github.com/ssmichael1/tetra3rs/pull/47))
- numeris 0.5.14 → 0.5.17; pyo3 0.29.2, numpy 0.29.0 ([#47](https://github.com/ssmichael1/tetra3rs/pull/47))

### Fixed

- Corrupt/tampered database and camera files and every Python pickle are rejected at load by new `validate()` methods (`SolverDatabase`, `StarCatalog`, `CameraModel`, distortion types) instead of panicking out-of-bounds mid-solve or inflating the key enumeration without bound ([#47](https://github.com/ssmichael1/tetra3rs/pull/47))
- Infinite loop / unbounded memory in the FOV sweep with `fov_max_error_rad: Some(f32::INFINITY)`; candidates implying a FOV outside `(0, π)` are skipped ([#47](https://github.com/ssmichael1/tetra3rs/pull/47))
- Hipparcos parser panicked on multi-byte UTF-8 straddling a column boundary ([#47](https://github.com/ssmichael1/tetra3rs/pull/47))
- `PolynomialDistortion` order/coefficient mismatches panicked out-of-bounds (`num_coeffs` u32 wrap, incl. the Python `order=2**32-1` hole); orders bounded by `MAX_POLY_ORDER` (12) ([#47](https://github.com/ssmichael1/tetra3rs/pull/47))
- Single-image `calibrate_camera` returned a fabricated perfect result on a no-fit; now `InvalidInput` ([#47](https://github.com/ssmichael1/tetra3rs/pull/47))
- Gaia loader: 32-bit star-count truncation; non-finite records skipped with a warning; 32-bit `width*height` overflow in the extraction buffer check ([#47](https://github.com/ssmichael1/tetra3rs/pull/47))

## 0.9.0 - 2026-07-03

Verification statistics recalibrated, centroid-extraction overhaul, and a review-pass robustness sweep ([#45](https://github.com/ssmichael1/tetra3rs/pull/45)).

### Added

- Fast-path `max_pixels` (default 10000) and opt-in `max_elongation`; the fast path now populates `Centroid.cov` ([#45](https://github.com/ssmichael1/tetra3rs/pull/45))
- `border_margin` on both extraction configs: drop blobs whose bounding box touches the image edge ([#45](https://github.com/ssmichael1/tetra3rs/pull/45))
- `deblend: DeblendMode` (CCL path; Python `deblend="off" | "reject"`) rejects blobs with more than one intensity peak; deterministic sub-pixel accuracy ensemble test ([#45](https://github.com/ssmichael1/tetra3rs/pull/45))
- `max_sharpness` (default 0.9; DAOFIND-style hot-pixel gate — set `None` for undersampled PSFs) and `saturation_level` (skip parabola refine on clipped peaks) on both extraction configs ([#45](https://github.com/ssmichael1/tetra3rs/pull/45))

### Changed

- **Breaking (Rust):** `CentroidExtractionConfig.use_8_connectivity` removed — both extraction paths share one run-length union-find detection core (8-connected by construction); CCL extraction 71.5 → ~26 ms on 2048² TESS frames, bit-identical moments ([#45](https://github.com/ssmichael1/tetra3rs/pull/45))
- **Breaking (wire format):** `Solution.image_width` / `image_height` removed (use `Solution.camera_model`); Python `SolveResult` pickles from 0.8.0 do not load ([#45](https://github.com/ssmichael1/tetra3rs/pull/45))
- **Breaking:** `calibrate_camera` returns `Result<CalibrateResult>` (`InvalidInput` instead of a fabricated model; Python `ValueError`); `PolynomialDistortion::new` takes `(order, scale, a, b)` — `ap`/`bp` inverse coefficients are zero-filled (Python keeps them as ignored kwargs); `star_from_*`, `SolveConfig::pixel_scale`, `StarCatalog` index fields made crate-private; `GaiaStar::{pmra, pmdec}` are `f32` ([#45](https://github.com/ssmichael1/tetra3rs/pull/45))
- **Behavior:** `matched_filter_sigma` defaults to `Some(1.5)` with the threshold auto-scaled by the kernel's noise suppression (no retuning when toggling); the filter convolves the unclamped residual ([#45](https://github.com/ssmichael1/tetra3rs/pull/45))
- **Behavior:** verification uses a measured-density null model with π·r² disc area, excludes the 4 hypothesis stars, applies sequential Bonferroni over candidates actually tested (`match_threshold` is now a per-solve false-accept budget; `Solution.prob` is the corrected p-value), re-verifies the refined attitude, and rebuilds candidate vectors at the measured FOV (15%-wrong FOV solves in ~36 µs vs ~21–40 ms). Sparse 6-star fields 0 → 71% solvable; weakly-evidenced solves that passed only on the old optimism may now fail — raise `match_threshold` to accept them ([#45](https://github.com/ssmichael1/tetra3rs/pull/45))
- Sub-pixel peak refinement fits log intensity when all 3×3 samples are positive (TESS pooled residual 0.132 → 0.077 px) ([#45](https://github.com/ssmichael1/tetra3rs/pull/45))
- Python `solve_from_centroids` no longer requires `fov_estimate_*` / `image_*` when `camera_model` is given ([#45](https://github.com/ssmichael1/tetra3rs/pull/45))
- Performance: fused background pass on the CCL path (~40% faster), FOV sweep step doubled to `4·match_radius·fov` (no-match 28.6 → 12.8 ms), rotation-matrix Phase-D re-association (~14% off easy solves), candidate-key enumeration skips ratio-invariant violations ([#45](https://github.com/ssmichael1/tetra3rs/pull/45))

### Fixed

- CCL-path noise estimator ran ~40% low (RMS about the lower-half mean instead of the median); `sigma_threshold` now means true Gaussian sigmas — multiply old values by ≈0.6 to keep the same depth; `FastCentroidConfig` unaffected ([#45](https://github.com/ssmichael1/tetra3rs/pull/45))
- `matched_centroid_indices` were shifted whenever the solver dropped non-finite centroids; now always index the caller's input slice ([#45](https://github.com/ssmichael1/tetra3rs/pull/45))
- Lost-in-space requires ≥ 5 centroids (4 is all hypothesis, no evidence) and returns `TooFew` immediately ([#45](https://github.com/ssmichael1/tetra3rs/pull/45))
- `saturation_level` compared against the raw pixel value on the CCL path, not the residual ([#45](https://github.com/ssmichael1/tetra3rs/pull/45))
- Gaia binary loader tolerates trailing bytes again; `pattern_max_error` validation matches its documented `(0, 0.25]` range ([#45](https://github.com/ssmichael1/tetra3rs/pull/45))
- Python: `calibrate_camera` accepts `SolveFailure` items in its list; native-`f64` centroid arrays parse zero-copy again; typed exceptions instead of aborts; big-endian and non-`f64` arrays accepted; `attitude_hint` normalized/validated; type stubs reconciled with the bindings; `CatalogStar` pickles ([#45](https://github.com/ssmichael1/tetra3rs/pull/45))
- Robustness sweep: degenerate images / bad extraction config error instead of panic; NaN-safe medians; candidate-key enumeration capped; hash-probe walk bounded; `GenerateDatabaseConfig::validate`; Gaia header validated before allocating ([#45](https://github.com/ssmichael1/tetra3rs/pull/45))

## 0.8.0 - 2026-06-13

Binary databases and pickles saved by 0.7.x do not load (`SolveResult`, `RadialDistortion`, `CameraModel` wire formats changed) — regenerate and re-pickle.

### Added

- `extract_centroids_fast` / `FastCentroidConfig`: single-pass run-length + union-find extractor, ~4–5× faster than the CCL path on 2048² TESS frames with equal solve accuracy; drop-in `ExtractionResult` ([#41](https://github.com/ssmichael1/tetra3rs/pull/41))

### Changed

- **Breaking:** centroid origin moved to the geometric image center `(W−1)/2, (H−1)/2` (was `W/2, H/2`), matching FITS / astropy / astrometry.net / OpenCV; removes a ½-pixel bias against those tools (issue #28) ([#41](https://github.com/ssmichael1/tetra3rs/pull/41))
- **Breaking (Rust):** `SolveResult` is `Result<Solution, SolveFailure>` — `Solution` fields are non-`Option`, `SolveStatus::MatchFound` is gone, `pixel_to_world` is infallible; `SolveConfig` derives FOV / image size / pixel scale from `camera_model` (`fov_estimate_rad`, `image_width`, `image_height` fields removed; new `with_camera_model()`); dead `refine_iterations` removed (Rust field + Python kwarg) ([#40](https://github.com/ssmichael1/tetra3rs/pull/40))
- **Breaking (Python):** `solve_from_centroids` returns a falsy `SolveFailure` (with `status`, `solve_time_ms`) instead of `None`; `SolveResult` attributes are no longer `Optional`; when `camera_model=` is passed it is authoritative over `fov_estimate_*` ([bc4f4bb](https://github.com/ssmichael1/tetra3rs/commit/bc4f4bb)) ([#40](https://github.com/ssmichael1/tetra3rs/pull/40))
- Radial calibration rewritten as an OpenCV-style intrinsics fit: free optical center `(cx, cy)`, focal-scale `γ` folded into `focal_length_px`, Brown-Conrady `(k1, k2, k3, p1, p2)`; new `RadialDistortion::center` field + `with_center()` (breaking wire format); TESS pooled fit 6.3 → 1.8 px ([943994a](https://github.com/ssmichael1/tetra3rs/commit/943994a))
- Calibrated `focal_length_px` is tan-consistent with `CameraModel::from_fov`; multi-image calibration excludes parity-outlier solves ([943994a](https://github.com/ssmichael1/tetra3rs/commit/943994a))
- One verification and one refinement pipeline shared by lost-in-space and tracking; final attitude derived directly from `(θ, CRVAL)`; deduplicated solver helpers ([#40](https://github.com/ssmichael1/tetra3rs/pull/40))
- CI: wheel builds parallelized across OS × Python ([#39](https://github.com/ssmichael1/tetra3rs/pull/39))

### Fixed

- Parity-flipped solves returned a reflection (det −1) as the attitude — `qicrs2cam` non-unit, residual stats garbage, `cd_matrix` sign convention wrong; `rotation_from_theta_crval` no longer takes `parity_flip` ([a504aaf](https://github.com/ssmichael1/tetra3rs/commit/a504aaf))
- No panics on degenerate input: failed Wahba SVD skips the candidate, NaN residuals don't panic MAD statistics, empty pattern catalog returns `NoMatch`; FOV sweep continues past cluster-buster `TooFew`; verification cone sized by the true image diagonal (portrait images); tracking solves are aberration-consistent ([#40](https://github.com/ssmichael1/tetra3rs/pull/40))
