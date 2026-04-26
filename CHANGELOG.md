# Changelog

## 0.7.0 (unreleased)

### Breaking changes

- **Serialization format changed from rkyv to [postcard](https://docs.rs/postcard).**
  Existing `.rkyv` databases saved by 0.6.x or earlier will fail to load.
  Regenerate via `SolverDatabase::generate_from_gaia(...)` (Rust) or
  `generate_from_gaia(...)` (Python). The same applies to any `.rkyv`
  `CameraModel` files saved with `CameraModel::save_to_file`. First-use
  regeneration is automatic for Python users whose databases live in the
  `gaia-catalog` package cache.
- **File-extension convention changed from `.rkyv` to `.bin`** in docs and
  examples. Existing user files keep working under any extension; the
  rename is cosmetic.
- **Pickle format changed.** Pickled `tetra3rs` objects (`SolverDatabase`,
  `CameraModel`, `SolveResult`, `CalibrateResult`, `ExtractionResult`,
  `Centroid`, `RadialDistortion`, `PolynomialDistortion`) now round-trip
  through postcard. Pickles produced by 0.6.x will not unpickle.
- **`SolverDatabase::pattern_catalog` is now a flat `Vec<PatternEntry>`-backed
  `PatternCatalog`** — the 0.6.0 sharding workaround for rkyv's 2 GB
  relative-offset limit is gone (postcard has no such limit). Public
  access via `.get(idx)` / `.get_mut(idx)` / `.len()` / `.is_empty()` is
  unchanged; the `PatternCatalog::SHARD_SIZE` constant and the
  `shards: Vec<Vec<PatternEntry>>` field are removed.
- **`SolverDatabase::to_rkyv_bytes` renamed to `SolverDatabase::to_bytes`.**

### New features

- **Lighter dependency footprint.** Around 10 crates removed from the build
  (`rkyv`, `rkyv_derive`, `bytecheck`, `bytecheck_derive`, `munge`,
  `munge_macro`, `ptr_meta`, `ptr_meta_derive`, `rancor`, `rend`,
  `simdutf8`) in exchange for 3 (`postcard`, `serde`, `serde_derive`).
- **Database files are portable.** postcard has a published wire-format
  spec, so a database written on one platform can be read by any
  postcard implementation — no longer Rust-locked.

### Other changes

- **`numeris` bumped to 0.5.10** to pick up native serde support for
  `Matrix<T, M, N>` and `Quaternion<T>`. The 0.4 → 0.5 bump is a minor
  API change at call sites: `Matrix::vecmul(&v)` is now the `*` operator
  (`m * v`), and `DynMatrix::zeros(rows, cols, fill)` /
  `DynVector::zeros(n, fill)` lost their fill argument
  (`zeros(rows, cols)` / `zeros(n)`). All internal call sites are
  updated.
- **`csv` dependency dropped.** The Gaia DR3 CSV reader
  (`catalogs::gaia::read_gaia_csv`) is now a hand-rolled ~30-line parser
  for the fixed 9-column unquoted schema produced by
  `scripts/download_gaia_catalog.py`. The CSV path through
  `SolverDatabase::generate_from_gaia` continues to work as before, just
  without the `csv`/`csv-core`/`bstr`/`aho-corasick`/`regex-automata`
  transitive tree.
- **Latent bug fix in CSV catalog loading.** The previous `csv::Reader`
  call had a stray `.skip(1)` after the header was already auto-skipped,
  silently dropping the brightest star from any CSV-loaded Gaia catalog.
  The new parser only skips the header line, so CSV-loaded catalogs now
  correctly include all stars. Binary (`.bin`) catalogs were never
  affected.
- **Public Rust function `extract_centroids(path, &config)` removed.**
  This was a 5-line convenience wrapper around `image::open(path)?` +
  `extract_centroids_from_image(...)`. Removing it lets the `image` dep
  drop its `jpeg` / `png` / `tiff` format features, killing a sizeable
  decoder dep tree (`zune-jpeg`, `png`, `tiff`, `fdeflate`, `weezl`,
  `moxcms`, `byteorder-lite`, `half`, `color_quant`, …). Callers who
  want file-path convenience should decode the file themselves with
  whichever `image` feature flags they need, then call
  [`extract_centroids_from_image`]. The Python
  `tetra3rs.extract_centroids(numpy_array, ...)` API is unchanged — it
  goes through `extract_centroids_from_raw` and never used the file path.
- **Connected-component labelling delegated to
  [`numeris::imageproc`](https://docs.rs/numeris).** The hand-rolled
  two-pass union-find in `centroid_extraction.rs` (~110 lines) is
  replaced by `numeris::imageproc::connected_components_with_label_buffer`,
  which also supplies per-blob area and bounding box. The
  `imageproc` feature on numeris is activated only when tetra3's
  `image` feature is on (`image = ["dep:image", "numeris/imageproc"]`),
  so non-image users don't pay for it. Intensity-weighted centroiding
  with per-blob local-background annulus refinement remains in tetra3.
- **`tracing-subscriber` moved from `[dependencies]` to
  `[dev-dependencies]`.** Library code emits log events via the
  lightweight `tracing` macros; configuring a subscriber is the
  application's job, not the library's. Downstream crates depending on
  `tetra3` no longer compile `tracing-subscriber` and its ~10
  transitive crates (`matchers`, `regex-automata`, `regex-syntax`,
  `nu-ansi-term`, `sharded-slab`, `lazy_static`, `smallvec`,
  `thread_local`, `tracing-log`).
- **`anyhow` replaced with a typed `tetra3::Error` enum.** All public
  functions that previously returned `anyhow::Result<T>` now return
  `tetra3::Result<T>` (alias for `Result<T, tetra3::Error>`). The
  `Error` enum has four variants — `Io`, `Postcard`, `InvalidCatalog`,
  `InvalidInput` — letting callers `match` on specific failure modes
  instead of receiving an opaque trait object. Implements `Display`,
  `std::error::Error`, and `From<std::io::Error>` /
  `From<postcard::Error>`. Built with `thiserror` (already in the dep
  tree via `postcard → cobs`, so no net dep added). The `anyhow` crate
  is dropped.
- **`SolverDatabase::generate_from_star_list` is now infallible** —
  signature changed from `anyhow::Result<Self>` to `Self`. The function
  never had any error sources; the `Result` wrapper was vestigial.
- **CSV catalog input dropped from `generate_from_gaia`.** The function
  previously branched on file extension / magic bytes between `.csv`
  and `.bin` paths; the `.csv` branch is gone. Callers must supply a
  `.bin` Gaia binary catalog (the format the bundled `gaia-catalog`
  package ships). The `read_gaia_csv` helper and its in-tree CSV
  parser are removed.
- **Stale Python `.pyi` stubs cleaned up.** The type stubs previously
  advertised `SolverDatabase.fit_radial_distortion()` /
  `.fit_polynomial_distortion()` methods that had no matching PyO3
  bindings (the stubs had drifted). Removed. The Rust functions
  `tetra3::distortion::fit::fit_radial_distortion` and
  `tetra3::distortion::fit::fit_polynomial_distortion` remain
  available as standalone Rust API.
- **`calibrate_camera` is now model-agnostic.** It can fit either a SIP
  polynomial (the existing default) or a Brown-Conrady radial
  `(k1, k2, k3)` distortion model. Selected via
  `CalibrateConfig::model: DistortionModelType { Polynomial { order } | Radial }`.
  Both models go through the same alternating per-image
  WCS-refinement + global-fit pipeline, so multi-image radial
  calibration (the canonical OpenCV / Zhang's-method shape) is now
  available. Python: pass `model="radial"` (or `"polynomial"`, the
  default) to `SolverDatabase.calibrate_camera()`. Breaking: the
  Rust `CalibrateConfig::polynomial_order: u32` field is replaced by
  `model: DistortionModelType`; callers need to update construction
  to `CalibrateConfig { model: DistortionModelType::Polynomial { order: 4 }, .. }`.
- **Direct `num-traits` dep removed.** tetra3's own code never used
  `num_traits`; it was pulled in transitively via `numeris` regardless.
  Manifest cleanup only.

### Notes on performance

postcard is not zero-copy, so loading a database now does an actual
deserialization pass instead of validating an archived layout in place.
In practice this is a non-regression: prior versions also fully
deserialized via `rkyv::from_bytes` rather than using zero-copy
`rkyv::access`, so nothing in the existing code path was benefiting from
zero-copy. If you need true mmap-style loading for very large databases
in the future, that's a separate change to consider.

### Internal

- `src/rkyv_numeris.rs` removed entirely. With numeris 0.5.10's native
  serde implementations of `Matrix<T, M, N>` and `Quaternion<T>`,
  `Option<Matrix2<f32>>` and `Option<Quaternion<f32>>` fields derive
  `Serialize` / `Deserialize` directly — no adapter shims needed.

## 0.6.1

### Fixes

- **`import tetra3rs` no longer crashes when installed without package
  metadata (issue #22).** The module-level call to
  `importlib.metadata.version("tetra3rs")` raised `PackageNotFoundError`
  in environments where the package is on `sys.path` without a
  corresponding `.dist-info/` directory — e.g. Pi OS image builds that
  copy the source tree instead of running `pip install`. `__version__`
  now falls back to `"0.0.0+unknown"` when metadata is unavailable.
  Properly pip-installed users see the correct version string as before.

## 0.6.0

### Fixes

- **Multiscale databases no longer crash on `save_to_file` (issue #13).** Databases
  covering a wide range of field-of-view scales (e.g. 0.5°–5°) can generate hash
  tables larger than 2 GB, which overflowed rkyv's default 32-bit relative-offset
  limit and caused serialization to panic with *"out of range integral type
  conversion attempted"*. The pattern catalog is now stored as
  [`PatternCatalog`](https://docs.rs/tetra3/0.6.0/tetra3/solver/struct.PatternCatalog.html)
  — a sharded container that splits its backing storage into independently-archived
  chunks of up to ~770 MB each. Probe logic is unchanged in spirit (one additional
  L1-resident dereference per probe, effectively zero runtime cost).

### Breaking changes

- **`.rkyv` database file format bumped.** Existing cached `.rkyv` files saved
  with 0.5.x or earlier will fail to load under 0.6.0. Regenerate via
  `SolverDatabase::generate_from_gaia(...)` (Rust) or
  `generate_from_gaia(...)` (Python). First-use regeneration is automatic for
  Python users whose databases live in the `gaia-catalog` package cache.
- **`SolverDatabase::pattern_catalog` field type** changed from
  `Vec<PatternEntry>` to `PatternCatalog`. Access slots via `.get(idx)` /
  `.get_mut(idx)` rather than `[idx]`; the hash-probe loop in user code (if
  any — most users don't access this field directly) needs a one-line update.
- **Python: upper-bounded `gaia-catalog<1.0`.** The bundled Gaia binary
  catalog format is unchanged in 0.6.0 and still works with
  `gaia-catalog` 0.1.x. Adding this upper bound is a forward guard: if
  `gaia-catalog` ever ships a breaking binary-format change under a
  `1.0` release, this prevents it from silently being installed under a
  `tetra3rs 0.6.x` that wouldn't be able to read it. No-op for users
  today. Older `tetra3rs` releases don't pin `gaia-catalog` at all —
  we'd protect 0.5.x users similarly via a future `0.5.2` patch if a
  breaking `gaia-catalog` release ever lands.

## 0.5.1

### New features

- **Tracking-mode solving via attitude hint.** `SolveConfig` gains an optional `attitude_hint: Option<Quaternion>` (plus `hint_uncertainty_rad` and `strict_hint`). When set, the solver projects catalog stars near the hinted boresight, nearest-neighbor matches them to centroids, runs Wahba SVD, and reuses the existing verification + WCS refine path — skipping the 4-star pattern hash entirely. Succeeds with as few as 3 matched stars (lost-in-space needs 4), is robust to pattern-hash failures from sparse / low-SNR fields, and on failure falls back to lost-in-space unless `strict_hint` is set. Intended for video-rate star trackers chaining solves frame-to-frame.

### Fixes

- `SolveResult::camera_model` now has `image_width` / `image_height` populated from the input `SolveConfig`. Previously these were inherited from `config.camera_model`, so a solve config built with `..Default::default()` (without explicitly constructing a `CameraModel`) would leave them zero, breaking downstream code that consumes the result's camera model.

### Other

- `CLAUDE.md` added to the repo root (guidance for Claude Code sessions in this repo).

## 0.5.0

### Precision improvements

- **True-pinhole pixel scale throughout.** The solver previously used the small-angle approximation `pixel_scale = fov / image_width` internally while storing `focal_length_px = (W/2) / tan(fov/2)` (true pinhole) on the result. At finite FOV the two differ by ~0.5%, producing ~100″ residuals at field corners if downstream code mixed them. The internal pipeline (`solve.rs`, `wcs_refine.rs`, `SolveConfig::pixel_scale`, distortion calibration, synthetic test generators) now uses `1/f` everywhere.
- **Newton iteration for polynomial undistort.** `PolynomialDistortion::undistort` now solves the forward polynomial by Newton iteration (2-4 iterations to machine precision) instead of evaluating a separately-fit inverse polynomial. A finite-order inverse polynomial cannot perfectly invert a finite-order forward polynomial, and the resulting asymmetry error amplified at field corners under tight match radii. Newton is exact (limited only by forward polynomial expressiveness) and eliminates the asymmetry.
- **TESS multi-image calibration**: average agreement with FITS WCS dropped from 0.81″ to **0.42″** across 10 sectors, with every sector improved or equal. Sector 17 specifically: RMSE 5.25″ → 2.56″.

### Breaking changes

- **`wcs_to_rotation` return value** — the returned FOV is now the *angular* FOV `2·atan(ps·W/2)` rather than the linear `ps·W`. Matches the convention of `fov_estimate_rad` elsewhere. Affects any external code calling this function directly; internal callers are all updated.
- **Removed `term_pairs_range` / `num_coeffs_range`** from `distortion::polynomial`. These `pub` helpers were unused in-tree and had no external users we're aware of.
- **`PolynomialDistortion::{ap_coeffs, bp_coeffs}`** are retained in the struct for binary-format compatibility but are zero-valued in any model produced by this crate. `fit_inverse_poly_ls` removed.

### Other

- `SolveConfig::pixel_scale()` return value is now `1/f` (true pinhole) instead of `fov / W` (linear); the two differ by ~0.5% at 15° FOV.

## 0.4.1

### New features

- **CameraModel save/load.** Added `save_to_file()` and `load_from_file()` methods to `CameraModel` for persisting camera intrinsics (including distortion) to disk using rkyv serialization. Available in both Rust and Python — models saved from one language can be loaded in the other.

### Other

- Added Gaia DR3 and Hipparcos 2 catalog attribution to LICENSE.

## 0.4.0

### Breaking changes

- **Gaia DR3 is now the default (and always-included) star catalog.** The `gaia` feature flag has been removed; Gaia support is always compiled in. Hipparcos support is now behind an optional `hipparcos` feature flag.
- **`Star.id` changed from `u64` to `i64`** to support negative source IDs for Hipparcos gap-fill stars in the merged catalog. This affects `matched_catalog_ids` arrays (`np.int64` in Python) and `get_star_by_id()`.
- **`generate_from_hipparcos` removed from Python bindings.** Use `generate_from_gaia()` instead.
- **Python dependency changed** from `hipparcos-catalog` to `gaia-catalog`.

### New features

- **Gaia DR3 + Hipparcos merged catalog.** The merged catalog uses Gaia for most stars and fills in the brightest stars (G < 4) from Hipparcos where Gaia saturates. Hipparcos positions are propagated from J1991.25 to the Gaia epoch (J2016.0).
- **Compact binary catalog format (.bin).** A custom 36-byte-per-star binary format (header + packed structs) reduces catalog size from 77 MB (CSV) to 17 MB. `generate_from_gaia()` auto-detects CSV vs binary format.
- **`gaia-catalog` PyPI package.** The merged catalog is bundled as a lightweight Python package (~15 MB wheel). `generate_from_gaia()` with no arguments automatically uses the bundled catalog.
- **`generate_from_gaia()` accepts optional `catalog_path`.** When `None` (default), uses the bundled `gaia-catalog` package. Accepts both `.csv` and `.bin` files.
- **`scripts/download_gaia_catalog.py`** downloads Gaia DR3 via TAP, merges with Hipparcos 2, and outputs either CSV or binary format (determined by file extension).

### Improvements

- **Switched from `nalgebra` to `numeris`** for linear algebra. `numeris` is a lightweight, `no_std`-compatible pure-Rust library for matrix/vector/quaternion operations, reducing dependency weight and improving suitability for embedded targets.
- **TESS test reliability.** Changed `match_radius` from `0.01` to `0.005` across TESS tests, fixing a false-match failure on the sparse field image.
- Removed `zip` dev-dependency (tests now download `.bin` directly instead of extracting from `.zip`).

### Migration guide

**Rust:**

Download the pre-built merged catalog (~17 MB, 482k stars to G-band magnitude 10, Gaia DR3 + Hipparcos bright-star gap-fill):
```sh
curl -o data/gaia_merged.bin "https://storage.googleapis.com/tetra3rs-testvecs/gaia_merged.bin"
```

Or generate your own with a custom magnitude limit:
```sh
python scripts/download_gaia_catalog.py --mag-limit 12.0 --output data/gaia_merged.bin
```

```rust
// Before (0.3.x)
let db = SolverDatabase::generate_from_hipparcos("data/hip2.dat", &config)?;

// After (0.4.0)
let db = SolverDatabase::generate_from_gaia("data/gaia_merged.bin", &config)?;
```

**Python:**
```python
# Before (0.3.x)
db = tetra3rs.SolverDatabase.generate_from_hipparcos()

# After (0.4.0)
db = tetra3rs.SolverDatabase.generate_from_gaia()  # uses bundled gaia-catalog
```

## 0.3.2

Initial public release with Hipparcos catalog support, centroid extraction, camera model, distortion calibration, WCS output, and stellar aberration correction.
