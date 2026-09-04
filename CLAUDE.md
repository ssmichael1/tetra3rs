# CLAUDE.md

Guidance for Claude Code when working in this repository. Complements the workspace-level `~/CLAUDE.md`.

## What this project is

**tetra3rs** is a lost-in-space star plate solver. Given star centroids (pixel coords, origin at image center), it identifies them against a catalog and returns camera attitude as a quaternion — no prior estimate required.

- Rust crate published as `tetra3` on crates.io
- Python wheels published as `tetra3rs` on PyPI
- Current version: 0.4.1 (alpha)
- Based on the tetra3 / cedar-solve algorithms (ESA / Steven Rosenthal)

## Build & test

```sh
# Rust
cargo build --release
cargo test                                   # default features
cargo test --features image                  # integration tests (download test data on first run)
cargo test --test skyview_solve_test --features image -- --nocapture
cargo test --test tess_solve_test --features image -- --nocapture

# Multi-threaded centroid extraction (rayon). Parallelizes the dominant
# local-background stage + full-image maps; also turns on numeris's rayon
# imageproc (matched-filter blur). Bit-identical results; ~1.9x sparse /
# ~1.45x dense on 8 cores. Combine with `image`.
cargo build --release --features image,parallel

# Python (maturin via setuptools-rust)
pip install -e .

# Solver profiling (feature-gated leaf timers; zero-cost when off)
cargo run --release --features profile --example profile_solve            # 2000 solves
cargo run --release --features profile --example profile_solve -- 5000    # n trials
# Scenario knobs (env): T3_SPURIOUS=K appends K false centroids per field;
# T3_RANDOM=1 makes each field fully random (forces no-match enumeration).
```

`[profile.test]` uses `opt-level = 3`.

### Pre-push gate (fmt + clippy)

A committed hook (`.githooks/pre-push`) blocks pushes **to `main`** unless
`cargo fmt --all --check` and `cargo clippy --all-targets --all-features -D
warnings` both pass (other branches are not gated). Enable it once per clone:

```sh
git config core.hooksPath .githooks
```

Keep `main` fmt- and clippy-clean (`--all-features`, so `image`/`hipparcos`/
`profile`-gated code is linted too). `git push --no-verify` bypasses it for a
one-off.

The `profile` feature wires thread-local leaf timers into the lost-in-space
solve path (`solver::profiling`, bucket names in `profiling::buckets`). It is a
no-op without the feature. Scope: the pattern search + `wcs_refine` only (not
tracking or DB generation). Profiling shows `wcs_refine` dominates easy solves
(~90%), with its Phase-D re-association (catalog query + project + match) the
largest internal cost.

## Module layout (`src/`)

| Module | Responsibility |
|---|---|
| `solver/` | Staged solve pipeline: `preprocess.rs` (CRPIX/undistort/brightness order) → hypothesis source (`pattern_search.rs`: 4-star geometric hash + FOV sweep + Wahba SVD; or `track.rs`: attitude hint) → `verify.rs` (measured-density binomial FPR) → `solve.rs` (driver, LIS acceptance stage, refinement + `Solution` assembly) |
| `solver/clock.rs` | Cfg-switched `Instant`: `std` everywhere except `wasm32-unknown-unknown`, where it is `web_time::Instant` (`performance.now()`). All solver timing (`solve_timeout_ms`, `solve_time_ms`, `timed!`) goes through it |
| `solver/wcs_refine.rs` | Constrained 3-DOF refinement: rotation angle θ + CRVAL offset (dξ₀, dη₀). Pixel scale locked. Sigma-clipped |
| `camera_model/` | `CameraModel` struct: `focal_length_px`, `image_width`, `image_height`, `crpix`, `parity_flip`, `distortion`. Methods `fov_deg()` / `fov_rad()`, `pixel_to_tanplane` / `tanplane_to_pixel` |
| `distortion/` | SIP polynomial + radial distortion; `calibrate_camera` for multi-image fits |
| `distortion/calibrate.rs` | Branches on `n_valid` solves: 1 image → pooled fit with the solve's own attitude; 2+ → alternating per-image WCS refine + global fit (3 outer iterations). Both call `fit::fit_pooled` |
| `distortion/fit.rs` | `fit_pooled` (polynomial / radial dispatch, before/after RMSE); `fit_polynomial_sigma_clip`, `fit_radial_centered_sigma_clip` loops |
| `distortion/polynomial.rs` | `term_pairs_range(min_order, max_order)` — SIP convention (order ≥ 2) |
| `centroid_extraction/` (feature `image`) | Local background subtraction, CCL, quadratic sub-pixel peak refinement |
| `centroid/`, `star/` | Data types |
| `starcatalog/` | Catalog management, proper-motion propagation |
| `catalogs/` | Gaia DR3 (+ optional Hipparcos). Merged catalog: Gaia primary, Hipparcos for G<4 |
| `aberration/` | First-order stellar aberration (~20″); `earth_barycentric_velocity` convenience |

Public re-exports in `src/lib.rs` — `CameraModel`, `SolveConfig`, `SolveResult`, `SolverDatabase`, `calibrate_camera`, `earth_barycentric_velocity`, etc.

## Conventions

- **Camera frame**: +X right, +Y down, +Z boresight (right-handed)
- **Centroids**: pixel coordinates, **origin at image center** (not top-left)
- **Float precision**: f32 throughout; f64 only in final SVD step (insufficient accuracy at f32)
- **Quaternions**: `numeris::Quaternion<f32>` (re-exported as `Quaternion`)
- **Pipeline**: pixel → subtract CRPIX → undistort → parity flip → divide by f → tangent plane → hash → SVD → verify → refine → WCS fit
- **Parity**: detected via determinant of initial rotation; when negative, negates centroid x before re-solving. `SolveResult.parity_flip` records this
- **Aberration**: optional, via `SolveConfig::observer_velocity_km_s`. Shifts solved pointing by up to 20″ — a bias correction, not a residual reduction

## Python bindings (`python/`)

- PyO3 0.28, setuptools-rust build backend
- `crate-type = cdylib`, depends on root crate with `image` feature
- All public types pickle via postcard: `SolverDatabase`, `CameraModel`, `SolveResult`, `CalibrateResult`, `ExtractionResult`, `Centroid`, `RadialDistortion`, `PolynomialDistortion`
- Gaia catalog bundled via the `gaia-catalog` PyPI package — no manual download needed
- Wheels: cibuildwheel, cp310–cp314, skips i686/musllinux

## Releasing (version bump — read before tagging)

A release version lives in **three** files that must be bumped in lockstep, or
publishing breaks in confusing ways:

| File | Field | Drives |
|---|---|---|
| `Cargo.toml` | `[package] version` | the `tetra3` **crates.io** crate |
| **`pyproject.toml`** (repo root) | `[project] version` | the `tetra3rs` **PyPI** wheel name (setuptools-rust reads this, **not** Cargo.toml) |
| `python/Cargo.toml` | `[package] version` | the PyO3 extension crate (`publish = false`); keep in sync |

⚠️ The easy trap (hit in 0.7.2): bumping `Cargo.toml` but **not** root
`pyproject.toml`. The crates.io publish succeeds while wheels build under the
*old* version and the PyPI upload fails with "file already exists." Always grep
`grep -rn '<old-version>' Cargo.toml pyproject.toml python/Cargo.toml` and
update `CHANGELOG.md` before tagging.

### Changelog convention

`CHANGELOG.md` is kept concise (same style as `../satkit`): only recent
releases, `### Added` / `### Changed` / `### Fixed` subsections, **one line
per item** ending in a PR link (`([#N](https://github.com/ssmichael1/tetra3rs/pull/N))`),
`**Breaking:**` prefix where applicable, no paragraphs. The full detail — root
cause, failure scenario, before/after metrics, tests — goes in the **PR
description**, not the changelog. Older releases live in git history
(`git show vX.Y.Z:CHANGELOG.md`).

Publishing:
- **crates.io (Rust):** manual — `cargo publish -p tetra3` (no CI automation).
- **PyPI (Python):** automated by `.github/workflows/publish.yml` on a `v*` tag
  push (`git tag vX.Y.Z && git push origin vX.Y.Z`). The tagged commit must
  already carry all three version bumps.

## Tests

| Test | FOV | Status |
|---|---|---|
| Unit (`cargo test`) | — | 40/40 passing (1 ignored) |
| `tests/integration_test.rs` | — | 4/4 (1000 noiseless + 1000 noisy + basic + save/load) |
| `tests/skyview_solve_test.rs` | 10° | 10/10 — synthetic NASA SkyView FITS, simple CDELT WCS |
| `tests/tess_solve_test.rs` | ~12° | 3-image basic (<30′), 3-image distortion fit (<1′), 10-image multi-sector calibration (RMSE <9″, FITS WCS agreement <3″) |

TESS images have significant optical distortion and SIP WCS. Science region trimmed from 2136×2078 → 2048×2048.

## Serialization & trust boundaries

All persistence is **postcard + serde** (no rkyv anywhere, despite older notes
elsewhere): `SolverDatabase` and `CameraModel` files, and every Python pickle.
Postcard parses untrusted bytes safely, but structure ≠ semantics — cross-field
invariants are enforced by `validate()` methods (`SolverDatabase`,
`StarCatalog`, `CameraModel`, `Distortion`/`RadialDistortion`/
`PolynomialDistortion`), called at every load/pickle boundary. When adding a
field whose value other code indexes by or divides by, extend the owning
`validate()`.

## Data assets (`data/`)

Downloaded on first integration-test run from GCS (`tetra3rs-testvecs` bucket). Not in git.

- `hipsolver_10_30.bin` (455M) — main solver database
- `test_tess_db.bin` (144M), `test_skyview_db.bin` (38M)
- `gaia_merged.bin` (17M) — binary Gaia+Hipparcos catalog
- `gaia_merged.csv` (81M) — CSV form (used by `scripts/`; no in-tree Rust CSV parser — the crate loads only the `.bin` GDR3 format)
- FITS test images (skyview, TESS). TIFF/JPEG variants exist in the bucket but aren't exercised in CI — the library no longer ships file-format decoders, so callers must decode files themselves before calling `extract_centroids_from_image`.

## Scripts (`scripts/`)

- `download_gaia_catalog.py` — fetch Gaia DR3 with custom magnitude limit
- `download_hip2.sh` — Hipparcos catalog
- `check-notebook-outputs.py` — validate tutorial notebooks

## Docs

`docs/` + `mkdocs.yml`, published at https://tetra3rs.dev/. Material theme, MathJax, mkdocstrings (Python API), mkdocs-jupyter (executes tutorial notebooks). Structure: getting-started, concepts (algorithm, coordinates, camera model, WCS, aberration), tutorials, API reference.

Rust API docs at https://docs.rs/tetra3.

## When adding features

- Exposing a new Rust type to Python: update `python/src/*.rs` wrapper, implement pickle (`__getstate__`/`__setstate__`) serializing **all** fields, re-export from `python/src/lib.rs`
- New `SolveConfig` / `SolveResult` field: update the Python wrapper struct, pickle impl, and type stubs
- Avoid `_ =>` catch-alls in matches on public enums — let the compiler flag new variants
