# tetra3rs

[![Crates.io](https://img.shields.io/crates/v/tetra3)](https://crates.io/crates/tetra3)
[![PyPI](https://img.shields.io/pypi/v/tetra3rs)](https://pypi.org/project/tetra3rs/)
[![docs.rs](https://img.shields.io/docsrs/tetra3)](https://docs.rs/tetra3)
[![Docs](https://img.shields.io/badge/docs-guide-blue)](https://tetra3rs.dev/)
[![License](https://img.shields.io/crates/l/tetra3)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange)]()

A fast, robust lost-in-space star plate solver written in Rust.

Given a set of star centroids extracted from a camera image, tetra3rs identifies the stars against a catalog and returns the camera's pointing direction as a quaternion — no prior attitude estimate required. The goal is to make this fast and robust enough for use in embedded systems such as star trackers on satellites.

**Documentation:** For tutorials, concept guides, and Python API reference, see the [tetra3rs documentation](https://tetra3rs.dev/). For Rust API docs, see [docs.rs](https://docs.rs/tetra3).

> [!IMPORTANT]
> **Status: Alpha** — The core solver is based on well-vetted algorithms but has only been tested against a limited set of images. The API is not yet stable and may change between releases.  Having said that, I've made it work on both low-SNR images taken with a camera in my backyard and with high-star-density images from more-complex telescopes.

> [!WARNING]
> **0.6.0 is a breaking release.** The `.rkyv` solver database file format changed (sharded pattern catalog to support multiscale databases > 2 GB), and the Rust `SolverDatabase::pattern_catalog` field type changed from `Vec<PatternEntry>` to `PatternCatalog`. `.rkyv` files written by 0.5.x or earlier will not load under 0.6.0 — regenerate via `generate_from_gaia`. See [CHANGELOG.md](CHANGELOG.md) for the full list.


## Features

- **Lost-in-space solving** — determines attitude from star patterns with no initial guess
- **Tracking mode** — when an attitude hint is available (e.g. the previous frame's solution), skip the 4-star pattern-hash phase and match centroids directly against catalog stars near the hinted boresight. Succeeds with as few as 3 stars, robust to sparse/low-SNR fields, with automatic fallback to lost-in-space if the hint is stale
- **Fast** — geometric hashing of 4-star patterns with breadth-first (brightest-first) search
- **Robust** — statistical verification via binomial false-positive probability
- **Multiscale** — supports a range of field-of-view scales in a single database
- **Proper motion** — propagates catalog star positions to any observation epoch
- **Zero-copy deserialization** — databases serialize with [rkyv](https://github.com/rkyv/rkyv) for instant loading. The pattern catalog is sharded so databases of any size (including wide-FOV-range multiscale databases that exceed 2 GB) can be saved and loaded safely
- **Centroid extraction** — detect stars from images with local background subtraction, connected-component labeling, and quadratic sub-pixel peak refinement (requires `image` feature)
- **Camera model** — unified intrinsics struct (focal length, optical center, parity, distortion) used throughout the pipeline
- **Distortion calibration** — fit SIP polynomial or radial distortion models from one or more solved images via `calibrate_camera`
- **WCS output** — solve results include FITS-standard WCS fields (CD matrix, CRVAL) and pixel↔sky coordinate conversion methods
- **Stellar aberration** — optional correction for the ~20" apparent shift in star positions caused by the observer's barycentric velocity, with a built-in convenience function for Earth's barycentric velocity

## Installation

### Rust

The crate is published on [crates.io](https://crates.io/crates/tetra3) as `tetra3`:

```sh
cargo add tetra3
```

### Python

Binary wheels are available on [PyPI](https://pypi.org/project/tetra3rs/) for Linux (x86_64, ARM64), macOS (ARM64), and Windows (x86_64):

```sh
pip install tetra3rs
```

To build from source (requires a Rust toolchain):

```sh
pip install .
```



> [!NOTE]
> All Python objects (`SolverDatabase`, `CameraModel`, `SolveResult`, `CalibrateResult`, `ExtractionResult`, `Centroid`, `RadialDistortion`, `PolynomialDistortion`) support `pickle` serialization via zero-copy [rkyv](https://github.com/rkyv/rkyv).

## Quick start

### Star catalog

tetra3rs uses a merged Gaia DR3 + Hipparcos catalog. The merged catalog uses Gaia for most stars and fills in the brightest stars (G < 4) from Hipparcos where Gaia saturates.

**Python:** The catalog is bundled in the [`gaia-catalog`](https://pypi.org/project/gaia-catalog/) package (installed automatically with `tetra3rs`). No manual download needed — just call `generate_from_gaia()` with no arguments.

**Rust:** Download the pre-built binary catalog:

```sh
mkdir -p data
curl -o data/gaia_merged.bin "https://storage.googleapis.com/tetra3rs-testvecs/gaia_merged.bin"
```

Or generate your own with a custom magnitude limit using `scripts/download_gaia_catalog.py`.

### Example

```rust
use tetra3::{GenerateDatabaseConfig, SolverDatabase, SolveConfig, Centroid, SolveStatus};

// Generate a database from the Gaia catalog
let config = GenerateDatabaseConfig {
    max_fov_deg: 20.0,
    epoch_proper_motion_year: Some(2025.0),
    ..Default::default()
};
let db = SolverDatabase::generate_from_gaia("data/gaia_merged.bin", &config)?;

// Save the database to disk for fast loading later
db.save_to_file("data/my_database.rkyv")?;

// ... or load a previously saved database
let db = SolverDatabase::load_from_file("data/my_database.rkyv")?;

// Solve from image centroids (pixel coordinates, origin at image center)
let centroids = vec![
    Centroid { x: 100.0, y: 200.0, mass: Some(50.0), cov: None },
    Centroid { x: -50.0, y: -10.0, mass: Some(45.0), cov: None },
    // ...
];

let solve_config = SolveConfig {
    fov_estimate_rad: (15.0_f32).to_radians(), // horizontal FOV
    image_width: 1024,
    image_height: 1024,
    fov_max_error_rad: Some((2.0_f32).to_radians()),
    ..Default::default()
};

let result = db.solve_from_centroids(&centroids, &solve_config);
if result.status == SolveStatus::MatchFound {
    let q = result.qicrs2cam.unwrap();
    println!("Attitude: {q}");
    println!("Matched {} stars in {:.1} ms",
        result.num_matches.unwrap(), result.solve_time_ms);
}
```

## Algorithm overview

1. **Pattern generation** — select combinations of 4 bright centroids; compute 6 pairwise angular separations and normalize into 5 edge ratios (a geometric invariant)
2. **Hash lookup** — quantize the edge ratios into a key and probe a precomputed hash table for matching catalog patterns
3. **Attitude estimation** — solve Wahba's problem via SVD to find the rotation from catalog (ICRS) to camera frame
4. **Verification** — project nearby catalog stars into the camera frame, count matches, and accept only if the false-positive probability (binomial CDF) is below threshold
5. **Refinement** — re-estimate the rotation using all matched star pairs via iterative SVD passes
6. **WCS fit** — constrained 3-DOF tangent-plane refinement (rotation angle θ + CRVAL offset) with sigma-clipping, producing FITS-standard WCS output (CD matrix, CRVAL)

### Parity flip detection

Some imaging systems produce mirror-reflected images (e.g. FITS files with `CDELT1 < 0`, or optics with an odd number of reflections). In these cases the initial rotation estimate yields a reflection (determinant < 0) rather than a proper rotation. The solver detects this by checking the determinant of the rotation matrix; when negative, it negates the x-coordinates of all centroid vectors and recomputes the rotation.

The `SolveResult` includes a `parity_flip` flag (`bool` / `True`/`False` in Python) indicating whether this correction was applied. This is critical for pixel↔sky coordinate conversions: when `parity_flip` is `True`, the mapping between pixel x-coordinates and camera-frame x must include a sign flip.

### Tracking mode

When you already have a rough attitude estimate — typically the previous frame's solution in a video-rate star tracker, a propagated gyro estimate, or a coarse attitude sensor — you can skip the lost-in-space pattern-hash phase entirely by passing an `attitude_hint`:

**Rust:**

```rust
use tetra3::SolveConfig;

// Reuse the camera model from the previous solve so the refined focal length carries over.
let config = SolveConfig {
    attitude_hint: prev_result.qicrs2cam,
    hint_uncertainty_rad: 1.0_f32.to_radians(),
    camera_model: prev_result.camera_model.clone().unwrap(),
    ..SolveConfig::new((15.0_f32).to_radians(), 1024, 1024)
};
let result = db.solve_from_centroids(&centroids, &config);
```

**Python:**

```python
# `attitude_hint` accepts either a 4-element [w, x, y, z] quaternion
# (Hamilton, scalar-first) or a 3×3 rotation matrix.
result = db.solve_from_centroids(
    centroids,
    fov_estimate_deg=15.0,
    image_shape=(1024, 1024),
    camera_model=prev_result.camera_model,
    attitude_hint=prev_result.quaternion,  # or .rotation_matrix_icrs_to_camera
    hint_uncertainty_deg=1.0,
)
```

The solver projects catalog stars near the hinted boresight, nearest-neighbor matches them to centroids, and runs the same Wahba SVD + verification + WCS refine path as lost-in-space. Tracking succeeds with as few as 3 matched stars (lost-in-space needs 4) and is robust to pattern-hash failures from sparse / low-SNR fields. On failure it falls back to lost-in-space automatically — set `strict_hint=True` (`strict_hint: true` in Rust) to opt out of the fallback.

See [`docs/concepts/tracking.md`](https://tetra3rs.dev/concepts/tracking/) for details on hint uncertainty, quaternion convention, and limitations.

### Stellar aberration correction

Stellar aberration is the apparent displacement of star positions caused by the finite speed of light combined with the observer's velocity — analogous to how rain appears to fall at an angle when you're moving. For Earth-based observers, this shifts apparent star positions by up to ~20" (v/c ≈ 10⁻⁴ rad). Without correction, the solved attitude is biased by up to ~20".

To correct for aberration, pass the observer's barycentric velocity (ICRS, km/s) via `SolveConfig::observer_velocity_km_s`. The solver applies a first-order correction (s' = s + β − s(s·β)) to all catalog star vectors before matching and refinement, producing an unbiased attitude.

The convenience function `earth_barycentric_velocity()` provides an approximate Earth velocity using a circular-orbit model (~0.5 km/s accuracy, sufficient for the ~20" effect):

> [!NOTE]
> Enabling aberration correction shifts the entire solved pointing by up to ~20", not just the within-field residuals. This is the physically correct result — without it, the reported attitude is biased by the observer's velocity. Most plate solvers (e.g. [astrometry.net](https://astrometry.net/)) do not account for aberration, so comparing results may show a systematic offset of up to ~20" when this correction is enabled.


> [!NOTE]
> For a near-Earth observer, Earth's orbital motion around the Sun (~30 km/s) dominates stellar aberration, producing ~20″ shifts. Earth's surface-rotation contribution (~0.46 km/s at the equator) is only ~0.3″ and can usually be neglected. **LEO orbital velocity (~7.5 km/s) is ~25% of Earth's orbital velocity and produces ~5″ additional direction error** — not negligible for star trackers on LEO spacecraft. Pass the observer's full barycentric velocity (Earth-around-Sun + spacecraft-around-Earth + ground-based surface rotation, as appropriate) to get an unbiased attitude.

**Rust:**

```rust
use tetra3::{earth_barycentric_velocity, SolveConfig};

// days since J2000.0 (2000 Jan 1 12:00 TT)
let v = earth_barycentric_velocity(9321.0);

let config = SolveConfig {
    observer_velocity_km_s: Some(v),
    ..SolveConfig::new((10.0_f32).to_radians(), 1024, 1024)
};
```

**Python:**

```python
from datetime import datetime
import tetra3rs

v = tetra3rs.earth_barycentric_velocity(datetime(2025, 7, 10))
result = db.solve_from_centroids(
    centroids,
    fov_estimate_deg=10.0,
    image_shape=img.shape,
    observer_velocity_km_s=v,
)
```

## Catalog support

| Catalog | Format | Notes |
|---------|--------|-------|
| Gaia DR3 + Hipparcos | `.bin` (binary) or `.csv` | Default; merged catalog with proper motion. Binary format bundled in [`gaia-catalog`](https://pypi.org/project/gaia-catalog/) PyPI package |
| Hipparcos only | `hip2.dat` | Legacy; requires `hipparcos` feature flag |

## Tests

Unit tests run with the default feature set:

```sh
cargo test
```

Integration tests require the `image` feature and test data files. Test data is automatically downloaded from Google Cloud Storage on first run and cached in `data/`:

```sh
cargo test --features image
```

### SkyView integration test

Solves 10 synthetic star field images (10° FOV) generated from NASA's [SkyView](https://skyview.gsfc.nasa.gov/) virtual observatory, which composites archival survey data into FITS images at any sky position. These use simple CDELT WCS (orthogonal, uniform pixel scale). Each image is solved and the resulting RA/Dec/Roll is compared against the FITS header WCS.

```sh
cargo test --test skyview_solve_test --features image -- --nocapture
```

### TESS integration test

Solves Full Frame Images (~12° FOV) from NASA's [TESS](https://tess.mit.edu/) (Transiting Exoplanet Survey Satellite), a space telescope that images large swaths of sky to detect exoplanets via stellar transits. TESS images have significant optical distortion and use CD-matrix WCS with SIP polynomial corrections. The science region is trimmed from the raw 2136×2078 frame to 2048×2048 before centroid extraction.

The test suite includes:

- **3-image basic solve** — solves each image and verifies the boresight is within 30' of the FITS WCS solution.
- **3-image distortion fit** — fits a 4th-order polynomial distortion model from each solved image, re-solves, and verifies the center pixel RA/Dec is within 1' of the FITS WCS solution.
- **10-image multi-image calibration** — solves 10 images from the same CCD (Camera 1, CCD 1) across different sectors with 4 tiered solve+calibrate passes (progressively tighter match radius and higher polynomial order). After calibration, all 10 images achieve RMSE < 9" and center pixel agreement with FITS WCS < 3".

```sh
cargo test --test tess_solve_test --features image -- --nocapture
```

## Roadmap (not in order)

- **Deeper Gaia catalog** — support fainter limiting magnitudes for narrow-FOV cameras

## Credits

This project is based upon the **tetra3** / **cedar-solve** algorithms.

- **[cedar-solve](https://github.com/smroid/cedar-solve)** — Steven Rosenthal's Python plate solver, which this implementation closely follows for the star quad generation and matching.  (excellent work!)
- **[tetra3](https://github.com/esa/tetra3)** — the original Python implementation by Gustav Pettersson at ESA
- **Paper**: G. Pettersson, "Tetra3: a fast and robust star identification algorithm," ESA GNC Conference, 2023

## License

MIT License. See [LICENSE](LICENSE) for details.

This project is a derivative of [tetra3](https://github.com/esa/tetra3) and [cedar-solve](https://github.com/smroid/cedar-solve), both licensed under Apache 2.0 (which in turn derive from [Tetra](https://github.com/brownj4/Tetra) by brownj4, MIT licensed). The upstream license notices are included in the LICENSE file.
