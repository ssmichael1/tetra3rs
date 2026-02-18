# tetra3rs

[![Crates.io](https://img.shields.io/crates/v/tetra3)](https://crates.io/crates/tetra3)
[![docs.rs](https://img.shields.io/docsrs/tetra3)](https://docs.rs/tetra3)
[![License](https://img.shields.io/crates/l/tetra3)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange)]()

A fast, robust lost-in-space star plate solver written in Rust.

> **Status: Alpha** — The core solver is based on well-vetted algorithms but has only been tested against a limited set of images. The API is not yet stable and may change between releases.  Having said that, I've made it work on both low-SNR images taken with a camera in my backyard and with high-star-density images from more-complex telescopes.

Given a set of star centroids extracted from a camera image, tetra3rs identifies the stars against a catalog and returns the camera's pointing direction as a quaternion — no prior attitude estimate required. The goal is to make this fast and robust enough for use in embedded systems such as star trackers on satellites.

## Features

- **Lost-in-space solving** — determines attitude from star patterns with no initial guess
- **Fast** — geometric hashing of 4-star patterns with breadth-first (brightest-first) search
- **Robust** — statistical verification via binomial false-positive probability
- **Multiscale** — supports a range of field-of-view scales in a single database
- **Proper motion** — propagates Hipparcos catalog positions to any observation epoch
- **Zero-copy deserialization** — databases serialize with [rkyv](https://github.com/rkyv/rkyv) for instant loading

## Installation

### Rust

The crate is published on [crates.io](https://crates.io/crates/tetra3) as `tetra3`:

```sh
cargo add tetra3
```

### Python

Python bindings are available via [PyO3](https://pyo3.rs/) in the `python/` directory. The package is not yet on PyPI; install from source using [maturin](https://www.maturin.rs/):

```sh
cd python
pip install maturin
maturin develop --release
```

This builds and installs the `tetra3rs` Python module into your current environment.

## Quick start

### Obtaining the Hipparcos catalog

Download `hip2.dat` from the [Hipparcos, the New Reduction (I/311)](http://cdsarc.u-strasbg.fr/ftp/I/311/) and place it at `data/hip2.dat`:

```sh
mkdir -p data
curl -o data/hip2.dat.gz "http://cdsarc.u-strasbg.fr/ftp/I/311/hip2.dat.gz"
gunzip data/hip2.dat.gz
```

### Example

```rust
use tetra3::{GenerateDatabaseConfig, SolverDatabase, SolveConfig, Centroid, SolveStatus};

// Generate a database from the Hipparcos catalog
let config = GenerateDatabaseConfig {
    max_fov_deg: 20.0,
    epoch_proper_motion_year: Some(2025.0),
    ..Default::default()
};
let db = SolverDatabase::generate_from_hipparcos("data/hip2.dat", &config)?;

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
    let q = result.quaternion.unwrap();
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
5. **Refinement** — re-estimate the rotation using all matched star pairs

## Catalog support

| Catalog | File | Notes |
|---------|------|-------|
| Hipparcos | `data/hip2.dat` | Default; includes proper motion |
| Gaia | `data/gaia_bright_stars.csv` | Requires `--features gaia` (incomplete) |

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

Solves 3 Full Frame Images (~12° FOV) from NASA's [TESS](https://tess.mit.edu/) (Transiting Exoplanet Survey Satellite), a space telescope that images large swaths of sky to detect exoplanets via stellar transits. TESS images have significant optical distortion and use CD-matrix WCS with SIP polynomial corrections. The science region is trimmed from the raw 2136×2078 frame to 2048×2048 before centroid extraction.

The solved boresight is compared against the true boresight computed from the full WCS (CRPIX, SIP, CD matrix, TAN deprojection) at the center of the science region. Because the solver assumes a perfect pinhole projection while TESS has up to ~65 px of SIP distortion at the corners, the boresight error is typically 1-3 arcminutes and the RMSE is ~3-4 arcminutes. This is a known limitation of the pinhole model on wide-field distorted optics; see the Roadmap for planned distortion correction support.

```sh
cargo test --test tess_solve_test --features image -- --nocapture
```

## Roadmap (not in order)

- **Tracking mode** — accept an initial attitude guess to restrict the search to nearby catalog stars, improving speed and robustness for sequential frames (e.g. star trackers solution on previous frame)
- **Image distortion estimation and correction** — the solver currently assumes a perfect pinhole (gnomonic) projection; cameras with significant optical distortion (e.g. TESS, wide-angle lenses) produce ~1-5' boresight error and elevated RMSE
- **Stellar aberration** — correct for the apparent shift in star positions caused by the observer's velocity (up to ~20" for Earth-orbiting spacecraft)
- **Gaia catalog support** — complete the Gaia bright star catalog import (`--features gaia`)
- **Tycho-2 catalog support** — import the Tycho-2 catalog (~2.5 million stars, fills the gap between Hipparcos and Gaia)

## Credits

This project is a Rust implementation of the **tetra3** / **cedar-solve** algorithm.

- **[cedar-solve](https://github.com/smroid/cedar-solve)** — Steven Rosenthal's Python plate solver, which this implementation closely follows (excellent work!)
- **[tetra3](https://github.com/esa/tetra3)** — the original Python implementation by Gustav Pettersson at ESA
- **Paper**: G. Pettersson, "Tetra3: a fast and robust star identification algorithm," ESA GNC Conference, 2023

## License

MIT License. See [LICENSE](LICENSE) for details.

This project is a derivative of [tetra3](https://github.com/esa/tetra3) and [cedar-solve](https://github.com/smroid/cedar-solve), both licensed under Apache 2.0 (which in turn derive from [Tetra](https://github.com/brownj4/Tetra) by brownj4, MIT licensed). The upstream license notices are included in the LICENSE file.
