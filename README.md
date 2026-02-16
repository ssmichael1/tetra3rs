# tetra3-rs

A fast, robust lost-in-space star plate solver written in Rust.

Given a set of star centroids extracted from a camera image, tetra3-rs identifies the stars against a catalog and returns the camera's pointing direction as a quaternion — no prior attitude estimate required.

## Features

- **Lost-in-space solving** — determines attitude from star patterns with no initial guess
- **Fast** — geometric hashing of 4-star patterns with breadth-first (brightest-first) search
- **Robust** — statistical verification via binomial false-positive probability
- **Multiscale** — supports a range of field-of-view scales in a single database
- **Proper motion** — propagates Hipparcos catalog positions to any observation epoch
- **Zero-copy deserialization** — databases serialize with [rkyv](https://github.com/rkyv/rkyv) for instant loading

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

// Solve from image centroids (positions in radians from boresight)
let centroids = vec![
    Centroid { x: 0.01, y: 0.02, mass: Some(50.0), cov: None },
    Centroid { x: 0.05, y: -0.01, mass: Some(45.0), cov: None },
    // ...
];

let solve_config = SolveConfig {
    fov_estimate: (15.0_f32).to_radians(),
    fov_max_error: Some((2.0_f32).to_radians()),
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
| Gaia | `data/gaia_bright_stars.csv` | Requires `--features gaia` |

## Credits

This project is a Rust implementation of the **tetra3** / **cedar-solve** algorithm.

- **[cedar-solve](https://github.com/smroid/cedar-solve)** — Steven Rosenthal's C++/Rust star plate solver, which this implementation closely follows
- **[tetra3](https://github.com/esa/tetra3)** — the original Python implementation by Gustav Pettersson at ESA
- **Paper**: G. Pettersson, "Tetra3: a fast and robust star identification algorithm," ESA GNC Conference, 2023

## License

MIT License. See [LICENSE](LICENSE) for details.

This project is a derivative of [tetra3](https://github.com/esa/tetra3) and [cedar-solve](https://github.com/smroid/cedar-solve), both licensed under Apache 2.0 (which in turn derive from [Tetra](https://github.com/brownj4/Tetra) by brownj4, MIT licensed). The upstream license notices are included in the LICENSE file.
