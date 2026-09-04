# Quick Start

## Python

### Generate a database

```python
import tetra3rs

# Generate from the bundled Gaia catalog (installed with tetra3rs)
db = tetra3rs.SolverDatabase.generate_from_gaia(
    max_fov_deg=20.0,
    epoch_proper_motion_year=2025.0,
)

# Save for fast loading later
db.save_to_file("my_database.bin")

# ... or load a previously saved database
db = tetra3rs.SolverDatabase.load_from_file("my_database.bin")
```

Saved databases start with a 6-byte header (`"T3DB"` plus a format version);
files written by 0.12 and earlier have no header and still load. Every load
validates the file's cross-field invariants, so a corrupt or tampered file is
rejected with an error rather than failing mid-solve.

### Solve from centroids

```python
# Solve from a list of centroids
result = db.solve_from_centroids(
    centroids,
    fov_estimate_deg=10.0,
    image_width=2048,
    image_height=1536,
)

if result:
    print(f"RA: {result.ra_deg:.4f}°, Dec: {result.dec_deg:.4f}°")
    print(f"Roll: {result.roll_deg:.2f}°")
    print(f"Matched {result.num_matches} stars, RMSE: {result.rmse_arcsec:.1f}\"")
else:
    # A falsy SolveFailure: status is 'no_match', 'timeout', or 'too_few'
    print(f"Solve failed: {result.status} after {result.solve_time_ms:.0f} ms")
```

### Extract centroids from an image

```python
import numpy as np
from astropy.io import fits

# Load a FITS image
hdul = fits.open("my_image.fits")
image = hdul[0].data.astype(np.float64)

# Extract star centroids
extraction = tetra3rs.extract_centroids(image, sigma_threshold=5.0)
print(f"Found {len(extraction.centroids)} stars")

# Solve
result = db.solve_from_centroids(
    extraction.centroids,
    fov_estimate_deg=10.0,
    image_shape=image.shape,
)
```

### Pixel ↔ sky coordinate conversion

```python
if result:
    # Pixel to sky
    ra, dec = result.pixel_to_world(100.0, 200.0)

    # Sky to pixel
    x, y = result.world_to_pixel(180.0, 45.0)

    # Vectorized with numpy arrays
    import numpy as np
    xs = np.array([0.0, 100.0, -100.0])
    ys = np.array([0.0, 50.0, -50.0])
    ras, decs = result.pixel_to_world(xs, ys)
```

## Rust

```rust
use tetra3::{GenerateDatabaseConfig, SolverDatabase, SolveConfig, Centroid};

// Generate a database from the Gaia catalog
let config = GenerateDatabaseConfig {
    max_fov_deg: 20.0,
    epoch_proper_motion_year: Some(2025.0),
    ..Default::default()
};
let db = SolverDatabase::generate_from_gaia("data/gaia_merged.bin", &config)?;

// Save and reload
db.save_to_file("data/my_database.bin")?;
let db = SolverDatabase::load_from_file("data/my_database.bin")?;

// Solve from image centroids (pixel coordinates, origin at image center)
let centroids = vec![
    Centroid { x: 100.0, y: 200.0, mass: Some(50.0), cov: None },
    Centroid { x: -50.0, y: -10.0, mass: Some(45.0), cov: None },
    // ...
];

let solve_config = SolveConfig {
    fov_max_error_rad: Some((2.0_f32).to_radians()),
    // 15° horizontal FOV estimate, 1024×1024 image
    ..SolveConfig::new((15.0_f32).to_radians(), 1024, 1024)
};

// The solve returns Result<Solution, SolveFailure>; a Solution's
// fields are all guaranteed present.
if let Ok(solution) = db.solve_from_centroids(&centroids, &solve_config) {
    println!("Attitude: {}", solution.qicrs2cam);
    println!("Matched {} stars in {:.1} ms",
        solution.num_matches, solution.solve_time_ms);
}
```
