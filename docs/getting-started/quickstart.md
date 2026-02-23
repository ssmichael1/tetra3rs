# Quick Start

## Python

### Generate a database

```python
import tetra3rs

# Generate from the Hipparcos catalog
db = tetra3rs.SolverDatabase.generate_from_hipparcos(
    "data/hip2.dat",
    max_fov_deg=20.0,
    epoch_proper_motion_year=2025.0,
)

# Save for fast loading later
db.save_to_file("my_database.bin")

# ... or load a previously saved database
db = tetra3rs.SolverDatabase.load_from_file("my_database.bin")
```

### Solve from centroids

```python
# Solve from a list of centroids
result = db.solve_from_centroids(
    centroids,
    fov_estimate_deg=10.0,
    image_width=2048,
    image_height=1536,
)

if result is not None:
    print(f"RA: {result.ra_deg:.4f}°, Dec: {result.dec_deg:.4f}°")
    print(f"Roll: {result.roll_deg:.2f}°")
    print(f"Matched {result.num_matches} stars, RMSE: {result.rmse_arcsec:.1f}\"")
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
if result is not None:
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
use tetra3::{GenerateDatabaseConfig, SolverDatabase, SolveConfig, Centroid, SolveStatus};

// Generate a database from the Hipparcos catalog
let config = GenerateDatabaseConfig {
    max_fov_deg: 20.0,
    epoch_proper_motion_year: Some(2025.0),
    ..Default::default()
};
let db = SolverDatabase::generate_from_hipparcos("data/hip2.dat", &config)?;

// Save and reload
db.save_to_file("data/my_database.rkyv")?;
let db = SolverDatabase::load_from_file("data/my_database.rkyv")?;

// Solve from image centroids (pixel coordinates, origin at image center)
let centroids = vec![
    Centroid { x: 100.0, y: 200.0, mass: Some(50.0), cov: None },
    Centroid { x: -50.0, y: -10.0, mass: Some(45.0), cov: None },
    // ...
];

let solve_config = SolveConfig {
    fov_estimate_rad: (15.0_f32).to_radians(),
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
