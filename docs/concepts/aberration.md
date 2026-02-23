# Stellar Aberration

Stellar aberration is the apparent displacement of star positions caused by the finite speed of light combined with the observer's velocity — analogous to how rain appears to fall at an angle when you're moving.

## Effect Size

For Earth-based observers, stellar aberration shifts apparent star positions by up to **~20″** ($v/c \approx 10^{-4}$ rad). Without correction, the solved attitude is biased by up to ~20″.

| Velocity source | Magnitude | Aberration |
|-----------------|-----------|------------|
| Earth's orbital velocity | ~30 km/s | ~20″ |
| LEO orbital velocity | ~7.5 km/s | ~5″ |
| Earth's rotation (equator) | ~0.46 km/s | ~0.3″ |

For most applications, Earth's orbital velocity dominates and the other contributions can be neglected.

## Correction

To correct for aberration, pass the observer's barycentric velocity (ICRS, km/s) via the `observer_velocity_km_s` parameter:

```python
from datetime import datetime
import tetra3rs

# Get Earth's approximate barycentric velocity
v = tetra3rs.earth_barycentric_velocity(datetime(2025, 7, 10))

# Pass to solver
result = db.solve_from_centroids(
    centroids,
    fov_estimate_deg=10.0,
    image_shape=image.shape,
    observer_velocity_km_s=v,
)
```

The solver applies a first-order correction ($\mathbf{s'} = \mathbf{s} + \boldsymbol{\beta} - \mathbf{s}(\mathbf{s} \cdot \boldsymbol{\beta})$) to all catalog star vectors before matching and refinement, producing an unbiased attitude.

### Rust

```rust
use tetra3::{earth_barycentric_velocity, SolveConfig};

// days since J2000.0 (2000 Jan 1 12:00 TT)
let v = earth_barycentric_velocity(9321.0);

let config = SolveConfig {
    observer_velocity_km_s: Some(v),
    ..SolveConfig::new((10.0_f32).to_radians(), 1024, 1024)
};
```

## Earth Barycentric Velocity

The convenience function `earth_barycentric_velocity()` provides an approximate Earth velocity using a circular-orbit model:

- **Accuracy**: ~0.5 km/s (~1.7%), sufficient for the ~20″ aberration effect (~0.3″ error)
- **Input**: Python `datetime` (UTC) or days since J2000.0 (Rust)
- **Output**: `[vx, vy, vz]` in km/s, ICRS equatorial frame

!!! note
    Enabling aberration correction shifts the entire solved pointing by up to ~20″, not just the within-field residuals. This is the physically correct result. Most plate solvers (e.g., astrometry.net) do not account for aberration, so comparing results may show a systematic offset.
