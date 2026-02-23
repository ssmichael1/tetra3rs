# WCS Output

Every successful solve produces FITS-standard World Coordinate System (WCS) fields, enabling direct conversion between pixel and sky coordinates.

## WCS Fields in SolveResult

| Property | Description |
|----------|-------------|
| `cd_matrix` | 2×2 CD matrix (tangent-plane radians per pixel) |
| `crval_ra_deg` | RA of the WCS reference point (degrees) |
| `crval_dec_deg` | Dec of the WCS reference point (degrees) |
| `crpix` | Optical center offset from image center `[x, y]` (pixels) |

The WCS uses a **gnomonic (TAN) projection** centered at `(CRVAL_RA, CRVAL_DEC)`.

## CD Matrix

The CD matrix maps pixel offsets (from CRPIX) to tangent-plane coordinates at CRVAL:

$$
\begin{pmatrix} \xi \\ \eta \end{pmatrix} = \mathbf{CD} \cdot \begin{pmatrix} \Delta x \\ \Delta y \end{pmatrix}
$$

where $\xi$ and $\eta$ are gnomonic tangent-plane coordinates in radians, and $(\Delta x, \Delta y)$ are pixel offsets from the optical center.

The CD matrix encodes pixel scale, rotation, and any skew. For a camera with uniform pixel scale and no skew, the CD matrix elements give:

- **Pixel scale**: $\sqrt{CD_{11}^2 + CD_{21}^2}$ radians/pixel
- **Position angle**: $\arctan(-CD_{11} / CD_{21})$

## Pixel ↔ Sky Conversions

`SolveResult` provides convenience methods that apply the full WCS pipeline (including distortion if present):

```python
# Pixel to sky (RA, Dec in degrees)
ra, dec = result.pixel_to_world(x, y)

# Sky to pixel (centered pixel coordinates)
x, y = result.world_to_pixel(ra_deg, dec_deg)
```

Both methods accept either scalar or numpy array inputs:

```python
import numpy as np

xs = np.array([0.0, 100.0, -200.0])
ys = np.array([0.0, -50.0, 150.0])
ras, decs = result.pixel_to_world(xs, ys)
```

!!! note
    Pixel coordinates use the image-center origin convention. See [Coordinate Conventions](coordinates.md) for details.

## WCS Refinement

After the initial attitude solve (SVD), tetra3rs performs a constrained 3-DOF tangent-plane refinement:

1. **θ** — in-plane rotation angle
2. **dξ₀, dη₀** — CRVAL offset in tangent-plane coordinates

The pixel scale is locked to the value from the initial solve. Sigma-clipping rejects outlier matches during the refinement. This produces a more accurate WCS that minimizes residuals across all matched stars.
