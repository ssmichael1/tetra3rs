# Camera Model

The `CameraModel` encapsulates the mapping between pixel coordinates and tangent-plane (gnomonic projection) coordinates. It bundles all camera intrinsics into a single struct used throughout the solving and calibration pipeline.

## Parameters

| Parameter | Description |
|-----------|-------------|
| `focal_length_px` | Focal length in pixels (= image_width / (2 × tan(FOV/2))) |
| `image_width` | Image width in pixels |
| `image_height` | Image height in pixels |
| `crpix` | Optical center offset from image center `[x, y]` in pixels |
| `parity_flip` | Whether the image x-axis is mirrored |
| `distortion` | Optional `RadialDistortion` or `PolynomialDistortion` model |

## Pixel → Tangent Plane Pipeline

The `pixel_to_tanplane()` method applies the following steps:

1. **Subtract CRPIX** — shift to optical-center-relative coordinates
2. **Undistort** — apply the inverse distortion model (if any)
3. **Parity flip** — negate x if `parity_flip` is True
4. **Divide by focal length** — convert from pixels to radians

The result is tangent-plane coordinates `(ξ, η)` in radians, suitable for gnomonic (TAN) projection.

## Creating a Camera Model

=== "From FOV"

    ```python
    import tetra3rs

    # Simple pinhole model from field of view
    cam = tetra3rs.CameraModel.from_fov(
        fov_deg=10.0,
        image_width=2048,
        image_height=1536,
    )
    ```

=== "With explicit parameters"

    ```python
    cam = tetra3rs.CameraModel(
        focal_length_px=11718.4,
        image_width=2048,
        image_height=1536,
        crpix=[2.5, -1.3],       # optical center offset
        parity_flip=True,         # mirrored image
    )
    ```

=== "With distortion"

    ```python
    # Pure radial (Brown-Conrady with k1, k2, k3; tangential set to 0)
    distortion = tetra3rs.RadialDistortion(k1=-7e-9, k2=2e-15)

    # Or full Brown-Conrady with tangential / decentering coefficients:
    # distortion = tetra3rs.RadialDistortion(k1=-7e-9, p1=5e-7, p2=-3e-7)

    cam = tetra3rs.CameraModel(
        focal_length_px=11718.4,
        image_width=2048,
        image_height=1536,
        distortion=distortion,
    )
    ```

## Using with the Solver

Pass a `CameraModel` to `solve_from_centroids()` to apply distortion correction and parity during solving:

```python
result = db.solve_from_centroids(
    centroids,
    fov_estimate_deg=10.0,
    image_shape=image.shape,
    camera_model=cam,
)
```

The returned `SolveResult` uses the camera model's distortion and parity for its `pixel_to_world()` and `world_to_pixel()` methods.

## Calibration

Use `calibrate_camera()` to fit a camera model from one or more solved images. See [CalibrateResult](../api/calibrate-result.md) for details.

The fit selects between two distortion models via the `model` parameter:

- `model="polynomial"` (default): SIP-like polynomial of the requested
  `order`. Captures arbitrary 2D distortion, including tangential and
  decentering effects, by fitting independent per-axis coefficients
  `A_pq, B_pq`. Order-0 terms absorb the optical-center offset; higher
  orders capture lens distortion. Preferred for off-axis CCDs and astronomy
  WCS pipelines.
- `model="radial"`: full Brown-Conrady — three radial coefficients
  `(k1, k2, k3)` plus two tangential coefficients `(p1, p2)`, with the
  optical center `(cx, cy)` fit jointly. Three-to-seven free parameters
  total; well-conditioned with relatively few matches and the standard
  model in computer-vision camera calibration.

## References

### Brown-Conrady (radial + tangential)

- **Conrady, A. E.** (1919). "[Decentred Lens-Systems](https://doi.org/10.1093/mnras/79.5.384)."
  *Monthly Notices of the Royal Astronomical Society*, 79(5): 384-390.
  — Original derivation of the tangential / decentering distortion form.
- **Brown, D. C.** (1966). "Decentering Distortion of Lenses."
  *Photogrammetric Engineering*, 32(3): 444-462. — Modernized the
  Conrady formulation; introduced the radial-plus-tangential form used today.
- **Brown, D. C.** (1971). "Close-Range Camera Calibration."
  *Photogrammetric Engineering*, 37(8): 855-866. — Calibration procedure;
  basis for the OpenCV / photogrammetry conventions.
- **Zhang, Z.** (2000). "[A Flexible New Technique for Camera Calibration](https://doi.org/10.1109/34.888718)."
  *IEEE TPAMI*, 22(11): 1330-1334. — Multi-image planar-target calibration
  that became the standard method, and the model implemented by OpenCV's
  `calibrateCamera`.
- **OpenCV documentation** for the equivalent `(k1, k2, k3, p1, p2)`
  formulation: <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>

### SIP polynomial

- **Shupe, D. L.; Moshir, M.; Li, J.; Makovoz, D.; Narron, R.; Hook, R. N.**
  (2005). "[The SIP Convention for Representing Distortion in FITS Image Headers](https://www.adass.org/adass/proceedings/adass04/reprints/P3-1-3.pdf)."
  *Astronomical Data Analysis Software and Systems XIV*, ASP Conference
  Series, 347: 491. — The original SIP specification.
- **FITS WCS SIP convention registry entry**:
  <https://fits.gsfc.nasa.gov/registry/sip.html>

