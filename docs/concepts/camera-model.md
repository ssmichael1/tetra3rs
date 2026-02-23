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
    distortion = tetra3rs.RadialDistortion(k1=-7e-9, k2=2e-15)
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
