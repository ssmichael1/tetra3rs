# Coordinate Conventions

## Camera Frame

tetra3rs uses a right-handed camera frame:

- **+X** — right
- **+Y** — down
- **+Z** — boresight (into the scene)

## Pixel Coordinates

All pixel coordinates use the **image center as origin**:

- `x = 0, y = 0` is the geometric center of the image
- `+X` is to the right
- `+Y` is downward

This applies to:

- Input centroids (both `Centroid` objects and numpy arrays)
- `SolveResult.pixel_to_world()` / `world_to_pixel()` inputs and outputs
- `ExtractionResult.centroids` returned by `extract_centroids()`

!!! note
    This convention differs from many image processing libraries where the origin is at the top-left corner. If your centroids use a top-left origin, subtract `(image_width/2, image_height/2)` before passing them to the solver.

## CRPIX — Optical Center Offset

The optical center of a real camera may not coincide with the geometric image center. The `crpix` parameter in `CameraModel` represents this offset:

- `crpix = [0, 0]` means the optical center is at the image center (default)
- `crpix = [dx, dy]` means the optical center is shifted by `(dx, dy)` pixels from the image center

The solver subtracts `crpix` from pixel coordinates before applying the projection.

## Sky Coordinates (ICRS)

Solved attitudes are in the International Celestial Reference System (ICRS):

- **RA** (Right Ascension) — in degrees, range [0°, 360°)
- **Dec** (Declination) — in degrees, range [−90°, +90°]
- **Roll** — position angle of camera +Y measured East of North, in degrees

## Rotation Representation

The solved attitude is available as:

- A **3×3 rotation matrix** (`rotation_matrix_icrs_to_camera`) mapping ICRS unit vectors to camera-frame unit vectors
- Decomposed **RA, Dec, Roll** angles

In the Rust API, the attitude is also available as a `UnitQuaternion` (`qicrs2cam`).
