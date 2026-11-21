# Coordinate Conventions

## Camera Frame

tetra3rs uses a right-handed camera frame:

- **+X** — right
- **+Y** — down
- **+Z** — boresight (into the scene)

## Pixel Coordinates

All pixel coordinates use the **geometric image center as the origin**.

Pixel *centers* sit at integer indices (a single bright pixel at column `c`,
row `r` produces a centroid at `x_px = c`, `y_px = r` before centering), so for
an image `W` pixels wide and `H` pixels tall the geometric center is at

$$
(c_x,\;c_y) = \left(\frac{W-1}{2},\;\frac{H-1}{2}\right),
$$

and the centered coordinates handed to / returned by the solver are
`x = c − (W−1)/2`, `y = r − (H−1)/2`:

- `x = 0, y = 0` is the geometric center of the image.
- For **even** dimensions this is the intersection of the four central pixels
  (e.g. for `W = 3840`, `c_x = 1919.5`, between columns 1919 and 1920).
- For **odd** dimensions it is the center of the single middle pixel.
- `+X` is to the right, `+Y` is downward.

This matches the geometric-center convention used by **FITS WCS, astropy,
astrometry.net, OpenCV, and scikit-image** (FITS' 1-indexed center
`(W+1)/2` is the same point as the 0-indexed `(W−1)/2`), so a `crpix` taken
from any of those tools can be used directly.

This applies to:

- Input centroids (both `Centroid` objects and numpy arrays)
- `SolveResult.pixel_to_world()` / `world_to_pixel()` inputs and outputs
- `ExtractionResult.centroids` returned by `extract_centroids()`

!!! note "Centered origin — not a corner origin"
    tetra3rs always uses a **center origin**: the boresight, the gnomonic
    projection, and the distortion field are all symmetric about the image
    center, so that is the natural reference for pointing geometry. This
    differs from image-processing libraries (numpy, OpenCV indexing), which
    use a **top-left corner** origin with the first pixel center at `(0, 0)`
    because it suits raster array access, not pointing.

    You only deal with the corner convention at the *input* boundary: if your
    centroids are corner-indexed (e.g. raw detector row/column), subtract
    `((W−1)/2, (H−1)/2)` to convert them to the center origin before passing
    them to the solver. `extract_centroids()` already returns center-origin
    coordinates.

!!! warning "Changed in 0.8.0"
    Earlier versions placed the origin at `(W/2, H/2)` — a half-pixel right of
    and below the geometric center. 0.8.0 moved it to `((W−1)/2, (H−1)/2)` to
    match the FITS / astropy / OpenCV convention. This removes a ~½-pixel bias
    when comparing tetra3rs solutions against those tools (it was internally
    consistent, so it never biased tetra3rs-only solves). See issue #28.

## CRPIX — projection reference offset

`crpix` in `CameraModel` shifts the **projection origin** away from the
geometric image center:

- `crpix = [0, 0]` (default) puts the projection origin at the geometric
  center, so the boresight (+Z) is the sky direction imaged at the center
  pixel.
- `crpix = [dx, dy]` moves the projection origin by `(dx, dy)` pixels; the
  solver subtracts `crpix` from each pixel coordinate before projecting.

!!! note "`crpix` is not the distortion center"
    `crpix` is the tangent-plane projection origin. The **optical axis** —
    where lens (radial/tangential) distortion is physically centered — is a
    separate quantity, carried by `RadialDistortion::center`. On mosaic
    cameras (TESS, Kepler, …) the optical axis can lie far off the detector
    (near a CCD corner), while `crpix` stays near the image center so the
    solver's FOV-radius queries and pattern geometry remain valid. See the
    [camera model](camera-model.md) page.

## Sky Coordinates (ICRS)

Solved attitudes are in the International Celestial Reference System (ICRS):

- **RA** (Right Ascension) — in degrees, range [0°, 360°)
- **Dec** (Declination) — in degrees, range [−90°, +90°]
- **Roll** — position angle of camera +Y measured East of North, in degrees

## Rotation Representation

The solved attitude is available as:

- A **3×3 rotation matrix** (`rotation_matrix_icrs_to_camera`) mapping ICRS
  unit vectors to camera-frame unit vectors.
- Decomposed **RA, Dec, Roll** angles.
- In the Rust API, a `UnitQuaternion` (`qicrs2cam`).

### What the attitude points at

The solved attitude's boresight — camera **+Z**, equivalently the reported
**RA/Dec** — is the sky direction that images onto the **projection origin**:
the center pixel `((W−1)/2, (H−1)/2)` when `crpix = [0, 0]`, or the `crpix`
location when set.

!!! important "Center pixel, not the optical boresight"
    +Z points at the sky direction landing on the **center pixel**, *not* at
    the camera's optical/lens axis. On a real camera the optical axis (the
    distortion center, `RadialDistortion::center`) generally falls on a
    *different* pixel, so it corresponds to a slightly different sky
    direction. tetra3rs always reports the attitude of the center-pixel
    boresight; if you need the pointing of the optical axis, project that
    pixel through `pixel_to_world()`.
