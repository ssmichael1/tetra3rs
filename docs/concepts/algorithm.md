# Algorithm Overview

tetra3rs implements a geometric hashing algorithm for lost-in-space star identification. The solver works in two phases: an offline **database generation** step that precomputes patterns from the star catalog, and an online **solve** step that matches observed star centroids against the database.

## Pipeline

### 1. Pattern Generation (offline)

Select combinations of 4 bright centroids and compute 6 pairwise angular separations. These are normalized into 5 edge ratios — a geometric invariant that is independent of the camera's pointing direction and rotation. The edge ratios are quantized into a hash key and stored in a precomputed hash table.

### 2. Hash Lookup

For each candidate 4-star pattern in the image, compute the same edge ratios and probe the hash table. Matching catalog patterns are returned as candidates.

### 3. Attitude Estimation

For each candidate match, solve Wahba's problem via SVD to find the optimal rotation from catalog coordinates (ICRS) to the camera frame. This gives a 3×3 rotation matrix (or equivalently a quaternion) that maps star unit vectors in the ICRS frame to unit vectors in the camera frame.

### 4. Verification

Project nearby catalog stars into the camera frame using the estimated rotation, count how many match observed centroids within a tolerance, and accept only if the false-positive probability (computed via the binomial CDF) is below a threshold. This statistical test ensures that the match is not a coincidence.

### 5. Refinement

Re-estimate the rotation using *all* matched star pairs (not just the initial 4) via iterative SVD passes. Each pass re-projects catalog stars and re-matches centroids using the improved rotation. The number of passes is controlled by `refine_iterations` (default 2).

### 6. WCS Fit

A constrained 3-DOF tangent-plane refinement (rotation angle θ + CRVAL offset) with sigma-clipping produces FITS-standard WCS output: a CD matrix and CRVAL reference point. This allows direct pixel↔sky coordinate conversions.

## Parity Flip Detection

Some imaging systems produce mirror-reflected images (e.g., FITS files with `CDELT1 < 0`, or optics with an odd number of reflections). In these cases the initial rotation estimate yields a reflection (determinant < 0) rather than a proper rotation.

The solver detects this by checking the determinant of the rotation matrix. When negative, it negates the x-coordinates of all centroid vectors and recomputes the rotation.

The `SolveResult` includes a `parity_flip` flag indicating whether this correction was applied. This is critical for pixel↔sky coordinate conversions: when `parity_flip` is `True`, the mapping between pixel x-coordinates and camera-frame x must include a sign flip.

## Search Strategy

The solver uses a breadth-first (brightest-first) search strategy. Brighter stars are more likely to be real detections (not noise), so trying patterns from the brightest centroids first maximizes the chance of an early match.

The search can be bounded by:

- **Timeout** (`solve_timeout_ms`) — stop after a time limit
- **Match threshold** (`match_threshold`) — accept the first match with false-positive probability below this value
- **FOV error** (`fov_max_error_rad`) — restrict the search to patterns consistent with the estimated field of view
