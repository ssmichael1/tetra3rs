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

Project nearby catalog stars into the camera frame using the estimated rotation, match them to observed centroids, and score the field with a per-star **likelihood ratio** Λ between "this attitude is right" and "it is wrong". A matched centroid contributes `q · A/(n·2πσ²) · exp(−r²/2σ²) + (1 − q)` — its closeness `r` relative to the *measured* density of projected catalog stars (`n` predictions over the frame area `A`) — and an unmatched one `1 − q`, where `q` is the fraction of detections that are catalog stars, estimated from the field separately for the bright centroids (rank below the number of catalog stars predicted in frame) and the faint remainder. A bright detection with no counterpart therefore counts *against* the attitude, while faint misses below the catalog cutoff are cheap. σ combines the stage's expected residual with the centroid's own covariance when the extractor supplies one. The 4 pattern stars that formed the candidate are excluded — they are the hypothesis being tested, not evidence for it. Because each factor is a true likelihood ratio, Λ has expectation 1 for a wrong attitude, so `1/Λ` is a valid p-value (Markov's inequality); a small BIC penalty pays for the two fitted fractions.

Each matched pattern also yields a *measured* FOV (from the ratio of catalog to image pattern scale), and the candidate's centroid vectors are rebuilt at that scale before verification — so a wrong FOV estimate (up to `fov_max_error_rad`) solves at essentially full speed, without a fine FOV sweep.

### 5. Refinement and WCS Fit

Refine the solution using *all* matched star pairs (not just the initial 4) via a constrained 3-DOF tangent-plane fit (rotation angle θ + CRVAL offset, pixel scale locked). The fit iterates internally: each pass re-fits the parameters, sigma-clips outliers, re-projects catalog stars, and re-matches centroids using the improved solution, stopping once the match set is stable. Acceptance is then decided by **re-verifying the refined attitude** with σ set by the refined RMSE: a true candidate's matches sit within a few RMSE (its p-value collapses when σ tightens), while a false candidate's coincidences are spread uniformly and cannot be aligned by the 3-DOF fit. The result is FITS-standard WCS output — a CD matrix and CRVAL reference point — allowing direct pixel↔sky coordinate conversions. The same normal equations give the solution's **attitude covariance**: `attitude_cov_rad2` is the 3×3 covariance of `[θ, ξ₀, η₀]` (roll, boresight tangent-plane offsets), `σ²·(JᵀJ)⁻¹` with `σ²` the residual variance over `2n − 3` degrees of freedom, and `attitude_sigma_rad()` returns the three 1σ values. When the solve was given an observer velocity, `observer_velocity_km_s` records it so `calibrate_camera` can apply the same aberration correction per image.

## Parity Flip Detection

Some imaging systems produce mirror-reflected images (e.g., FITS files with `CDELT1 < 0`, or optics with an odd number of reflections). In these cases the initial rotation estimate yields a reflection (determinant < 0) rather than a proper rotation.

The solver detects this by checking the determinant of the rotation matrix. When negative, it negates the x-coordinates of all centroid vectors and recomputes the rotation.

The `SolveResult` includes a `parity_flip` flag indicating whether this correction was applied. This is critical for pixel↔sky coordinate conversions: when `parity_flip` is `True`, the mapping between pixel x-coordinates and camera-frame x must include a sign flip.

## Search Strategy

The solver uses a breadth-first (brightest-first) search strategy. Brighter stars are more likely to be real detections (not noise), so trying patterns from the brightest centroids first maximizes the chance of an early match. Only the brightest `pattern_checking_stars` well-separated centroids (default 24) form patterns at all — the database stores patterns among each field's brightest stars, so those are the quads that can hit — which bounds a fruitless search at `C(24, 4) = 10,626` patterns per FOV value; fainter centroids still take part in verification.

The search can be bounded by:

- **Timeout** (`solve_timeout_ms`) — stop after a wall-clock limit
- **Pattern budget** (`max_patterns_checked`) — stop after testing a fixed
  number of image patterns, whichever of the two limits trips first. This
  bounds the search by work rather than time, so the outcome is reproducible
  across machines and the search stays finite on targets with no clock
- **Match threshold** (`match_threshold`) — a per-solve false-positive *budget*: the k-th candidate tested is accepted when its corrected p-value `p·k = k/Λ` falls below this value (a sequential Bonferroni correction over the candidates actually tested). `SolveResult.prob` reports the corrected p-value. Raising the budget (e.g. to `1e-3`) lets very sparse fields solve, at an explicitly chosen false-positive risk
- **FOV error** (`fov_max_error_rad`) — restrict the search to patterns consistent with the estimated field of view
