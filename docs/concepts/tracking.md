# Tracking Mode

The lost-in-space (LIS) pipeline described in [Algorithm Overview](algorithm.md) takes no prior information about the camera's pointing — it identifies the attitude from scratch on every frame. That's exactly what you want on the first frame, but when you're solving a sequence of frames (e.g. from a star-tracker camera capturing at video rate), each solve tells you *very* nearly what the next frame's attitude will be.

**Tracking mode** uses the previous frame's attitude as a hint to skip the 4-star pattern-hash phase entirely.

## When to use it

- Frame-to-frame star-tracker solves.
- Any situation where you have a rough attitude estimate from a propagated IMU, a coarse attitude sensor, or a prior solve.
- Low-SNR images where the pattern-hash phase might fail but catalog stars would still be identifiable given a hint.

## How it differs from LIS

| Step | Lost-in-space | Tracking |
|---|---|---|
| 1. Pattern matching | 4-star geometric hash lookup | Skipped |
| 2. Correspondence | Hash-derived candidates | Catalog stars near hinted boresight, nearest-neighbor to centroids |
| 3. Rotation estimate | SVD on 4-star quad | SVD on all initial correspondences |
| 4. Verification | Binomial false-positive test | Same |
| 5. WCS refinement | Same |

The key differences:

- **Minimum star count drops from 4 to 3.** LIS needs at least 4 stars to form one pattern; tracking only needs enough correspondences for Wahba's problem (3 is the theoretical minimum, though more is better).
- **No hash-quantization risk.** The LIS hash-table step is sensitive to the FOV estimate matching reality and to the pattern's edge ratios falling cleanly into bins. Tracking bypasses this entirely.
- **Speed.** LIS iterates candidate 4-star combinations until one matches; tracking is a single cone query + correspondence pass. On easy frames the speedup is modest (LIS is already millisecond-fast); on hard frames where LIS would have iterated many candidates, it can be substantial.

## API

### Python

```python
# Frame 1: lost-in-space
result1 = db.solve_from_centroids(
    centroids1, fov_estimate_deg=15, image_shape=(1024, 1024), camera_model=cam
)

# Frame 2: seed tracking with frame 1's attitude
result2 = db.solve_from_centroids(
    centroids2,
    fov_estimate_deg=15,
    image_shape=(1024, 1024),
    camera_model=result1.camera_model,   # reuse refined focal length
    attitude_hint=result1.quaternion,    # or result1.rotation_matrix_icrs_to_camera
    hint_uncertainty_deg=1.0,            # how off the hint might be
)
```

`attitude_hint` accepts either a 4-element `[w, x, y, z]` quaternion
(`result.quaternion` or a plain list/ndarray) or a 3×3 rotation matrix
(`result.rotation_matrix_icrs_to_camera` or any 3×3 ndarray).

### Quaternion convention

tetra3rs uses the **Hamilton convention** with the real part first:

```
q = [w, x, y, z]    with    q = w + x·i + y·j + z·k
```

- **Element order.** `w` is the scalar (real) component; `(x, y, z)` is the
  vector (imaginary) component. This matches `numpy.quaternion`, `scipy`'s
  quaternion conventions for `Rotation.as_quat(scalar_first=True)`, and most
  aerospace / attitude-estimation literature. It does **not** match scipy's
  default `Rotation.as_quat()` (which is `[x, y, z, w]` — scalar last) —
  when in doubt, check the element magnitudes: for small rotations `w ≈ 1`
  and the vector components are small.
- **Unit.** `w² + x² + y² + z² = 1`. `SolveResult.quaternion` is always
  unit-length; you should ensure hints passed in are too.
- **Sense.** The quaternion rotates a vector **from the ICRS frame into the
  camera frame**: `camera_vec = q ⊗ icrs_vec ⊗ q*` (where `⊗` is quaternion
  multiplication and `q*` is the conjugate). Equivalently,
  `R · icrs_vec = camera_vec` where `R` is
  `SolveResult.rotation_matrix_icrs_to_camera`. To invert the sense (e.g.
  to get a rotation from camera to ICRS), negate the vector part:
  `q_inv = [w, -x, -y, -z]`.
- **Composition.** To combine hints (e.g. apply a small delta rotation to a
  prior result), use Hamilton-order quaternion multiplication
  `q_new = q_delta ⊗ q_prior`, where applying `q_new` first rotates by
  `q_prior` then by `q_delta`. Neither NumPy nor scipy ship a Hamilton
  quaternion multiply in the standard API — the
  [integration test][test-source] has a worked example.

[test-source]: https://github.com/ssmichael1/tetra3rs/blob/main/python/tests/test_solve.py

### Rust

```rust
use tetra3::{SolveConfig, SolverDatabase};

let config = SolveConfig {
    fov_estimate_rad: 15.0_f32.to_radians(),
    image_width: 1024,
    image_height: 1024,
    attitude_hint: prev_result.qicrs2cam,  // Option<Quaternion>
    hint_uncertainty_rad: 1.0_f32.to_radians(),
    camera_model: prev_result.camera_model.clone().unwrap(),
    ..Default::default()
};
let result = db.solve_from_centroids(&centroids, &config);
```

## Fallback behavior

By default, if the hinted solve fails (too few matches, verification rejection), the solver **falls back to lost-in-space** on the same frame. This means tracking mode is strictly more powerful than LIS — you never lose the ability to solve from scratch, you just try the fast path first.

Set `strict_hint=True` (Python) / `strict_hint: true` (Rust) to disable the fallback. Useful when you specifically want to know whether the hint was good — for example, to detect that the camera has slewed and the prior attitude is stale.

## Hint uncertainty

The `hint_uncertainty_rad` / `hint_uncertainty_deg` parameter controls how wide the catalog cone search and initial pixel match radius are. Larger values tolerate staler or less accurate hints, at the cost of pulling in more candidate stars and widening the correspondence search.

- **Default: 1°.** Good for short-propagation hints (e.g. the previous frame at 30 fps).
- **Tighter (e.g. 0.1°):** Faster matching, but requires the hint to be that accurate. Typical after a few frames of lock-in.
- **Looser (e.g. 5°):** Tolerates long propagation gaps or coarse attitude sensors. Still much faster than LIS on most frames.

If the hint falls outside this cone (e.g. after an uncommanded slew), the hinted solve fails and the solver falls back to LIS.

## Limitations

- **Requires a reasonable hint.** If the hint is off by more than `hint_uncertainty_rad`, tracking will fail. With the default fallback this is safe but slow (LIS takes over); with `strict_hint=True` the solve returns failure.
- **No multiscale search.** LIS's FOV sweep doesn't run under tracking — the focal length from the provided camera model is trusted. If your focal length is wrong, tracking will fail where LIS might have recovered by sweeping.
- **Same verification threshold as LIS.** Tracking doesn't relax the statistical false-positive test, so very sparse fields (where only 2–3 catalog stars are visible) still fail verification even though Wahba's problem is solvable.
