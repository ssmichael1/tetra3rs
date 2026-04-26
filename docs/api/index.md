# API Reference

Python API documentation for tetra3rs, auto-generated from type stubs.

## Core Classes

| Class | Description |
|-------|-------------|
| [`SolverDatabase`](solver-database.md) | Star pattern database — generate, save/load, and solve |
| [`CameraModel`](camera-model.md) | Camera intrinsics — focal length, optical center, parity, distortion |
| [`SolveResult`](solve-result.md) | Plate-solve result — attitude, WCS, matched stars, pixel↔sky conversions |
| [`CalibrateResult`](calibrate-result.md) | Camera calibration result — fitted camera model and statistics |

## Centroid Extraction

| Symbol | Description |
|--------|-------------|
| [`extract_centroids()`](extraction.md) | Extract star centroids from an image array |
| [`ExtractionResult`](extraction.md#tetra3rs.ExtractionResult) | Extraction result with centroids and image statistics |
| [`Centroid`](extraction.md#tetra3rs.Centroid) | A single star centroid with position, brightness, and shape |

## Distortion Models

| Class | Description |
|-------|-------------|
| [`RadialDistortion`](distortion.md#tetra3rs.RadialDistortion) | Radial (k1, k2, k3) distortion model |
| [`PolynomialDistortion`](distortion.md#tetra3rs.PolynomialDistortion) | SIP-like polynomial distortion model |

## Module-Level Functions

| Function | Description |
|----------|-------------|
| [`earth_barycentric_velocity()`](functions.md#tetra3rs.earth_barycentric_velocity) | Approximate Earth barycentric velocity for aberration correction |
| [`undistort_centroids()`](functions.md#tetra3rs.undistort_centroids) | Apply inverse distortion to a list of centroids |
| [`distort_centroids()`](functions.md#tetra3rs.distort_centroids) | Apply forward distortion to a list of centroids |
