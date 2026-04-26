# tetra3rs

[![Crates.io](https://img.shields.io/crates/v/tetra3)](https://crates.io/crates/tetra3)
[![PyPI](https://img.shields.io/pypi/v/tetra3rs)](https://pypi.org/project/tetra3rs/)
[![docs.rs](https://img.shields.io/docsrs/tetra3)](https://docs.rs/tetra3)
[![License](https://img.shields.io/crates/l/tetra3)](https://github.com/ssmichael1/tetra3rs/blob/main/LICENSE)

A fast, robust lost-in-space star plate solver written in Rust with Python bindings.

Given a set of star centroids extracted from a camera image, tetra3rs identifies the stars against a catalog and returns the camera's pointing direction as a quaternion — no prior attitude estimate required.

!!! warning "Alpha Status"
    The core solver is based on well-vetted algorithms but has only been tested against a limited set of images. The API is not yet stable and may change between releases.

## Features

- **Lost-in-space solving** — determines attitude from star patterns with no initial guess
- **Fast** — geometric hashing of 4-star patterns with breadth-first (brightest-first) search
- **Robust** — statistical verification via binomial false-positive probability
- **Multiscale** — supports a range of field-of-view scales in a single database
- **Proper motion** — propagates Gaia DR3 / Hipparcos catalog positions to any observation epoch
- **Compact binary databases** — databases serialize with [postcard](https://docs.rs/postcard) in a portable, lightweight format
- **Centroid extraction** — detect stars from in-memory pixel data with local background subtraction, connected-component labeling, and quadratic sub-pixel peak refinement (accepts a decoded `DynamicImage` or a raw `&[f32]` pixel buffer)
- **Camera model** — unified intrinsics struct (focal length, optical center, parity, distortion) used throughout the pipeline
- **Distortion calibration** — fit SIP polynomial or Brown-Conrady radial distortion models from one or more solved images via `calibrate_camera`
- **WCS output** — solve results include FITS-standard WCS fields (CD matrix, CRVAL) and pixel↔sky coordinate conversion methods
- **Stellar aberration** — optional correction for the ~20″ apparent shift caused by the observer's barycentric velocity

## Quick Links

| | |
|---|---|
| **[Installation](getting-started/installation.md)** | Install from PyPI or build from source |
| **[Quick Start](getting-started/quickstart.md)** | Get solving in minutes |
| **[Concepts](concepts/algorithm.md)** | Understand the algorithm and coordinate conventions |
| **[API Reference](api/index.md)** | Full Python API documentation |
| **[Tutorials](tutorials/index.md)** | Interactive Jupyter notebook examples |

## Credits

This project is based upon the **tetra3** / **cedar-solve** algorithms:

- **[cedar-solve](https://github.com/smroid/cedar-solve)** — Steven Rosenthal's Python plate solver
- **[tetra3](https://github.com/esa/tetra3)** — the original Python implementation by Gustav Pettersson at ESA
- **Paper**: G. Pettersson, "Tetra3: a fast and robust star identification algorithm," ESA GNC Conference, 2023

## License

MIT License. See [LICENSE](https://github.com/ssmichael1/tetra3rs/blob/main/LICENSE) for details.
