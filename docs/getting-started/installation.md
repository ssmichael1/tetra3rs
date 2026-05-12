# Installation

## Python

Binary wheels are available on [PyPI](https://pypi.org/project/tetra3rs/) for Linux (x86_64, ARM64), macOS (ARM64), and Windows (x86_64):

```sh
pip install tetra3rs
```

### Build from source

Building from source requires a [Rust toolchain](https://rustup.rs/):

```sh
git clone https://github.com/ssmichael1/tetra3rs.git
cd tetra3rs
pip install .
```

## Rust

The crate is published on [crates.io](https://crates.io/crates/tetra3) as `tetra3`:

```sh
cargo add tetra3
```

To enable centroid extraction from images, add the `image` feature:

```sh
cargo add tetra3 --features image
```

## Star Catalog

tetra3rs solves against a merged Gaia DR3 + Hipparcos catalog. Python users get this bundled automatically via the [`gaia-catalog`](https://pypi.org/project/gaia-catalog/) PyPI package — no setup needed. Rust users (and Python users who want a deeper magnitude limit) can download a pre-built binary or generate their own.

See [Star Catalog](catalog.md) for the pre-built download, custom-mag-limit scripts (ESA TAP up to G ≲ 10, Flatiron flathub for fainter), and the Hipparcos bright-star merge.

!!! note
    The catalog is also downloaded automatically when running the Rust integration tests (`cargo test --features image`).
