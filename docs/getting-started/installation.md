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

tetra3rs generates its pattern database from a merged Gaia DR3 + Hipparcos catalog (~482k stars to G-band magnitude 10). Gaia provides most stars; Hipparcos fills in the brightest stars (G < 4) where Gaia saturates.

**Python:** The catalog is bundled in the [`gaia-catalog`](https://pypi.org/project/gaia-catalog/) package, which is installed automatically with `tetra3rs`. No manual download needed — just call `generate_from_gaia()` with no arguments.

**Rust:** Download the pre-built binary catalog (~17 MB):

```sh
mkdir -p data
curl -o data/gaia_merged.bin "https://storage.googleapis.com/tetra3rs-testvecs/gaia_merged.bin"
```

Or generate your own with a custom magnitude limit:

```sh
pip install astroquery astropy
python scripts/download_gaia_catalog.py --mag-limit 12.0 --output data/gaia_merged.bin
```

!!! note
    The catalog is also downloaded automatically when running the Rust integration tests (`cargo test --features image`).
