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

## Hipparcos Catalog

tetra3rs generates its pattern database from the Hipparcos catalog.

**Python:** The catalog is bundled automatically via the [`hipparcos-catalog`](https://pypi.org/project/hipparcos-catalog/) dependency — no manual download needed.

**Rust:** Download `hip2.dat` for use with the Rust API:

```sh
mkdir -p data
curl -o data/hip2.dat.gz "http://cdsarc.u-strasbg.fr/ftp/I/311/hip2.dat.gz"
gunzip data/hip2.dat.gz
```

!!! note
    The Hipparcos catalog is also downloaded automatically when running the Rust integration tests (`cargo test --features image`).
