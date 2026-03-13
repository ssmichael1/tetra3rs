# Changelog

## 0.4.0

### Breaking changes

- **Gaia DR3 is now the default (and always-included) star catalog.** The `gaia` feature flag has been removed; Gaia support is always compiled in. Hipparcos support is now behind an optional `hipparcos` feature flag.
- **`Star.id` changed from `u64` to `i64`** to support negative source IDs for Hipparcos gap-fill stars in the merged catalog. This affects `matched_catalog_ids` arrays (`np.int64` in Python) and `get_star_by_id()`.
- **`generate_from_hipparcos` removed from Python bindings.** Use `generate_from_gaia()` instead.
- **Python dependency changed** from `hipparcos-catalog` to `gaia-catalog`.

### New features

- **Gaia DR3 + Hipparcos merged catalog.** The merged catalog uses Gaia for most stars and fills in the brightest stars (G < 4) from Hipparcos where Gaia saturates. Hipparcos positions are propagated from J1991.25 to the Gaia epoch (J2016.0).
- **Compact binary catalog format (.bin).** A custom 36-byte-per-star binary format (header + packed structs) reduces catalog size from 77 MB (CSV) to 17 MB. `generate_from_gaia()` auto-detects CSV vs binary format.
- **`gaia-catalog` PyPI package.** The merged catalog is bundled as a lightweight Python package (~15 MB wheel). `generate_from_gaia()` with no arguments automatically uses the bundled catalog.
- **`generate_from_gaia()` accepts optional `catalog_path`.** When `None` (default), uses the bundled `gaia-catalog` package. Accepts both `.csv` and `.bin` files.
- **`scripts/download_gaia_catalog.py`** downloads Gaia DR3 via TAP, merges with Hipparcos 2, and outputs either CSV or binary format (determined by file extension).

### Improvements

- **Switched from `nalgebra` to `numeris`** for linear algebra. `numeris` is a lightweight, `no_std`-compatible pure-Rust library for matrix/vector/quaternion operations, reducing dependency weight and improving suitability for embedded targets.
- **TESS test reliability.** Changed `match_radius` from `0.01` to `0.005` across TESS tests, fixing a false-match failure on the sparse field image.
- Removed `zip` dev-dependency (tests now download `.bin` directly instead of extracting from `.zip`).

### Migration guide

**Rust:**

Download the pre-built merged catalog (~17 MB, 482k stars to G-band magnitude 10, Gaia DR3 + Hipparcos bright-star gap-fill):
```sh
curl -o data/gaia_merged.bin "https://storage.googleapis.com/tetra3rs-testvecs/gaia_merged.bin"
```

Or generate your own with a custom magnitude limit:
```sh
python scripts/download_gaia_catalog.py --mag-limit 12.0 --output data/gaia_merged.bin
```

```rust
// Before (0.3.x)
let db = SolverDatabase::generate_from_hipparcos("data/hip2.dat", &config)?;

// After (0.4.0)
let db = SolverDatabase::generate_from_gaia("data/gaia_merged.bin", &config)?;
```

**Python:**
```python
# Before (0.3.x)
db = tetra3rs.SolverDatabase.generate_from_hipparcos()

# After (0.4.0)
db = tetra3rs.SolverDatabase.generate_from_gaia()  # uses bundled gaia-catalog
```

## 0.3.2

Initial public release with Hipparcos catalog support, centroid extraction, camera model, distortion calibration, WCS output, and stellar aberration correction.
