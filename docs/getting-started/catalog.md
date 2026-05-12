# Star Catalog

tetra3rs solves against a merged **Gaia DR3 + Hipparcos** catalog. Gaia DR3 provides the bulk of stars; Hipparcos 2 fills in the few dozen brightest stars (G < 4) where Gaia saturates. Positions are stored at Gaia's reference epoch (J2016.0) and propagated to the observation epoch at solve time using per-star proper motion.

The on-disk format is a compact little-endian binary (header `GDR3`, 36 bytes per star) used by both the Rust crate and the Python [`gaia-catalog`](https://pypi.org/project/gaia-catalog/) PyPI package. The same script can also write a CSV form.

## Pre-built download

The easiest path. A pre-built catalog with G < 10 (~17 MB, ~482k stars) is hosted on Google Cloud Storage:

```sh
mkdir -p data
curl -o data/gaia_merged.bin "https://storage.googleapis.com/tetra3rs-testvecs/gaia_merged.bin"
```

Python users get the same file bundled in [`gaia-catalog`](https://pypi.org/project/gaia-catalog/), which is installed automatically with `tetra3rs` — no curl needed.

## Generate your own

If you need a different magnitude limit (e.g. fainter stars for narrow-FOV cameras) or want to regenerate at a more recent reference epoch, two download scripts live in `scripts/`:

| Script | Backend | Practical mag limit | Setup |
|---|---|---|---|
| `download_gaia_catalog.py` | ESA Gaia Archive TAP server | G ≲ 11.5 (3M-row hard cap) | `pip install astroquery astropy` |
| `download_gaia_flatiron.py` | [Flatiron Institute flathub](https://flathub.flatironinstitute.org/gaiadr3) | full Gaia DR3 | flathub from GitHub (see below) |

Both scripts produce byte-compatible output and share the Hipparcos-2 bright-star merge — the flathub script imports the merge and writer functions from the ESA script.

### ESA TAP (shallow catalogs)

The canonical Gaia source. Default settings produce the same ~17 MB / ~482k-star catalog as the pre-built download:

```sh
pip install astroquery astropy
bash scripts/download_hip2.sh
python scripts/download_gaia_catalog.py --mag-limit 10.0 --output data/gaia_merged.bin
```

!!! warning "3-million-row cap"
    The ESA TAP server advertises a **hard** output limit of 3,000,000 rows per async job for anonymous users (see `https://gea.esac.esa.int/tap-server/tap/capabilities`). Cumulative Gaia DR3 counts: G<10 ≈ 482k, G<11 ≈ 1.25M, G<12 ≈ 3.09M, G<13 ≈ 7.37M. So anonymous queries top out around **G ≈ 11.5**; fainter mags return a silently-truncated result. The script already uses `launch_job_async`, but the cap is server-side and `MAXREC` cannot raise it.

    Workarounds: register for a free [Gaia archive account](https://www.cosmos.esa.int/web/gaia-users/register) (the per-job limit is higher for authenticated users), or use the flathub script below, which has no faint-end cap.

### Flatiron flathub (faint catalogs)

flathub hosts the full Gaia DR3 catalog and streams `.npy` payloads directly. It serves the same query as the ESA TAP backend without the faint-end cap.

flathub is **not on PyPI**. Install from the upstream repo:

```sh
# one-shot from git
pip install "flathub @ git+https://github.com/flatironinstitute/flathub.git@prod#subdirectory=py"

# or clone + local install
git clone https://github.com/flatironinstitute/flathub.git
pip install ./flathub/py
```

You'll also need the Hipparcos 2 catalog for the bright-star merge:

```sh
bash scripts/download_hip2.sh
```

Then run the downloader:

```sh
pip install astropy scipy
python scripts/download_gaia_flatiron.py --mag-limit 14.0 --output data/gaia_merged.bin
```

!!! note "NumPy 2.0 shim"
    The flathub client currently calls `numpy.DataSource`, which NumPy 2.0 removed. The script shims that symbol back at import time, so no patch to flathub itself is needed.

!!! warning "Large downloads"
    A G < 14 catalog is ~17M stars, ~1 GB on disk; the response is served as a single streaming `.npy`. A corporate / filtering proxy may drop or truncate the connection — run the script on a direct connection if possible.

## What's in the merged catalog

Records, in either output format, carry: `source_id`, `ra`, `dec`, `phot_g_mean_mag`, `pmra`, `pmdec` (binary), plus `phot_bp_mean_mag`, `phot_rp_mean_mag`, `parallax` (CSV only).

The Hipparcos merge propagates Hipparcos positions from the Hipparcos reference epoch (J1991.25) to the Gaia DR3 reference epoch (J2016.0) using each star's proper motion, then adds Hipparcos stars brighter than the bright-threshold (G ~ 4 by default) that have no Gaia counterpart within the match radius (5″ by default). Synthesized Hipparcos entries are stored with **negative** `source_id` values so they're distinguishable from real Gaia IDs. The relevant CLI flags are `--bright-threshold` and `--match-radius` (both scripts).

See the docstrings in `scripts/download_gaia_catalog.py` and `scripts/download_gaia_flatiron.py` for the full option set and cross-match details.
