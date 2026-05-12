#!/usr/bin/env python3
"""
Download Gaia DR3 from the Flatiron Institute's flathub service.

flathub (https://flathub.flatironinstitute.org) hosts the full Gaia DR3
catalog and returns numpy arrays directly over HTTP. It is preferred over
``download_gaia_catalog.py`` (which queries ESA's TAP server) when pulling
faint stars in bulk: ESA's TAP service throttles / truncates large faint-end
queries around G ~ 10, while flathub serves the same query without that cap.

Output and Hipparcos gap-fill logic mirror ``download_gaia_catalog.py`` so
the file is interchangeable with the sibling script:

  - .bin  compact binary used by tetra3rs and the gaia-catalog Python package
  - .csv  same columns as the sibling: source_id, ra, dec,
          phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
          parallax, pmra, pmdec
  - Bright Hipparcos 2 stars (estimated G < --bright-threshold) without a
    Gaia counterpart are added with negative source_ids to fill Gaia's
    bright-star saturation gap. Hipparcos positions are propagated from
    J1991.25 to the Gaia DR3 epoch (J2016.0) before merging.

Requirements:
    numpy, astropy, scipy, requests (pip-installable).

    flathub is not on PyPI — install it from the Flatiron Institute's repo
    per the instructions at
    https://github.com/flatironinstitute/flathub/tree/prod/py
    (e.g. clone the repo and `pip install ./py`, or
    `pip install "flathub @ git+https://github.com/flatironinstitute/flathub.git@prod#subdirectory=py"`).

    The flathub client currently calls ``numpy.DataSource`` directly, which
    NumPy 2.0 removed; this script shims that attribute back at import time
    so no patch to flathub itself is needed.

Usage:
    python download_gaia_flatiron.py                                 # mag 10, binary
    python download_gaia_flatiron.py --mag-limit 14 --output data/gaia14.bin
    python download_gaia_flatiron.py --mag-limit 12 --output data/gaia12.csv
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Reuse Hipparcos loader, merge, and writers from the sibling script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_gaia_catalog import (  # noqa: E402
    GAIA_EPOCH,
    load_hipparcos,
    merge_catalogs,
    propagate_hipparcos_to_epoch,
    write_merged_binary,
    write_merged_csv,
)


# Brightest Gaia DR3 source has phot_g_mean_mag ~ 1.732. flathub rejects
# range queries that fall below the field's catalog-wide minimum, so we use
# the exact lower bound advertised by the flathub schema.
GAIA_G_MIN = 1.7316069602966309

GAIA_FIELDS = [
    "source_id",
    "ra",
    "dec",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "parallax",
    "pmra",
    "pmdec",
]


def query_gaia_flathub(mag_limit: float) -> np.ndarray:
    """Pull Gaia DR3 stars with G < mag_limit from flathub as a structured array."""
    # flathub's client calls numpy.DataSource, which was removed in NumPy 2.0
    # and is still reachable at numpy.lib.npyio.DataSource. Patch it back on
    # before the import so the client can run unmodified.
    if not hasattr(np, "DataSource"):
        np.DataSource = np.lib.npyio.DataSource  # type: ignore[attr-defined]
    import flathub

    print(f"Querying flathub Gaia DR3 for stars with G < {mag_limit}...")
    gaiadr3 = flathub.Catalog(
        "gaiadr3", endpoint="https://flathub.flatironinstitute.org/api"
    )
    arr = gaiadr3.numpy(
        fields=GAIA_FIELDS,
        phot_g_mean_mag=(GAIA_G_MIN, mag_limit),
    )
    print(f"  Retrieved {len(arr)} Gaia stars")
    return arr


def main():
    parser = argparse.ArgumentParser(
        description="Download Gaia DR3 from flathub and merge with Hipparcos 2 bright stars"
    )
    parser.add_argument(
        "--mag-limit",
        type=float,
        default=10.0,
        help="Limiting G-band magnitude (default: 10.0)",
    )
    parser.add_argument(
        "--bright-threshold",
        type=float,
        default=4.0,
        help="G-mag threshold below which Hipparcos fills Gaia gaps (default: 4.0)",
    )
    parser.add_argument(
        "--match-radius",
        type=float,
        default=5.0,
        help="Cross-match radius in arcseconds (default: 5.0)",
    )
    parser.add_argument(
        "--hip2",
        type=str,
        default="data/hip2.dat",
        help="Path to hip2.dat file (default: data/hip2.dat)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gaia_merged.bin",
        help="Output path; .bin for binary or .csv for CSV (default: data/gaia_merged.bin)",
    )
    args = parser.parse_args()

    if not Path(args.hip2).exists():
        print(f"Error: {args.hip2} not found. Run scripts/download_hip2.sh first.")
        sys.exit(1)

    gaia_table = query_gaia_flathub(args.mag_limit)

    hip_table = load_hipparcos(args.hip2)
    propagate_hipparcos_to_epoch(hip_table, GAIA_EPOCH)

    hip_only, hip_gmag_only = merge_catalogs(
        gaia_table,
        hip_table,
        bright_threshold=args.bright_threshold,
        match_radius_arcsec=args.match_radius,
    )

    if args.output.endswith(".bin"):
        write_merged_binary(gaia_table, hip_only, hip_gmag_only, args.output)
    else:
        write_merged_csv(gaia_table, hip_only, hip_gmag_only, args.output)


if __name__ == "__main__":
    main()
