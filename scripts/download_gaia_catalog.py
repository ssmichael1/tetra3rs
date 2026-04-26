#!/usr/bin/env python3
"""
Download Gaia DR3 star catalog and merge with Hipparcos 2 for bright stars.

Gaia saturates for very bright stars (G <~ 3), so Hipparcos 2 is used to fill
in those missing entries. Hipparcos positions are propagated from J1991.25 to
the Gaia epoch (J2016.0) before merging. Hipparcos gap-fill stars are assigned
negative source_ids.

Output format is determined by the file extension:
  - .bin  — compact binary (default, ~17 MB for mag 10); used by tetra3rs and
            the gaia-catalog Python package
  - .csv  — CSV with columns: source_id, ra, dec, phot_g_mean_mag,
            phot_bp_mean_mag, phot_rp_mean_mag, parallax, pmra, pmdec

Requirements:
    pip install astroquery astropy

Usage:
    python download_gaia_catalog.py                                      # binary, mag 10
    python download_gaia_catalog.py --mag-limit 12.0 --output gaia.bin   # binary, mag 12
    python download_gaia_catalog.py --output data/gaia_merged.csv        # CSV
"""

import argparse
import ssl
import sys
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


def _fix_ssl():
    """Work around SSL certificate verification failures with ESA TAP server.

    This is a known issue (see ESA Gaia Archive FAQ). If certifi certificates
    are available, point the default SSL context at them. As a last resort,
    fall back to an unverified context.
    """
    try:
        import certifi

        ssl._create_default_https_context = lambda: ssl.create_default_context(
            cafile=certifi.where()
        )
    except ImportError:
        # No certifi — fall back to unverified (prints a warning)
        print(
            "WARNING: certifi not installed; disabling SSL verification. "
            "Install certifi (`pip install certifi`) for secure connections."
        )
        ssl._create_default_https_context = ssl._create_unverified_context


def query_gaia(mag_limit: float, row_limit: int = -1) -> "astropy.table.Table":
    """Query Gaia DR3 via TAP for stars brighter than mag_limit in G-band."""
    _fix_ssl()
    from astroquery.gaia import Gaia

    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    Gaia.ROW_LIMIT = row_limit

    query = f"""
    SELECT
        source_id,
        ra, dec,
        phot_g_mean_mag,
        phot_bp_mean_mag,
        phot_rp_mean_mag,
        parallax,
        pmra, pmdec
    FROM gaiadr3.gaia_source
    WHERE phot_g_mean_mag < {mag_limit:.2f}
    ORDER BY phot_g_mean_mag ASC
    """

    print(f"Querying Gaia DR3 for stars with G < {mag_limit}...")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            job = Gaia.launch_job_async(query)
            table = job.get_results()
            print(f"  Retrieved {len(table)} Gaia stars")
            return table
        except Exception as e:
            print(f"  Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                import time
                wait = 10 * attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def load_hipparcos(hip2_path: str) -> "astropy.table.Table":
    """Parse hip2.dat fixed-width format into an astropy Table.

    Uses the same column positions as tetra3rs src/catalogs/hipparcos.rs.
    Positions are at the Hipparcos reference epoch J1991.25.
    """
    from astropy.table import Table

    hip_ids = []
    ras = []
    decs = []
    hpmags = []
    bvs = []
    pmras = []   # mu_alpha*cos(delta), mas/yr
    pmdecs = []  # mu_delta, mas/yr

    with open(hip2_path, "r") as f:
        for line in f:
            if len(line) < 171:
                continue
            try:
                hip = int(line[0:7])
                ra_rad = float(line[15:29])
                dec_rad = float(line[29:43])
                pmra = float(line[51:60])    # mu_alpha*cos(delta) mas/yr
                pmdec = float(line[60:69])   # mu_delta mas/yr
                hpmag = float(line[129:137])
                bv_str = line[152:159].strip()
                bv = float(bv_str) if bv_str else 0.0
            except ValueError:
                continue

            # Convert Hp magnitude to Johnson V using ESA polynomial
            # (same formula as hipparcos.rs hp_to_v)
            delta = 0.304 * bv - 0.202 * bv**2 + 0.107 * bv**3 - 0.045 * bv**4
            vmag = hpmag - delta

            hip_ids.append(hip)
            ras.append(np.degrees(ra_rad))
            decs.append(np.degrees(dec_rad))
            hpmags.append(vmag)
            bvs.append(bv)
            pmras.append(pmra)
            pmdecs.append(pmdec)

    t = Table()
    t["hip"] = np.array(hip_ids, dtype=np.int32)
    t["ra"] = np.array(ras, dtype=np.float64)
    t["dec"] = np.array(decs, dtype=np.float64)
    t["vmag"] = np.array(hpmags, dtype=np.float32)
    t["bv"] = np.array(bvs, dtype=np.float32)
    t["pmra"] = np.array(pmras, dtype=np.float64)    # mas/yr
    t["pmdec"] = np.array(pmdecs, dtype=np.float64)  # mas/yr
    print(f"  Loaded {len(t)} Hipparcos 2 stars from {hip2_path}")
    return t


# Hipparcos reference epoch
HIPPARCOS_EPOCH = 1991.25
# Gaia DR3 reference epoch
GAIA_EPOCH = 2016.0


def propagate_hipparcos_to_epoch(hip_table, target_epoch: float = GAIA_EPOCH):
    """Propagate Hipparcos positions from J1991.25 to a target epoch using proper motion.

    Modifies the table in-place, updating ra and dec columns.
    Uses the same convention as tetra3rs: pmra is mu_alpha*cos(delta),
    so we divide by cos(dec) to get the true RA rate.
    Skips proper motion near poles (|dec| > 87°) to avoid numerical instability.
    """
    dt = target_epoch - HIPPARCOS_EPOCH
    mas_to_deg = 1.0 / (3600.0 * 1000.0)

    dec_rad = np.radians(hip_table["dec"])
    cos_dec = np.cos(dec_rad)

    # Mask for stars away from poles
    safe = np.abs(cos_dec) > 0.05  # |dec| < ~87°

    # pmra is mu_alpha*cos(delta) in mas/yr; divide by cos(dec) for true RA rate
    dra = np.where(safe, hip_table["pmra"] / cos_dec * mas_to_deg * dt, 0.0)
    ddec = np.where(safe, hip_table["pmdec"] * mas_to_deg * dt, 0.0)

    hip_table["ra"] = hip_table["ra"] + dra
    hip_table["dec"] = hip_table["dec"] + ddec

    print(
        f"  Propagated Hipparcos positions from J{HIPPARCOS_EPOCH} to J{target_epoch} "
        f"(dt={dt:.2f} yr)"
    )


def estimate_gaia_g_from_hip(vmag: np.ndarray, bv: np.ndarray) -> np.ndarray:
    """Rough V-band to Gaia G-band conversion using B-V color.

    Based on Jordi+ 2010 / Evans+ 2018 polynomial fits. Accuracy ~0.1 mag,
    which is sufficient for identifying bright-star gaps.
    """
    # G - V ≈ -0.0176 - 0.00686*BV - 0.1732*BV^2 (Evans+ 2018, Table A.2)
    g_minus_v = -0.0176 - 0.00686 * bv - 0.1732 * bv**2
    return vmag + g_minus_v


def merge_catalogs(
    gaia_table,
    hip_table,
    bright_threshold: float = 4.0,
    match_radius_arcsec: float = 5.0,
):
    """Merge Gaia and Hipparcos catalogs.

    Hipparcos stars brighter than `bright_threshold` (in estimated G-band) that
    have no Gaia counterpart within `match_radius_arcsec` are added to fill
    Gaia's bright-star gap.
    """
    # Estimate G-mag for Hipparcos stars
    hip_gmag_est = estimate_gaia_g_from_hip(hip_table["vmag"], hip_table["bv"])

    # Select only bright Hipparcos stars as candidates for gap-filling
    bright_mask = hip_gmag_est < bright_threshold
    hip_bright = hip_table[bright_mask]
    hip_gmag_bright = hip_gmag_est[bright_mask]
    print(f"  {len(hip_bright)} Hipparcos stars brighter than G ~ {bright_threshold}")

    # Cross-match to find Hipparcos stars NOT in Gaia
    hip_coords = SkyCoord(
        ra=hip_bright["ra"] * u.deg, dec=hip_bright["dec"] * u.deg
    )
    gaia_coords = SkyCoord(
        ra=np.array(gaia_table["ra"]) * u.deg,
        dec=np.array(gaia_table["dec"]) * u.deg,
    )

    # For each bright Hipparcos star, find nearest Gaia neighbor
    idx, sep, _ = hip_coords.match_to_catalog_sky(gaia_coords)
    no_match = sep > match_radius_arcsec * u.arcsec

    hip_only = hip_bright[no_match]
    hip_gmag_only = hip_gmag_bright[no_match]
    print(f"  {len(hip_only)} bright Hipparcos stars have no Gaia match (added)")

    if len(hip_only) > 0:
        print("  Bright Hipparcos-only stars:")
        for i in range(len(hip_only)):
            print(
                f"    HIP {hip_only['hip'][i]:6d}  "
                f"V={hip_only['vmag'][i]:5.2f}  "
                f"G~{hip_gmag_only[i]:5.2f}  "
                f"({hip_only['ra'][i]:8.4f}, {hip_only['dec'][i]:+8.4f})"
            )

    return hip_only, hip_gmag_only


def write_merged_csv(gaia_table, hip_only, hip_gmag_only, output_path: str):
    """Write merged catalog in the CSV format expected by gaia.rs."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_id", "ra", "dec",
            "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag",
            "parallax", "pmra", "pmdec",
        ])

        # Write Gaia stars
        for row in gaia_table:
            writer.writerow([
                int(row["source_id"]),
                float(row["ra"]),
                float(row["dec"]),
                float(row["phot_g_mean_mag"]),
                _val_or_empty(row, "phot_bp_mean_mag"),
                _val_or_empty(row, "phot_rp_mean_mag"),
                _val_or_empty(row, "parallax"),
                _val_or_empty(row, "pmra"),
                _val_or_empty(row, "pmdec"),
            ])

        # Write Hipparcos gap-fill stars with synthetic source_ids
        # Use negative IDs to distinguish from real Gaia source_ids
        for i in range(len(hip_only)):
            hip_id = int(hip_only["hip"][i])
            writer.writerow([
                -hip_id,  # negative = Hipparcos origin
                float(hip_only["ra"][i]),
                float(hip_only["dec"][i]),
                float(hip_gmag_only[i]),  # estimated G-mag
                "",  # no BP
                "",  # no RP
                "",  # no parallax
                float(hip_only["pmra"][i]),   # mas/yr, mu_alpha*cos(delta)
                float(hip_only["pmdec"][i]),  # mas/yr
            ])

    total = len(gaia_table) + len(hip_only)
    print(f"\nWrote {total} stars to {output_path}")
    print(f"  ({len(gaia_table)} Gaia + {len(hip_only)} Hipparcos gap-fill)")


def _val_or_empty(row, col):
    """Return value or empty string if masked/nan."""
    val = row[col]
    try:
        if np.ma.is_masked(val) or np.isnan(val):
            return ""
    except (TypeError, ValueError):
        pass
    return float(val)


def _safe_float(row, col, default=0.0):
    """Return float value, or default if masked/nan."""
    val = row[col]
    try:
        if np.ma.is_masked(val) or np.isnan(val):
            return default
    except (TypeError, ValueError):
        pass
    return float(val)


def write_merged_binary(gaia_table, hip_only, hip_gmag_only, output_path: str):
    """Write merged catalog in the compact binary format expected by gaia.rs.

    Binary format:
        Header:  b'GDR3' + version (u32 LE, 1) + num_stars (u64 LE)
        Per star: source_id (i64 LE) + ra (f64 LE) + dec (f64 LE)
                  + mag (f32 LE) + pmra (f32 LE) + pmdec (f32 LE)
        = 36 bytes per star
    """
    import struct

    total = len(gaia_table) + len(hip_only)

    with open(output_path, "wb") as f:
        # Header
        f.write(b"GDR3")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<Q", total))

        # Gaia stars
        for row in gaia_table:
            f.write(struct.pack(
                "<qddfff",
                int(row["source_id"]),
                float(row["ra"]),
                float(row["dec"]),
                float(row["phot_g_mean_mag"]),
                _safe_float(row, "pmra"),
                _safe_float(row, "pmdec"),
            ))

        # Hipparcos gap-fill stars
        for i in range(len(hip_only)):
            f.write(struct.pack(
                "<qddfff",
                -int(hip_only["hip"][i]),
                float(hip_only["ra"][i]),
                float(hip_only["dec"][i]),
                float(hip_gmag_only[i]),
                float(hip_only["pmra"][i]),
                float(hip_only["pmdec"][i]),
            ))

    print(f"\nWrote {total} stars to {output_path}")
    print(f"  ({len(gaia_table)} Gaia + {len(hip_only)} Hipparcos gap-fill)")


def main():
    parser = argparse.ArgumentParser(
        description="Download Gaia DR3 catalog and merge with Hipparcos 2 bright stars"
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
        help="Output path; use .csv for CSV or .bin for binary (default: data/gaia_merged.bin)",
    )
    args = parser.parse_args()

    # Validate hip2.dat exists
    if not Path(args.hip2).exists():
        print(f"Error: {args.hip2} not found. Run scripts/download_hip2.sh first.")
        sys.exit(1)

    # Step 1: Query Gaia DR3
    gaia_table = query_gaia(args.mag_limit)

    # Step 2: Load Hipparcos 2 and propagate to Gaia epoch (J2016.0)
    hip_table = load_hipparcos(args.hip2)
    propagate_hipparcos_to_epoch(hip_table, GAIA_EPOCH)

    # Step 3: Merge (add bright Hipparcos stars missing from Gaia)
    hip_only, hip_gmag_only = merge_catalogs(
        gaia_table,
        hip_table,
        bright_threshold=args.bright_threshold,
        match_radius_arcsec=args.match_radius,
    )

    # Step 4: Write output (format based on file extension)
    if args.output.endswith(".bin"):
        write_merged_binary(gaia_table, hip_only, hip_gmag_only, args.output)
    else:
        write_merged_csv(gaia_table, hip_only, hip_gmag_only, args.output)


if __name__ == "__main__":
    main()
