"""Shared fixtures and helpers for tetra3rs Python tests."""

import math
import os
import urllib.request

import numpy as np
import pytest

import tetra3rs

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
GCS_BASE = "https://storage.googleapis.com/tetra3rs-testvecs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def download_if_missing(local_path, gcs_key):
    """Download a test data file from GCS if not already present."""
    if os.path.exists(local_path):
        return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    url = f"{GCS_BASE}/{gcs_key}"
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, local_path)
    return local_path


def angular_sep_deg(ra1, dec1, ra2, dec2):
    """Angular separation in degrees between two (RA, Dec) positions."""
    ra1, dec1 = math.radians(ra1), math.radians(dec1)
    ra2, dec2 = math.radians(ra2), math.radians(dec2)
    cos_sep = math.sin(dec1) * math.sin(dec2) + math.cos(dec1) * math.cos(
        dec2
    ) * math.cos(ra1 - ra2)
    cos_sep = max(-1.0, min(1.0, cos_sep))
    return math.degrees(math.acos(cos_sep))


def project_stars_tan(stars, ra0_deg, dec0_deg, f_px, image_size):
    """Project catalog stars into pixel coords via TAN (gnomonic) projection.

    Returns a list of Centroid objects for stars within the image bounds.
    """
    ra0 = math.radians(ra0_deg)
    dec0 = math.radians(dec0_deg)
    cos_dec0 = math.cos(dec0)
    sin_dec0 = math.sin(dec0)
    half = image_size / 2.0

    centroids = []
    for s in stars:
        ra = math.radians(s.ra_deg)
        dec = math.radians(s.dec_deg)
        cos_dec = math.cos(dec)
        sin_dec = math.sin(dec)
        cos_dra = math.cos(ra - ra0)
        sin_dra = math.sin(ra - ra0)

        denom = sin_dec * sin_dec0 + cos_dec * cos_dec0 * cos_dra
        if denom <= 0:
            continue
        xi = (cos_dec * sin_dra) / denom
        eta = (sin_dec * cos_dec0 - cos_dec * sin_dec0 * cos_dra) / denom

        x = xi * f_px
        y = eta * f_px
        if abs(x) < half and abs(y) < half:
            centroids.append(
                tetra3rs.Centroid(x=x, y=y, brightness=10.0 - s.magnitude)
            )
    return centroids


def fits_pixel_to_radec(headers, pixel_x, pixel_y):
    """Convert full-frame 0-indexed pixel coords to RA/Dec using CD+SIP WCS.

    Implements: pixel → SIP forward → CD matrix → TAN deprojection.
    """
    crpix1 = headers["CRPIX1"]
    crpix2 = headers["CRPIX2"]
    crval1 = headers["CRVAL1"]
    crval2 = headers["CRVAL2"]

    # Convert to 1-indexed FITS convention, then offset from CRPIX
    u = (pixel_x + 1.0) - crpix1
    v = (pixel_y + 1.0) - crpix2

    # Apply SIP distortion (forward: pixel → corrected pixel)
    a_order = headers.get("A_ORDER")
    b_order = headers.get("B_ORDER")
    if a_order is not None and b_order is not None:
        f = 0.0
        for p in range(a_order + 1):
            for q in range(a_order - p + 1):
                c = headers.get(f"A_{p}_{q}", 0.0)
                if c != 0.0:
                    f += c * u**p * v**q
        g = 0.0
        for p in range(b_order + 1):
            for q in range(b_order - p + 1):
                c = headers.get(f"B_{p}_{q}", 0.0)
                if c != 0.0:
                    g += c * u**p * v**q
        u, v = u + f, v + g

    # CD matrix → intermediate world coordinates (degrees)
    cd11 = headers.get("CD1_1", 0.0)
    cd12 = headers.get("CD1_2", 0.0)
    cd21 = headers.get("CD2_1", 0.0)
    cd22 = headers.get("CD2_2", 0.0)
    xi_deg = cd11 * u + cd12 * v
    eta_deg = cd21 * u + cd22 * v

    # TAN (gnomonic) deprojection
    xi = math.radians(xi_deg)
    eta = math.radians(eta_deg)
    ra0 = math.radians(crval1)
    dec0 = math.radians(crval2)

    denom = math.cos(dec0) - eta * math.sin(dec0)
    ra = ra0 + math.atan2(xi, denom)
    dec = math.atan2(
        math.sin(dec0) + eta * math.cos(dec0),
        math.sqrt(xi**2 + denom**2),
    )
    return math.degrees(ra) % 360.0, math.degrees(dec)


def read_fits_image(path, hdu_index=0):
    """Minimal FITS reader — returns (data, headers) for a 2D image HDU.

    Handles float32 (BITPIX=-32) and int16 (BITPIX=16) with BZERO/BSCALE.
    """
    with open(path, "rb") as f:
        current_hdu = 0
        while True:
            headers = {}
            end_found = False
            while not end_found:
                block = f.read(2880)
                if len(block) < 2880:
                    raise ValueError(
                        f"Unexpected end of file in HDU {current_hdu}"
                    )
                for i in range(36):
                    card = block[i * 80 : (i + 1) * 80].decode(
                        "ascii", errors="replace"
                    )
                    keyword = card[:8].strip()
                    if keyword == "END":
                        end_found = True
                        break
                    if len(card) < 10 or card[8:10] != "= ":
                        continue
                    value_str = card[10:].strip()
                    if value_str.startswith("'"):
                        end_q = value_str.find("'", 1)
                        if end_q > 0:
                            headers[keyword] = value_str[1:end_q].strip()
                        continue
                    if "/" in value_str:
                        value_str = value_str[: value_str.index("/")].strip()
                    value_str = value_str.strip()
                    if not value_str:
                        continue
                    if value_str == "T":
                        headers[keyword] = True
                    elif value_str == "F":
                        headers[keyword] = False
                    else:
                        try:
                            headers[keyword] = int(value_str)
                        except ValueError:
                            try:
                                headers[keyword] = float(value_str)
                            except ValueError:
                                headers[keyword] = value_str

            naxis = headers.get("NAXIS", 0)
            bitpix = headers.get("BITPIX", 8)
            data_bytes = 0
            if naxis > 0:
                npixels = 1
                for ax in range(1, naxis + 1):
                    npixels *= headers.get(f"NAXIS{ax}", 1)
                data_bytes = npixels * abs(bitpix) // 8
            data_bytes += headers.get("PCOUNT", 0)

            if current_hdu == hdu_index and data_bytes > 0:
                naxis1 = headers["NAXIS1"]
                naxis2 = headers["NAXIS2"]
                raw = f.read(data_bytes)
                if bitpix == -32:
                    data = np.frombuffer(raw, dtype=">f4").reshape(
                        naxis2, naxis1
                    )
                elif bitpix == -64:
                    data = np.frombuffer(raw, dtype=">f8").reshape(
                        naxis2, naxis1
                    )
                elif bitpix == 16:
                    data = (
                        np.frombuffer(raw, dtype=">i2")
                        .reshape(naxis2, naxis1)
                        .astype(np.float32)
                    )
                    data = data * headers.get("BSCALE", 1) + headers.get(
                        "BZERO", 0
                    )
                elif bitpix == 32:
                    data = (
                        np.frombuffer(raw, dtype=">i4")
                        .reshape(naxis2, naxis1)
                        .astype(np.float32)
                    )
                    data = data * headers.get("BSCALE", 1) + headers.get(
                        "BZERO", 0
                    )
                else:
                    raise ValueError(f"Unsupported BITPIX: {bitpix}")
                data = np.where(np.isfinite(data), data, 0.0).astype(
                    np.float32
                )
                return data, headers

            # Skip data + padding, advance to next HDU
            padded = ((data_bytes + 2879) // 2880) * 2880
            f.seek(padded, 1)
            current_hdu += 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def ensure_gaia_catalog():
    """Ensure gaia_merged.bin is available, downloading from GCS if needed."""
    bin_path = os.path.join(DATA_DIR, "gaia_merged.bin")
    download_if_missing(bin_path, "gaia_merged.bin")
    return bin_path


@pytest.fixture(scope="session")
def gaia_catalog_path():
    """Ensure Gaia merged catalog is available, downloading from GCS if needed."""
    try:
        return ensure_gaia_catalog()
    except Exception as e:
        pytest.skip(f"Could not obtain Gaia catalog: {e}")


@pytest.fixture(scope="session")
def skyview_db(gaia_catalog_path):
    """Load (or generate and cache) a database suitable for 10deg SkyView images."""
    cache_path = os.path.join(DATA_DIR, "test_skyview_db.rkyv")
    if os.path.exists(cache_path):
        return tetra3rs.SolverDatabase.load_from_file(cache_path)
    db = tetra3rs.SolverDatabase.generate_from_gaia(
        gaia_catalog_path,
        max_fov_deg=15.0,
        star_max_magnitude=7.0,
        pattern_max_error=0.005,
        patterns_per_lattice_field=150,
        verification_stars_per_fov=50,
        epoch_proper_motion_year=2000.0,
        catalog_nside=8,
    )
    os.makedirs(DATA_DIR, exist_ok=True)
    db.save_to_file(cache_path)
    return db


SKYVIEW_IMAGES = [
    "orion_region_10deg.fits",
    "pleiades_region_10deg.fits",
    "cygnus_region_10deg.fits",
    "andromeda_m31_10deg.fits",
    "cassiopeia_10deg.fits",
]


@pytest.fixture(scope="session")
def skyview_image_paths():
    """Ensure SkyView FITS test images are available."""
    paths = {}
    for name in SKYVIEW_IMAGES:
        local = os.path.join(DATA_DIR, "skyview_10deg_test_images", name)
        gcs_key = f"skyview_10deg_test_images/{name}"
        try:
            download_if_missing(local, gcs_key)
            paths[name] = local
        except Exception:
            pass
    if not paths:
        pytest.skip("No SkyView test images available")
    return paths


# TESS same-CCD sectors (matching the notebook)
TESS_SECTORS = [1, 2, 3, 4, 5, 6, 13, 14, 15, 17]


@pytest.fixture(scope="session")
def tess_db(gaia_catalog_path):
    """Load (or generate and cache) a database suitable for TESS ~12deg FOV images."""
    cache_path = os.path.join(DATA_DIR, "test_tess_db.rkyv")
    if os.path.exists(cache_path):
        return tetra3rs.SolverDatabase.load_from_file(cache_path)
    db = tetra3rs.SolverDatabase.generate_from_gaia(
        gaia_catalog_path,
        max_fov_deg=14.0,
        pattern_max_error=0.005,
        lattice_field_oversampling=100,
        patterns_per_lattice_field=500,
        verification_stars_per_fov=3000,
        epoch_proper_motion_year=2018.0,
    )
    os.makedirs(DATA_DIR, exist_ok=True)
    db.save_to_file(cache_path)
    return db


@pytest.fixture(scope="session")
def tess_image_paths():
    """Ensure TESS same-CCD test images are available."""
    paths = {}
    for sector in TESS_SECTORS:
        name = f"sector{sector:02d}_cam1_ccd1.fits"
        local = os.path.join(DATA_DIR, "tess_same_ccd", name)
        gcs_key = f"tess_same_ccd/{name}"
        try:
            download_if_missing(local, gcs_key)
            paths[sector] = local
        except Exception:
            pass
    if not paths:
        pytest.skip("No TESS test images available")
    return paths
