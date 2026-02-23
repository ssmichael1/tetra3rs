"""SkyView image integration tests — extract centroids from real FITS images and solve.

Requires hip2.dat and SkyView test images (downloaded automatically from GCS).
"""

import pytest

import tetra3rs

from .conftest import angular_sep_deg, read_fits_image


class TestSkyViewSolve:
    """Extract centroids from SkyView FITS images and solve for attitude."""

    @pytest.mark.parametrize(
        "filename",
        [
            "orion_region_10deg.fits",
            "pleiades_region_10deg.fits",
            "cygnus_region_10deg.fits",
            "andromeda_m31_10deg.fits",
            "cassiopeia_10deg.fits",
        ],
    )
    def test_solve_skyview_image(self, skyview_db, skyview_image_paths, filename):
        if filename not in skyview_image_paths:
            pytest.skip(f"{filename} not available")

        path = skyview_image_paths[filename]
        image, headers = read_fits_image(path, hdu_index=0)
        h, w = image.shape

        # Get ground-truth WCS from FITS header
        crval_ra = headers["CRVAL1"]
        crval_dec = headers["CRVAL2"]
        cdelt1 = headers["CDELT1"]
        cdelt2 = headers["CDELT2"]
        fov_h_deg = abs(cdelt2) * w

        # Extract centroids
        result = tetra3rs.extract_centroids(
            image,
            sigma_threshold=10.0,
            min_pixels=3,
            max_pixels=10000,
            max_centroids=200,
            local_bg_block_size=64,
            max_elongation=3.0,
        )
        assert len(result.centroids) >= 4, (
            f"Only {len(result.centroids)} centroids extracted"
        )

        # Apply parity correction: negate x when CDELT1 < 0 (RA increases left)
        parity_x = -1.0 if cdelt1 < 0 else 1.0
        parity_y = -1.0 if cdelt2 < 0 else 1.0
        centroids = [
            tetra3rs.Centroid(
                x=c.x * parity_x,
                y=c.y * parity_y,
                brightness=c.brightness,
            )
            for c in result.centroids
        ]

        # Solve
        solve = skyview_db.solve_from_centroids(
            centroids,
            fov_estimate_deg=fov_h_deg,
            image_width=w,
            image_height=h,
            fov_max_error_deg=3.0,
            match_radius=0.01,
            match_threshold=1e-5,
            solve_timeout_ms=60_000,
        )

        assert solve is not None, f"{filename}: no solution found"
        assert solve.status == "match_found"
        assert solve.num_matches >= 4

        # Boresight should be within 30' of FITS WCS center
        error_deg = angular_sep_deg(solve.ra_deg, solve.dec_deg, crval_ra, crval_dec)
        error_arcmin = error_deg * 60.0
        assert error_arcmin < 30.0, (
            f"{filename}: boresight error {error_arcmin:.1f}' > 30'"
        )

        # Extra checks for Orion: pixel↔world conversion, FOV, match count
        if filename == "orion_region_10deg.fits":
            ra, dec = solve.pixel_to_world(0.0, 0.0)
            center_err = angular_sep_deg(ra, dec, crval_ra, crval_dec)
            assert center_err < 1.0, (
                f"Center pixel RA/Dec error: {center_err:.3f}°"
            )
            assert solve.fov_deg is not None
            assert abs(solve.fov_deg - fov_h_deg) < 1.0
            assert solve.num_matches >= 10
            assert solve.rmse_arcsec < 60.0
