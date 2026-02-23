"""TESS multi-image calibration test — mirrors tess_multi_image.ipynb.

Extracts centroids from 10 TESS FFI images (same CCD across sectors),
iteratively solves + calibrates with progressively tighter parameters,
and verifies final RMSE and WCS agreement.

Requires hip2.dat and TESS same-CCD images (downloaded from GCS).
"""

import numpy as np
import pytest

import tetra3rs

from .conftest import (
    TESS_SECTORS,
    angular_sep_deg,
    fits_pixel_to_radec,
    read_fits_image,
)


class TestTessMultiImageCalibration:
    """Multi-image tiered calibration on TESS Camera 1 CCD 1."""

    def test_tiered_calibration(self, tess_db, tess_image_paths):
        available = sorted(tess_image_paths.keys())
        if len(available) < 5:
            pytest.skip(f"Only {len(available)} TESS images available, need >= 5")

        # Use all available sectors that are in the notebook list
        sectors = [s for s in TESS_SECTORS if s in tess_image_paths]

        # -- Extract centroids from all images --
        all_centroids = []
        all_headers = []
        sci_size = 2048

        for sector in sectors:
            path = tess_image_paths[sector]
            # TESS data is in HDU 1; trim to 2048x2048 science region
            image, headers = read_fits_image(path, hdu_index=1)
            image = image[:2048, 44:2092].copy()

            ext = tetra3rs.extract_centroids(
                image,
                sigma_threshold=300.0,
                min_pixels=4,
                max_pixels=10000,
                local_bg_block_size=16,
                max_elongation=6.0,
            )
            all_centroids.append(ext.centroids)
            all_headers.append(headers)

        # -- Tiered solve + calibrate (matching notebook) --
        pass_configs = [
            # (match_radius, refine_iterations, cal_order, fov_error_deg)
            (0.01, 10, 3, 0.5),
            (0.003, 10, 5, 0.5),
            (0.002, 10, 6, 0.5),
            (0.001, 10, 6, 0.5),
        ]

        camera_model = None

        for match_radius, refine_iter, cal_order, fov_err in pass_configs:
            results = []
            fov_est = camera_model.fov_deg if camera_model else 11.8

            for centroids in all_centroids:
                result = tess_db.solve_from_centroids(
                    centroids,
                    fov_estimate_deg=fov_est,
                    fov_max_error_deg=fov_err,
                    image_shape=(sci_size, sci_size),
                    match_radius=match_radius,
                    match_threshold=1e-5,
                    refine_iterations=refine_iter,
                    camera_model=camera_model,
                )
                results.append(result)

            cal = tess_db.calibrate_camera(
                results,
                all_centroids,
                image_shape=(sci_size, sci_size),
                order=cal_order,
            )
            camera_model = cal.camera_model

        # -- Verify final results --
        # All images should be solved
        n_solved = sum(1 for r in results if r is not None)
        assert n_solved == len(sectors), (
            f"Only {n_solved}/{len(sectors)} images solved"
        )

        # Check per-image quality
        arcsec_per_px = results[0].fov_deg * 3600 / sci_size

        for idx, (result, headers, sector) in enumerate(
            zip(results, all_headers, sectors)
        ):
            assert result is not None, f"Sector {sector}: no solution"
            assert result.status == "match_found"

            rmse = result.rmse_arcsec
            rmse_px = rmse / arcsec_per_px

            # Compare center pixel against FITS WCS
            solved_ra, solved_dec = result.pixel_to_world(0.0, 0.0)

            # FITS WCS reference: center of the 2048x2048 science region
            # in full-frame coords is (44 + 1024, 1024)
            wcs_ra, wcs_dec = fits_pixel_to_radec(
                headers, 44.0 + sci_size / 2.0, sci_size / 2.0
            )
            sep = angular_sep_deg(solved_ra, solved_dec, wcs_ra, wcs_dec) * 3600

            print(
                f"  Sector {sector:2d}: {result.num_matches:3d} matches, "
                f'RMSE={rmse:.2f}" ({rmse_px:.3f} px), '
                f'vs WCS={sep:.2f}"'
            )

            assert rmse < 15.0, (
                f'Sector {sector}: RMSE {rmse:.1f}" exceeds 15"'
            )
            assert sep < 10.0, (
                f'Sector {sector}: WCS separation {sep:.1f}" exceeds 10"'
            )
