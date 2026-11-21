"""TESS multi-image calibration test — mirrors tess_multi_image.ipynb.

Extracts centroids from 10 TESS FFI images (same CCD across sectors),
iteratively solves + calibrates with progressively tighter parameters,
and verifies final RMSE and WCS agreement.

Requires Gaia catalog and TESS same-CCD images (downloaded from GCS).
"""

import pytest
import tetra3rs

from .conftest import (
    TESS_SECTORS,
    angular_sep_deg,
    fits_pixel_to_radec,
    read_fits_image,
)

# Extraction settings for TESS full-frame images (mirrors `tess_extract_config`
# in the Rust TESS tests). The PSF is undersampled (sigma ~0.75 px); the field
# is crowded, so only high-SNR stars are wanted and the background grid is
# coarse; and saturated stars bleed along columns, so the elongation gate is
# loosened to keep the mildly bled bright stars the solver's patterns need.
TESS_EXTRACT = dict(
    psf_sigma=0.8,
    sigma_threshold=50.0,
    bg_block_size=128,
    max_elongation=6.0,
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
**TESS_EXTRACT)
            all_centroids.append(ext.centroids)
            all_headers.append(headers)

        # -- Tiered solve + calibrate (matching notebook) --
        pass_configs = [
            # (match_radius, cal_order, fov_error_deg)
            (0.005, 3, 0.5),
            (0.005, 4, 0.5),
            (0.003, 5, 0.5),
            (0.002, 6, 0.5),
        ]

        camera_model = None

        for match_radius, cal_order, fov_err in pass_configs:
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
        n_solved = sum(1 for r in results if r)
        assert n_solved == len(sectors), f"Only {n_solved}/{len(sectors)} images solved"

        # Check per-image quality
        arcsec_per_px = results[0].fov_deg * 3600 / sci_size

        for idx, (result, headers, sector) in enumerate(
            zip(results, all_headers, sectors)
        ):
            assert result, f"Sector {sector}: no solution ({result})"
            assert result.status == "match_found"

            rmse = result.rmse_arcsec
            rmse_px = rmse / arcsec_per_px

            # Compare center pixel against FITS WCS
            solved_ra, solved_dec = result.pixel_to_world(0.0, 0.0)

            # FITS WCS reference: geometric center of the 2048x2048 science
            # region (pixel-center origin (W-1)/2, issue #28) in full-frame
            # 0-indexed coords is (44 + 1023.5, 1023.5) — matching the origin of
            # pixel_to_world(0, 0) above.
            wcs_ra, wcs_dec = fits_pixel_to_radec(
                headers, 44.0 + (sci_size - 1) / 2.0, (sci_size - 1) / 2.0
            )
            sep = angular_sep_deg(solved_ra, solved_dec, wcs_ra, wcs_dec) * 3600

            print(
                f"  Sector {sector:2d}: {result.num_matches:3d} matches, "
                f'RMSE={rmse:.2f}" ({rmse_px:.3f} px), '
                f'vs WCS={sep:.2f}"'
            )

            assert rmse < 15.0, f'Sector {sector}: RMSE {rmse:.1f}" exceeds 15"'
            assert sep < 10.0, f'Sector {sector}: WCS separation {sep:.1f}" exceeds 10"'

    def test_radial_calibration(self, tess_db, tess_image_paths):
        """Multi-image Brown-Conrady (radial + tangential) calibration on TESS.

        Companion to ``test_tiered_calibration`` (SIP polynomial). The
        ``"radial"`` model fits the full Brown-Conrady form
        ``(cx, cy, k1, k2, k3, p1, p2)`` jointly via Levenberg-Marquardt:
        radial coefficients capture the symmetric component, tangential
        coefficients (p1, p2) capture decentering. The test verifies that:

          1. ``calibrate_camera(model="radial", ...)`` runs end-to-end
             through the alternating-refinement multi-image path.
          2. The fitted CameraModel carries a RadialDistortion (not None,
             not PolynomialDistortion).
          3. Calibration meaningfully reduces residuals vs. the
             un-calibrated solve.
          4. All sectors re-solve with attitude agreeing with the FITS WCS
             to within ~2 arcminutes per image.

        Bounds are looser than the polynomial test because the 7-parameter
        Brown-Conrady model captures less distortion than a 4th-order SIP
        polynomial — but tight enough to reliably distinguish a working fit
        from a regression.
        """
        sectors = [s for s in TESS_SECTORS if s in tess_image_paths]
        if len(sectors) < 5:
            pytest.skip(
                f"Only {len(sectors)} TESS images available, need >= 5 for radial calibration"
            )

        sci_size = 2048
        all_centroids = []
        all_headers = []
        for sector in sectors:
            path = tess_image_paths[sector]
            image, headers = read_fits_image(path, hdu_index=1)
            image = image[:2048, 44:2092].copy()
            ext = tetra3rs.extract_centroids(
                image,
**TESS_EXTRACT)
            all_centroids.append(ext.centroids)
            all_headers.append(headers)

        # -- Initial solve (no camera model yet) --
        initial_results = []
        for centroids in all_centroids:
            result = tess_db.solve_from_centroids(
                centroids,
                fov_estimate_deg=11.8,
                fov_max_error_deg=0.5,
                image_shape=(sci_size, sci_size),
                # 0.005 (matching the Rust TESS tests): at 0.01 with this
                # database's 3000 verification stars, a wrong attitude
                # coincidence-matches ~90% of centroids, leaving verification
                # statistically uninformative for raw-distortion frames.
                match_radius=0.005,
                match_threshold=1e-5,
            )
            initial_results.append(result)

        n_initially_solved = sum(
            1 for r in initial_results if r.status == "match_found"
        )
        assert n_initially_solved >= 3, (
            f"Need ≥3 initial solves to fit radial; got {n_initially_solved}"
        )

        # -- Radial calibration (multi-image alternating refinement) --
        cal = tess_db.calibrate_camera(
            initial_results,
            all_centroids,
            image_shape=(sci_size, sci_size),
            model="radial",
        )

        # Verify the fitted model is a RadialDistortion
        assert isinstance(cal.camera_model.distortion, tetra3rs.RadialDistortion), (
            f"expected RadialDistortion, got {type(cal.camera_model.distortion).__name__}"
        )
        rd = cal.camera_model.distortion
        print(
            f"  Radial fit: k1={rd.k1:.3e}, k2={rd.k2:.3e}, k3={rd.k3:.3e}, "
            f"RMSE {cal.rmse_before_px:.3f} -> {cal.rmse_after_px:.3f} px, "
            f"{cal.n_inliers}/{cal.n_inliers + cal.n_outliers} inliers"
        )

        # Calibration must actually improve things
        assert cal.rmse_after_px < cal.rmse_before_px, (
            f"radial calibration didn't reduce RMSE: "
            f"{cal.rmse_before_px:.3f} -> {cal.rmse_after_px:.3f} px"
        )
        # 3-parameter radial leaves residuals on TESS (decentering not captured),
        # but should still bring the pooled fit under a few pixels.
        assert cal.rmse_after_px < 5.0, (
            f"radial calibration RMSE {cal.rmse_after_px:.2f} px exceeds 5 px "
            f"— suspiciously bad fit for a multi-image radial pass"
        )

        # -- Re-solve with the radial-calibrated camera model --
        # match_radius=0.005 (matching the Rust TESS solve configs): the
        # radial model's ~3 px residuals sit comfortably inside the ~10 px
        # radius, and against this database's 3000 verification stars a
        # 0.01 radius saturates the coincidence null (a wrong attitude
        # matches most centroids by chance), leaving verification
        # uninformative. (An older comment here justified 0.01 by
        # wrong-attitude alternate solves at 0.005 — a failure mode of the
        # pre-0.9 verification statistics that no longer exists.)
        results = []
        for centroids in all_centroids:
            result = tess_db.solve_from_centroids(
                centroids,
                fov_estimate_deg=cal.camera_model.fov_deg,
                fov_max_error_deg=0.5,
                image_shape=(sci_size, sci_size),
                match_radius=0.005,
                match_threshold=1e-5,
                camera_model=cal.camera_model,
            )
            results.append(result)

        n_solved = sum(1 for r in results if r.status == "match_found")
        assert n_solved == len(sectors), (
            f"Only {n_solved}/{len(sectors)} re-solved with radial model"
        )

        # Per-image WCS check. Brown-Conrady captures TESS's radial AND
        # tangential distortion, so all sectors should solve with attitude
        # agreement within ~2 arcminutes. Bound is loose vs. the polynomial
        # test (~0.5" agreement) because a 7-parameter Brown-Conrady fit
        # leaves more residual than a 4th-order SIP polynomial, but tight
        # enough to reliably catch a regression that breaks the fit.
        max_wcs_sep_arcsec = 150.0  # 2.5 arcmin
        for result, headers, sector in zip(results, all_headers, sectors):
            rmse = result.rmse_arcsec
            solved_ra, solved_dec = result.pixel_to_world(0.0, 0.0)
            # Geometric center (W-1)/2 (issue #28), matching pixel_to_world(0, 0).
            wcs_ra, wcs_dec = fits_pixel_to_radec(
                headers, 44.0 + (sci_size - 1) / 2.0, (sci_size - 1) / 2.0
            )
            sep = angular_sep_deg(solved_ra, solved_dec, wcs_ra, wcs_dec) * 3600
            print(
                f"  Sector {sector:2d}: {result.num_matches:4d} matches, "
                f'RMSE={rmse:.2f}", vs WCS={sep:.2f}"'
            )
            assert sep < max_wcs_sep_arcsec, (
                f'Sector {sector}: WCS separation {sep:.1f}" exceeds '
                f'{max_wcs_sep_arcsec}"'
            )
