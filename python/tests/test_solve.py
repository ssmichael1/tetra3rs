"""Database and solve tests — requires Gaia catalog."""

import math
import os
import pickle
import tempfile

import numpy as np
import pytest

import tetra3rs

from .conftest import angular_sep_deg, project_stars_tan


# ---------------------------------------------------------------------------
# Database generation and persistence
# ---------------------------------------------------------------------------


class TestDatabaseGeneration:
    def test_generate_and_properties(self, skyview_db):
        assert skyview_db.num_stars > 1000
        assert skyview_db.num_patterns > 10_000
        assert abs(skyview_db.max_fov_deg - 15.0) < 0.1

    def test_save_load_roundtrip(self, skyview_db):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            skyview_db.save_to_file(path)
            db2 = tetra3rs.SolverDatabase.load_from_file(path)
            assert db2.num_stars == skyview_db.num_stars
            assert db2.num_patterns == skyview_db.num_patterns
        finally:
            os.unlink(path)

    def test_pickle_roundtrip(self, skyview_db):
        data = pickle.dumps(skyview_db)
        db2 = pickle.loads(data)
        assert db2.num_stars == skyview_db.num_stars
        assert db2.num_patterns == skyview_db.num_patterns


# ---------------------------------------------------------------------------
# Catalog access
# ---------------------------------------------------------------------------


class TestCatalogAccess:
    def test_get_star_by_index(self, skyview_db):
        star = skyview_db.get_star(0)
        assert 0 <= star.ra_deg < 360
        assert -90 <= star.dec_deg <= 90
        assert star.magnitude < 2.0  # brightest star in catalog

    def test_get_star_by_id(self, skyview_db):
        # Sirius = HIP 32349
        star = skyview_db.get_star_by_id(32349)
        if star is not None:
            assert abs(star.ra_deg - 101.287) < 0.5
            assert abs(star.dec_deg - (-16.716)) < 0.5
            assert star.magnitude < 0.0

    def test_cone_search(self, skyview_db):
        # Search 5° around Orion belt region
        stars = skyview_db.cone_search(83.0, -1.0, 5.0)
        assert len(stars) > 5
        # All returned stars should be within the search radius
        for s in stars:
            sep = angular_sep_deg(s.ra_deg, s.dec_deg, 83.0, -1.0)
            assert sep <= 5.1, f"Star at {sep:.2f}° exceeds radius"

    def test_cone_search_sorted_by_brightness(self, skyview_db):
        stars = skyview_db.cone_search(83.0, -1.0, 5.0)
        mags = [s.magnitude for s in stars]
        assert mags == sorted(mags), "Cone search should return brightest first"


# ---------------------------------------------------------------------------
# Solve from synthetic centroids
# ---------------------------------------------------------------------------


class TestSolveFromCentroids:
    """Solve using catalog stars projected via TAN projection."""

    # Test several sky regions for robustness
    @pytest.mark.parametrize(
        "ra, dec, label",
        [
            (83.0, -1.0, "Orion"),
            (56.75, 24.1, "Pleiades"),
            (310.0, 45.0, "Cygnus"),
            (0.0, 89.0, "North pole"),
            (180.0, -60.0, "Southern sky"),
        ],
    )
    def test_solve_synthetic(self, skyview_db, ra, dec, label):
        fov_deg = 10.0
        image_size = 2048
        f_px = image_size / (2.0 * math.tan(math.radians(fov_deg / 2.0)))

        stars = skyview_db.cone_search(ra, dec, fov_deg)
        centroids = project_stars_tan(stars[:50], ra, dec, f_px, image_size)
        assert len(centroids) >= 4, f"{label}: too few centroids ({len(centroids)})"

        result = skyview_db.solve_from_centroids(
            centroids,
            fov_estimate_deg=fov_deg,
            image_width=image_size,
            image_height=image_size,
            fov_max_error_deg=3.0,
            solve_timeout_ms=60_000,
        )

        assert result is not None, f"{label}: no solution"
        assert result.status == "match_found"
        error = angular_sep_deg(result.ra_deg, result.dec_deg, ra, dec)
        assert error < 0.5, f"{label}: boresight error {error:.3f}° > 0.5°"

    def test_solve_with_numpy_array(self, skyview_db):
        """Verify centroids can be passed as Nx2 numpy array."""
        ra, dec = 83.0, -1.0
        fov_deg = 10.0
        image_size = 2048
        f_px = image_size / (2.0 * math.tan(math.radians(fov_deg / 2.0)))

        stars = skyview_db.cone_search(ra, dec, fov_deg)
        centroid_objs = project_stars_tan(stars[:50], ra, dec, f_px, image_size)
        arr = np.array([[c.x, c.y] for c in centroid_objs], dtype=np.float64)

        result = skyview_db.solve_from_centroids(
            arr,
            fov_estimate_deg=fov_deg,
            image_width=image_size,
            image_height=image_size,
            fov_max_error_deg=3.0,
        )
        assert result is not None
        assert result.status == "match_found"

    def test_solve_with_numpy_array_3col(self, skyview_db):
        """Verify centroids can be passed as Nx3 numpy array (x, y, brightness)."""
        ra, dec = 83.0, -1.0
        fov_deg = 10.0
        image_size = 2048
        f_px = image_size / (2.0 * math.tan(math.radians(fov_deg / 2.0)))

        stars = skyview_db.cone_search(ra, dec, fov_deg)
        centroid_objs = project_stars_tan(stars[:50], ra, dec, f_px, image_size)
        arr = np.array(
            [[c.x, c.y, c.brightness or 0] for c in centroid_objs],
            dtype=np.float64,
        )

        result = skyview_db.solve_from_centroids(
            arr,
            fov_estimate_deg=fov_deg,
            image_width=image_size,
            image_height=image_size,
            fov_max_error_deg=3.0,
        )
        assert result is not None
        assert result.status == "match_found"


# ---------------------------------------------------------------------------
# SolveResult properties and coordinate conversion
# ---------------------------------------------------------------------------


class TestSolveResult:
    @pytest.fixture()
    def orion_result(self, skyview_db):
        """Solve Orion region and return the result."""
        ra, dec = 83.0, -1.0
        fov_deg = 10.0
        image_size = 2048
        f_px = image_size / (2.0 * math.tan(math.radians(fov_deg / 2.0)))

        stars = skyview_db.cone_search(ra, dec, fov_deg)
        centroids = project_stars_tan(stars[:50], ra, dec, f_px, image_size)

        result = skyview_db.solve_from_centroids(
            centroids,
            fov_estimate_deg=fov_deg,
            image_width=image_size,
            image_height=image_size,
            fov_max_error_deg=3.0,
        )
        assert result is not None
        return result

    def test_basic_properties(self, orion_result):
        r = orion_result
        assert isinstance(r.ra_deg, float)
        assert isinstance(r.dec_deg, float)
        assert isinstance(r.roll_deg, float)
        assert isinstance(r.solve_time_ms, float)
        assert r.solve_time_ms > 0

    def test_match_quality(self, orion_result):
        r = orion_result
        assert isinstance(r.num_matches, int)
        assert r.num_matches >= 4
        assert isinstance(r.rmse_arcsec, float)
        assert r.rmse_arcsec > 0
        assert r.rmse_arcsec < 60  # better than 1 arcmin
        assert isinstance(r.probability, float)

    def test_rotation_matrix(self, orion_result):
        R = orion_result.rotation_matrix_icrs_to_camera
        assert R.shape == (3, 3)
        # Should be a proper rotation (orthogonal, det=+1)
        assert abs(np.linalg.det(R) - 1.0) < 1e-4
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-4)

    def test_matched_arrays(self, orion_result):
        r = orion_result
        assert len(r.matched_centroids) == r.num_matches
        assert len(r.matched_catalog_ids) == r.num_matches
        # Catalog IDs should be nonzero (positive = Gaia, negative = Hipparcos gap-fill)
        assert all(cid != 0 for cid in r.matched_catalog_ids)

    def test_wcs_fields(self, orion_result):
        r = orion_result
        assert r.crval_ra_deg is not None
        assert r.crval_dec_deg is not None
        assert r.cd_matrix is not None
        assert r.cd_matrix.shape == (2, 2)
        assert r.theta_deg is not None

    def test_fov(self, orion_result):
        r = orion_result
        assert r.fov_deg is not None
        assert abs(r.fov_deg - 10.0) < 1.0

    def test_pixel_to_world_center(self, orion_result):
        """Center pixel should map to near the boresight."""
        r = orion_result
        result = r.pixel_to_world(0.0, 0.0)
        assert result is not None
        ra, dec = result
        error = angular_sep_deg(ra, dec, r.ra_deg, r.dec_deg)
        assert error < 0.5

    def test_world_to_pixel_center(self, orion_result):
        """Boresight RA/Dec should map back to near image center."""
        r = orion_result
        result = r.world_to_pixel(r.ra_deg, r.dec_deg)
        assert result is not None
        x, y = result
        assert abs(x) < 10.0
        assert abs(y) < 10.0

    def test_pixel_world_roundtrip(self, orion_result):
        """pixel→world→pixel should round-trip within ~1 pixel."""
        r = orion_result
        x0, y0 = 200.0, -150.0
        ra, dec = r.pixel_to_world(x0, y0)
        x1, y1 = r.world_to_pixel(ra, dec)
        assert abs(x1 - x0) < 1.0, f"x roundtrip error: {abs(x1-x0):.2f} px"
        assert abs(y1 - y0) < 1.0, f"y roundtrip error: {abs(y1-y0):.2f} px"

    def test_pixel_to_world_arrays(self, orion_result):
        """pixel_to_world accepts numpy arrays."""
        r = orion_result
        xs = np.array([0.0, 100.0, -100.0])
        ys = np.array([0.0, 50.0, -50.0])
        result = r.pixel_to_world(xs, ys)
        assert result is not None
        ras, decs = result
        assert len(ras) == 3
        assert len(decs) == 3

    def test_solve_result_pickle(self, orion_result):
        r2 = pickle.loads(pickle.dumps(orion_result))
        assert abs(r2.ra_deg - orion_result.ra_deg) < 1e-6
        assert abs(r2.dec_deg - orion_result.dec_deg) < 1e-6
        assert r2.num_matches == orion_result.num_matches


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_single_image_calibrate(self, skyview_db):
        """Solve and calibrate from a single synthetic image."""
        ra, dec = 83.0, -1.0
        fov_deg = 10.0
        image_size = 2048
        f_px = image_size / (2.0 * math.tan(math.radians(fov_deg / 2.0)))

        stars = skyview_db.cone_search(ra, dec, fov_deg)
        centroids = project_stars_tan(stars[:50], ra, dec, f_px, image_size)

        result = skyview_db.solve_from_centroids(
            centroids,
            fov_estimate_deg=fov_deg,
            image_width=image_size,
            image_height=image_size,
            fov_max_error_deg=3.0,
        )
        assert result is not None

        cal = skyview_db.calibrate_camera(
            result,
            centroids,
            image_width=image_size,
            image_height=image_size,
            order=2,
        )

        assert isinstance(cal, tetra3rs.CalibrateResult)
        assert cal.camera_model is not None
        assert cal.rmse_after_px <= cal.rmse_before_px + 0.01
        assert cal.n_inliers > 0

    def test_calibrate_result_pickle(self, skyview_db):
        ra, dec = 83.0, -1.0
        fov_deg = 10.0
        image_size = 2048
        f_px = image_size / (2.0 * math.tan(math.radians(fov_deg / 2.0)))

        stars = skyview_db.cone_search(ra, dec, fov_deg)
        centroids = project_stars_tan(stars[:50], ra, dec, f_px, image_size)

        result = skyview_db.solve_from_centroids(
            centroids,
            fov_estimate_deg=fov_deg,
            image_width=image_size,
            image_height=image_size,
            fov_max_error_deg=3.0,
        )

        cal = skyview_db.calibrate_camera(
            result,
            centroids,
            image_width=image_size,
            image_height=image_size,
            order=2,
        )
        cal2 = pickle.loads(pickle.dumps(cal))
        assert cal2.n_inliers == cal.n_inliers
        assert abs(cal2.rmse_after_px - cal.rmse_after_px) < 1e-6


class TestTrackingMode:
    """Solve with an attitude hint (tracking mode)."""

    def _make_centroids(self, skyview_db, ra, dec, fov_deg, image_size):
        f_px = image_size / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
        stars = skyview_db.cone_search(ra, dec, fov_deg)
        return project_stars_tan(stars[:50], ra, dec, f_px, image_size)

    @pytest.mark.parametrize("hint_kind", ["quaternion", "rotation_matrix"])
    def test_tracking_with_hint(self, skyview_db, hint_kind):
        """LIS solve → perturb attitude → re-solve with hint. Must agree with LIS."""
        ra, dec, fov_deg, image_size = 83.0, -1.0, 10.0, 2048
        centroids = self._make_centroids(skyview_db, ra, dec, fov_deg, image_size)
        assert len(centroids) >= 4

        # Step 1: lost-in-space solve
        lis = skyview_db.solve_from_centroids(
            centroids,
            fov_estimate_deg=fov_deg,
            image_width=image_size,
            image_height=image_size,
            fov_max_error_deg=3.0,
        )
        assert lis is not None, "LIS solve failed"

        # Step 2: perturb the recovered attitude by 15' around a random axis.
        # Small-angle quaternion: q_pert = [cos(θ/2), sin(θ/2)·axis]
        perturb_rad = math.radians(15.0 / 60.0)
        rng = np.random.default_rng(42)
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        half = perturb_rad / 2.0
        q_pert = np.array([math.cos(half), *(math.sin(half) * axis)])
        qw, qx, qy, qz = lis.quaternion
        # Hamilton product q_pert * lis.quaternion
        pw, px, py, pz = q_pert
        hinted_quat = np.array([
            pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
        ])

        # Pass either the quaternion or the rotation matrix form.
        if hint_kind == "quaternion":
            hint = hinted_quat
        else:
            # Build the rotation matrix from the perturbed quaternion.
            w, x, y, z = hinted_quat
            hint = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])

        tracked = skyview_db.solve_from_centroids(
            centroids,
            fov_estimate_deg=fov_deg,
            image_width=image_size,
            image_height=image_size,
            camera_model=lis.camera_model,
            attitude_hint=hint,
            hint_uncertainty_deg=1.0,
            strict_hint=True,  # no LIS fallback — we want to test tracking specifically
        )

        assert tracked is not None, f"Tracking solve failed with {hint_kind} hint"
        # On noiseless synthetic data, tracking and LIS converge to the same
        # fixed point of wcs_refine — agreement should be well below 1″.
        # We use 1″ as a loose bound that catches gross regressions but
        # tolerates any numerical scatter on real-data use cases.
        sep_arcsec = angular_sep_deg(lis.ra_deg, lis.dec_deg, tracked.ra_deg, tracked.dec_deg) * 3600.0
        assert sep_arcsec < 1.0, (
            f"Tracked boresight disagrees with LIS by {sep_arcsec:.3f}″ ({hint_kind})"
        )

    def test_tracking_fallback_to_lis_on_bad_hint(self, skyview_db):
        """A wildly wrong hint should fall back to LIS (default behavior)."""
        ra, dec, fov_deg, image_size = 83.0, -1.0, 10.0, 2048
        centroids = self._make_centroids(skyview_db, ra, dec, fov_deg, image_size)

        # Bogus hint pointing 180° away from the actual field.
        bad_hint = np.array([0.0, 1.0, 0.0, 0.0])  # 180° rotation about X

        result = skyview_db.solve_from_centroids(
            centroids,
            fov_estimate_deg=fov_deg,
            image_width=image_size,
            image_height=image_size,
            fov_max_error_deg=3.0,
            attitude_hint=bad_hint,
            hint_uncertainty_deg=0.5,  # tight — hint must be near truth
            # strict_hint=False → fallback to LIS
        )

        assert result is not None, "Should have fallen back to LIS"
        error = angular_sep_deg(result.ra_deg, result.dec_deg, ra, dec)
        assert error < 0.5, f"LIS fallback solved wrong field (error {error:.3f}°)"

    def test_strict_hint_fails_on_bad_hint(self, skyview_db):
        """With strict_hint=True, a bad hint should produce None."""
        ra, dec, fov_deg, image_size = 83.0, -1.0, 10.0, 2048
        centroids = self._make_centroids(skyview_db, ra, dec, fov_deg, image_size)
        bad_hint = np.array([0.0, 1.0, 0.0, 0.0])

        result = skyview_db.solve_from_centroids(
            centroids,
            fov_estimate_deg=fov_deg,
            image_width=image_size,
            image_height=image_size,
            attitude_hint=bad_hint,
            hint_uncertainty_deg=0.5,
            strict_hint=True,
        )
        assert result is None, "strict_hint should suppress LIS fallback on bad hint"

    def test_attitude_hint_invalid_shape_raises(self, skyview_db):
        """Passing a hint with wrong shape should raise ValueError."""
        centroids = self._make_centroids(skyview_db, 83.0, -1.0, 10.0, 2048)
        with pytest.raises(ValueError):
            skyview_db.solve_from_centroids(
                centroids,
                fov_estimate_deg=10.0,
                image_width=2048,
                image_height=2048,
                attitude_hint=np.zeros(5),  # wrong size
            )

    def test_hint_uncertainty_both_raises(self, skyview_db):
        centroids = self._make_centroids(skyview_db, 83.0, -1.0, 10.0, 2048)
        with pytest.raises(ValueError):
            skyview_db.solve_from_centroids(
                centroids,
                fov_estimate_deg=10.0,
                image_width=2048,
                image_height=2048,
                attitude_hint=np.array([1.0, 0.0, 0.0, 0.0]),
                hint_uncertainty_deg=1.0,
                hint_uncertainty_rad=0.01,
            )


