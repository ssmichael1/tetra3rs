"""Basic API tests — no test data files needed."""

import math
import pickle
from datetime import datetime

import numpy as np
import pytest

import tetra3rs


# ---------------------------------------------------------------------------
# Import and module structure
# ---------------------------------------------------------------------------


class TestImport:
    def test_module_name(self):
        assert tetra3rs.__name__ == "tetra3rs"

    def test_all_classes_exist(self):
        for name in [
            "CameraModel",
            "CalibrateResult",
            "CatalogStar",
            "Centroid",
            "ExtractionResult",
            "PolynomialDistortion",
            "RadialDistortion",
            "SolveResult",
            "SolverDatabase",
        ]:
            assert hasattr(tetra3rs, name), f"Missing: {name}"

    def test_all_functions_exist(self):
        for name in ["earth_barycentric_velocity", "extract_centroids"]:
            assert callable(getattr(tetra3rs, name)), f"Not callable: {name}"


# ---------------------------------------------------------------------------
# CameraModel
# ---------------------------------------------------------------------------


class TestCameraModel:
    def test_construction(self):
        cm = tetra3rs.CameraModel(
            focal_length_px=1000.0, image_width=2048, image_height=2048
        )
        assert cm.focal_length_px == 1000.0
        assert cm.image_width == 2048
        assert cm.image_height == 2048
        assert cm.parity_flip is False
        assert cm.distortion is None

    def test_from_fov(self):
        cm = tetra3rs.CameraModel.from_fov(
            fov_deg=10.0, image_width=2048, image_height=2048
        )
        expected_f = 2048.0 / (2.0 * math.tan(math.radians(5.0)))
        assert abs(cm.focal_length_px - expected_f) < 1.0
        assert abs(cm.fov_deg - 10.0) < 0.01

    def test_crpix_default(self):
        cm = tetra3rs.CameraModel(
            focal_length_px=1000.0, image_width=2048, image_height=2048
        )
        crpix = cm.crpix
        assert abs(crpix[0]) < 1e-6
        assert abs(crpix[1]) < 1e-6

    def test_crpix_custom(self):
        cm = tetra3rs.CameraModel(
            focal_length_px=1000.0,
            image_width=2048,
            image_height=2048,
            crpix=[5.0, -3.0],
        )
        crpix = cm.crpix
        assert abs(crpix[0] - 5.0) < 1e-6
        assert abs(crpix[1] - (-3.0)) < 1e-6

    def test_parity_flip(self):
        cm = tetra3rs.CameraModel(
            focal_length_px=1000.0,
            image_width=2048,
            image_height=2048,
            parity_flip=True,
        )
        assert cm.parity_flip is True

    def test_pixel_scale(self):
        cm = tetra3rs.CameraModel(
            focal_length_px=1000.0, image_width=2048, image_height=2048
        )
        assert abs(cm.pixel_scale() - 1.0 / 1000.0) < 1e-8

    def test_pixel_tanplane_roundtrip(self):
        cm = tetra3rs.CameraModel(
            focal_length_px=1000.0, image_width=2048, image_height=2048
        )
        xi, eta = cm.pixel_to_tanplane(100.0, -50.0)
        px, py = cm.tanplane_to_pixel(xi, eta)
        assert abs(px - 100.0) < 1e-4
        assert abs(py - (-50.0)) < 1e-4

    def test_pixel_tanplane_roundtrip_with_distortion(self):
        # Use a small k1 appropriate for pixel coordinates (~1024 px half-width)
        dist = tetra3rs.RadialDistortion(k1=-1e-8)
        cm = tetra3rs.CameraModel(
            focal_length_px=1000.0,
            image_width=2048,
            image_height=2048,
            distortion=dist,
        )
        xi, eta = cm.pixel_to_tanplane(100.0, -50.0)
        px, py = cm.tanplane_to_pixel(xi, eta)
        assert abs(px - 100.0) < 0.5
        assert abs(py - (-50.0)) < 0.5

    def test_pickle_roundtrip(self):
        cm = tetra3rs.CameraModel(
            focal_length_px=1234.5,
            image_width=1024,
            image_height=768,
            crpix=[2.0, -1.5],
            parity_flip=True,
        )
        cm2 = pickle.loads(pickle.dumps(cm))
        assert cm2.focal_length_px == cm.focal_length_px
        assert cm2.image_width == cm.image_width
        assert cm2.image_height == cm.image_height
        assert cm2.parity_flip == cm.parity_flip
        assert abs(cm2.crpix[0] - cm.crpix[0]) < 1e-6

    def test_pickle_with_radial_distortion(self):
        dist = tetra3rs.RadialDistortion(k1=-1e-8, k2=1e-16)
        cm = tetra3rs.CameraModel(
            focal_length_px=1000.0,
            image_width=2048,
            image_height=2048,
            distortion=dist,
        )
        cm2 = pickle.loads(pickle.dumps(cm))
        assert cm2.distortion is not None
        assert abs(cm2.distortion.k1 - (-1e-8)) < 1e-15


# ---------------------------------------------------------------------------
# Centroid
# ---------------------------------------------------------------------------


class TestCentroid:
    def test_construction(self):
        c = tetra3rs.Centroid(x=10.5, y=-20.0, brightness=500.0)
        assert c.x == pytest.approx(10.5, abs=1e-5)
        assert c.y == pytest.approx(-20.0, abs=1e-5)
        assert c.brightness == pytest.approx(500.0, abs=1e-3)

    def test_no_brightness(self):
        c = tetra3rs.Centroid(x=1.0, y=2.0)
        assert c.brightness is None

    def test_with_offset(self):
        c = tetra3rs.Centroid(x=10.0, y=20.0, brightness=100.0)
        c2 = c.with_offset(5.0, -3.0)
        assert abs(c2.x - 15.0) < 1e-6
        assert abs(c2.y - 17.0) < 1e-6
        assert c2.brightness == 100.0

    def test_pickle_roundtrip(self):
        c = tetra3rs.Centroid(x=1.5, y=-2.5, brightness=42.0)
        c2 = pickle.loads(pickle.dumps(c))
        assert c2.x == c.x
        assert c2.y == c.y
        assert c2.brightness == c.brightness


# ---------------------------------------------------------------------------
# RadialDistortion
# ---------------------------------------------------------------------------


class TestRadialDistortion:
    def test_construction(self):
        d = tetra3rs.RadialDistortion(k1=-1e-8, k2=1e-16, k3=-1e-24)
        assert d.k1 == pytest.approx(-1e-8)
        assert d.k2 == pytest.approx(1e-16)
        assert d.k3 == pytest.approx(-1e-24)

    def test_defaults(self):
        d = tetra3rs.RadialDistortion()
        assert d.k1 == 0.0
        assert d.k2 == 0.0
        assert d.k3 == 0.0

    def test_distort_undistort_roundtrip(self):
        # Coefficients must be small for pixel-coordinate inputs
        d = tetra3rs.RadialDistortion(k1=-1e-8, k2=1e-16)
        x, y = 100.0, 200.0
        xd, yd = d.distort(x, y)
        xu, yu = d.undistort(xd, yd)
        assert abs(xu - x) < 0.1
        assert abs(yu - y) < 0.1

    def test_zero_distortion_is_identity(self):
        d = tetra3rs.RadialDistortion()
        x, y = 50.0, -75.0
        xd, yd = d.distort(x, y)
        assert abs(xd - x) < 1e-10
        assert abs(yd - y) < 1e-10

    def test_pickle_roundtrip(self):
        d = tetra3rs.RadialDistortion(k1=-1e-8, k2=1e-16, k3=-1e-24)
        d2 = pickle.loads(pickle.dumps(d))
        assert d2.k1 == pytest.approx(d.k1)
        assert d2.k2 == pytest.approx(d.k2)
        assert d2.k3 == pytest.approx(d.k3)

    def test_centroid_distort_undistort(self):
        d = tetra3rs.RadialDistortion(k1=-1e-8)
        c = tetra3rs.Centroid(x=100.0, y=200.0, brightness=50.0)
        cd = c.distort(d)
        cu = cd.undistort(d)
        assert abs(cu.x - c.x) < 0.1
        assert abs(cu.y - c.y) < 0.1


# ---------------------------------------------------------------------------
# PolynomialDistortion
# ---------------------------------------------------------------------------


class TestPolynomialDistortion:
    def test_construction_order2(self):
        # Order 2: all terms with p+q <= 2 → (order+1)*(order+2)/2 = 6
        n = 6
        zeros = np.zeros(n, dtype=np.float64)
        d = tetra3rs.PolynomialDistortion(
            order=2,
            scale=1024.0,
            a_coeffs=zeros,
            b_coeffs=zeros,
            ap_coeffs=zeros,
            bp_coeffs=zeros,
        )
        assert d.order == 2
        assert d.scale == 1024.0
        assert d.num_coeffs == n

    def test_zero_polynomial_is_identity(self):
        n = 6
        zeros = np.zeros(n, dtype=np.float64)
        d = tetra3rs.PolynomialDistortion(
            order=2,
            scale=1024.0,
            a_coeffs=zeros,
            b_coeffs=zeros,
            ap_coeffs=zeros,
            bp_coeffs=zeros,
        )
        x, y = 100.0, -50.0
        xd, yd = d.distort(x, y)
        assert abs(xd - x) < 1e-10
        assert abs(yd - y) < 1e-10

    def test_pickle_roundtrip(self):
        n = 6
        a = np.zeros(n, dtype=np.float64)
        a[3] = 0.001  # a non-zero coefficient
        zeros = np.zeros(n, dtype=np.float64)
        d = tetra3rs.PolynomialDistortion(
            order=2,
            scale=1024.0,
            a_coeffs=a,
            b_coeffs=zeros,
            ap_coeffs=zeros,
            bp_coeffs=zeros,
        )
        d2 = pickle.loads(pickle.dumps(d))
        assert d2.order == d.order
        assert d2.scale == d.scale
        np.testing.assert_array_equal(d2.a_coeffs, d.a_coeffs)


# ---------------------------------------------------------------------------
# earth_barycentric_velocity
# ---------------------------------------------------------------------------


class TestEarthBarycentricVelocity:
    def test_returns_3_floats(self):
        v = tetra3rs.earth_barycentric_velocity(datetime(2025, 7, 10))
        assert len(v) == 3
        assert all(isinstance(vi, float) for vi in v)

    def test_speed_is_30_km_s(self):
        v = tetra3rs.earth_barycentric_velocity(datetime(2025, 7, 10))
        speed = math.sqrt(sum(vi**2 for vi in v))
        assert 25.0 < speed < 35.0, f"Speed {speed:.1f} km/s outside [25, 35]"

    def test_opposite_at_6_months(self):
        v1 = tetra3rs.earth_barycentric_velocity(datetime(2025, 1, 1))
        v2 = tetra3rs.earth_barycentric_velocity(datetime(2025, 7, 1))
        dot = sum(a * b for a, b in zip(v1, v2))
        assert dot < 0, "Velocity should be roughly anti-parallel at 6-month separation"


# ---------------------------------------------------------------------------
# extract_centroids (with synthetic image)
# ---------------------------------------------------------------------------


class TestExtractCentroids:
    def test_gaussian_spots(self):
        """Create an image with Gaussian spots and verify extraction."""
        h, w = 512, 512
        image = np.random.normal(100, 5, (h, w)).astype(np.float32)

        # Add 5 bright Gaussian spots
        spots = [(100, 200), (300, 150), (250, 400), (50, 50), (450, 300)]
        for cy, cx in spots:
            yy, xx = np.mgrid[cy - 10 : cy + 11, cx - 10 : cx + 11]
            yy = np.clip(yy, 0, h - 1)
            xx = np.clip(xx, 0, w - 1)
            g = 5000 * np.exp(
                -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0**2)
            )
            image[yy, xx] += g.astype(np.float32)

        result = tetra3rs.extract_centroids(
            image, sigma_threshold=5.0, min_pixels=3, max_centroids=20
        )

        assert isinstance(result, tetra3rs.ExtractionResult)
        assert result.image_width == w
        assert result.image_height == h
        assert result.background_sigma > 0
        assert len(result.centroids) >= 3  # should find most of the 5 spots

        # Centroids should be centered (origin at image center)
        for c in result.centroids:
            assert abs(c.x) <= w / 2 + 1
            assert abs(c.y) <= h / 2 + 1

    def test_extraction_result_pickle(self):
        """ExtractionResult supports pickle."""
        h, w = 64, 64
        image = np.random.normal(100, 5, (h, w)).astype(np.float32)
        image[32, 32] = 10000  # bright pixel

        result = tetra3rs.extract_centroids(image, sigma_threshold=3.0)
        result2 = pickle.loads(pickle.dumps(result))
        assert result2.image_width == result.image_width
        assert len(result2.centroids) == len(result.centroids)
