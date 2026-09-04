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
        for name in [
            "earth_barycentric_velocity",
            "extract_centroids",
            "extract_centroids_fast",
        ]:
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

    def test_tangential_construction(self):
        d = tetra3rs.RadialDistortion(k1=-7e-9, k2=2e-15, p1=5e-7, p2=-3e-7)
        assert d.k1 == pytest.approx(-7e-9)
        assert d.k2 == pytest.approx(2e-15)
        assert d.k3 == 0.0
        assert d.p1 == pytest.approx(5e-7)
        assert d.p2 == pytest.approx(-3e-7)

    def test_tangential_defaults_zero(self):
        d = tetra3rs.RadialDistortion(k1=-1e-8)
        assert d.p1 == 0.0
        assert d.p2 == 0.0

    def test_tangential_changes_distortion(self):
        radial_only = tetra3rs.RadialDistortion(k1=-1e-8)
        with_tang = tetra3rs.RadialDistortion(k1=-1e-8, p1=5e-7, p2=-3e-7)
        x, y = 300.0, 400.0
        xr, yr = radial_only.distort(x, y)
        xt, yt = with_tang.distort(x, y)
        assert (xr, yr) != (xt, yt)

    def test_tangential_distort_undistort_roundtrip(self):
        d = tetra3rs.RadialDistortion(k1=-7e-9, k2=2e-15, p1=5e-7, p2=-3e-7)
        for x, y in [(100.0, 200.0), (-500.0, 300.0), (1024.0, -512.0)]:
            xd, yd = d.distort(x, y)
            xu, yu = d.undistort(xd, yd)
            assert abs(xu - x) < 1e-4
            assert abs(yu - y) < 1e-4

    def test_tangential_pickle_roundtrip(self):
        d = tetra3rs.RadialDistortion(k1=-7e-9, k2=2e-15, k3=-1e-24, p1=5e-7, p2=-3e-7)
        d2 = pickle.loads(pickle.dumps(d))
        assert d2.k1 == pytest.approx(d.k1)
        assert d2.k2 == pytest.approx(d.k2)
        assert d2.k3 == pytest.approx(d.k3)
        assert d2.p1 == pytest.approx(d.p1)
        assert d2.p2 == pytest.approx(d.p2)


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

    def test_construct_without_inverse_coeffs(self):
        """ap/bp are optional — the model inverts numerically."""
        n = 6
        a = np.zeros(n, dtype=np.float64)
        a[3] = 0.01
        b = np.zeros(n, dtype=np.float64)
        d = tetra3rs.PolynomialDistortion(order=2, scale=1024.0, a_coeffs=a, b_coeffs=b)
        # Forward → inverse should still round-trip via Newton iteration.
        xd, yd = d.distort(120.0, -80.0)
        xu, yu = d.undistort(xd, yd)
        assert abs(xu - 120.0) < 1e-6
        assert abs(yu + 80.0) < 1e-6


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

    def test_fast_gaussian_spots(self):
        """The single-pass fast path recovers spots over a gradient."""
        h, w = 512, 512
        # Background with a left-to-right gradient + noise.
        gradient = np.linspace(50, 250, w, dtype=np.float32)[None, :]
        image = (np.random.normal(0, 5, (h, w)) + gradient).astype(np.float32)

        spots = [(100, 200), (300, 150), (250, 400), (50, 50), (450, 300)]
        for cy, cx in spots:
            yy, xx = np.mgrid[cy - 10 : cy + 11, cx - 10 : cx + 11]
            yy = np.clip(yy, 0, h - 1)
            xx = np.clip(xx, 0, w - 1)
            g = 5000 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0**2))
            image[yy, xx] += g.astype(np.float32)

        result = tetra3rs.extract_centroids_fast(
            image, sigma_threshold=5.0, bg_grid=64, max_centroids=20
        )
        assert isinstance(result, tetra3rs.ExtractionResult)
        assert result.image_width == w and result.image_height == h
        assert result.background_sigma > 0
        assert len(result.centroids) >= 3

        # Brightest of the 5 injected spots should be recovered to ~1 px.
        found = [(c.x + w / 2, c.y + h / 2) for c in result.centroids]
        for cy, cx in spots:
            nearest = min(((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5 for fx, fy in found)
            assert nearest < 1.0, f"spot ({cx},{cy}) nearest detection {nearest:.2f} px"

    def test_degenerate_image_raises_not_panics(self):
        """Zero-size / 1-wide images and bad config raise, not abort."""
        empty = np.zeros((0, 0), dtype=np.float32)
        with pytest.raises((ValueError, TypeError)):
            tetra3rs.extract_centroids(empty)
        tiny = np.zeros((1, 1), dtype=np.float32)
        with pytest.raises(ValueError):
            tetra3rs.extract_centroids(tiny)
        img = np.zeros((16, 16), dtype=np.float32)
        with pytest.raises(ValueError):
            tetra3rs.extract_centroids(img, local_bg_block_size=0)

    def test_non_ndarray_raises_typeerror(self):
        """A plain list gives a clear TypeError, not an AttributeError."""
        with pytest.raises(TypeError):
            tetra3rs.extract_centroids([[1.0, 2.0], [3.0, 4.0]])

    def test_big_endian_matches_native(self):
        """Big-endian input (as FITS yields) extracts the same as native."""
        h, w = 128, 128
        native = np.full((h, w), 100.0, dtype=np.float32)
        yy, xx = np.mgrid[54:75, 54:75]
        native[yy, xx] += (
            5000 * np.exp(-((yy - 64) ** 2 + (xx - 64) ** 2) / (2 * 2.0**2))
        ).astype(np.float32)
        big_endian = native.astype(">f4")
        assert not big_endian.dtype.isnative
        r_native = tetra3rs.extract_centroids(native, sigma_threshold=5.0)
        r_big = tetra3rs.extract_centroids(big_endian, sigma_threshold=5.0)
        assert len(r_native.centroids) == len(r_big.centroids) >= 1
        assert r_native.image_width == r_big.image_width
        # Same values in → same centroid position out.
        assert abs(r_native.centroids[0].x - r_big.centroids[0].x) < 1e-4

    @pytest.mark.parametrize(
        "extract",
        [tetra3rs.extract_centroids, tetra3rs.extract_centroids_fast],
        ids=["ccl", "fast"],
    )
    def test_layout_and_dtype_give_identical_centroids(self, extract):
        """C-contiguous input takes a slice fast path; strided views and
        other dtypes must produce bit-identical centroids."""
        h, w = 256, 320
        rng = np.random.default_rng(7)
        image = rng.normal(500, 8, (h, w))
        yy, xx = np.mgrid[0:h, 0:w]
        for cy, cx in [(40, 60), (120, 200), (200, 90), (180, 280)]:
            image += 20000 * np.exp(
                -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.8**2)
            )
        u16 = np.ascontiguousarray(np.clip(image, 0, 65535).astype(np.uint16))
        assert u16.flags.c_contiguous

        def key(img):
            r = extract(img, sigma_threshold=5.0, max_centroids=20)
            assert len(r.centroids) >= 3
            return [(c.x, c.y, c.brightness) for c in r.centroids]

        ref = key(u16)
        # Same pixels, non-contiguous layouts (strided fallback path).
        fortran = np.asfortranarray(u16)
        assert not fortran.flags.c_contiguous
        assert key(fortran) == ref
        padded = np.zeros((h, w + 8), dtype=np.uint16)
        padded[:, 4 : 4 + w] = u16
        window = padded[:, 4 : 4 + w]  # row stride != w: not contiguous
        assert not window.flags.c_contiguous
        assert key(window) == ref
        # Same pixels, other dtypes (each exactly representable in f32).
        for dtype in (np.int16, np.float32, np.float64):
            assert key(u16.astype(dtype)) == ref, dtype
        # A step-sliced view vs. its contiguous copy.
        view = u16[::2, ::2]
        assert not view.flags.c_contiguous
        assert key(view) == key(np.ascontiguousarray(view))

    def test_fast_result_pickle_and_drop_in(self):
        """Fast path returns a pickleable ExtractionResult with usable centroids."""
        h, w = 64, 64
        image = np.random.normal(100, 5, (h, w)).astype(np.float32)
        # A star-shaped 3x3 patch (a bare 2-pixel spike is a hot-pixel pair,
        # which the default sharpness gate now correctly rejects).
        image[32, 32] = 10000
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            image[32 + dr, 32 + dc] = 5000
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            image[32 + dr, 32 + dc] = 2500

        result = tetra3rs.extract_centroids_fast(image, sigma_threshold=4.0)
        assert len(result.centroids) >= 1
        result2 = pickle.loads(pickle.dumps(result))
        assert result2.image_width == result.image_width
        assert len(result2.centroids) == len(result.centroids)


class TestArgumentValidation:
    """Regression tests: degenerate arguments raise ValueError instead of
    panicking (PanicException) or silently poisoning downstream math."""

    def test_from_fov_rejects_degenerate_fov(self):
        for bad_fov in [0.0, -10.0, 180.0, 360.0, float("nan"), float("inf")]:
            with pytest.raises(ValueError):
                tetra3rs.CameraModel.from_fov(bad_fov, 1024, 768)
        with pytest.raises(ValueError):
            tetra3rs.CameraModel.from_fov(10.0, 0, 768)

    def test_camera_model_rejects_degenerate_intrinsics(self):
        with pytest.raises(ValueError):
            tetra3rs.CameraModel(0.0, 1024, 768)
        with pytest.raises(ValueError):
            tetra3rs.CameraModel(float("nan"), 1024, 768)
        with pytest.raises(ValueError):
            tetra3rs.CameraModel(5000.0, 0, 768)

    def test_polynomial_distortion_rejects_huge_order(self):
        # Used to construct via u32 wrap-around and panic on first distort().
        with pytest.raises(ValueError):
            tetra3rs.PolynomialDistortion(2**32 - 1, 1.0, [], [])
        with pytest.raises(ValueError):
            tetra3rs.PolynomialDistortion(100, 1.0, [0.0], [0.0])

    def test_distortion_rejects_non_finite(self):
        with pytest.raises(ValueError):
            tetra3rs.RadialDistortion(k1=float("nan"))
        n = 15  # order 4
        bad = [0.0] * n
        bad[3] = float("inf")
        with pytest.raises(ValueError):
            tetra3rs.PolynomialDistortion(4, 512.0, bad, [0.0] * n)

    def test_corrupt_camera_model_pickle_raises(self):
        cam = tetra3rs.CameraModel.from_fov(10.0, 1024, 768)
        blob = pickle.dumps(cam)
        # Truncating the payload must give a clean exception, not a panic on
        # a later method call.
        with pytest.raises(Exception) as excinfo:
            pickle.loads(blob[:-4])
        assert "Panic" not in type(excinfo.value).__name__


class TestCentroidExtractor:
    """`CentroidExtractor` reuses buffers across frames; output must equal
    `extract_centroids` bit for bit, including after a larger frame."""

    @staticmethod
    def _scene(w, h, seed):
        rng = np.random.default_rng(seed)
        image = rng.normal(100.0, 3.0, size=(h, w)).astype(np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        for _ in range(10):
            cx = rng.uniform(8, w - 8)
            cy = rng.uniform(8, h - 8)
            amp = rng.uniform(500, 3000)
            image += (
                amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.6**2))
            ).astype(np.float32)
        return image

    @staticmethod
    def _flat(result):
        return [
            (c.x, c.y, c.brightness, None if c.cov is None else c.cov.tobytes())
            for c in result.centroids
        ] + [
            result.background_mean,
            result.background_sigma,
            result.threshold,
            result.num_blobs_raw,
            result.image_width,
            result.image_height,
        ]

    def test_reuse_matches_free_function(self):
        extractor = tetra3rs.CentroidExtractor()
        assert repr(extractor) == "CentroidExtractor()"
        # big -> small -> big, matched filter on and off, local/global background
        frames = [
            ((200, 160), dict(matched_filter_sigma=1.5)),
            ((96, 80), dict(matched_filter_sigma=1.5)),
            ((96, 80), dict(matched_filter_sigma=None)),
            ((200, 160), dict(local_bg_block_size=None)),
            ((130, 70), {}),
        ]
        for i, ((w, h), kw) in enumerate(frames):
            image = self._scene(w, h, seed=i)
            fresh = tetra3rs.extract_centroids(image, sigma_threshold=5.0, **kw)
            reused = extractor.extract(image, sigma_threshold=5.0, **kw)
            assert len(fresh.centroids) >= 2
            assert self._flat(fresh) == self._flat(reused)

    def test_pickle_gives_a_working_extractor(self):
        extractor = tetra3rs.CentroidExtractor()
        image = self._scene(120, 90, seed=3)
        before = extractor.extract(image, sigma_threshold=5.0)
        restored = pickle.loads(pickle.dumps(extractor))
        assert isinstance(restored, tetra3rs.CentroidExtractor)
        after = restored.extract(image, sigma_threshold=5.0)
        assert self._flat(before) == self._flat(after)

    def test_rejects_bad_input_like_free_function(self):
        extractor = tetra3rs.CentroidExtractor()
        with pytest.raises(ValueError):
            extractor.extract(np.zeros((1, 1), dtype=np.float32))
        with pytest.raises(ValueError):
            extractor.extract(np.zeros((32, 32), dtype=np.float32), deblend="maybe")
