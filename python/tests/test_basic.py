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


def _star_field(w, h, stars, psf_sigma, noise=3.0, seed=0, background=100.0):
    """Noise frame with Gaussian stars at ``stars`` = [(x, y, amplitude), ...]
    (array coordinates: x = column, y = row, pixel centers at integers)."""
    rng = np.random.default_rng(seed)
    image = rng.normal(background, noise, (h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    for x, y, amp in stars:
        image += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * psf_sigma**2))
    return image.astype(np.float32)


def _random_stars(w, h, n, seed, amp=(500.0, 3000.0), margin=12):
    rng = np.random.default_rng(seed)
    return [
        (rng.uniform(margin, w - margin), rng.uniform(margin, h - margin), rng.uniform(*amp))
        for _ in range(n)
    ]


def _nearest_px(result, x, y):
    """Distance from array position (x, y) to the nearest centroid."""
    w, h = result.image_width, result.image_height
    return min(
        (
            math.hypot(c.x + (w - 1) / 2 - x, c.y + (h - 1) / 2 - y)
            for c in result.centroids
        ),
        default=math.inf,
    )


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
            image, sigma_threshold=5.0, psf_sigma=2.0, max_centroids=20
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
        """ExtractionResult (and its centroids) round-trip through pickle."""
        w, h = 200, 160
        image = _star_field(w, h, _random_stars(w, h, 8, seed=11), psf_sigma=1.5)

        result = tetra3rs.extract_centroids(image, sigma_threshold=5.0)
        assert len(result.centroids) >= 5
        result2 = pickle.loads(pickle.dumps(result))
        assert result2.image_width == result.image_width
        assert result2.image_height == result.image_height
        assert result2.background_mean == result.background_mean
        assert result2.background_sigma == result.background_sigma
        assert result2.threshold == result.threshold
        assert result2.num_blobs_raw == result.num_blobs_raw
        assert result2.psf_sigma_estimate == result.psf_sigma_estimate
        assert len(result2.centroids) == len(result.centroids)
        for c, c2 in zip(result.centroids, result2.centroids):
            assert (c2.x, c2.y, c2.brightness) == (c.x, c.y, c.brightness)
            assert np.array_equal(c2.cov, c.cov)
            assert np.array_equal(c2.shape, c.shape)
        # A single centroid pickles on its own too, shape included.
        c = result.centroids[0]
        c2 = pickle.loads(pickle.dumps(c))
        assert np.array_equal(c2.cov, c.cov) and np.array_equal(c2.shape, c.shape)

    def test_spots_over_gradient(self):
        """Spots are recovered over a background gradient."""
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

        result = tetra3rs.extract_centroids(
            image, sigma_threshold=5.0, psf_sigma=2.0, bg_block_size=64, max_centroids=20
        )
        assert isinstance(result, tetra3rs.ExtractionResult)
        assert result.image_width == w and result.image_height == h
        assert result.background_sigma > 0
        assert len(result.centroids) >= 5

        # Every injected spot is recovered to well under a pixel, and the
        # five spots are the five brightest detections.
        for cy, cx in spots:
            nearest = _nearest_px(result, cx, cy)
            assert nearest < 0.1, f"spot ({cx},{cy}) nearest detection {nearest:.3f} px"
        top5 = tetra3rs.extract_centroids(image, psf_sigma=2.0, max_centroids=5)
        assert len(top5.centroids) == 5
        for cy, cx in spots:
            assert _nearest_px(top5, cx, cy) < 0.1

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
            tetra3rs.extract_centroids(img, bg_block_size=0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(psf_sigma=0.0),
            dict(psf_sigma=-1.0),
            dict(psf_sigma=float("nan")),
            dict(psf_sigma=float("inf")),
            dict(psf_sigma=10.5),
            dict(detection_binning=0),
            dict(detection_binning=3),
            dict(sigma_threshold=0.0),
            dict(sigma_threshold=float("nan")),
        ],
        ids=repr,
    )
    def test_invalid_config_raises_valueerror(self, kwargs):
        img = np.zeros((64, 64), dtype=np.float32)
        with pytest.raises(ValueError):
            tetra3rs.extract_centroids(img, **kwargs)
        with pytest.raises(ValueError):
            tetra3rs.CentroidExtractor().extract(img, **kwargs)

    def test_valid_config_extremes_accepted(self):
        img = np.zeros((64, 64), dtype=np.float32)
        tetra3rs.extract_centroids(img, psf_sigma=10.0)
        tetra3rs.extract_centroids(img, psf_sigma=0.3, detection_binning=1)
        tetra3rs.extract_centroids(img, psf_sigma=0.8, detection_binning=2)
        tetra3rs.extract_centroids(img, max_elongation=None)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(min_pixels=3),
            dict(local_bg_block_size=64),
            dict(matched_filter_sigma=1.5),
            dict(max_sharpness=0.9),
            dict(deblend="off"),
        ],
        ids=repr,
    )
    def test_removed_arguments_are_rejected(self, kwargs):
        """Arguments of the pre-rewrite extractor fail loudly, not silently."""
        img = np.zeros((64, 64), dtype=np.float32)
        with pytest.raises(TypeError):
            tetra3rs.extract_centroids(img, **kwargs)
        with pytest.raises(TypeError):
            tetra3rs.CentroidExtractor().extract(img, **kwargs)
        assert not hasattr(tetra3rs, "extract_centroids_fast")

    def test_only_sigma_threshold_is_positional(self):
        img = np.zeros((64, 64), dtype=np.float32)
        tetra3rs.extract_centroids(img, 5.0)
        with pytest.raises(TypeError):
            tetra3rs.extract_centroids(img, 5.0, 1.5)
        with pytest.raises(TypeError):
            tetra3rs.CentroidExtractor().extract(img, 5.0, 1.5)

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

    @pytest.mark.parametrize("detection_binning", [1, 2], ids=["full", "binned"])
    def test_layout_and_dtype_give_identical_centroids(self, detection_binning):
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
            r = tetra3rs.extract_centroids(
                img,
                sigma_threshold=5.0,
                max_centroids=20,
                detection_binning=detection_binning,
            )
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

    def test_positions_are_subpixel_accurate(self):
        """Bright stars at random sub-pixel phases are located to a few
        hundredths of a pixel, and brightness orders them."""
        w, h = 400, 300
        stars = _random_stars(w, h, 30, seed=1)
        result = tetra3rs.extract_centroids(_star_field(w, h, stars, 1.5, seed=1))
        assert len(result.centroids) == len(stars)
        errors = [_nearest_px(result, x, y) for x, y, _ in stars]
        assert max(errors) < 0.05, f"max position error {max(errors):.4f} px"
        fluxes = [c.brightness for c in result.centroids]
        assert fluxes == sorted(fluxes, reverse=True)
        assert all(f > 0 for f in fluxes)

    def test_detection_binning_does_not_move_bright_stars(self):
        """Binning only affects detection: positions are measured on the
        unbinned pixels, so bright stars land in the same place either way."""
        w, h = 400, 300
        stars = _random_stars(w, h, 30, seed=2)
        image = _star_field(w, h, stars, 1.5, seed=2)
        full = tetra3rs.extract_centroids(image, detection_binning=1)
        binned = tetra3rs.extract_centroids(image, detection_binning=2)
        assert len(full.centroids) == len(binned.centroids) == len(stars)
        for a in full.centroids:
            # Pair by position (near-equal fluxes may sort differently).
            b = min(binned.centroids, key=lambda c: math.hypot(a.x - c.x, a.y - c.y))
            assert math.hypot(a.x - b.x, a.y - b.y) < 1e-3
            assert b.brightness == pytest.approx(a.brightness, rel=1e-3)
        for x, y, _ in stars:
            assert _nearest_px(full, x, y) < 0.05
            assert _nearest_px(binned, x, y) < 0.05

        def positions(r):
            return [(c.x, c.y) for c in r.centroids]

        # None picks binning 2 at psf_sigma >= 1.25 and 1 below it.
        assert positions(tetra3rs.extract_centroids(image)) == positions(binned)
        narrow_auto = tetra3rs.extract_centroids(image, psf_sigma=1.0)
        narrow_full = tetra3rs.extract_centroids(image, psf_sigma=1.0, detection_binning=1)
        assert positions(narrow_auto) == positions(narrow_full)

    @pytest.mark.parametrize("true_sigma", [1.0, 1.5, 2.0, 2.5])
    def test_psf_sigma_estimate_recovers_the_psf_width(self, true_sigma):
        """`psf_sigma_estimate` reports the PSF width measured from the frame,
        including when the configured `psf_sigma` is not quite right."""
        w, h = 400, 300
        image = _star_field(w, h, _random_stars(w, h, 30, seed=1), true_sigma, seed=1)
        matched = tetra3rs.extract_centroids(image, psf_sigma=true_sigma)
        assert matched.psf_sigma_estimate == pytest.approx(true_sigma, rel=0.03)
        default = tetra3rs.extract_centroids(image)  # psf_sigma=1.5
        assert default.psf_sigma_estimate == pytest.approx(true_sigma, rel=0.03)

    def test_psf_sigma_estimate_is_none_without_enough_stars(self):
        w, h = 200, 160
        blank = _star_field(w, h, [], 1.5)
        assert tetra3rs.extract_centroids(blank).psf_sigma_estimate is None
        few = _star_field(w, h, [(50.3, 40.6, 2000.0), (150.7, 110.2, 2000.0)], 1.5)
        result = tetra3rs.extract_centroids(few)
        assert len(result.centroids) == 2
        assert result.psf_sigma_estimate is None

    def test_cov_is_position_uncertainty_and_shape_is_star_size(self):
        """`cov` is how well the position is known (tiny for a bright star,
        larger for a faint one); `shape` is the star's second moments
        (about psf_sigma² on the diagonal, whatever the brightness)."""
        w, h = 240, 120
        psf_sigma = 1.5
        stars = [(60.3, 60.6, 5000.0), (180.7, 59.2, 50.0)]
        result = tetra3rs.extract_centroids(_star_field(w, h, stars, psf_sigma, seed=4))
        assert len(result.centroids) == 2
        bright, faint = result.centroids  # brightest first
        assert _nearest_px(result, 60.3, 60.6) < 0.05

        for c in (bright, faint):
            assert c.cov.shape == (2, 2) and c.cov.dtype == np.float64
            assert c.shape.shape == (2, 2) and c.shape.dtype == np.float64
            assert np.allclose(c.cov, c.cov.T) and np.allclose(c.shape, c.shape.T)
            assert c.cov[0, 0] > 0 and c.cov[1, 1] > 0

        # Shape: the PSF's variance, independent of brightness.
        assert bright.shape[0, 0] == pytest.approx(psf_sigma**2, rel=0.05)
        assert bright.shape[1, 1] == pytest.approx(psf_sigma**2, rel=0.05)
        assert abs(bright.shape[0, 1]) < 0.05 * psf_sigma**2

        # Position covariance: orders of magnitude below the shape for a
        # bright star, and it scales as 1/SNR² (amplitudes differ 100x).
        assert bright.cov[0, 0] < 1e-3 * bright.shape[0, 0]
        ratio = faint.cov[0, 0] / bright.cov[0, 0]
        assert 100**2 / 4 < ratio < 100**2 * 4, ratio
        # ...and it is honest: the read-noise bound for a Gaussian PSF is
        # sigma_pos = sqrt(2) * psf_sigma / SNR per axis, with matched-filter
        # SNR = amplitude * sqrt(pi) * psf_sigma / noise.
        snr = 50.0 * math.sqrt(math.pi) * psf_sigma / 3.0
        bound = (math.sqrt(2) * psf_sigma / snr) ** 2
        for var in (faint.cov[0, 0], faint.cov[1, 1]):
            assert 0.8 * bound < var < 1.5 * bound, (var, bound)

        # Centroids built by hand carry neither.
        manual = tetra3rs.Centroid(1.0, 2.0, 3.0)
        assert manual.cov is None and manual.shape is None
        # with_offset keeps both.
        moved = bright.with_offset(1.0, -2.0)
        assert np.array_equal(moved.cov, bright.cov)
        assert np.array_equal(moved.shape, bright.shape)

    def test_reject_hot_pixels(self):
        """Single-pixel spikes are dropped by default and kept on request;
        real stars are unaffected."""
        w, h = 400, 300
        stars = _random_stars(w, h, 10, seed=3)
        image = _star_field(w, h, stars, 1.5, seed=3)
        hot = [(50, 60), (200, 150), (350, 250), (120, 280)]
        for x, y in hot:
            image[y, x] += 4000.0

        default = tetra3rs.extract_centroids(image)
        assert len(default.centroids) == len(stars)
        for x, y in hot:
            assert _nearest_px(default, x, y) > 3.0
        for x, y, _ in stars:
            assert _nearest_px(default, x, y) < 0.05

        kept = tetra3rs.extract_centroids(image, reject_hot_pixels=False)
        assert len(kept.centroids) == len(stars) + len(hot)
        for x, y in hot:
            assert _nearest_px(kept, x, y) < 0.5

    def test_close_pair_is_separated(self):
        """Two stars sharing one above-threshold region come out as two
        centroids (there is no `deblend` switch any more)."""
        w, h = 200, 200
        stars = [(96.0, 100.3, 3000.0), (104.0, 100.3, 2000.0)]
        result = tetra3rs.extract_centroids(_star_field(w, h, stars, 1.5, seed=5))
        assert len(result.centroids) == 2
        for x, y, _ in stars:
            assert _nearest_px(result, x, y) < 0.1

    def test_saturated_star_gives_one_centroid(self):
        """A clipped (flat-topped) star falls back to the center of mass:
        one centroid, at the star, with or without `saturation_level`."""
        w, h = 200, 200
        image = _star_field(w, h, [(100.4, 99.7, 200000.0)], 1.5, seed=5)
        image = np.minimum(image, 4095.0).astype(np.float32)
        for saturation_level in (None, 4095.0):
            result = tetra3rs.extract_centroids(image, saturation_level=saturation_level)
            assert len(result.centroids) == 1
            assert _nearest_px(result, 100.4, 99.7) < 0.2

    def test_max_elongation_rejects_a_streak(self):
        w, h = 200, 200
        image = _star_field(w, h, [(60.3, 60.6, 3000.0)], 1.5, seed=6)
        yy, xx = np.mgrid[0:h, 0:w]
        # A 30 px horizontal trail with a Gaussian cross-section.
        image += (
            1500.0 * np.exp(-((yy - 140.0) ** 2) / (2 * 1.5**2)) * ((xx >= 90) & (xx < 120))
        ).astype(np.float32)
        result = tetra3rs.extract_centroids(image)
        assert len(result.centroids) == 1
        assert _nearest_px(result, 60.3, 60.6) < 0.05
        ungated = tetra3rs.extract_centroids(image, max_elongation=None)
        assert len(ungated.centroids) > 1

    def test_border_margin(self):
        w, h = 200, 160
        stars = [(20.3, 80.6, 3000.0), (100.7, 80.2, 3000.0)]
        image = _star_field(w, h, stars, 1.5, seed=8)
        assert len(tetra3rs.extract_centroids(image).centroids) == 2
        inner = tetra3rs.extract_centroids(image, border_margin=30)
        assert len(inner.centroids) == 1
        assert _nearest_px(inner, 100.7, 80.2) < 0.05

    def test_background_statistics(self):
        """`background_sigma` is the raw-pixel noise; `threshold` sits on the
        smoothed detection image, so it is far closer to the background than
        `sigma_threshold` raw sigmas would be."""
        w, h = 400, 300
        image = _star_field(w, h, _random_stars(w, h, 10, seed=9), 1.5, noise=5.0, seed=9)
        result = tetra3rs.extract_centroids(image, sigma_threshold=5.0)
        assert result.background_mean == pytest.approx(100.0, abs=0.5)
        assert result.background_sigma == pytest.approx(5.0, rel=0.15)
        assert result.background_mean < result.threshold
        assert result.threshold < result.background_mean + 5.0 * result.background_sigma
        # Regions, not stars: one region can hold several centroids.
        assert 1 <= result.num_blobs_raw <= 10


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
            (
                c.x,
                c.y,
                c.brightness,
                None if c.cov is None else c.cov.tobytes(),
                None if c.shape is None else c.shape.tobytes(),
            )
            for c in result.centroids
        ] + [
            result.background_mean,
            result.background_sigma,
            result.threshold,
            result.num_blobs_raw,
            result.psf_sigma_estimate,
            result.image_width,
            result.image_height,
        ]

    def test_reuse_matches_free_function(self):
        extractor = tetra3rs.CentroidExtractor()
        assert repr(extractor) == "CentroidExtractor()"
        # big -> small -> big; binned and full-resolution detection, a
        # different kernel, a different background grid
        frames = [
            ((200, 160), dict(detection_binning=2)),
            ((96, 80), dict(detection_binning=2)),
            ((96, 80), dict(detection_binning=1)),
            ((200, 160), dict(psf_sigma=1.0, bg_block_size=32)),
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
            extractor.extract(np.zeros((32, 32), dtype=np.float32), psf_sigma=0.0)
        with pytest.raises(ValueError):
            extractor.extract(np.zeros((32, 32), dtype=np.float32), detection_binning=4)
        # A failed call leaves the extractor usable.
        image = self._scene(120, 90, seed=3)
        assert self._flat(extractor.extract(image)) == self._flat(
            tetra3rs.extract_centroids(image)
        )
