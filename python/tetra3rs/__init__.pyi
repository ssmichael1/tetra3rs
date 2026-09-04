"""
tetra3rs: Fast star plate solver

A Rust implementation of the tetra3 star plate solving algorithm,
exposed to Python via PyO3.
"""

from __future__ import annotations

__version__: str
__git_hash__: str

import datetime
import numpy as np
import numpy.typing as npt
from typing import Optional, Union, final, overload

@final
class CameraModel:
    """Camera intrinsics model: focal length, optical center, parity, and distortion.

    Encapsulates the mapping between pixel coordinates and tangent-plane coordinates.
    Supports ``pickle`` serialization.

    Example::

        cam = tetra3rs.CameraModel.from_fov(fov_deg=10.0, image_width=2048, image_height=1536)
        xi, eta = cam.pixel_to_tanplane(100.0, 200.0)
    """

    def __reduce__(self) -> tuple: ...
    @staticmethod
    def _from_pickle_bytes(data: bytes) -> "CameraModel": ...

    def __new__(
        cls,
        focal_length_px: float,
        image_width: int,
        image_height: int,
        crpix: Optional[list[float]] = None,
        parity_flip: bool = False,
        distortion: Optional[Union["RadialDistortion", "PolynomialDistortion"]] = None,
    ) -> "CameraModel":
        """Create a camera model with explicit parameters.

        Args:
            focal_length_px: Focal length in pixels.
            image_width: Image width in pixels.
            image_height: Image height in pixels.
            crpix: Optical center offset from image center [x, y]. Default [0, 0].
            parity_flip: Whether x-axis is flipped. Default False.
            distortion: A RadialDistortion or PolynomialDistortion. Default None.
        """
        ...

    @staticmethod
    def from_fov(fov_deg: float, image_width: int, image_height: int) -> "CameraModel":
        """Create a camera model from a horizontal field of view and image dimensions.

        Args:
            fov_deg: Horizontal field of view in degrees.
            image_width: Image width in pixels.
            image_height: Image height in pixels.

        Returns:
            CameraModel with no distortion, crpix=[0,0], parity_flip=False.
        """
        ...

    @property
    def focal_length_px(self) -> float:
        """Focal length in pixels."""
        ...

    @property
    def image_width(self) -> int:
        """Image width in pixels."""
        ...

    @property
    def image_height(self) -> int:
        """Image height in pixels."""
        ...

    @property
    def crpix(self) -> list[float]:
        """Optical center offset from image center [x, y] in pixels."""
        ...

    @property
    def parity_flip(self) -> bool:
        """Whether the image x-axis is flipped."""
        ...

    @property
    def distortion(self) -> Optional[Union["RadialDistortion", "PolynomialDistortion"]]:
        """The distortion model, or None if no distortion."""
        ...

    @property
    def fov_deg(self) -> float:
        """Horizontal field of view in degrees."""
        ...

    def pixel_scale(self) -> float:
        """Pixel scale in radians per pixel."""
        ...

    def pixel_to_tanplane(self, px: float, py: float) -> tuple[float, float]:
        """Convert pixel coordinates to tangent-plane coordinates (xi, eta) in radians."""
        ...

    def tanplane_to_pixel(self, xi: float, eta: float) -> tuple[float, float]:
        """Convert tangent-plane coordinates to pixel coordinates."""
        ...

    def save_to_file(self, path: str) -> None:
        """Save the camera model to a file (postcard binary format).

        Args:
            path: File path to write to.
        """
        ...

    @staticmethod
    def load_from_file(path: str) -> "CameraModel":
        """Load a camera model from a file (postcard binary format).

        Args:
            path: File path to read from.

        Returns:
            CameraModel loaded from the file.
        """
        ...


@final
class CalibrateResult:
    """Result of camera calibration.

    Returned by ``SolverDatabase.calibrate_camera``.
    Supports ``pickle`` serialization.
    """

    def __reduce__(self) -> tuple: ...
    @staticmethod
    def _from_pickle_bytes(data: bytes) -> "CalibrateResult": ...

    @property
    def camera_model(self) -> CameraModel:
        """The fitted CameraModel (focal length, crpix, distortion)."""
        ...

    @property
    def rmse_before_px(self) -> float:
        """RMS residual in pixels before calibration."""
        ...

    @property
    def rmse_after_px(self) -> float:
        """RMS residual in pixels after calibration."""
        ...

    @property
    def n_inliers(self) -> int:
        """Number of inlier star matches used in the fit."""
        ...

    @property
    def n_outliers(self) -> int:
        """Number of outlier star matches rejected by sigma clipping."""
        ...

    @property
    def iterations(self) -> int:
        """Number of sigma-clip iterations performed."""
        ...


@final
class SolveResult:
    """Result of a successful plate-solve.

    Returned by ``SolverDatabase.solve_from_centroids`` on a successful match.
    Contains the camera attitude, matched stars, and error statistics.
    Supports ``pickle`` serialization.
    """

    def __reduce__(self) -> tuple: ...
    @staticmethod
    def _from_pickle_bytes(data: bytes) -> "SolveResult": ...
    def __bool__(self) -> bool:
        """Always ``True`` — lets ``if result:`` distinguish success from a
        (falsy) ``SolveFailure``."""
        ...

    @property
    def rotation_matrix_icrs_to_camera(self) -> npt.NDArray[np.float64]:
        """3x3 rotation matrix from ICRS to camera frame."""
        ...

    @property
    def quaternion(self) -> npt.NDArray[np.float64]:
        """Attitude quaternion as a 4-element ``[w, x, y, z]`` array.

        **Convention.** Hamilton, scalar first: ``q = w + x·i + y·j + z·k``
        with ``w² + x² + y² + z² = 1``. This matches ``scipy.spatial.
        transform.Rotation.as_quat(scalar_first=True)`` and is the usual
        convention in aerospace / attitude literature. It does **not**
        match scipy's default (scalar-last) ordering.

        **Sense.** Rotates a vector from the ICRS frame into the camera
        frame: ``camera_vec = q ⊗ icrs_vec ⊗ q*``. Equivalently,
        ``rotation_matrix_icrs_to_camera @ icrs_vec == camera_vec``.

        Suitable for feeding back as ``attitude_hint`` on the next frame's
        ``solve_from_centroids`` call (tracking mode).
        """
        ...

    @property
    def ra_deg(self) -> float:
        """Right ascension of the boresight in degrees [0, 360)."""
        ...

    @property
    def dec_deg(self) -> float:
        """Declination of the boresight in degrees [-90, 90]."""
        ...

    @property
    def roll_deg(self) -> float:
        """Roll angle: position angle of camera +Y measured East of North, in degrees."""
        ...

    @property
    def fov_deg(self) -> float:
        """Solved horizontal field of view in degrees."""
        ...

    @property
    def num_matches(self) -> int:
        """Number of matched star pairs."""
        ...

    @property
    def rmse_arcsec(self) -> float:
        """Root mean square error of matched stars in arcseconds."""
        ...

    @property
    def p90e_arcsec(self) -> float:
        """90th percentile error in arcseconds."""
        ...

    @property
    def max_err_arcsec(self) -> float:
        """Maximum match error in arcseconds."""
        ...

    @property
    def probability(self) -> float:
        """False-positive probability (lower is better)."""
        ...

    @property
    def solve_time_ms(self) -> float:
        """Time taken to solve in milliseconds."""
        ...

    @property
    def matched_centroids(self) -> npt.NDArray[np.uint64]:
        """Indices of matched centroids in the input array."""
        ...

    @property
    def matched_catalog_ids(self) -> npt.NDArray[np.int64]:
        """Catalog IDs of matched stars (negative IDs = Hipparcos gap-fill stars)."""
        ...

    @property
    def status(self) -> str:
        """Always 'match_found' when a result is returned."""
        ...

    @property
    def parity_flip(self) -> bool:
        """Whether the image x-axis was flipped to achieve a proper rotation.

        When ``True``, the rotation matrix assumes negated x-coordinates.
        Pixel-to-sky and sky-to-pixel conversions must account for this.
        """
        ...

    @property
    def camera_model(self) -> CameraModel:
        """The camera model used for the solve (focal length, image dimensions,
        optical center, distortion, and parity)."""
        ...

    @property
    def cd_matrix(self) -> npt.NDArray[np.float64]:
        """WCS CD matrix as a 2x2 numpy array (tangent-plane radians per pixel).

        Maps pixel offsets from CRPIX to gnomonic tangent-plane coordinates
        at CRVAL.
        """
        ...

    @property
    def theta_deg(self) -> float:
        """Fitted rotation angle in degrees (camera roll in tangent plane).

        The angle from the tangent-plane ξ (East) axis to the camera +X axis,
        measured counter-clockwise. When ``parity_flip`` is ``True``,
        "camera +X" means the x-negated (mirror-corrected) axis — the same
        frame the quaternion rotates into.
        """
        ...

    @property
    def observer_velocity_km_s(self) -> Optional[npt.NDArray[np.float64]]:
        """Observer velocity (km/s, ICRS) the solve corrected stellar aberration
        for — the ``observer_velocity_km_s`` it was called with — or ``None``.
        ``SolverDatabase.calibrate_camera`` reuses it per image so differential
        aberration is not fitted as lens distortion.
        """
        ...

    @property
    def attitude_cov_rad2(self) -> npt.NDArray[np.float64]:
        """Covariance of the refined attitude parameters ``[theta, xi0, eta0]``
        as a 3x3 array in rad²: roll about the boresight and the tangent-plane
        offsets of the boresight (East, North at CRVAL).

        Estimated from the refinement's normal equations and the observed
        centroid scatter (``sigma² · (JᵀJ)⁻¹`` with
        ``sigma² = Σ residual² / (2n − 3)``). The diagonal is ``inf`` when the
        fit is unconstrained (fewer than 2 matched stars).
        """
        ...

    @property
    def attitude_sigma_rad(self) -> npt.NDArray[np.float64]:
        """1-sigma uncertainties ``[sigma_theta, sigma_xi, sigma_eta]`` of the
        refined attitude in radians (square roots of the diagonal of
        ``attitude_cov_rad2``). Boresight pointing uncertainty is
        ``hypot(sigma_xi, sigma_eta)``.
        """
        ...

    @property
    def crval_ra_deg(self) -> float:
        """WCS reference point RA in degrees.

        The tangent point of the gnomonic (TAN) projection, close to the boresight.
        """
        ...

    @property
    def crval_dec_deg(self) -> float:
        """WCS reference point Dec in degrees."""
        ...

    @property
    def crpix(self) -> npt.NDArray[np.float64]:
        """Optical center offset from the geometric image center, in pixels [x, y]."""
        ...

    @overload
    def pixel_to_world(self, x: float, y: float) -> tuple[float, float]: ...
    @overload
    def pixel_to_world(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Convert centered pixel coordinates to world coordinates (RA, Dec in degrees).

        Pixel coordinates use the same convention as solver centroids:
        origin at the image center, +X right, +Y down. Scalars in, scalars out;
        1D numpy arrays in, 1D numpy arrays out (NaN where undefined).
        """
        ...

    @overload
    def world_to_pixel(self, ra_deg: float, dec_deg: float) -> Optional[tuple[float, float]]: ...
    @overload
    def world_to_pixel(
        self, ra_deg: npt.NDArray[np.float64], dec_deg: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Convert world coordinates (RA, Dec in degrees) to centered pixel coordinates.

        Returns pixel coordinates in the same convention as solver centroids:
        origin at the image center, +X right, +Y down. Scalars in, scalars out
        (None if the point is behind the camera); 1D numpy arrays in, arrays out
        (NaN for points behind the camera).
        """
        ...

@final
class SolveFailure:
    """A failed plate-solve attempt: why it failed and how long it took.

    Returned by ``SolverDatabase.solve_from_centroids`` when no solution was
    found. Falsy, so ``if result:`` cleanly separates success from failure::

        result = db.solve_from_centroids(centroids, ...)
        if result:
            print(result.ra_deg, result.dec_deg)
        else:
            print(f"solve failed: {result.status} after {result.solve_time_ms:.0f} ms")

    Supports ``pickle`` serialization.
    """

    def __reduce__(self) -> tuple: ...
    @staticmethod
    def _from_pickle_bytes(data: bytes) -> "SolveFailure": ...
    def __bool__(self) -> bool:
        """Always ``False`` — lets ``if result:`` distinguish a (truthy)
        ``SolveResult`` from a failure."""
        ...

    @property
    def status(self) -> str:
        """Why the solve produced no solution.

        One of ``'no_match'`` (all pattern combinations exhausted),
        ``'timeout'`` (``solve_timeout_ms`` reached), ``'too_few'``
        (fewer than 4 usable centroids), or ``'invalid_config'``
        (degenerate camera model or non-finite matching parameters —
        nothing was searched).
        """
        ...

    @property
    def solve_time_ms(self) -> float:
        """Wall-clock time spent before giving up, in milliseconds."""
        ...

@final
class CatalogStar:
    """A star from the solver catalog.

    Returned by ``SolverDatabase.get_star``, ``get_star_by_id``, and ``cone_search``.
    """

    @property
    def id(self) -> int:
        """Catalog identifier (e.g. Hipparcos number)."""
        ...

    @property
    def ra_deg(self) -> float:
        """Right ascension in degrees [0, 360)."""
        ...

    @property
    def dec_deg(self) -> float:
        """Declination in degrees [-90, 90]."""
        ...

    @property
    def magnitude(self) -> float:
        """Visual magnitude."""
        ...

@final
class ExtractionResult:
    """Result of centroid extraction from an image.

    Returned by ``extract_centroids``.
    Supports ``pickle`` serialization.
    """

    def __reduce__(self) -> tuple: ...
    @staticmethod
    def _from_pickle_bytes(data: bytes) -> "ExtractionResult": ...

    @property
    def centroids(self) -> list[Centroid]:
        """List of detected centroids, sorted by brightness (brightest first)."""
        ...

    @property
    def image_width(self) -> int:
        """Width of the input image in pixels."""
        ...

    @property
    def image_height(self) -> int:
        """Height of the input image in pixels."""
        ...

    @property
    def background_mean(self) -> float:
        """Estimated background mean."""
        ...

    @property
    def background_sigma(self) -> float:
        """Estimated background standard deviation."""
        ...

    @property
    def threshold(self) -> float:
        """Detection threshold used."""
        ...

    @property
    def num_blobs_raw(self) -> int:
        """Number of raw blobs before filtering."""
        ...

@final
class Centroid:
    """A detected star centroid with position, brightness, and shape.

    Returned by ``extract_centroids``. Centroids use pixel coordinates
    with the origin at the image center, +X right, +Y down.
    Supports ``pickle`` serialization.
    """

    def __reduce__(self) -> tuple: ...
    @staticmethod
    def _from_pickle_bytes(data: bytes) -> "Centroid": ...

    def __new__(cls, x: float, y: float, brightness: Optional[float] = None) -> "Centroid":
        """Create a new Centroid.

        Args:
            x: X position in pixels (origin at image center, +X right).
            y: Y position in pixels (origin at image center, +Y down).
            brightness: Integrated intensity above background (optional).
        """
        ...

    @property
    def x(self) -> float:
        """X position in pixels (origin at image center, +X right)."""
        ...

    @property
    def y(self) -> float:
        """Y position in pixels (origin at image center, +Y down)."""
        ...

    @property
    def brightness(self) -> Optional[float]:
        """Integrated intensity above background."""
        ...

    @property
    def cov(self) -> Optional[npt.NDArray[np.float64]]:
        """2x2 intensity-weighted covariance matrix [[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]] in pixels squared.

        The eigenvalues give the squared semi-axes of the intensity profile,
        and the eigenvectors give the orientation.
        """
        ...

    def with_offset(self, dx: float, dy: float) -> Centroid:
        """Return a new Centroid with position shifted by (dx, dy).

        Preserves brightness and covariance.
        """
        ...

    def undistort(
        self, distortion: Union[RadialDistortion, PolynomialDistortion]
    ) -> Centroid:
        """Remove lens distortion from this centroid's position (distorted → ideal).

        Returns a new Centroid at the corrected position.
        Brightness and covariance are preserved.
        """
        ...

    def distort(
        self, distortion: Union[RadialDistortion, PolynomialDistortion]
    ) -> Centroid:
        """Apply lens distortion to this centroid's position (ideal → distorted).

        Returns a new Centroid at the distorted position.
        Brightness and covariance are preserved.
        """
        ...

@final
class SolverDatabase:
    """A star pattern database for plate solving.

    Generate from the Gaia or Hipparcos catalog, or load a previously saved database.
    Supports ``pickle`` serialization.

    Example::

        db = tetra3rs.SolverDatabase.generate_from_gaia()  # uses bundled gaia-catalog
        db.save_to_file("my_db.bin")
        db = tetra3rs.SolverDatabase.load_from_file("my_db.bin")
    """

    def __reduce__(self) -> tuple: ...
    @staticmethod
    def _from_pickle_bytes(data: bytes) -> "SolverDatabase": ...

    @staticmethod
    def generate_from_gaia(
        catalog_path: Optional[str] = None,
        max_fov_deg: float = 30.0,
        min_fov_deg: Optional[float] = None,
        star_max_magnitude: Optional[float] = None,
        pattern_max_error: float = 0.001,
        lattice_field_oversampling: int = 100,
        patterns_per_lattice_field: int = 50,
        verification_stars_per_fov: int = 150,
        multiscale_step: float = 1.5,
        epoch_proper_motion_year: Optional[float] = 2025.0,
        catalog_nside: int = 16,
    ) -> SolverDatabase:
        """Generate a database from a Gaia DR3 catalog file.

        Accepts either:
        - A CSV file (``.csv``) with columns:
          ``source_id,ra,dec,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag,parallax,pmra,pmdec``
        - A binary file (``.bin``) from the ``gaia-catalog`` package.

        Use ``scripts/download_gaia_catalog.py`` to produce a merged Gaia + Hipparcos CSV,
        or ``pip install gaia-catalog`` for the pre-built binary catalog.
        Negative source_ids indicate Hipparcos gap-fill stars.

        Args:
            catalog_path: Path to the Gaia catalog file (CSV or binary).
                If None, uses the bundled catalog from the ``gaia-catalog`` package.
            max_fov_deg: Maximum field of view in degrees.
            min_fov_deg: Minimum field of view in degrees.
                None means same as max_fov_deg (single-scale).
            star_max_magnitude: Faintest star to include (G-band). None = auto.
            pattern_max_error: Maximum edge-ratio error.
            lattice_field_oversampling: Lattice field oversampling factor.
            patterns_per_lattice_field: Patterns per lattice field.
            verification_stars_per_fov: Verification stars per FOV.
            multiscale_step: Multiscale step factor.
            epoch_proper_motion_year: Year for proper motion propagation.
            catalog_nside: HEALPix nside for catalog spatial indexing.

        Returns:
            A new SolverDatabase instance.
        """
        ...

    def save_to_file(self, path: str) -> None:
        """Save the database to a file.

        Args:
            path: Output file path.
        """
        ...

    @staticmethod
    def load_from_file(path: str) -> SolverDatabase:
        """Load a database from a file.

        Args:
            path: Path to a previously saved database file.

        Returns:
            A SolverDatabase loaded from disk.
        """
        ...

    def solve_from_centroids(
        self,
        centroids: Union[list[Centroid], npt.NDArray[np.float64]],
        fov_estimate_deg: Optional[float] = None,
        fov_estimate_rad: Optional[float] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        image_shape: Optional[tuple[int, int]] = None,
        fov_max_error_deg: Optional[float] = None,
        fov_max_error_rad: Optional[float] = None,
        match_radius: float = 0.01,
        match_threshold: float = 1e-5,
        solve_timeout_ms: Optional[int] = 5000,
        max_patterns_checked: Optional[int] = 10_000_000,
        pattern_checking_stars: int = 24,
        match_max_error: Optional[float] = None,
        camera_model: Optional[CameraModel] = None,
        observer_velocity_km_s: Optional[list[float]] = None,
        attitude_hint: Optional[
            Union[list[float], npt.NDArray[np.float64]]
        ] = None,  # [w, x, y, z] quaternion or 3x3 rotation matrix
        hint_uncertainty_deg: Optional[float] = None,
        hint_uncertainty_rad: Optional[float] = None,
        strict_hint: bool = False,
    ) -> Union[SolveResult, SolveFailure]:
        """Solve for camera attitude given star centroids.

        Runs **lost-in-space** (pattern-hash search) by default, or **tracking**
        (direct correspondence from a prior estimate) when ``attitude_hint`` is
        given — with automatic fallback to lost-in-space. The arguments group by
        which mode reads them; arguments for one mode are ignored in the other:

        * **Both modes:** ``camera_model`` / ``fov_estimate_*`` / ``image_*``,
          ``match_radius``, ``match_threshold``, ``solve_timeout_ms``,
          ``max_patterns_checked``,
          ``observer_velocity_km_s``.
        * **Lost-in-space only:** ``fov_max_error_*``, ``match_max_error``,
          ``pattern_checking_stars``.
        * **Tracking only** (ignored unless ``attitude_hint`` is set):
          ``attitude_hint``, ``hint_uncertainty_*``, ``strict_hint``.

        Args:
            centroids: Either a list of Centroid objects (from extract_centroids),
                or an Nx2/Nx3 numpy array of centroid positions in pixels.
                Columns are (x, y) or (x, y, brightness).
                Origin is at the image center, +X right, +Y down.
            fov_estimate_deg: Estimated horizontal field of view in degrees.
            fov_estimate_rad: Estimated horizontal field of view in radians.
                Exactly one of fov_estimate_deg or fov_estimate_rad must be provided.
                Used (with the image dimensions) to build a pinhole camera model
                when camera_model is not given; ignored when it is — the model's
                focal length then defines the FOV estimate.
            image_width: Image width in pixels.
            image_height: Image height in pixels.
            image_shape: Image shape as (height, width) tuple (numpy convention).
                Can be used instead of image_width/image_height.
            fov_max_error_deg: Maximum FOV error in degrees. None = no limit.
            fov_max_error_rad: Maximum FOV error in radians. None = no limit.
                At most one of fov_max_error_deg or fov_max_error_rad can be provided.
                Lost-in-space only — tracking takes its scale from the hint.
            match_radius: Match distance as fraction of FOV.
            match_threshold: False-positive probability budget for accepting
                a solution (candidate p-values are tested against this with a
                sequential multiple-comparison correction). Raising it (e.g.
                1e-3) accepts weaker evidence — useful for very sparse fields
                at increased false-positive risk.
            solve_timeout_ms: Wall-clock timeout in milliseconds. None = no
                timeout. Default 5000.
            max_patterns_checked: Maximum number of image 4-star patterns the
                lost-in-space search tests (summed over the FOV sweep) before
                giving up with ``'timeout'``. Bounds the search by work rather
                than wall time, so the outcome is the same on every machine;
                composes with ``solve_timeout_ms`` (whichever trips first).
                None = unbounded. Lost-in-space only.
            pattern_checking_stars: Number of brightest (well-separated)
                centroids the lost-in-space search forms 4-star patterns
                from; fainter centroids still take part in verification.
                Bounds a no-match search at C(N, 4) patterns per FOV value.
                Raise it when many of the brightest detections are not
                catalog stars. Must be >= 4. Lost-in-space only.
            match_max_error: Maximum edge-ratio error. None = use database value.
                Values below the database's pattern quantization error are clamped up to it.
                Lost-in-space only — tracking does not use the pattern hash.
            camera_model: A CameraModel specifying focal length, image
                dimensions, optical center, distortion, and parity. When
                provided it is the single source of camera geometry
                (fov_estimate and image dimensions are ignored). None = simple
                pinhole model built from fov_estimate and the image dimensions.
            observer_velocity_km_s: Observer's barycentric velocity as
                [vx, vy, vz] in km/s (ICRS/GCRF frame). When set, catalog
                positions are aberration-corrected to apparent positions,
                removing the ~20" bias from Earth's orbital velocity.
                None = no correction (default).
            attitude_hint: Optional attitude hint. Accepts either:

                * a 4-element ``[w, x, y, z]`` quaternion (list or 1D ndarray),
                  Hamilton / scalar-first convention — same as
                  ``SolveResult.quaternion``. This matches
                  ``scipy.spatial.transform.Rotation.as_quat(scalar_first=True)``;
                  it does **not** match scipy's default (scalar-last) ordering.
                * a 3×3 rotation matrix (2D ndarray) — same as
                  ``SolveResult.rotation_matrix_icrs_to_camera``.

                Either form must rotate a vector from the ICRS frame into the
                camera frame. When provided, the solver skips the 4-star
                pattern-hash phase and instead projects nearby catalog stars
                via the hint, nearest-neighbor matches them to centroids, and
                runs the same verification + WCS refine path as lost-in-space.
                Typical use case: video-rate tracking where each frame's solve
                seeds the next. Succeeds with as few as 3 matched stars
                (vs. 4 for LIS). On failure, falls back to lost-in-space
                unless ``strict_hint=True``.
            hint_uncertainty_deg: Angular uncertainty of the attitude hint,
                in degrees.
            hint_uncertainty_rad: Angular uncertainty of the attitude hint,
                in radians. At most one of the two may be provided; default 1°
                if neither is set. Used to size the catalog cone search and
                the initial pixel match radius. Ignored unless ``attitude_hint``
                is set.
            strict_hint: If True, do not fall back to lost-in-space if the
                hinted solve fails. Default False. Ignored unless
                ``attitude_hint`` is set.

        Returns:
            A SolveResult on success, or a (falsy) SolveFailure carrying the
            failure reason (``status``: ``'no_match'`` / ``'timeout'`` /
            ``'too_few'`` / ``'invalid_config'``) and ``solve_time_ms``. Use
            ``if result:`` to distinguish the two.
        """
        ...

    @property
    def num_stars(self) -> int:
        """Number of stars in the catalog."""
        ...

    @property
    def num_patterns(self) -> int:
        """Number of patterns in the database."""
        ...

    @property
    def max_fov_deg(self) -> float:
        """Maximum FOV the database was built for (degrees)."""
        ...

    @property
    def min_fov_deg(self) -> float:
        """Minimum FOV the database was built for (degrees)."""
        ...

    @property
    def parameters(self) -> dict:
        """Database generation parameters as a dict.

        Returns the settings baked into this database when it was generated
        (stars per FOV, lattice field oversampling, pattern quantization, FOV
        range, magnitude limit, epochs). Read from the stored properties, so
        they reflect the actual database on disk -- not solve-time options.
        """
        ...

    def print_parameters(self) -> None:
        """Print the database generation parameters to stdout.

        Human-readable dump of the settings this database was built with --
        stars per FOV, lattice field oversampling, pattern quantization, FOV
        range, magnitude limit, and epochs. See ``parameters`` for the same
        data as a dict.
        """
        ...

    def get_star(self, index: int) -> CatalogStar:
        """Get a catalog star by its internal index (0-based, brightness order).

        Args:
            index: Star index in [0, num_stars).

        Returns:
            CatalogStar at that index.

        Raises:
            IndexError: If index is out of range.
        """
        ...

    def get_star_by_id(self, catalog_id: int) -> Optional[CatalogStar]:
        """Get a catalog star by its catalog ID (e.g. Hipparcos number).

        Args:
            catalog_id: The catalog identifier to search for.

        Returns:
            CatalogStar with that ID, or None if not found.
        """
        ...

    def cone_search(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
    ) -> list[CatalogStar]:
        """Query catalog stars within an angular radius of a sky position.

        Args:
            ra_deg: Right ascension of cone center in degrees.
            dec_deg: Declination of cone center in degrees.
            radius_deg: Search radius in degrees.

        Returns:
            List of CatalogStar objects within the cone, sorted by brightness.
        """
        ...

    def calibrate_camera(
        self,
        solve_results: Union[SolveResult, list[Union[SolveResult, SolveFailure]]],
        centroids: Union[
            list[Centroid],
            npt.NDArray[np.float64],
            list[Union[list[Centroid], npt.NDArray[np.float64]]],
        ],
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        image_shape: Optional[tuple[int, int]] = None,
        model: str = "polynomial",
        order: int = 4,
        max_iterations: int = 10,
        sigma_clip: float = 3.0,
        convergence_threshold_px: float = 0.01,
    ) -> CalibrateResult:
        """Calibrate a camera model from one or more plate-solve results.

        Fits a global CameraModel (focal length, optical center, distortion) by
        alternating per-image constrained WCS refinement with a global
        least-squares fit.

        The distortion model is selected by ``model``:

        - ``"polynomial"`` (default) — SIP-like polynomial of the given
          ``order``. Captures arbitrary 2D distortion including tangential
          and decentering terms; preferred for off-axis CCDs (e.g. TESS).
        - ``"radial"`` — Brown-Conrady ``(k1, k2, k3)``. Three parameters
          total — well-conditioned with few matches and the standard model
          in computer-vision camera calibration. Assumes distortion is
          symmetric about the optical center.

        Args:
            solve_results: A SolveResult, or a list that may mix SolveResult
                and SolveFailure objects — failures are skipped, so solve
                outputs can be passed straight through without filtering.
            centroids: Matching centroids (list of Centroid lists, or single list).
            image_width: Image width in pixels.
            image_height: Image height in pixels.
            image_shape: Image shape as (height, width) tuple (numpy convention).
                Can be used instead of image_width/image_height.
            model: Distortion model — ``"polynomial"`` or ``"radial"``.
                Default ``"polynomial"``.
            order: Polynomial distortion order (2-6). Ignored unless
                ``model="polynomial"``. Default 4.
            max_iterations: Maximum outer iterations. Default 10.
            sigma_clip: Sigma threshold for outlier rejection. Default 3.0.
            convergence_threshold_px: Stop when RMSE change < this (pixels).
                Default 0.01.

        Returns:
            CalibrateResult with camera_model, rmse_before_px, rmse_after_px,
            n_inliers, n_outliers, and iterations.
        """
        ...

@final
class RadialDistortion:
    """Brown-Conrady radial+tangential distortion model.

    ``x_d = x · (1 + k1·r² + k2·r⁴ + k3·r⁶) + 2·p1·x·y + p2·(r² + 2·x²)``
    ``y_d = y · (1 + k1·r² + k2·r⁴ + k3·r⁶) + p1·(r² + 2·y²) + 2·p2·x·y``

    Coordinates are in pixels, origin at the image center; the model shifts
    into its own ``center`` frame (the optical axis, ``(0, 0)`` by default)
    internally. Tangential coefficients ``p1, p2`` default to 0 — set them
    to model lens decentering or sensor tilt. On mosaic cameras (e.g. TESS)
    the optical axis can sit far off the detector center — ``center``
    carries that offset. Supports ``pickle`` serialization.

    Example::

        d = tetra3rs.RadialDistortion(k1=-7e-9, k2=2e-15)
        d = tetra3rs.RadialDistortion(k1=-7e-9, p1=5e-7, p2=-3e-7)
        x_undistorted, y_undistorted = d.undistort(100.0, 200.0)

    References:
      - Conrady, A. E. (1919). "Decentred Lens-Systems."
        *Monthly Notices of the Royal Astronomical Society*, 79(5): 384-390.
        — Original derivation of the tangential / decentering form.
        https://doi.org/10.1093/mnras/79.5.384
      - Brown, D. C. (1966). "Decentering Distortion of Lenses."
        *Photogrammetric Engineering*, 32(3): 444-462. — Modernized form.
      - Brown, D. C. (1971). "Close-Range Camera Calibration."
        *Photogrammetric Engineering*, 37(8): 855-866. — Calibration procedure.
      - Zhang, Z. (2000). "A Flexible New Technique for Camera Calibration."
        *IEEE TPAMI*, 22(11): 1330-1334. — Multi-image planar calibration
        (the standard method, used in OpenCV ``calibrateCamera``).
        https://doi.org/10.1109/34.888718
      - OpenCV documentation for the equivalent ``(k1, k2, k3, p1, p2)``
        formulation: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
    """

    def __reduce__(self) -> tuple: ...
    @staticmethod
    def _from_pickle_bytes(data: bytes) -> "RadialDistortion": ...

    def __new__(
        cls,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        p1: float = 0.0,
        p2: float = 0.0,
        center: tuple[float, float] = (0.0, 0.0),
    ) -> "RadialDistortion": ...
    @property
    def k1(self) -> float:
        """First radial coefficient."""
        ...

    @property
    def k2(self) -> float:
        """Second radial coefficient."""
        ...

    @property
    def k3(self) -> float:
        """Third radial coefficient."""
        ...

    @property
    def p1(self) -> float:
        """First tangential / decentering coefficient."""
        ...

    @property
    def p2(self) -> float:
        """Second tangential / decentering coefficient."""
        ...

    @property
    def center(self) -> tuple[float, float]:
        """Distortion center (optical axis) as ``(cx, cy)`` in pixels,
        image-center-origin frame."""
        ...

    def distort(self, x: float, y: float) -> tuple[float, float]:
        """Forward distortion: ideal → distorted."""
        ...

    def undistort(self, x: float, y: float) -> tuple[float, float]:
        """Inverse distortion: distorted → ideal."""
        ...

@final
class PolynomialDistortion:
    """SIP-like polynomial distortion model with independent x,y correction terms.

    Forward:  x_d = x + Σ A_pq · (x/s)^p · (y/s)^q   (0 ≤ p+q ≤ order)
    Inverse:  x_i = x_d + Σ AP_pq · (x_d/s)^p · (y_d/s)^q

    Where s = scale = image_width/2.

    Includes all polynomial terms from order 0:
      - (p+q = 0): constant offset — optical center shift
      - (p+q = 1): linear terms — residual scale & rotation
      - (p+q ≥ 2): higher-order distortion

    Total coefficients per axis: (order+1)(order+2)/2.

    Typically constructed by `SolverDatabase.calibrate_camera()` (which fits
    the polynomial internally and returns it via the camera model), or
    directly from coefficient arrays (e.g. extracted from a FITS WCS SIP
    model). Supports ``pickle`` serialization.

    References:
      - Shupe, D. L.; Moshir, M.; Li, J.; Makovoz, D.; Narron, R.;
        Hook, R. N. (2005). "The SIP Convention for Representing
        Distortion in FITS Image Headers." *Astronomical Data Analysis
        Software and Systems XIV*, ASP Conference Series, 347: 491.
        — Original SIP specification.
        https://www.adass.org/adass/proceedings/adass04/reprints/P3-1-3.pdf
      - FITS WCS SIP convention registry entry:
        https://fits.gsfc.nasa.gov/registry/sip.html

    The convention here is SIP-like — same A_pq, B_pq polynomial basis on
    normalized pixel coordinates. Standard SIP starts at order 2 (the
    linear part is absorbed into the WCS CD matrix); this implementation
    also includes order 0 / 1 terms so a single fit can absorb optical-
    center offset and residual scale/rotation from the matched-points data.
    """

    def __reduce__(self) -> tuple: ...
    @staticmethod
    def _from_pickle_bytes(data: bytes) -> "PolynomialDistortion": ...

    def __new__(
        cls,
        order: int,
        scale: float,
        a_coeffs: list[float],
        b_coeffs: list[float],
        ap_coeffs: Optional[list[float]] = None,
        bp_coeffs: Optional[list[float]] = None,
    ) -> "PolynomialDistortion":
        """Create a polynomial distortion model from forward coefficient arrays.

        a_coeffs and b_coeffs must each have exactly (order+1)(order+2)/2
        elements, covering all terms from p+q=0 (constant offset) through
        p+q=order.

        Args:
            order: Polynomial order (2–6 typical).
            scale: Normalization scale (typically image_width / 2).
            a_coeffs: Forward A coefficients (x correction, ideal → distorted).
            b_coeffs: Forward B coefficients (y correction, ideal → distorted).
            ap_coeffs: Deprecated and ignored — the model inverts numerically.
            bp_coeffs: Deprecated and ignored. See ap_coeffs.
        """
        ...

    @property
    def order(self) -> int:
        """Polynomial order."""
        ...

    @property
    def scale(self) -> float:
        """Normalization scale (typically image_width / 2)."""
        ...

    @property
    def num_coeffs(self) -> int:
        """Number of polynomial coefficients per axis."""
        ...

    @property
    def a_coeffs(self) -> npt.NDArray[np.float64]:
        """Forward A coefficients (x correction, ideal → distorted)."""
        ...

    @property
    def b_coeffs(self) -> npt.NDArray[np.float64]:
        """Forward B coefficients (y correction, ideal → distorted)."""
        ...

    @property
    def ap_coeffs(self) -> npt.NDArray[np.float64]:
        """Inverse AP coefficients (x correction, distorted → ideal)."""
        ...

    @property
    def bp_coeffs(self) -> npt.NDArray[np.float64]:
        """Inverse BP coefficients (y correction, distorted → ideal)."""
        ...

    def distort(self, x: float, y: float) -> tuple[float, float]:
        """Forward distortion: ideal → distorted."""
        ...

    def undistort(self, x: float, y: float) -> tuple[float, float]:
        """Inverse distortion: distorted → ideal."""
        ...

@final
class CentroidExtractor:
    """The :func:`extract_centroids` pipeline with its working buffers kept
    between calls.

    :func:`extract_centroids` allocates its full-image buffers (~48 MB at
    2048²) fresh on every call, and the first touch of each page costs more
    than the allocation (~0.3–0.5 ms per 2048² frame). An extractor reuses
    them, resizing only when the frame size changes, so a frame loop pays
    that once. Results are bit-identical to :func:`extract_centroids`.

    The buffers are sized for the largest frame seen and freed with the
    object. Use one extractor per thread. Pickling keeps no buffers — an
    unpickled extractor starts empty.

    Example::

        extractor = tetra3rs.CentroidExtractor()
        for frame in frames:
            result = extractor.extract(frame, sigma_threshold=5.0)
    """

    def __init__(self) -> None: ...
    def extract(
        self,
        image: npt.NDArray,
        sigma_threshold: float = 5.0,
        min_pixels: int = 3,
        max_pixels: int = 10000,
        max_centroids: Optional[int] = None,
        local_bg_block_size: Optional[int] = 64,
        max_elongation: Optional[float] = 3.0,
        matched_filter_sigma: Optional[float] = 1.5,
        max_sharpness: Optional[float] = 0.9,
        saturation_level: Optional[float] = None,
        deblend: str = "off",
        border_margin: int = 0,
    ) -> ExtractionResult:
        """Extract star centroids from a 2D image array, reusing this
        extractor's buffers.

        Takes exactly the arguments of :func:`extract_centroids`, with the
        same defaults.

        Args:
            image: 2D numpy array (height x width) of pixel values.
                Supported dtypes: float64, float32, uint16, int16, uint8.
            sigma_threshold: Detection threshold in sigma above background.
            min_pixels: Minimum blob size in pixels.
            max_pixels: Maximum blob size in pixels.
            max_centroids: Maximum number of centroids to return. None = all.
            local_bg_block_size: Block size for local background estimation.
                None = global background only.
            max_elongation: Maximum blob elongation ratio. None = disabled.
            matched_filter_sigma: Apply a Gaussian matched filter of this sigma
                (in pixels) before thresholding (~2x point-source SNR for a
                sigma~1.5 px PSF). Used only to form the detection mask, so
                photometry is unaffected, and the threshold is automatically
                scaled for the filtered noise level — no retuning needed.
                None = disabled. Default 1.5.
            max_sharpness: Reject blobs whose peak sharpness
                ``(peak - mean(8 neighbors)) / peak`` exceeds this — values near
                1 are hot pixels / cosmic rays, not stars. A critically sampled
                PSF scores ~0.5; strongly undersampled optics up to ~0.85.
                Set to None for severely undersampled data (PSF FWHM below
                ~1.5 px), where real stars are indistinguishable from hot
                pixels. Default 0.9.
            saturation_level: Pixel value at or above which the sensor is
                saturated; such blobs skip sub-pixel peak refinement (a flat top
                has no meaningful maximum) and keep the center-of-mass position.
                None = disabled.
            deblend: Policy for blobs with more than one distinct intensity
                peak (blended pairs centroid to a wrong midpoint position).
                "off" keeps them merged; "reject" drops them — the safe choice
                for plate solving. Saturated blobs are exempt.
            border_margin: Drop blobs whose bounding box comes within this many
                pixels of an image edge (truncated PSFs bias the center-of-mass
                inward).

        Returns:
            ExtractionResult with centroids and image statistics.
        """
        ...

def extract_centroids(
    image: npt.NDArray,
    sigma_threshold: float = 5.0,
    min_pixels: int = 3,
    max_pixels: int = 10000,
    max_centroids: Optional[int] = None,
    local_bg_block_size: Optional[int] = 64,
    max_elongation: Optional[float] = 3.0,
    matched_filter_sigma: Optional[float] = 1.5,
    max_sharpness: Optional[float] = 0.9,
    saturation_level: Optional[float] = None,
    deblend: str = "off",
    border_margin: int = 0,
) -> ExtractionResult:
    """Extract star centroids from a 2D image array.

    Args:
        image: 2D numpy array (height x width) of pixel values.
            Supported dtypes: float64, float32, uint16, int16, uint8.
        sigma_threshold: Detection threshold in sigma above background.
        min_pixels: Minimum blob size in pixels.
        max_pixels: Maximum blob size in pixels.
        max_centroids: Maximum number of centroids to return. None = all.
        local_bg_block_size: Block size for local background estimation.
            None = global background only.
        max_elongation: Maximum blob elongation ratio. None = disabled.
        matched_filter_sigma: Apply a Gaussian matched filter of this sigma
            (in pixels) before thresholding (~2x point-source SNR for a
            sigma~1.5 px PSF). Used only to form the detection mask, so
            photometry is unaffected, and the threshold is automatically
            scaled for the filtered noise level — no retuning needed.
            None = disabled. Default 1.5.
        max_sharpness: Reject blobs whose peak sharpness
            ``(peak - mean(8 neighbors)) / peak`` exceeds this — values near
            1 are hot pixels / cosmic rays, not stars. A critically sampled
            PSF scores ~0.5; strongly undersampled optics up to ~0.85.
            Set to None for severely undersampled data (PSF FWHM below
            ~1.5 px), where real stars are indistinguishable from hot
            pixels. Default 0.9.
        saturation_level: Pixel value at or above which the sensor is
            saturated; such blobs skip sub-pixel peak refinement (a flat top
            has no meaningful maximum) and keep the center-of-mass position.
            None = disabled.
        deblend: Policy for blobs with more than one distinct intensity
            peak (blended pairs centroid to a wrong midpoint position).
            "off" keeps them merged; "reject" drops them — the safe choice
            for plate solving. Saturated blobs are exempt.
        border_margin: Drop blobs whose bounding box comes within this many
            pixels of an image edge (truncated PSFs bias the center-of-mass
            inward).

    Returns:
        ExtractionResult with centroids and image statistics.
    """
    ...

def extract_centroids_fast(
    image: npt.NDArray,
    sigma_threshold: float = 5.0,
    bg_grid: int = 64,
    min_pixels: int = 2,
    max_centroids: Optional[int] = None,
    max_sharpness: Optional[float] = 0.9,
    saturation_level: Optional[float] = None,
    max_pixels: int = 10000,
    max_elongation: Optional[float] = None,
    border_margin: int = 0,
) -> ExtractionResult:
    """Fast single-pass centroid extraction — the "adequate star tracker" path.

    Reads each pixel once (coarse-grid background + run-length connected-
    component moments), so it is several times faster than
    :func:`extract_centroids` — at the cost of faint-star sensitivity and
    sub-pixel accuracy (~0.1 px on bright stars). Use :func:`extract_centroids`
    for calibration or faint-star work. Returns the same ``ExtractionResult``,
    so it is a drop-in for ``solve_from_centroids``.

    Args:
        image: 2D numpy array (height x width) of pixel values.
            Supported dtypes: float64, float32, uint16, int16, uint8.
        sigma_threshold: Detection threshold in noise sigmas above the local
            background.
        bg_grid: Coarse background-grid block size in pixels (tracks gradients
            such as vignetting / Milky Way).
        min_pixels: Minimum pixels in a region; rejects hot pixels.
        max_centroids: Maximum number of centroids to return, brightest first.
            None = all.
        max_sharpness: Reject regions whose peak sharpness
            ``(peak - mean(8 neighbors)) / peak`` exceeds this — values near
            1 are hot pixels / cosmic rays, not stars. Set to None for
            severely undersampled data (PSF FWHM below ~1.5 px).
            Default 0.9.
        saturation_level: Pixel value at or above which the sensor is
            saturated; such regions skip sub-pixel peak refinement and keep
            the center-of-mass position. None = disabled.
        max_pixels: Maximum region size in pixels — without it a satellite
            trail or horizon glow becomes the *brightest* centroid handed to
            the solver.
        max_elongation: Maximum elongation ratio (major/minor axis) from
            intensity-weighted second moments — rejects streaks too small
            for max_pixels. Off by default (noisy for few-pixel regions);
            enable (3.0-5.0) with min_pixels raised to ~5+. None = disabled.
        border_margin: Drop regions whose bounding box comes within this
            many pixels of an image edge (truncated PSFs bias the
            center-of-mass inward).

    Returns:
        ExtractionResult with centroids and image statistics.
    """
    ...

def earth_barycentric_velocity(dt: datetime.datetime) -> list[float]:
    """Approximate Earth barycentric velocity in km/s (ICRS equatorial frame).

    Uses a circular-orbit approximation. Accuracy ~0.5 km/s (~1.7%),
    sufficient for stellar aberration correction (~20" effect, ~0.3" error).

    Args:
        dt: Observation time as a ``datetime.datetime`` (UTC).
            The ~69 s offset between UTC and TT is negligible for this
            approximation.

    Returns:
        [vx, vy, vz] in km/s, ICRS equatorial frame. Pass directly to
        ``solve_from_centroids(observer_velocity_km_s=...)``.

    Example::

        from datetime import datetime
        v = tetra3rs.earth_barycentric_velocity(datetime(2025, 7, 10))
        result = db.solve_from_centroids(centroids, ..., observer_velocity_km_s=v)
    """
    ...
