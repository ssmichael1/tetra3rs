"""
tetra3rs: Fast star plate solver

A Rust implementation of the tetra3 star plate solving algorithm,
exposed to Python via PyO3.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Optional, Union

class SolveResult:
    """Result of a successful plate-solve.

    Returned by ``SolverDatabase.solve_from_centroids`` on a successful match.
    Contains the camera attitude, matched stars, and error statistics.
    """

    @property
    def rotation_matrix_icrs_to_camera(self) -> npt.NDArray[np.float64]:
        """3x3 rotation matrix from ICRS to camera frame."""
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
    def fov_deg(self) -> Optional[float]:
        """Solved horizontal field of view in degrees."""
        ...

    @property
    def num_matches(self) -> Optional[int]:
        """Number of matched star pairs."""
        ...

    @property
    def rmse_arcsec(self) -> Optional[float]:
        """Root mean square error of matched stars in arcseconds."""
        ...

    @property
    def p90e_arcsec(self) -> Optional[float]:
        """90th percentile error in arcseconds."""
        ...

    @property
    def max_err_arcsec(self) -> Optional[float]:
        """Maximum match error in arcseconds."""
        ...

    @property
    def probability(self) -> Optional[float]:
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
    def matched_catalog_ids(self) -> npt.NDArray[np.uint64]:
        """Catalog IDs of matched stars."""
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
    def distortion(self) -> Optional[Union["RadialDistortion", "PolynomialDistortion"]]:
        """The distortion model used during solving, if any.

        Returns a ``RadialDistortion`` or ``PolynomialDistortion`` instance,
        or ``None`` if no distortion was applied.
        """
        ...

    from typing import overload

    @overload
    def pixel_to_world(self, x: float, y: float) -> Optional[tuple[float, float]]: ...
    @overload
    def pixel_to_world(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

    def pixel_to_world(
        self,
        x: Union[float, npt.NDArray[np.float64]],
        y: Union[float, npt.NDArray[np.float64]],
    ) -> Union[Optional[tuple[float, float]], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Convert centered pixel coordinates to world coordinates (RA, Dec in degrees).

        Pixel coordinates use the same convention as solver centroids:
        origin at the image center, +X right, +Y down.

        Args:
            x: X pixel coordinate(s). Scalar or 1D numpy array.
            y: Y pixel coordinate(s). Scalar or 1D numpy array.

        Returns:
            (ra_deg, dec_deg): Tuple of RA and Dec in degrees.
                Scalars if input is scalar, numpy arrays if input is array.
                Array elements are NaN where the transform is undefined.
                Returns None for scalar input if the point is degenerate.
        """
        ...

    @overload
    def world_to_pixel(self, ra_deg: float, dec_deg: float) -> Optional[tuple[float, float]]: ...
    @overload
    def world_to_pixel(
        self, ra_deg: npt.NDArray[np.float64], dec_deg: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

    def world_to_pixel(
        self,
        ra_deg: Union[float, npt.NDArray[np.float64]],
        dec_deg: Union[float, npt.NDArray[np.float64]],
    ) -> Union[Optional[tuple[float, float]], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Convert world coordinates (RA, Dec in degrees) to centered pixel coordinates.

        Returns pixel coordinates in the same convention as solver centroids:
        origin at the image center, +X right, +Y down.

        Args:
            ra_deg: Right ascension in degrees. Scalar or 1D numpy array.
            dec_deg: Declination in degrees. Scalar or 1D numpy array.

        Returns:
            (x, y): Tuple of pixel coordinates.
                Scalars if input is scalar, numpy arrays if input is array.
                Array elements are NaN for points behind the camera.
                Returns None for scalar input if the point is behind the camera.
        """
        ...

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

class ExtractionResult(TypedDict):
    """Result dictionary returned by extract_centroids."""

    centroids: list[Centroid]
    """List of detected centroids, sorted by brightness (brightest first)."""
    image_width: int
    """Width of the input image in pixels."""
    image_height: int
    """Height of the input image in pixels."""
    background_mean: float
    """Estimated background mean."""
    background_sigma: float
    """Estimated background standard deviation."""
    threshold: float
    """Detection threshold used."""
    num_blobs_raw: int
    """Number of raw blobs before filtering."""

class Centroid:
    """A detected star centroid with position, brightness, and shape.

    Returned by ``extract_centroids``. Centroids use pixel coordinates
    with the origin at the image center, +X right, +Y down.
    """

    def __init__(self, x: float, y: float, brightness: Optional[float] = None) -> None:
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

    def undistort(self, distortion: RadialDistortion) -> Centroid:
        """Remove lens distortion from this centroid's position (distorted → ideal).

        Returns a new Centroid at the corrected position.
        Brightness and covariance are preserved.
        """
        ...

    def distort(self, distortion: RadialDistortion) -> Centroid:
        """Apply lens distortion to this centroid's position (ideal → distorted).

        Returns a new Centroid at the distorted position.
        Brightness and covariance are preserved.
        """
        ...

class SolverDatabase:
    """A star pattern database for plate solving.

    Generate from the Hipparcos catalog, or load a previously saved database.

    Example::

        db = tetra3rs.SolverDatabase.generate_from_hipparcos("data/hip2.dat")
        db.save_to_file("my_db.bin")
        db = tetra3rs.SolverDatabase.load_from_file("my_db.bin")
    """

    @staticmethod
    def generate_from_hipparcos(
        catalog_path: str,
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
        """Generate a database from the Hipparcos catalog file.

        Args:
            catalog_path: Path to the hip2.dat file.
            max_fov_deg: Maximum field of view in degrees.
            min_fov_deg: Minimum field of view in degrees.
                None means same as max_fov_deg (single-scale).
            star_max_magnitude: Faintest star to include. None = auto.
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
        match_max_error: Optional[float] = None,
        refine_iterations: int = 2,
        distortion: Optional[Union[RadialDistortion, PolynomialDistortion]] = None,
    ) -> Optional[SolveResult]:
        """Solve for camera attitude given star centroids.

        Args:
            centroids: Either a list of Centroid objects (from extract_centroids),
                or an Nx2/Nx3 numpy array of centroid positions in pixels.
                Columns are (x, y) or (x, y, brightness).
                Origin is at the image center, +X right, +Y down.
            fov_estimate_deg: Estimated horizontal field of view in degrees.
            fov_estimate_rad: Estimated horizontal field of view in radians.
                Exactly one of fov_estimate_deg or fov_estimate_rad must be provided.
            image_width: Image width in pixels.
            image_height: Image height in pixels.
            image_shape: Image shape as (height, width) tuple (numpy convention).
                Can be used instead of image_width/image_height.
            fov_max_error_deg: Maximum FOV error in degrees. None = no limit.
            fov_max_error_rad: Maximum FOV error in radians. None = no limit.
                At most one of fov_max_error_deg or fov_max_error_rad can be provided.
            match_radius: Match distance as fraction of FOV.
            match_threshold: False-positive probability threshold.
            solve_timeout_ms: Timeout in milliseconds. None = no timeout.
            match_max_error: Maximum edge-ratio error. None = use database value.
            refine_iterations: Number of iterative SVD refinement passes.
                Each pass re-projects catalog stars and re-matches centroids
                using the refined rotation. Default 2.
            distortion: Lens distortion model to apply to centroids before solving.
                When provided, observed centroid pixel coordinates are undistorted
                before being converted to unit vectors.

        Returns:
            A SolveResult on success, or None if no match was found.
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

    def fit_radial_distortion(
        self,
        solve_results: Union[SolveResult, list[SolveResult]],
        centroids: Union[
            list[Centroid],
            npt.NDArray[np.float64],
            list[Union[list[Centroid], npt.NDArray[np.float64]]],
        ],
        image_width: int,
        sigma_clip: float = 3.0,
        max_iterations: int = 20,
        stage2_threshold_px: Optional[float] = 5.0,
    ) -> DistortionFitResult:
        """Fit a radial distortion model (k1, k2, k3) from solve results.

        Args:
            solve_results: A SolveResult or list of SolveResult objects.
            centroids: Matching centroids.
            image_width: Image width in pixels.
            sigma_clip: Sigma threshold for outlier rejection.
            max_iterations: Maximum sigma-clip iterations.
            stage2_threshold_px: Loose pixel threshold for second-stage
                recovery. None to disable.

        Returns:
            DistortionFitResult with the fitted radial model and statistics.
        """
        ...

    def fit_polynomial_distortion(
        self,
        solve_results: Union[SolveResult, list[SolveResult]],
        centroids: Union[
            list[Centroid],
            npt.NDArray[np.float64],
            list[Union[list[Centroid], npt.NDArray[np.float64]]],
        ],
        image_width: int,
        order: int = 4,
        sigma_clip: float = 3.0,
        max_iterations: int = 20,
        stage2_threshold_px: Optional[float] = 5.0,
    ) -> DistortionFitResult:
        """Fit a polynomial (SIP-like) distortion model from solve results.

        This model captures arbitrary 2D distortion including radial, tangential,
        and cross-terms — suitable for wide-field cameras like TESS where the
        optical center is offset from the CCD center.

        Args:
            solve_results: A SolveResult or list of SolveResult objects.
            centroids: Matching centroids.
            image_width: Image width in pixels.
            order: Polynomial order (2-6). Default 4.
            sigma_clip: Sigma threshold for outlier rejection.
            max_iterations: Maximum sigma-clip iterations.
            stage2_threshold_px: Loose pixel threshold for second-stage
                recovery. None to disable.

        Returns:
            DistortionFitResult with the fitted polynomial model and statistics.
        """
        ...

class RadialDistortion:
    """Radial lens distortion model: r_d = r × (1 + k1·r² + k2·r⁴ + k3·r⁶).

    Coordinates are in pixels relative to the optical center (image center).

    Example::

        d = tetra3rs.RadialDistortion(k1=-7e-9, k2=2e-15)
        x_undistorted, y_undistorted = d.undistort(100.0, 200.0)
    """

    def __init__(
        self,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
    ) -> None: ...
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

    def distort(self, x: float, y: float) -> tuple[float, float]:
        """Forward distortion: ideal → distorted."""
        ...

    def undistort(self, x: float, y: float) -> tuple[float, float]:
        """Inverse distortion: distorted → ideal."""
        ...

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

    Typically fitted from solve results via
    ``SolverDatabase.fit_polynomial_distortion()``, or constructed directly
    from coefficient arrays (e.g. extracted from a FITS WCS SIP model).
    """

    def __init__(
        self,
        order: int,
        scale: float,
        a_coeffs: list[float],
        b_coeffs: list[float],
        ap_coeffs: list[float],
        bp_coeffs: list[float],
    ) -> None:
        """Create a polynomial distortion model from coefficient arrays.

        Each coefficient array must have exactly (order+1)(order+2)/2 elements,
        covering all terms from p+q=0 (constant offset) through p+q=order.

        Args:
            order: Polynomial order (2–6 typical).
            scale: Normalization scale (typically image_width / 2).
            a_coeffs: Forward A coefficients (x correction, ideal → distorted).
            b_coeffs: Forward B coefficients (y correction, ideal → distorted).
            ap_coeffs: Inverse AP coefficients (x correction, distorted → ideal).
            bp_coeffs: Inverse BP coefficients (y correction, distorted → ideal).
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

class DistortionFitResult:
    """Result of a distortion fitting procedure.

    Returned by ``SolverDatabase.fit_radial_distortion`` or
    ``SolverDatabase.fit_polynomial_distortion``.
    """

    @property
    def model(self) -> Optional[Union[RadialDistortion, PolynomialDistortion]]:
        """The fitted distortion model."""
        ...

    @property
    def rmse_before_px(self) -> float:
        """RMS pixel residual before distortion correction."""
        ...

    @property
    def rmse_after_px(self) -> float:
        """RMS pixel residual after distortion correction."""
        ...

    @property
    def n_inliers(self) -> int:
        """Number of inlier matches in the final fit."""
        ...

    @property
    def n_outliers(self) -> int:
        """Number of rejected outliers."""
        ...

    @property
    def iterations(self) -> int:
        """Number of sigma-clip iterations performed."""
        ...

    @property
    def inlier_mask(self) -> npt.NDArray[np.bool_]:
        """Boolean inlier mask (True = inlier, False = outlier)."""
        ...

def extract_centroids(
    image: npt.NDArray,
    sigma_threshold: float = 5.0,
    min_pixels: int = 3,
    max_pixels: int = 10000,
    max_centroids: Optional[int] = None,
    local_bg_block_size: Optional[int] = 64,
    max_elongation: Optional[float] = 3.0,
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

    Returns:
        A dict with 'centroids' (list of Centroid objects sorted by brightness),
        'image_width', 'image_height', 'background_mean',
        'background_sigma', 'threshold', and 'num_blobs_raw'.
    """
    ...

def undistort_centroids(
    centroids: list[Centroid],
    distortion: Union[RadialDistortion, PolynomialDistortion],
) -> list[Centroid]:
    """Apply distortion correction to a list of centroids (distorted → ideal).

    Returns a new list with corrected positions; brightness and covariance are preserved.

    Args:
        centroids: List of Centroid objects.
        distortion: A RadialDistortion model.

    Returns:
        A new list of Centroid objects with undistorted positions.
    """
    ...

def distort_centroids(
    centroids: list[Centroid],
    distortion: Union[RadialDistortion, PolynomialDistortion],
) -> list[Centroid]:
    """Apply forward distortion to a list of centroids (ideal → distorted).

    Returns a new list with distorted positions; brightness and covariance are preserved.

    Args:
        centroids: List of Centroid objects.
        distortion: A RadialDistortion model.

    Returns:
        A new list of Centroid objects with distorted positions.
    """
    ...
