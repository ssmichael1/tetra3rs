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
