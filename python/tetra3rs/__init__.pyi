"""
tetra3rs: Fast star plate solver

A Rust implementation of the tetra3 star plate solving algorithm,
exposed to Python via PyO3.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Optional, TypedDict

class SolveResult(TypedDict):
    """Result dictionary returned by SolverDatabase.solve_from_centroids."""

    rotation_matrix: npt.NDArray[np.float64]
    """3x3 rotation matrix from ICRS to camera frame."""
    fov_deg: Optional[float]
    """Solved horizontal field of view in degrees."""
    num_matches: Optional[int]
    """Number of matched star pairs."""
    rmse_arcsec: Optional[float]
    """Root mean square error of matched stars in arcseconds."""
    p90e_arcsec: Optional[float]
    """90th percentile error in arcseconds."""
    max_err_arcsec: Optional[float]
    """Maximum match error in arcseconds."""
    probability: Optional[float]
    """False-positive probability (lower is better)."""
    solve_time_ms: float
    """Time taken to solve in milliseconds."""
    matched_centroids: npt.NDArray[np.uint64]
    """Indices of matched centroids in the input array."""
    matched_catalog_ids: npt.NDArray[np.uint32]
    """Hipparcos catalog IDs of matched stars."""
    status: str
    """Always 'match_found' when a result is returned."""

class ExtractionResult(TypedDict):
    """Result dictionary returned by extract_centroids."""

    centroids: npt.NDArray[np.float64]
    """Nx3 array of (x, y, brightness) in centered pixel coordinates."""
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
        centroids: npt.NDArray[np.float64],
        fov_estimate: float,
        image_width: int,
        image_height: int,
        fov_max_error: Optional[float] = None,
        match_radius: float = 0.01,
        match_threshold: float = 1e-5,
        solve_timeout_ms: Optional[int] = 5000,
        match_max_error: Optional[float] = None,
    ) -> Optional[SolveResult]:
        """Solve for camera attitude given star centroids.

        Args:
            centroids: Nx2 or Nx3 numpy array of centroid positions in pixels.
                Columns are (x, y) or (x, y, brightness).
                Origin is at the image center, +X right, +Y down.
            fov_estimate: Estimated horizontal field of view in degrees.
            image_width: Image width in pixels.
            image_height: Image height in pixels.
            fov_max_error: Maximum FOV error in degrees. None = no limit.
            match_radius: Match distance as fraction of FOV.
            match_threshold: False-positive probability threshold.
            solve_timeout_ms: Timeout in milliseconds. None = no timeout.
            match_max_error: Maximum edge-ratio error. None = use database value.

        Returns:
            A dict with solve results, or None if no match was found.
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
    image: npt.NDArray[np.float64],
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
        sigma_threshold: Detection threshold in sigma above background.
        min_pixels: Minimum blob size in pixels.
        max_pixels: Maximum blob size in pixels.
        max_centroids: Maximum number of centroids to return. None = all.
        local_bg_block_size: Block size for local background estimation.
            None = global background only.
        max_elongation: Maximum blob elongation ratio. None = disabled.

    Returns:
        A dict with 'centroids' (Nx3 array of x, y, brightness in centered
        pixel coords), 'image_width', 'image_height', 'background_mean',
        'background_sigma', 'threshold', and 'num_blobs_raw'.
    """
    ...
