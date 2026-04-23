from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tetra3rs")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from .tetra3rs import *  # type: ignore  # noqa: F403
from .tetra3rs import (
    CameraModel,
    CalibrateResult,
    CatalogStar,
    Centroid,
    ExtractionResult,
    PolynomialDistortion,
    RadialDistortion,
    SolveResult,
    SolverDatabase,
    earth_barycentric_velocity,
    extract_centroids,
    __git_hash__,
)

__all__ = [
    "CameraModel",
    "CalibrateResult",
    "CatalogStar",
    "Centroid",
    "ExtractionResult",
    "PolynomialDistortion",
    "RadialDistortion",
    "SolveResult",
    "SolverDatabase",
    "earth_barycentric_velocity",
    "extract_centroids",
    "__git_hash__",
]
