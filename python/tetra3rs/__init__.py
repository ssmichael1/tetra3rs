from importlib.metadata import version

__version__ = version("tetra3rs")

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
