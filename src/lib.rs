//! Tetra4: Fast and robust star extraction and astrometry library
//!

/// Raw star catalogs; currently Tycho-2 & Hipparcos
pub(crate) mod catalogs;
mod centroid;
pub mod solver;
pub mod star;
pub mod starcatalog;

pub use centroid::*;
pub use solver::{
    DatabaseProperties, GenerateDatabaseConfig, SolveConfig, SolveResult, SolveStatus,
    SolverDatabase,
};
pub use star::*;
pub use starcatalog::*;

// Commonly used types
// Note: 32-bit floats are sufficient for most of the math
// We switch to 64-bit for the SVD used in the final solver step,
// as 32-bit floats have shown to be insufficiently accurate for that step.
pub type Quaternion = nalgebra::UnitQuaternion<f32>;
pub type Vector3 = nalgebra::Vector3<f32>;
pub type Matrix2 = nalgebra::Matrix2<f32>;
