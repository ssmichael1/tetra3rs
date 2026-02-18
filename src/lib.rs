//! # tetra3
//!
//! A fast, robust **lost-in-space star plate solver** written in Rust.
//!
//! > **Status: Alpha** — The core solver is based on well-vetted algorithms but has
//! > only been tested against a limited set of images. The API is not yet stable and
//! > may change between releases. Having said that, it has been made to work on both
//! > low-SNR images taken with a backyard camera and high-star-density images from
//! > more complex telescopes.
//!
//! Given a set of star centroids extracted from a camera image, `tetra3` identifies
//! the stars against a catalog and returns the camera's pointing direction as a
//! quaternion — no prior attitude estimate required.
//!
//! ## Features
//!
//! - **Lost-in-space solving** — determines attitude from star patterns with no initial guess
//! - **Fast** — geometric hashing of 4-star patterns with breadth-first (brightest-first) search
//! - **Robust** — statistical verification via binomial false-positive probability
//! - **Multiscale** — supports a range of field-of-view scales in a single database
//! - **Proper motion** — propagates Hipparcos catalog positions to any observation epoch
//! - **Zero-copy deserialization** — databases serialize with [rkyv](https://docs.rs/rkyv)
//!   for instant loading
//!
//! ## Example
//!
//! ```no_run
//! use tetra3::{GenerateDatabaseConfig, SolverDatabase, SolveConfig, Centroid, SolveStatus};
//!
//! // Generate a database from the Hipparcos catalog
//! let config = GenerateDatabaseConfig {
//!     max_fov_deg: 20.0,
//!     epoch_proper_motion_year: Some(2025.0),
//!     ..Default::default()
//! };
//! let db = SolverDatabase::generate_from_hipparcos("data/hip2.dat", &config).unwrap();
//!
//! // Save for fast loading later, or load a previously saved database
//! db.save_to_file("data/my_database.rkyv").unwrap();
//! let db = SolverDatabase::load_from_file("data/my_database.rkyv").unwrap();
//!
//! // Solve from image centroids (pixel coordinates, origin at image center)
//! let centroids = vec![
//!     Centroid { x: 100.0, y: 200.0, mass: Some(50.0), cov: None },
//!     Centroid { x: -50.0, y: -10.0, mass: Some(45.0), cov: None },
//!     // ... more centroids ...
//! ];
//!
//! let solve_config = SolveConfig {
//!     fov_estimate_rad: (15.0_f32).to_radians(), // horizontal FOV
//!     image_width: 1024,
//!     image_height: 1024,
//!     fov_max_error_rad: Some((2.0_f32).to_radians()),
//!     ..Default::default()
//! };
//!
//! let result = db.solve_from_centroids(&centroids, &solve_config);
//! if result.status == SolveStatus::MatchFound {
//!     let q = result.qicrs2cam.unwrap();
//!     println!("Attitude: {q}");
//!     println!("Matched {} stars in {:.1} ms",
//!         result.num_matches.unwrap(), result.solve_time_ms);
//! }
//! ```
//!
//! ## Algorithm overview
//!
//! 1. **Pattern generation** — select combinations of 4 bright centroids; compute 6 pairwise
//!    angular separations and normalize into 5 edge ratios (a geometric invariant)
//! 2. **Hash lookup** — quantize the edge ratios into a key and probe a precomputed hash
//!    table for matching catalog patterns
//! 3. **Attitude estimation** — solve Wahba's problem via SVD to find the rotation from
//!    catalog (ICRS) to camera frame
//! 4. **Verification** — project nearby catalog stars into the camera frame, count matches,
//!    and accept only if the false-positive probability (binomial CDF) is below threshold
//! 5. **Refinement** — re-estimate the rotation using all matched star pairs
//!
//! ## Credits
//!
//! This crate is a Rust implementation of the **tetra3** / **cedar-solve** algorithm:
//!
//! - [**tetra3**](https://github.com/esa/tetra3) — the original Python implementation by
//!   Gustav Pettersson at ESA
//! - [**cedar-solve**](https://github.com/smroid/cedar-solve) — Steven Rosenthal's C++/Rust
//!   star plate solver, which this implementation closely follows
//! - **Paper**: G. Pettersson, "Tetra3: a fast and robust star identification algorithm,"
//!   ESA GNC Conference, 2023
//!
//! This Rust implementation was developed by Steven Michael with assistance from
//! [Claude Code](https://claude.ai/claude-code) (Anthropic).
//!

/// Raw star catalogs; currently Tycho-2 & Hipparcos
pub(crate) mod catalogs;
mod centroid;
#[cfg(feature = "image")]
pub mod centroid_extraction;
pub mod distortion;
pub mod solver;
pub mod star;
pub mod starcatalog;

pub use centroid::*;
#[cfg(feature = "image")]
pub use centroid_extraction::{
    extract_centroids, extract_centroids_from_image, extract_centroids_from_raw,
    CentroidExtractionConfig, CentroidExtractionResult,
};
pub use distortion::{
    Distortion, DistortionFitConfig, DistortionFitResult, RadialDistortion,
};
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
