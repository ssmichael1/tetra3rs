//! Portable monotonic clock for the solve path.
//!
//! On native targets this is exactly [`std::time::Instant`]. On
//! `wasm32-unknown-unknown` there is no monotonic clock in `std` —
//! `std::time::Instant::now()` aborts the module — so we substitute a no-op
//! clock that always reports zero elapsed time.
//!
//! The practical consequence for a WASM build: `solve_time_ms` in the result is
//! reported as `0.0`, and [`SolveConfig::solve_timeout_ms`] never trips (elapsed
//! is always below any positive budget). Measure wall-clock time on the host
//! (e.g. `performance.now()`) instead. A production browser build that needs a
//! real in-WASM clock can swap this for a `wasm-bindgen`-backed
//! `performance.now()` source (see the `web-time` crate).
//!
//! [`SolveConfig::solve_timeout_ms`]: super::SolveConfig::solve_timeout_ms

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::Instant;

#[cfg(target_arch = "wasm32")]
pub(crate) use wasm_clock::Instant;

#[cfg(target_arch = "wasm32")]
mod wasm_clock {
    use std::time::Duration;

    /// Drop-in stand-in for `std::time::Instant` on `wasm32-unknown-unknown`,
    /// where `std` has no monotonic clock. Always reports zero elapsed time.
    #[derive(Debug, Clone, Copy)]
    pub struct Instant;

    impl Instant {
        #[inline]
        pub fn now() -> Self {
            Instant
        }

        #[inline]
        pub fn elapsed(&self) -> Duration {
            Duration::ZERO
        }
    }
}
