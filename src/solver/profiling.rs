//! Lightweight, feature-gated solver profiling.
//!
//! Enabled by the `profile` cargo feature. When disabled, the `timed!` macro
//! and the `count` / `record` calls compile to nothing, so there is zero cost
//! in normal builds.
//!
//! Accumulation is per-thread into a small map keyed by a static bucket name
//! (see [`buckets`] for the canonical names). A bucket holds
//! `(total_nanoseconds, count)`. Leaf operations in the solver hot loop are
//! wrapped so the attribution survives `-O3` inlining (a sampling profiler
//! cannot see inlined helpers like `compute_sorted_edge_angles`).
//!
//! ## Scope & caveats
//!
//! - **Instrumented paths:** only the lost-in-space solve (`pattern_search::PatternSearch`)
//!   and its `wcs_refine`. The tracking/hint path (`track.rs`) and database
//!   generation are **not** instrumented — profiling a tracking workload yields
//!   empty buckets for the solve loop.
//! - **Threading:** accumulators are `thread_local`, so [`snapshot`] reports only
//!   the calling thread. A solve is single-threaded today; if solves are ever run
//!   in parallel, per-thread results would need to be merged by the caller.
//! - **Observer cost:** counting inside very hot leaves (hundreds of calls per
//!   solve) adds a few ns per call via the `RefCell` borrow, which
//!   slightly inflates the *absolute* timing of the enclosing span. Counts
//!   themselves are exact; relative splits from `timed!` spans around larger
//!   blocks are reliable.
//!
//! Typical use from a harness:
//! ```ignore
//! tetra3::solver::profiling::reset();
//! for c in centroid_sets { db.solve_from_centroids(c, &cfg); }
//! for (name, ns, n) in tetra3::solver::profiling::snapshot() { ... }
//! ```

use std::cell::RefCell;
use std::collections::HashMap;

/// Canonical bucket names — the single source of truth shared by the
/// instrumentation sites and any reporting harness. Using these consts instead
/// of string literals keeps the two in sync and makes typos a compile error.
pub mod buckets {
    // ── solver hot loop (pattern_search::PatternSearch::pass_at_fov) ──
    /// Image-side sorted edge angles + ratios (the N×N-precompute target).
    pub const IMAGE_EDGES: &str = "image_edges";
    /// Catalog-side sorted edge angles + ratios per surviving candidate.
    pub const CAT_EDGES: &str = "cat_edges";
    /// Candidate pattern-key range enumeration + sort, per combination.
    pub const KEY_ENUM: &str = "key_enum";
    /// Wahba SVD rotation estimate (incl. parity recompute).
    pub const SVD: &str = "svd";
    /// Verification cone query (catalog stars near boresight).
    pub const VERIFY_QUERY: &str = "verify_query";
    /// Verification centroid↔catalog greedy matching.
    pub const VERIFY_MATCH: &str = "verify_match";
    /// Full `wcs_refine` call (parent span of the `WCS_*` buckets below).
    pub const WCS_REFINE: &str = "wcs_refine";

    // ── counts (no time) ──
    pub const FOV_PASS: &str = "fov_pass";
    pub const COMBOS: &str = "combos";
    pub const CANDIDATES: &str = "candidates";
    pub const RATIO_PASS: &str = "ratio_pass";
    pub const VERIFY_QUERY_STARS: &str = "verify_query_stars";

    // ── wcs_refine internals ──
    /// Number of outer refinement iterations.
    pub const WCS_OUTER: &str = "wcs_outer";
    /// Number of inner least-squares iterations.
    pub const WCS_INNER: &str = "wcs_inner";
    /// Phase-D re-association: catalog cone query (timed).
    pub const WCS_REASSOC_QUERY: &str = "wcs_reassoc_query";
    /// Phase-D re-association: project nearby catalog stars to pixels (timed).
    pub const WCS_REASSOC_PROJECT: &str = "wcs_reassoc_project";
    /// Phase-D re-association: pixel-space greedy matching (timed).
    pub const WCS_REASSOC_MATCH: &str = "wcs_reassoc_match";
    /// Phase-D re-association invocations (count).
    pub const WCS_REASSOC_CALL: &str = "wcs_reassoc_call";
    /// Phase-D nearby catalog stars projected (count).
    pub const WCS_REASSOC_STARS: &str = "wcs_reassoc_stars";
}

thread_local! {
    static ACC: RefCell<HashMap<&'static str, (u128, u64)>> = RefCell::new(HashMap::new());
}

/// Add `ns` nanoseconds and `count` operations to `bucket`.
pub fn record(bucket: &'static str, ns: u128, count: u64) {
    ACC.with(|a| {
        let mut m = a.borrow_mut();
        let e = m.entry(bucket).or_insert((0, 0));
        e.0 += ns;
        e.1 += count;
    });
}

/// Increment `bucket`'s operation count by `n` without adding time.
pub fn count(bucket: &'static str, n: u64) {
    record(bucket, 0, n);
}

/// Clear all accumulated buckets on this thread.
pub fn reset() {
    ACC.with(|a| a.borrow_mut().clear());
}

/// Snapshot the accumulators as `(bucket, total_ns, count)` triples.
pub fn snapshot() -> Vec<(&'static str, u128, u64)> {
    ACC.with(|a| a.borrow().iter().map(|(k, v)| (*k, v.0, v.1)).collect())
}
