//! Thread pools for the multi-threaded paths.
//!
//! The extension is built with tetra3's `parallel` feature, so the
//! extraction hot paths and database loading (pattern-table decode and
//! validation) fan out through rayon. They are never run on rayon's *global*
//! pool. Every such call goes through [`run`] or [`run_default`], which
//! execute it inside a pool owned by this module, for two reasons:
//!
//! - **Thread count per call.** A parallel region uses the pool it is
//!   running in, so `threads=N` is a pool of N workers and `threads=1` makes
//!   every parallel helper run on a single worker.
//! - **Fork safety.** Worker threads do not survive `fork()`: a child
//!   inherits a pool object whose workers do not exist, and anything
//!   submitted to it waits forever. The global pool can never be rebuilt,
//!   so one extraction or database load in a parent would hang every later
//!   one in its forked children (`multiprocessing`'s `fork` start method,
//!   `os.fork`) — including unpickling a database handed to a worker.
//!   The pools here are tagged with the process id that built them, and a
//!   process that finds another id abandons them and builds its own.
//!
//! Anything else in the extension that reaches rayon-parallel tetra3 code
//! must go through one of the two as well, or it reintroduces the fork
//! hazard (`python/tests/test_threads.py` hang-checks the known paths).

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

/// With `threads=None`, frames with fewer pixels than this run on one
/// thread: waking the workers costs more than it saves below roughly
/// 512×512 (measured crossover between 512×384 and 640×480).
const AUTO_SINGLE_THREAD_BELOW_PIXELS: usize = 1 << 18;

/// An explicit `threads=` above this is refused rather than spawned.
const MAX_THREADS: usize = 1024;

/// Pools kept alive at once (one per distinct thread count, least recently
/// used dropped first), so alternating between a few settings does not
/// rebuild a pool per call and sweeping many does not accumulate threads.
const MAX_POOLS: usize = 4;

/// The pools of one process: `pid` is the process that built them.
struct Pools {
    pid: u32,
    /// `(thread count, pool)`, most recently used last. A count of 0 is
    /// rayon's default (`RAYON_NUM_THREADS`, else the available cores).
    pools: Vec<(usize, Arc<ThreadPool>)>,
}

/// Locked only while attached to the interpreter and never across
/// `Python::detach`, so the thread that calls `fork()` — which is attached —
/// can never find it held by another thread and inherit it locked.
static POOLS: Mutex<Option<Pools>> = Mutex::new(None);

/// The pool with `threads` workers (0 = rayon's default), built on first use.
fn pool(threads: usize) -> PyResult<Arc<ThreadPool>> {
    let mut guard = POOLS.lock().unwrap_or_else(|e| e.into_inner());
    let pid = std::process::id();
    if guard.as_ref().is_some_and(|p| p.pid != pid) {
        // Forked child: these pools' workers exist only in the parent. Leak
        // them rather than drop them — dropping would signal threads that
        // are not there, through state copied mid-use.
        std::mem::forget(guard.take());
    }
    let state = guard.get_or_insert_with(|| Pools {
        pid,
        pools: Vec::new(),
    });
    if let Some(i) = state.pools.iter().position(|(n, _)| *n == threads) {
        let entry = state.pools.remove(i);
        let pool = Arc::clone(&entry.1);
        state.pools.push(entry);
        return Ok(pool);
    }
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("tetra3rs-{i}"))
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("could not start the thread pool: {e}")))?;
    let pool = Arc::new(pool);
    if state.pools.len() == MAX_POOLS {
        // Its workers exit once idle; a call still running on it keeps it
        // alive through its own `Arc`.
        state.pools.remove(0);
    }
    state.pools.push((threads, Arc::clone(&pool)));
    Ok(pool)
}

/// Run `f` with the interpreter detached, inside a pool sized by the
/// Python-level `threads` argument: `None` uses every available core (one
/// thread for frames of fewer than [`AUTO_SINGLE_THREAD_BELOW_PIXELS`]
/// `pixels`), `Some(n)` exactly `n` threads. Results do not depend on the
/// choice — the parallel paths are bit-identical to the sequential ones.
pub(crate) fn run<T, F>(py: Python<'_>, threads: Option<usize>, pixels: usize, f: F) -> PyResult<T>
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    let threads = match threads {
        Some(0) => {
            return Err(PyValueError::new_err(
                "threads must be at least 1 (or None to use all cores)",
            ))
        }
        Some(n) if n > MAX_THREADS => {
            return Err(PyValueError::new_err(format!(
                "threads must be at most {MAX_THREADS}, got {n}"
            )))
        }
        Some(n) => n,
        None if pixels < AUTO_SINGLE_THREAD_BELOW_PIXELS => 1,
        None => 0,
    };
    let pool = pool(threads)?;
    Ok(py.detach(|| pool.install(f)))
}

/// Run `f` with the interpreter detached, inside the default pool (every
/// available core, or `RAYON_NUM_THREADS`) — for the database paths, which
/// take no `threads` argument.
pub(crate) fn run_default<T, F>(py: Python<'_>, f: F) -> PyResult<T>
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    let pool = pool(0)?;
    Ok(py.detach(|| pool.install(f)))
}
