"""The `threads=` argument of the extraction functions, and fork safety.

The extension is built with tetra3's `parallel` feature and runs every
multi-threaded path (extraction, database loading) on a thread pool it owns.
Worker threads do not survive `fork()`, so a pool inherited from a parent
must be abandoned, not used — otherwise the first such call in a forked child
waits forever.
"""

import multiprocessing
import os
import pickle
import warnings

import numpy as np
import pytest

import tetra3rs

# Generous: a healthy child finishes in well under a second.
FORK_TIMEOUT_S = 120


def _frame(width, height, n_stars, seed):
    """Gaussian-noise frame with Gaussian stars (float32)."""
    rng = np.random.default_rng(seed)
    img = rng.normal(100.0, 5.0, (height, width)).astype(np.float32)
    yy, xx = np.mgrid[-6:7, -6:7]
    for _ in range(n_stars):
        x = rng.uniform(8, width - 8)
        y = rng.uniform(8, height - 8)
        amp = 10 ** rng.uniform(1.5, 3.5)
        cx, cy = int(round(x)), int(round(y))
        psf = amp * np.exp(-((xx + cx - x) ** 2 + (yy + cy - y) ** 2) / (2 * 1.5**2))
        img[cy - 6 : cy + 7, cx - 6 : cx + 7] += psf.astype(np.float32)
    return img


# Above and below the size under which `threads=None` picks one thread.
LARGE = _frame(900, 700, 80, seed=3)
SMALL = _frame(200, 150, 8, seed=4)


def _key(result):
    """Everything an extraction returns, exactly (floats compared bit for bit)."""
    return (
        [
            (c.x, c.y, c.brightness, None if c.cov is None else c.cov.tobytes())
            for c in result.centroids
        ],
        result.image_width,
        result.image_height,
        result.background_mean,
        result.background_sigma,
        result.threshold,
        result.num_blobs_raw,
    )


def _extractors():
    extractor = tetra3rs.CentroidExtractor()
    return {
        "extract_centroids": tetra3rs.extract_centroids,
        "extract_centroids_fast": tetra3rs.extract_centroids_fast,
        "CentroidExtractor.extract": extractor.extract,
    }


def _all_results(frames=(LARGE, SMALL), thread_settings=(None, 1, 2)):
    return {
        (name, i, threads): _key(fn(frame, threads=threads))
        for name, fn in _extractors().items()
        for i, frame in enumerate(frames)
        for threads in thread_settings
    }


@pytest.mark.parametrize("name", list(_extractors()))
def test_threads_do_not_change_results(name):
    fn = _extractors()[name]
    for frame in (LARGE, SMALL):
        reference = _key(fn(frame, threads=1))
        assert len(reference[0]) > 0
        for threads in (None, 2, 3, 7):
            assert _key(fn(frame, threads=threads)) == reference, f"threads={threads}"
        # The default is `threads=None`.
        assert _key(fn(frame)) == reference


@pytest.mark.parametrize("name", list(_extractors()))
def test_threads_validation(name):
    fn = _extractors()[name]
    with pytest.raises(ValueError):
        fn(LARGE, threads=0)
    with pytest.raises(ValueError):
        fn(LARGE, threads=1_000_000)
    with pytest.raises((OverflowError, ValueError)):
        fn(LARGE, threads=-1)
    # Keyword-only.
    with pytest.raises(TypeError):
        if name == "extract_centroids_fast":
            fn(LARGE, 5.0, 64, 2, None, 0.9, None, 10000, None, 0, 2)
        else:
            fn(LARGE, 5.0, 3, 10000, None, 64, 3.0, 1.5, 0.9, None, "off", 0, 2)


def test_many_thread_counts_reuse_a_bounded_set_of_pools():
    # More distinct counts than pools are kept; each must still work, and
    # going back to an evicted count must rebuild it.
    reference = _key(tetra3rs.extract_centroids(LARGE, threads=1))
    for threads in (1, 2, 3, 4, 5, 6, 2, 1, None, 6):
        assert _key(tetra3rs.extract_centroids(LARGE, threads=threads)) == reference


def _run_in_fork(target, *args):
    """Run `target(conn, *args)` in a forked child; return what it sends.

    Fails (instead of hanging the test run) if the child does not finish.
    """
    ctx = multiprocessing.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=target, args=(child_conn, *args))
    with warnings.catch_warnings():
        # Python 3.12+ warns about fork() in a multi-threaded process; the
        # pool's workers are exactly the threads this test is about.
        warnings.simplefilter("ignore", DeprecationWarning)
        proc.start()
    child_conn.close()
    try:
        if not parent_conn.poll(FORK_TIMEOUT_S):
            pytest.fail(
                "forked child did not finish: it is waiting on a thread pool "
                "whose workers did not survive fork()"
            )
        payload = parent_conn.recv()
    finally:
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
    assert proc.exitcode == 0
    return payload


def _child_extract(conn):
    conn.send(_all_results())
    conn.close()


needs_fork = pytest.mark.skipif(
    not hasattr(os, "fork") or "fork" not in multiprocessing.get_all_start_methods(),
    reason="platform has no fork()",
)


@needs_fork
def test_extraction_in_forked_child():
    # Start the parent's pools first (default, 1 and 2 threads): the child
    # inherits them with no workers behind them.
    expected = _all_results()
    assert _run_in_fork(_child_extract) == expected
    # The parent's pools are unaffected by the fork.
    assert _all_results() == expected


def _child_fork_again(conn):
    first = _all_results()
    second = _run_in_fork(_child_extract)
    conn.send((first, second))
    conn.close()


@needs_fork
def test_extraction_in_forked_grandchild():
    expected = _all_results()
    first, second = _run_in_fork(_child_fork_again)
    assert first == expected
    assert second == expected


def _child_load_database(conn, blob, path):
    unpickled = pickle.loads(blob)
    loaded = tetra3rs.SolverDatabase.load_from_file(path)
    conn.send((unpickled.num_patterns, loaded.num_patterns))
    conn.close()


@needs_fork
def test_database_load_in_forked_child(skyview_db, tmp_path):
    # Database decode + validation are multi-threaded too. The common case:
    # a parent that has loaded a database hands it to forked workers, which
    # unpickle it.
    path = str(tmp_path / "db.bin")
    skyview_db.save_to_file(path)
    blob = pickle.dumps(skyview_db)
    n = skyview_db.num_patterns
    # The parent runs both load paths, so its pool has live worker threads.
    assert pickle.loads(blob).num_patterns == n
    assert tetra3rs.SolverDatabase.load_from_file(path).num_patterns == n
    assert _run_in_fork(_child_load_database, blob, path) == (n, n)
