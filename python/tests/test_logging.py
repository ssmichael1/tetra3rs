"""Rust `log` records forwarded to Python's `logging` (python/src/logging.rs)."""

import logging

import numpy as np

import tetra3rs

LOGGER = "tetra3.solver.solve"


def _invalid_solve(db):
    # A NaN match radius fails config validation, which logs one warning
    # from `tetra3::solver::solve` before any search work.
    result = db.solve_from_centroids(
        np.zeros((4, 2)),
        fov_estimate_deg=10.0,
        image_width=1024,
        image_height=1024,
        match_radius=float("nan"),
    )
    assert not result


def _records(caplog):
    return [r for r in caplog.records if r.name == LOGGER and r.levelno == logging.WARNING]


def test_warning_reaches_python_logging(skyview_db, caplog):
    tetra3rs.reset_log_cache()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _invalid_solve(skyview_db)
    records = _records(caplog)
    assert len(records) == 1
    assert "solve_from_centroids" in records[0].getMessage()
    assert records[0].pathname.endswith("solve.rs")


def test_level_cache_and_reset(skyview_db, caplog):
    logger = logging.getLogger(LOGGER)
    old_level = logger.level
    try:
        logger.setLevel(logging.ERROR)
        tetra3rs.reset_log_cache()
        _invalid_solve(skyview_db)
        assert not _records(caplog)

        # Cached level still says ERROR until the cache is reset.
        logger.setLevel(logging.WARNING)
        _invalid_solve(skyview_db)
        assert not _records(caplog)

        tetra3rs.reset_log_cache()
        _invalid_solve(skyview_db)
        assert len(_records(caplog)) == 1
    finally:
        logger.setLevel(old_level)
        tetra3rs.reset_log_cache()
