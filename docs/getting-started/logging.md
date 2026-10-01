# Logging

tetra3rs logs through the Rust [`log`](https://docs.rs/log) facade. Nothing is
printed unless the host application installs a logger.

| Level | What it reports |
|---|---|
| `WARN` | Invalid solve configuration; catalog records skipped while loading |
| `INFO` | Database generation progress (catalog loading, magnitude cut, pattern FOV scales) |
| `DEBUG` | Per-solve detail: FOV sweep, pattern counts, candidate attitudes, WCS refinement iterations |

Log targets follow the Rust module path: `tetra3::solver::solve`,
`tetra3::solver::wcs_refine`, `tetra3::solver::database`, and so on.

## Python

Records are forwarded to Python's standard
[`logging`](https://docs.python.org/3/library/logging.html) module. Logger
names are the module paths with `::` replaced by `.` (`tetra3.solver.solve`),
so the whole library lives under the `tetra3` logger:

```python
import logging
import tetra3rs

logging.basicConfig(level=logging.WARNING)
logging.getLogger("tetra3").setLevel(logging.DEBUG)          # everything
logging.getLogger("tetra3.solver.wcs_refine").setLevel(logging.INFO)  # quiet one module
```

Each logger's effective level is cached the first time tetra3rs logs to it, so
a disabled message costs nothing inside a solve (solves release the GIL, and
checking Python's level would mean taking it back for every message). As a
consequence, **a level change made after tetra3rs has started logging is not
seen until you call [`reset_log_cache()`](../api/functions.md)**:

```python
result = db.solve_from_centroids(...)          # levels cached here

logging.getLogger("tetra3").setLevel(logging.DEBUG)
tetra3rs.reset_log_cache()                      # pick up the new level
result = db.solve_from_centroids(...)          # now logs DEBUG detail
```

Configuring logging before the first solve or database build needs no reset.

## Rust

Install any `log`-compatible logger in your application. With
[`env_logger`](https://docs.rs/env_logger):

```rust
env_logger::init();
```

```sh
RUST_LOG=tetra3=debug cargo run --release
RUST_LOG=tetra3::solver::database=info,tetra3=warn cargo run --release
```

Applications using [`tracing`](https://docs.rs/tracing) receive tetra3's
records too: `tracing_subscriber`'s `init()` / `try_init()` install its
`log` bridge by default (the `tracing-log` feature).
