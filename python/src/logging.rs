//! Bridge from the Rust `log` facade to Python's `logging`.
//!
//! Records go to `logging.getLogger(target)` with `::` replaced by `.`
//! (e.g. `tetra3.solver.solve`). Each target's Python logger and effective
//! level are cached on first use, so a disabled record is rejected by a map
//! lookup without attaching to the interpreter — solves run detached and log
//! from rayon workers. `reset()` drops the cache after a level change.
//!
//! The cache is an `ArcSwap`, not a lock: readers include detached rayon
//! workers, which `fork()` does not pause, and a child that inherited a lock
//! mid-read would block its first cache update forever. Loads and swaps never
//! wait, so an interrupted reader costs the child nothing.

use std::collections::HashMap;
use std::sync::Arc;

use arc_swap::ArcSwap;

use log::{Level, LevelFilter, Log, Metadata, Record};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

struct CacheEntry {
    logger: Py<PyAny>,
    filter: LevelFilter,
}

type Cache = HashMap<String, Arc<CacheEntry>>;

pub(crate) struct PyLogger {
    cache: ArcSwap<Cache>,
}

impl PyLogger {
    /// Install as the global `log` logger. Fails only if a logger is already
    /// set (e.g. another extension in the process installed one first).
    pub(crate) fn install() -> Result<&'static Self, log::SetLoggerError> {
        let logger: &'static Self = Box::leak(Box::new(Self {
            cache: ArcSwap::from_pointee(Cache::new()),
        }));
        log::set_logger(logger)?;
        log::set_max_level(LevelFilter::Debug);
        Ok(logger)
    }

    /// Forget cached loggers and levels so the next record re-reads them.
    pub(crate) fn reset(&self) {
        self.cache.store(Arc::default());
    }

    /// The cached Python logger for `target`, looking it up (and caching it
    /// with its effective level) on a miss.
    fn logger_for<'py>(&self, py: Python<'py>, target: &str) -> PyResult<Bound<'py, PyAny>> {
        if let Some(entry) = self.cache.load().get(target) {
            return Ok(entry.logger.bind(py).clone());
        }

        let logger = py
            .import(intern!(py, "logging"))?
            .call_method1(intern!(py, "getLogger"), (target.replace("::", "."),))?;
        let level: u32 = logger
            .call_method0(intern!(py, "getEffectiveLevel"))?
            .extract()?;
        let entry = Arc::new(CacheEntry {
            logger: logger.clone().unbind(),
            filter: level_filter(level),
        });
        // Copy-on-write; `rcu` retries if another thread swapped in between.
        self.cache.rcu(|cache| {
            let mut cache = Cache::clone(cache);
            cache.insert(target.to_owned(), Arc::clone(&entry));
            cache
        });
        Ok(logger)
    }

    fn emit(&self, py: Python<'_>, record: &Record) -> PyResult<()> {
        let logger = self.logger_for(py, record.target())?;
        let level = py_level(record.level());
        if !logger
            .call_method1(intern!(py, "isEnabledFor"), (level,))?
            .is_truthy()?
        {
            return Ok(());
        }
        let py_record = logger.call_method1(
            intern!(py, "makeRecord"),
            (
                logger.getattr(intern!(py, "name"))?,
                level,
                record.file(),
                record.line().unwrap_or_default(),
                record.args().to_string(),
                PyTuple::empty(py), // args
                py.None(),          // exc_info
            ),
        )?;
        logger.call_method1(intern!(py, "handle"), (py_record,))?;
        Ok(())
    }
}

impl Log for PyLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        // Uncached targets pass; `emit` checks them against Python.
        self.cache
            .load()
            .get(metadata.target())
            .is_none_or(|entry| metadata.level() <= entry.filter)
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        // `try_attach` declines during interpreter shutdown; the record is
        // dropped rather than touching a finalizing interpreter.
        Python::try_attach(|py| {
            // Logging must not clobber an exception the caller is propagating.
            let pending = PyErr::take(py);
            if let Err(err) = self.emit(py, record) {
                err.write_unraisable(py, None);
            }
            if let Some(pending) = pending {
                pending.restore(py);
            }
        });
    }

    fn flush(&self) {}
}

/// Python `logging` level number for a Rust log level.
fn py_level(level: Level) -> u32 {
    match level {
        Level::Error => 40,
        Level::Warn => 30,
        Level::Info => 20,
        Level::Debug => 10,
        Level::Trace => 5,
    }
}

/// Most verbose Rust level a Python logger with effective level `level`
/// lets through (a record passes when its Python level is ≥ `level`).
fn level_filter(level: u32) -> LevelFilter {
    match level {
        0..=5 => LevelFilter::Trace,
        6..=10 => LevelFilter::Debug,
        11..=20 => LevelFilter::Info,
        21..=30 => LevelFilter::Warn,
        31..=40 => LevelFilter::Error,
        _ => LevelFilter::Off,
    }
}
