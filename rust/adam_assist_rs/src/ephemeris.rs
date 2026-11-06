//! Serialized ownership wrapper for the upstream ASSIST ephemeris.
//!
//! Upstream ASSIST's SPK parser uses process-global `strtok` state while an
//! ephemeris is created. Serialization therefore belongs at the lowest public
//! raw constructor exported by this crate, not only in higher-level provider
//! constructors: raw [`Ephemeris::from_paths`] and [`crate::AssistPropagator`]
//! callers in one linked crate instance participate in the same process-global
//! lock. Provider construction holds the same guard while observing eager DAF
//! regions immediately around parsing. Propagation and ephemeris queries are
//! intentionally not serialized. Direct `libassist_sys::Ephemeris`
//! construction, separately linked crate/library copies, and external upstream
//! ASSIST C/FFI callers are outside this guarantee.

use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

static EPHEMERIS_CONSTRUCTION_LOCK: Mutex<()> = Mutex::new(());

pub(crate) fn construction_guard() -> MutexGuard<'static, ()> {
    // A Rust panic cannot cross the underlying C call, and this mutex protects
    // no Rust data invariant. If unrelated caller code panicked while holding
    // the guard, recovering the lock is safer than permanently disabling every
    // future constructor in the process.
    EPHEMERIS_CONSTRUCTION_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Owned ASSIST ephemeris with process-global serialized construction.
///
/// All methods other than construction delegate to the canonical
/// [`libassist_sys::Ephemeris`]. Propagation does not take the construction
/// lock and remains parallel.
pub struct Ephemeris(libassist_sys::Ephemeris);

impl Ephemeris {
    /// Load an ASSIST ephemeris from planetary and small-body SPK paths.
    ///
    /// Calls in every thread through this wrapper (including all crate-owned
    /// provider routes) share one process-global lock around upstream parsing.
    /// The lock is released before this function returns and is never held
    /// during propagation or queries. Direct `libassist_sys` and separately
    /// linked copies are outside this guarantee.
    pub fn from_paths(
        planets: &Path,
        asteroids: &Path,
    ) -> std::result::Result<Self, libassist_sys::Error> {
        let _construction = construction_guard();
        Self::from_paths_while_construction_locked(planets, asteroids)
    }

    /// Construct while the caller holds [`construction_guard`].
    ///
    /// This crate-private route lets path-backed [`crate::AssistData`] observe
    /// the small SPK regions ASSIST parses eagerly immediately before and after
    /// the upstream call, without taking an outer second lock.
    pub(crate) fn from_paths_while_construction_locked(
        planets: &Path,
        asteroids: &Path,
    ) -> std::result::Result<Self, libassist_sys::Error> {
        libassist_sys::Ephemeris::from_paths(planets, asteroids).map(Self)
    }
}

impl Deref for Ephemeris {
    type Target = libassist_sys::Ephemeris;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Ephemeris {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_lock_recovers_from_poison_without_deadlock() {
        let _ = std::panic::catch_unwind(|| {
            let _guard = construction_guard();
            panic!("intentional construction-lock poison test");
        });
        drop(construction_guard());
    }
}
