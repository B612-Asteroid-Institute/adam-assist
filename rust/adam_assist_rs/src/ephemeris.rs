//! Serialized ownership wrapper for the upstream ASSIST ephemeris.
//!
//! Upstream ASSIST's SPK parser uses process-global `strtok` state while an
//! ephemeris is created. Serialization therefore belongs at the lowest public
//! raw constructor exported by this crate, not only in higher-level provider
//! constructors: raw [`Ephemeris::from_paths`] and [`crate::AssistPropagator`]
//! callers must participate in the same lock.

use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

static EPHEMERIS_CONSTRUCTION_LOCK: Mutex<()> = Mutex::new(());

fn construction_guard() -> MutexGuard<'static, ()> {
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
    /// Calls in every thread and every adam-assist construction route share
    /// one process-global lock around upstream parsing. The lock is released
    /// before this function returns and is never held during propagation.
    pub fn from_paths(
        planets: &Path,
        asteroids: &Path,
    ) -> std::result::Result<Self, libassist_sys::Error> {
        let _construction = construction_guard();
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
