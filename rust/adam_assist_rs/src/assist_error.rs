//! adam-assist-owned error normalization over the canonical sys crates.
//!
//! The sys crates correctly preserve their layer boundary by nesting REBOUND
//! integration exits inside `libassist_sys::Error::Reb`. The adapter flattens
//! those exits because its public propagation contract classifies them as
//! per-row failures while setup/ephemeris failures abort the backend call.

/// Errors used by adam-assist propagation orchestration.
#[derive(Debug, thiserror::Error)]
pub enum AssistError {
    /// REBOUND stopped because the simulation contains no particles.
    #[error("integration ended: no particles remain in the simulation")]
    NoParticles,
    /// REBOUND stopped at a configured close encounter.
    #[error("integration ended: close encounter")]
    CloseEncounter,
    /// REBOUND stopped because a particle escaped.
    #[error("integration ended: particle escape")]
    Escape,
    /// REBOUND stopped because particles collided.
    #[error("integration ended: collision")]
    Collision,
    /// REBOUND returned an unclassified integration status code.
    #[error("REBOUND integration failed with status {0}")]
    IntegrationFailed(i32),
    /// ASSIST could not load or query its ephemeris.
    #[error("ASSIST ephemeris error: {0}")]
    EphemerisError(String),
    /// Another ASSIST orchestration error.
    #[error("{0}")]
    Other(String),
}

impl From<librebound_sys::Error> for AssistError {
    fn from(error: librebound_sys::Error) -> Self {
        match error {
            librebound_sys::Error::NoParticles => Self::NoParticles,
            librebound_sys::Error::CloseEncounter => Self::CloseEncounter,
            librebound_sys::Error::Escape => Self::Escape,
            librebound_sys::Error::Collision => Self::Collision,
            librebound_sys::Error::IntegrationFailed(status) => Self::IntegrationFailed(status),
            librebound_sys::Error::Other(message) => Self::Other(message),
        }
    }
}

impl From<libassist_sys::Error> for AssistError {
    fn from(error: libassist_sys::Error) -> Self {
        match error {
            libassist_sys::Error::Reb(error) => error.into(),
            libassist_sys::Error::EphemerisError(message) => Self::EphemerisError(message),
            libassist_sys::Error::Other(message) => Self::Other(message),
        }
    }
}

/// Result type returned by public adam-assist operations.
pub type AssistResult<T> = std::result::Result<T, AssistError>;
