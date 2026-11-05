//! ASSIST propagation orchestration built on the canonical `libassist-sys`
//! and `librebound-sys` binding and RAII layers.
//!
//! Coordinate conversion, propagation, covariance transport, pooling, and
//! ephemeris-data ownership live together here so every public path uses the
//! same validated dynamics and resource contracts.
//!
//! Two propagation entry points:
//!
//! * [`assist_propagate_single`] — one-shot free function. Creates a fresh
//!   `AssistSim`, propagates, and tears down on return.
//! * [`PropagatorPool`] — stateful container that reuses the underlying
//!   `AssistSim` (and its REBOUND simulation, ASSIST extras, and IAS15
//!   allocations) across many orbits with the same force-model +
//!   variational-dimension configuration.

// Index-based loops are kept verbatim from the upstream matrix math for
// auditability of the move; iterator rewrites would obscure the row/column
// index correspondence in the 6x6/6x9 covariance products.
#![allow(clippy::needless_range_loop)]

use crate::{AssistError as Error, AssistResult as Result, Ephemeris};
use libassist_sys::{ffi, AssistSim};
use librebound_sys::{IntegratorConfig, Simulation};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{File, Metadata};
use std::io::{BufReader, Read, Seek, SeekFrom};
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Mutex, OnceLock};

// ─── Data bundle ────────────────────────────────────────────────────────────

#[cfg(feature = "kernel-data")]
const DE440_SIZE_BYTES: u64 = 119_799_808;
#[cfg(feature = "kernel-data")]
const DE440_SHA256: &str = "a4ce9bf9b3282becc9f4b2ac3cebe03a2ae7599981aabd7265fd8482fff7c4b5";
#[cfg(feature = "kernel-data")]
const SB441_N16_SIZE_BYTES: u64 = 645_727_232;
#[cfg(feature = "kernel-data")]
const SB441_N16_SHA256: &str = "919d612ce3c72a78fc7158f9120156542d0f21e6b8b052e4c1339c759747fd90";

#[derive(Clone, Debug, Eq, PartialEq)]
struct FileIdentity {
    size: u64,
    sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StableFileIdentity {
    size: u64,
    modified: Option<std::time::SystemTime>,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    mode: u32,
    #[cfg(unix)]
    hard_links: u64,
    #[cfg(unix)]
    modified_seconds: i64,
    #[cfg(unix)]
    modified_nanoseconds: i64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
}

impl StableFileIdentity {
    fn from_metadata(metadata: &Metadata) -> Self {
        Self {
            size: metadata.len(),
            modified: metadata.modified().ok(),
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            mode: metadata.mode(),
            #[cfg(unix)]
            hard_links: metadata.nlink(),
            #[cfg(unix)]
            modified_seconds: metadata.mtime(),
            #[cfg(unix)]
            modified_nanoseconds: metadata.mtime_nsec(),
            #[cfg(unix)]
            changed_seconds: metadata.ctime(),
            #[cfg(unix)]
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }

    #[cfg(unix)]
    fn same_file(&self, other: &Self) -> bool {
        self.device == other.device && self.inode == other.inode
    }

    #[cfg(not(unix))]
    fn same_file(&self, other: &Self) -> bool {
        self == other
    }
}

const DAF_RECORD_LENGTH: usize = 1024;

/// The immutable load-time facts on which ASSIST's eager parsing and fixed
/// mmap length depend. Deliberately excludes mutable metadata such as mtime,
/// ctime, mode and hard-link count.
#[derive(Clone, Debug, Eq, PartialEq)]
struct LoadedKernelFileIdentity {
    size: u64,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    eager_regions_sha256: String,
}

impl LoadedKernelFileIdentity {
    fn from_metadata_and_digest(metadata: &Metadata, eager_regions_sha256: String) -> Self {
        Self {
            size: metadata.len(),
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            eager_regions_sha256,
        }
    }

    fn matches_metadata(&self, metadata: &Metadata) -> bool {
        if self.size != metadata.len() {
            return false;
        }
        #[cfg(unix)]
        {
            self.device == metadata.dev() && self.inode == metadata.ino()
        }
        #[cfg(not(unix))]
        {
            true
        }
    }

    fn matches_stable_identity(&self, identity: &StableFileIdentity) -> bool {
        if self.size != identity.size {
            return false;
        }
        #[cfg(unix)]
        {
            self.device == identity.device && self.inode == identity.inode
        }
        #[cfg(not(unix))]
        {
            true
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LoadedKernelPairIdentity {
    planets: LoadedKernelFileIdentity,
    asteroids: LoadedKernelFileIdentity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BindingMismatch {
    EagerContent,
    MappingIdentity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct KernelFileMetadata {
    handle: StableFileIdentity,
    path: StableFileIdentity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct KernelPairMetadata {
    planets: KernelFileMetadata,
    asteroids: KernelFileMetadata,
}

struct KernelFile {
    label: &'static str,
    canonical_path: PathBuf,
    file: File,
}

impl KernelFile {
    fn open(path: &Path, label: &'static str) -> Result<Self> {
        let canonical_path = path.canonicalize().map_err(|error| {
            Error::Other(format!(
                "failed to resolve {label} kernel {}: {error}",
                path.display()
            ))
        })?;
        let file = File::open(&canonical_path)
            .map_err(|error| Error::Other(format!("failed to open {label} kernel: {error}")))?;
        file.metadata()
            .map_err(|error| Error::Other(format!("failed to inspect {label} kernel: {error}")))?;
        Ok(Self {
            label,
            canonical_path,
            file,
        })
    }

    fn loader_path(&self) -> PathBuf {
        #[cfg(target_os = "linux")]
        {
            PathBuf::from(format!("/proc/self/fd/{}", self.file.as_raw_fd()))
        }
        #[cfg(all(unix, not(target_os = "linux")))]
        {
            PathBuf::from(format!("/dev/fd/{}", self.file.as_raw_fd()))
        }
        #[cfg(not(unix))]
        {
            self.canonical_path.clone()
        }
    }

    fn metadata(&self) -> Result<KernelFileMetadata> {
        let handle = StableFileIdentity::from_metadata(&self.file.metadata().map_err(|error| {
            Error::Other(format!("failed to inspect {} kernel: {error}", self.label))
        })?);
        let path = StableFileIdentity::from_metadata(
            &std::fs::metadata(&self.canonical_path).map_err(|error| {
                Error::Other(format!(
                    "failed to re-inspect {} kernel {}: {error}",
                    self.label,
                    self.canonical_path.display()
                ))
            })?,
        );
        Ok(KernelFileMetadata { handle, path })
    }

    fn read_record(
        file: &File,
        record_number: u32,
        label: &str,
    ) -> Result<[u8; DAF_RECORD_LENGTH]> {
        if record_number == 0 {
            return Err(Error::Other(format!(
                "invalid zero DAF record number in {label} kernel"
            )));
        }
        let mut file = file.try_clone().map_err(|error| {
            Error::Other(format!("failed to clone {label} kernel handle: {error}"))
        })?;
        let offset = u64::from(record_number - 1) * DAF_RECORD_LENGTH as u64;
        file.seek(SeekFrom::Start(offset)).map_err(|error| {
            Error::Other(format!(
                "failed to seek to DAF record {record_number} in {label} kernel: {error}"
            ))
        })?;
        let mut record = [0_u8; DAF_RECORD_LENGTH];
        file.read_exact(&mut record).map_err(|error| {
            Error::Other(format!(
                "failed to read complete DAF record {record_number} from {label} kernel: {error}"
            ))
        })?;
        Ok(record)
    }

    fn summary_record_number(value: f64, file_records: u64, label: &str) -> Result<u32> {
        if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > u32::MAX as f64 {
            return Err(Error::Other(format!(
                "invalid DAF summary-record pointer {value:?} in {label} kernel"
            )));
        }
        let record = value as u32;
        if record != 0 && u64::from(record) > file_records {
            return Err(Error::Other(format!(
                "DAF summary-record pointer {record} exceeds {label} kernel length"
            )));
        }
        Ok(record)
    }

    /// Digest exactly the small regions upstream ASSIST parses eagerly: file
    /// record, comment records and the complete linked summary-record chain.
    fn loaded_identity(&self) -> Result<LoadedKernelFileIdentity> {
        let metadata = self.file.metadata().map_err(|error| {
            Error::Other(format!("failed to inspect {} kernel: {error}", self.label))
        })?;
        let size = metadata.len();
        let file_records = size / DAF_RECORD_LENGTH as u64;
        if size < DAF_RECORD_LENGTH as u64 || size % DAF_RECORD_LENGTH as u64 != 0 {
            return Err(Error::Other(format!(
                "{} kernel length {size} is not a complete DAF record sequence",
                self.label
            )));
        }

        let file_record = Self::read_record(&self.file, 1, self.label)?;
        if !file_record.starts_with(b"DAF/SPK") {
            return Err(Error::Other(format!(
                "{} kernel does not have a DAF/SPK file record",
                self.label
            )));
        }
        if &file_record[88..96] != b"LTL-IEEE" {
            return Err(Error::Other(format!(
                "{} kernel uses an unsupported DAF binary format",
                self.label
            )));
        }
        let fward = u32::from_le_bytes(file_record[76..80].try_into().unwrap());
        if fward < 2 || u64::from(fward) > file_records {
            return Err(Error::Other(format!(
                "invalid first DAF summary record {fward} in {} kernel",
                self.label
            )));
        }

        let mut hasher = Sha256::new();
        hasher.update(b"adam-assist-eager-spk-regions-v1\0");
        let mut update = |kind: u8, record_number: u32, record: &[u8]| {
            hasher.update([kind]);
            hasher.update(record_number.to_le_bytes());
            hasher.update(record);
        };
        update(b'F', 1, &file_record);
        for record_number in 2..fward {
            let record = Self::read_record(&self.file, record_number, self.label)?;
            update(b'C', record_number, &record);
        }

        let mut visited = HashSet::new();
        let mut record_number = fward;
        while record_number != 0 {
            if !visited.insert(record_number) {
                return Err(Error::Other(format!(
                    "cyclic DAF summary-record chain at record {record_number} in {} kernel",
                    self.label
                )));
            }
            let record = Self::read_record(&self.file, record_number, self.label)?;
            update(b'S', record_number, &record);
            let next = f64::from_le_bytes(record[0..8].try_into().unwrap());
            record_number = Self::summary_record_number(next, file_records, self.label)?;
        }

        Ok(LoadedKernelFileIdentity::from_metadata_and_digest(
            &metadata,
            format!("{:x}", hasher.finalize()),
        ))
    }

    fn binding_error(
        &self,
        trust_established: bool,
        mismatch: BindingMismatch,
        detail: &str,
    ) -> Error {
        let kind = match mismatch {
            BindingMismatch::EagerContent => "eager content",
            BindingMismatch::MappingIdentity => "mapping identity",
        };
        if trust_established {
            let recovery = match mismatch {
                BindingMismatch::EagerContent => "restore the trusted bytes",
                BindingMismatch::MappingIdentity => {
                    "restore the original inode or reconstruct the provider"
                }
            };
            Error::Other(format!(
                "ASSIST {} kernel {kind} changed after provenance was established: {detail}; {recovery}",
                self.label
            ))
        } else {
            Error::Other(format!(
                "{} kernel {kind} no longer matches the bytes ASSIST loaded before provenance was established: {detail}",
                self.label
            ))
        }
    }

    fn validate_loaded_binding(
        &self,
        expected_metadata: &KernelFileMetadata,
        observed: &LoadedKernelFileIdentity,
        loaded: &LoadedKernelFileIdentity,
        trust_established: bool,
    ) -> Result<()> {
        if !expected_metadata.handle.same_file(&expected_metadata.path) {
            return Err(self.binding_error(
                trust_established,
                BindingMismatch::MappingIdentity,
                "the canonical path was replaced with a different inode than the retained ASSIST mapping",
            ));
        }
        let handle_metadata = self.file.metadata().map_err(|error| {
            Error::Other(format!("failed to inspect {} kernel: {error}", self.label))
        })?;
        if !loaded.matches_metadata(&handle_metadata) {
            return Err(self.binding_error(
                trust_established,
                BindingMismatch::MappingIdentity,
                "the retained mapping length, device, or inode changed",
            ));
        }
        if observed != loaded {
            return Err(self.binding_error(
                trust_established,
                BindingMismatch::EagerContent,
                "the eagerly parsed DAF records changed",
            ));
        }
        Ok(())
    }

    fn hash_open_file(&self, file: &File, source: &str) -> Result<FileIdentity> {
        let mut reader = BufReader::new(file.try_clone().map_err(|error| {
            Error::Other(format!(
                "failed to clone {} kernel {source}: {error}",
                self.label
            ))
        })?);
        reader.seek(SeekFrom::Start(0)).map_err(|error| {
            Error::Other(format!(
                "failed to rewind {} kernel {source}: {error}",
                self.label
            ))
        })?;
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 1024 * 1024];
        let mut size = 0_u64;
        loop {
            let count = reader.read(&mut buffer).map_err(|error| {
                Error::Other(format!(
                    "failed to read {} kernel {source}: {error}",
                    self.label
                ))
            })?;
            if count == 0 {
                break;
            }
            size += count as u64;
            hasher.update(&buffer[..count]);
        }
        Ok(FileIdentity {
            size,
            sha256: format!("{:x}", hasher.finalize()),
        })
    }

    fn content_identity(
        &self,
        expected_metadata: &KernelFileMetadata,
        loaded_identity: Option<&LoadedKernelFileIdentity>,
        trust_established: bool,
    ) -> Result<Option<FileIdentity>> {
        // Metadata movement is not a content failure. Tell the bounded outer
        // loop to take another complete snapshot instead of poisoning the
        // provider with a transient first-hash race.
        if self.metadata()? != *expected_metadata {
            return Ok(None);
        }

        // Path-backed providers bind their first full digest to the file
        // record, comments, summary chain, mapped length and retained inode
        // observed immediately around ASSIST construction. Read those eager
        // regions both before and after the full stream so the baseline and
        // digest belong to one stable observation attempt.
        let eager_before = match loaded_identity {
            Some(loaded) => {
                let observed = self.loaded_identity()?;
                self.validate_loaded_binding(
                    expected_metadata,
                    &observed,
                    loaded,
                    trust_established,
                )?;
                Some(observed)
            }
            None => None,
        };
        let handle_identity = self.hash_open_file(&self.file, "retained handle")?;
        if let (Some(loaded), Some(before)) = (loaded_identity, eager_before.as_ref()) {
            let after = self.loaded_identity()?;
            if &after != before {
                return Ok(None);
            }
            self.validate_loaded_binding(expected_metadata, &after, loaded, trust_established)?;
        }

        let path_file = File::open(&self.canonical_path).map_err(|error| {
            Error::Other(format!(
                "failed to reopen {} kernel {}: {error}",
                self.label,
                self.canonical_path.display()
            ))
        })?;
        let opened_path_metadata =
            StableFileIdentity::from_metadata(&path_file.metadata().map_err(|error| {
                Error::Other(format!(
                    "failed to inspect reopened {} kernel: {error}",
                    self.label
                ))
            })?);
        if opened_path_metadata != expected_metadata.path {
            return Ok(None);
        }
        let path_identity = if expected_metadata.handle.same_file(&expected_metadata.path) {
            handle_identity.clone()
        } else {
            self.hash_open_file(&path_file, "canonical path")?
        };
        if self.metadata()? != *expected_metadata {
            return Ok(None);
        }
        if handle_identity != path_identity {
            return Err(self.binding_error(
                trust_established,
                BindingMismatch::MappingIdentity,
                "the canonical path no longer identifies the retained ASSIST mapping",
            ));
        }
        Ok(Some(handle_identity))
    }
}

fn kernel_pair_fingerprint(planets: &FileIdentity, asteroids: &FileIdentity) -> String {
    let canonical = format!(
        "adam-assist-kernel-data-v1\nplanets_size={}\nplanets_sha256={}\nasteroids_size={}\nasteroids_sha256={}",
        planets.size, planets.sha256, asteroids.size, asteroids.sha256
    );
    format!("{:x}", Sha256::digest(canonical.as_bytes()))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct KernelPairIdentity {
    planets: FileIdentity,
    asteroids: FileIdentity,
    fingerprint: String,
}

struct KernelFiles {
    planets: KernelFile,
    asteroids: KernelFile,
    loaded_identity: Option<LoadedKernelPairIdentity>,
    identity: OnceLock<KernelPairIdentity>,
    trusted_metadata: Mutex<Option<KernelPairMetadata>>,
    #[cfg(test)]
    compute_count: AtomicUsize,
}

impl KernelFiles {
    fn open(planets_path: &Path, asteroids_path: &Path) -> Result<Self> {
        let planets = KernelFile::open(planets_path, "planetary")?;
        let asteroids = KernelFile::open(asteroids_path, "small-body")?;
        Ok(Self {
            planets,
            asteroids,
            loaded_identity: None,
            identity: OnceLock::new(),
            // Mutable metadata is intentionally not trusted at construction.
            // Path-backed AssistData attaches a small immutable baseline for
            // the records ASSIST parsed eagerly and its fixed mmap identity.
            trusted_metadata: Mutex::new(None),
            #[cfg(test)]
            compute_count: AtomicUsize::new(0),
        })
    }

    fn loader_paths(&self) -> (PathBuf, PathBuf) {
        (self.planets.loader_path(), self.asteroids.loader_path())
    }

    fn metadata(&self) -> Result<KernelPairMetadata> {
        Ok(KernelPairMetadata {
            planets: self.planets.metadata()?,
            asteroids: self.asteroids.metadata()?,
        })
    }

    fn observe_loaded_identity(&self) -> Result<LoadedKernelPairIdentity> {
        let before = self.metadata()?;
        if !before.planets.handle.same_file(&before.planets.path)
            || !before.asteroids.handle.same_file(&before.asteroids.path)
        {
            return Err(Error::Other(
                "ASSIST kernel path changed before the ephemeris load completed".to_string(),
            ));
        }
        let observed = LoadedKernelPairIdentity {
            planets: self.planets.loaded_identity()?,
            asteroids: self.asteroids.loaded_identity()?,
        };
        let after = self.metadata()?;
        if !after.planets.handle.same_file(&after.planets.path)
            || !after.asteroids.handle.same_file(&after.asteroids.path)
            || !observed
                .planets
                .matches_stable_identity(&before.planets.handle)
            || !observed
                .planets
                .matches_stable_identity(&after.planets.handle)
            || !observed
                .asteroids
                .matches_stable_identity(&before.asteroids.handle)
            || !observed
                .asteroids
                .matches_stable_identity(&after.asteroids.handle)
        {
            return Err(Error::Other(
                "ASSIST kernel load identity changed during eager-region observation".to_string(),
            ));
        }
        Ok(observed)
    }

    fn bind_loaded_identity(&mut self, identity: LoadedKernelPairIdentity) {
        self.loaded_identity = Some(identity);
    }

    fn verified_identity(
        &self,
        trusted_metadata: Option<&KernelPairMetadata>,
        trusted_identity: Option<&KernelPairIdentity>,
    ) -> Result<(KernelPairIdentity, KernelPairMetadata)> {
        const MAX_STABLE_METADATA_ATTEMPTS: usize = 3;
        for _ in 0..MAX_STABLE_METADATA_ATTEMPTS {
            let before = self.metadata()?;
            #[cfg(test)]
            self.compute_count.fetch_add(1, AtomicOrdering::SeqCst);
            let planets = match (trusted_identity, trusted_metadata) {
                (Some(identity), Some(metadata)) if before.planets == metadata.planets => {
                    identity.planets.clone()
                }
                _ => match self.planets.content_identity(
                    &before.planets,
                    self.loaded_identity
                        .as_ref()
                        .map(|identity| &identity.planets),
                    trusted_identity.is_some(),
                )? {
                    Some(identity) => identity,
                    None => continue,
                },
            };
            let asteroids = match (trusted_identity, trusted_metadata) {
                (Some(identity), Some(metadata)) if before.asteroids == metadata.asteroids => {
                    identity.asteroids.clone()
                }
                _ => match self.asteroids.content_identity(
                    &before.asteroids,
                    self.loaded_identity
                        .as_ref()
                        .map(|identity| &identity.asteroids),
                    trusted_identity.is_some(),
                )? {
                    Some(identity) => identity,
                    None => continue,
                },
            };
            let after = self.metadata()?;
            if after != before {
                continue;
            }
            let fingerprint = kernel_pair_fingerprint(&planets, &asteroids);
            let observed = KernelPairIdentity {
                planets,
                asteroids,
                fingerprint,
            };
            if let Some(identity) = trusted_identity {
                if observed != *identity {
                    return Err(Error::Other(
                        "ASSIST kernel bytes changed after provenance was established: complete content digest mismatch"
                            .to_string(),
                    ));
                }
            }
            return Ok((observed, after));
        }
        Err(Error::Other(
            "ASSIST kernel metadata did not remain stable during content verification".to_string(),
        ))
    }

    fn identity(&self) -> Result<&KernelPairIdentity> {
        let mut trusted_metadata = self
            .trusted_metadata
            .lock()
            .map_err(|_| Error::Other("ASSIST kernel provenance lock was poisoned".to_string()))?;
        if let Some(identity) = self.identity.get() {
            let current_metadata = self.metadata()?;
            if trusted_metadata.as_ref() != Some(&current_metadata) {
                let (_, refreshed_metadata) =
                    self.verified_identity(trusted_metadata.as_ref(), Some(identity))?;
                *trusted_metadata = Some(refreshed_metadata);
            }
            return Ok(identity);
        }

        // Trust begins only after a complete stable content observation. A
        // failed or racing first hash leaves both cells uninitialized, so a
        // later call can retry rather than inheriting a permanently cached
        // transient error.
        let (computed, refreshed_metadata) = self.verified_identity(None, None)?;
        self.identity
            .set(computed)
            .map_err(|_| Error::Other("ASSIST kernel identity initialization raced".to_string()))?;
        *trusted_metadata = Some(refreshed_metadata);
        Ok(self
            .identity
            .get()
            .expect("successful identity initialization must populate OnceLock"))
    }

    fn verify_identity(&self) -> Result<()> {
        self.identity().map(|_| ())
    }

    #[cfg(test)]
    fn compute_count(&self) -> usize {
        self.compute_count.load(AtomicOrdering::SeqCst)
    }
}

#[cfg(feature = "kernel-data")]
fn verify_kernel_identity(
    path: &Path,
    label: &str,
    identity: &FileIdentity,
    expected_size: u64,
    expected_sha256: &str,
) -> Result<()> {
    if identity.size != expected_size {
        return Err(Error::Other(format!(
            "{label} kernel size mismatch for {}: expected {expected_size}, got {}",
            path.display(),
            identity.size
        )));
    }
    if identity.sha256 != expected_sha256 {
        return Err(Error::Other(format!(
            "{label} kernel SHA-256 mismatch for {}: expected {expected_sha256}, got {}",
            path.display(),
            identity.sha256
        )));
    }
    Ok(())
}

/// The data resource every propagation entry point needs: the JPL SPK
/// [`Ephemeris`], loaded once (from `de440.bsp` + `sb441-n16.bsp`) and shared
/// for the lifetime of the process (`Ephemeris` is `Send + Sync`).
pub struct AssistData {
    /// JPL SPK ephemeris.
    pub ephem: Ephemeris,
    kernel_files: Option<KernelFiles>,
}

impl AssistData {
    /// Wrap an already loaded ASSIST ephemeris.
    ///
    /// The original kernel bytes are unavailable on this route, so this value
    /// cannot be exported as a provenance-complete external OD provider. Use
    /// [`Self::from_paths`] when a capsule data fingerprint is required.
    /// The exported raw [`Ephemeris::from_paths`] constructor and every
    /// path-owned provider constructor share the same process-global parser
    /// lock, so wrapping a raw ephemeris here cannot race provider creation.
    pub fn new(ephem: Ephemeris) -> Self {
        Self {
            ephem,
            kernel_files: None,
        }
    }

    /// Load the planetary and small-body ASSIST ephemeris kernels explicitly.
    ///
    /// Kernel files are immutable package-managed resources and must remain
    /// unchanged for the provider's lifetime. Construction opens stable handles
    /// and binds the DAF file record, comments, complete summary chain, mapped
    /// length, and retained file identity around upstream parsing. Ordinary
    /// detectable replacement, truncation, or corruption returns an error.
    /// Concurrent mutation is unsupported: these consistency checks are not an
    /// adversarial filesystem security boundary. Full-content trust is
    /// established from a stable snapshot at first actual use. Chebyshev data
    /// bytes unread during construction remain live through the mapping and are
    /// intentionally adopted by that first complete fingerprint. Construct a
    /// new provider after any legitimate kernel update.
    pub fn from_paths(
        planets_path: impl AsRef<Path>,
        asteroids_path: impl AsRef<Path>,
    ) -> Result<Self> {
        Self::from_paths_with_load_hook(planets_path.as_ref(), asteroids_path.as_ref(), || Ok(()))
    }

    fn from_paths_with_load_hook(
        planets_path: &Path,
        asteroids_path: &Path,
        after_initial_observation: impl FnOnce() -> Result<()>,
    ) -> Result<Self> {
        let mut kernel_files = KernelFiles::open(planets_path, asteroids_path)?;
        let (planets_loader, asteroids_loader) = kernel_files.loader_paths();

        // Keep the eager-region observations and upstream parsing in one
        // serialized construction window. This baseline is small (file record,
        // comments and summary chain), not the lazy 765 MiB full-content hash.
        let _construction = crate::ephemeris::construction_guard();
        let before = kernel_files.observe_loaded_identity()?;
        after_initial_observation()?;
        let ephem =
            Ephemeris::from_paths_while_construction_locked(&planets_loader, &asteroids_loader)?;
        let after = kernel_files.observe_loaded_identity()?;
        if before != after {
            return Err(Error::Other(
                "ASSIST kernel eagerly parsed content changed during ephemeris construction; retry construction"
                    .to_string(),
            ));
        }
        kernel_files.bind_loaded_identity(after);
        Ok(Self {
            ephem,
            kernel_files: Some(kernel_files),
        })
    }

    /// Return the lazily computed immutable identity for both loaded kernels.
    ///
    /// The first actual kernel use verifies the immutable load-time binding,
    /// then hashes a stable snapshot of the current retained handles and
    /// matching paths. Benign hard-link, chmod, or touch changes after successful
    /// construction are accepted without trusting stale mutable metadata. A
    /// first-hash race leaves the provider uninitialized for retry. Once trust
    /// exists, equal bytes refresh metadata only on the same retained mapping.
    /// Replacing the path with another inode requires restoring the original
    /// inode or reconstructing the provider. Different content fails closed
    /// without replacing trust, and exact restoration recovers.
    pub fn data_fingerprint(&self) -> Result<&str> {
        self.kernel_files
            .as_ref()
            .ok_or_else(|| {
                Error::Other(
                    "ASSIST provider has no kernel content fingerprint; construct it from paths"
                        .to_string(),
                )
            })?
            .identity()
            .map(|identity| identity.fingerprint.as_str())
    }

    pub(crate) fn verify_claimed_kernel_identity(&self) -> Result<()> {
        match &self.kernel_files {
            Some(files) => files.verify_identity(),
            None => Ok(()),
        }
    }

    /// Resolve DE440 plus SB441-n16 using adam-core's pure-Rust override ->
    /// installed-Python -> cache -> verified-wheel policy, then verify the
    /// pinned file size and SHA-256 before returning the ephemeris.
    #[cfg(feature = "kernel-data")]
    pub fn from_default_kernels() -> Result<Self> {
        let resolver = adam_core_rs_kernel_data::Resolver::from_env();
        let (planets, asteroids) = resolver
            .assist_ephemeris_paths()
            .map_err(|error| Error::Other(format!("failed to resolve ASSIST kernels: {error}")))?;
        let data = Self::from_paths(&planets, &asteroids)?;
        let identity = data
            .kernel_files
            .as_ref()
            .expect("from_paths records kernel handles")
            .identity()?;
        verify_kernel_identity(
            &planets,
            "DE440",
            &identity.planets,
            DE440_SIZE_BYTES,
            DE440_SHA256,
        )?;
        verify_kernel_identity(
            &asteroids,
            "SB441-n16",
            &identity.asteroids,
            SB441_N16_SIZE_BYTES,
            SB441_N16_SHA256,
        )?;
        Ok(data)
    }
}

// ─── Orbit container ────────────────────────────────────────────────────────

/// Marsden-Sekanina non-gravitational force parameters.
///
/// The non-gravitational acceleration uses the RTN (radial/transverse/normal)
/// decomposition with a g(r) model:
///
/// ```text
/// a_ng = g(r) * (A1·r̂ + A2·t̂ + A3·n̂)
/// g(r) = α * (r/r₀)^(-m) * (1 + (r/r₀)^n)^(-k)
/// ```
///
/// Default model parameters (α=1, m=2, n=5.093, k=0, r₀=1 AU) give
/// g(r) = r⁻² — a pure inverse-square law.
#[derive(Debug, Clone)]
pub struct NonGravParams {
    /// Radial non-grav coefficient A1 (AU/day²).
    pub a1: f64,
    /// Transverse non-grav coefficient A2 (AU/day²).
    pub a2: f64,
    /// Normal non-grav coefficient A3 (AU/day²).
    pub a3: f64,
    /// g(r) normalization. Default: 1.0.
    pub alpha: Option<f64>,
    /// g(r) exponent k. Default: 0.0.
    pub nk: Option<f64>,
    /// g(r) exponent m. Default: 2.0.
    pub nm: Option<f64>,
    /// g(r) exponent n. Default: 5.093.
    pub nn: Option<f64>,
    /// g(r) scale distance in AU. Default: 1.0.
    pub r0: Option<f64>,
}

impl NonGravParams {
    /// Create non-grav params with the default g(r) = r⁻² model.
    pub fn new(a1: f64, a2: f64, a3: f64) -> Self {
        Self {
            a1,
            a2,
            a3,
            alpha: None,
            nk: None,
            nm: None,
            nn: None,
            r0: None,
        }
    }
}

/// A test particle orbit: state vector + epoch + optional non-gravitational parameters.
///
/// All coordinates are **heliocentric ecliptic J2000** (AU, AU/day) with **MJD TDB** epochs.
#[derive(Debug, Clone)]
pub struct Orbit {
    /// Heliocentric ecliptic J2000 state [x, y, z, vx, vy, vz] (AU, AU/day).
    pub state: [f64; 6],
    /// Epoch (MJD TDB).
    pub epoch: f64,
    /// Optional non-gravitational force parameters.
    pub non_grav: Option<NonGravParams>,
}

impl Orbit {
    /// Create an orbit with no non-gravitational forces.
    pub fn new(state: [f64; 6], epoch: f64) -> Self {
        Self {
            state,
            epoch,
            non_grav: None,
        }
    }

    /// Create an orbit with non-gravitational force parameters.
    pub fn with_non_grav(state: [f64; 6], epoch: f64, non_grav: NonGravParams) -> Self {
        Self {
            state,
            epoch,
            non_grav: Some(non_grav),
        }
    }
}

// ─── Frame rotations ────────────────────────────────────────────────────────
//
// ASSIST works in **barycentric equatorial ICRF** (AU, AU/day); this
// orchestration layer's contract is **heliocentric ecliptic J2000**.

/// J2000 mean obliquity of the ecliptic (radians).
/// ε = 23°26′21.448″ = 84381.448″ (DE430/DE431) — the same constant adam-core
/// uses (`Constants.OBLIQUITY = 84381.448 * π / (180 * 3600)`), evaluated in
/// f64 with that exact expression so the value is bit-identical to adam-core.
pub const OBLIQUITY_J2000: f64 = 0.409_092_804_222_328_9;

/// cos(ε) — f64 cosine of `OBLIQUITY_J2000`, bit-identical to adam-core's
/// rotation matrices (`np.cos(Constants.OBLIQUITY)`).
const COS_EPS: f64 = 0.917_482_062_069_181_8;
/// sin(ε) — f64 sine of `OBLIQUITY_J2000`, bit-identical to adam-core's
/// rotation matrices (`np.sin(Constants.OBLIQUITY)`).
const SIN_EPS: f64 = 0.397_777_155_931_913_65;

/// Rotate a 6-element state vector `[x, y, z, vx, vy, vz]` from equatorial to ecliptic.
pub fn equatorial_to_ecliptic(state: &[f64; 6]) -> [f64; 6] {
    let [x, y, z, vx, vy, vz] = *state;
    [
        x,
        COS_EPS * y + SIN_EPS * z,
        -SIN_EPS * y + COS_EPS * z,
        vx,
        COS_EPS * vy + SIN_EPS * vz,
        -SIN_EPS * vy + COS_EPS * vz,
    ]
}

/// Rotate a 6-element state vector from ecliptic to equatorial.
pub fn ecliptic_to_equatorial(state: &[f64; 6]) -> [f64; 6] {
    let [x, y, z, vx, vy, vz] = *state;
    [
        x,
        COS_EPS * y - SIN_EPS * z,
        SIN_EPS * y + COS_EPS * z,
        vx,
        COS_EPS * vy - SIN_EPS * vz,
        SIN_EPS * vy + COS_EPS * vz,
    ]
}

/// Rotate a 6×6 matrix from equatorial to ecliptic: R × M × R^T.
pub fn rotate_matrix_eq_to_ecl(m: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let r = rotation_matrix_eq_to_ecl();
    mat6x6_multiply(&mat6x6_multiply(&r, m), &transpose6x6(&r))
}

fn rotation_matrix_eq_to_ecl() -> [[f64; 6]; 6] {
    let mut r = [[0.0f64; 6]; 6];
    r[0][0] = 1.0;
    r[1][1] = COS_EPS;
    r[1][2] = SIN_EPS;
    r[2][1] = -SIN_EPS;
    r[2][2] = COS_EPS;
    r[3][3] = 1.0;
    r[4][4] = COS_EPS;
    r[4][5] = SIN_EPS;
    r[5][4] = -SIN_EPS;
    r[5][5] = COS_EPS;
    r
}

fn transpose6x6(m: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut t = [[0.0f64; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            t[i][j] = m[j][i];
        }
    }
    t
}

fn mat6x6_multiply(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut c = [[0.0f64; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..6 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

// ─── Propagation results ────────────────────────────────────────────────────

/// Result of propagating to a single epoch.
#[derive(Debug, Clone)]
pub struct PropagatedState {
    /// Heliocentric ecliptic J2000 state [x, y, z, vx, vy, vz] (AU, AU/day).
    pub state: [f64; 6],
    /// Epoch (MJD TDB).
    pub epoch: f64,
    /// 6×6 state transition matrix Φ(t, t₀) = ∂x(t)/∂x₀, row-major, in
    /// heliocentric ecliptic J2000. Populated when `compute_stm` is true.
    pub stm: Option<[[f64; 6]; 6]>,
    /// 6×3 partials ∂x(t)/∂(A1, A2, A3), row-major (6 state rows, 3 parameter
    /// columns), in heliocentric ecliptic J2000. Populated when `compute_stm`
    /// is true *and* the orbit carries non-gravitational parameters.
    /// Columns are ordered A1, A2, A3.
    pub nongrav_partials: Option<[[f64; 3]; 6]>,
}

impl PropagatedState {
    /// Linearly propagate a 6×6 initial-state covariance to this epoch:
    ///
    /// ```text
    ///   P(t) = Φ · P₀ · Φᵀ
    /// ```
    ///
    /// where `Φ` is `self.stm`. Returns `None` when `stm` is not populated
    /// (i.e., the orbit was propagated with `compute_stm = false`).
    pub fn propagate_covariance(&self, p0: &[[f64; 6]; 6]) -> Option<[[f64; 6]; 6]> {
        self.stm.as_ref().map(|stm| covariance_6x6(stm, p0))
    }

    /// Linearly propagate a 9×9 initial covariance over
    /// `(x, y, z, vx, vy, vz, A1, A2, A3)` to the 6×6 state covariance at this
    /// epoch:
    ///
    /// ```text
    ///   J = [ Φ | G ]         (6×9, Φ = stm, G = nongrav_partials)
    ///   P(t) = J · P₀ · Jᵀ
    /// ```
    ///
    /// Returns `None` unless both `stm` and `nongrav_partials` are populated.
    pub fn propagate_covariance_with_nongrav(&self, p0: &[[f64; 9]; 9]) -> Option<[[f64; 6]; 6]> {
        let stm = self.stm.as_ref()?;
        let ng = self.nongrav_partials.as_ref()?;
        Some(covariance_9x9(stm, ng, p0))
    }
}

/// P = Φ · P₀ · Φᵀ, where Φ and P₀ are 6×6 (row-major).
fn covariance_6x6(stm: &[[f64; 6]; 6], p0: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut tmp = [[0.0f64; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            let mut s = 0.0;
            for k in 0..6 {
                s += stm[i][k] * p0[k][j];
            }
            tmp[i][j] = s;
        }
    }
    let mut out = [[0.0f64; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            let mut s = 0.0;
            for k in 0..6 {
                // stm[j][k] is (Φᵀ)[k][j]
                s += tmp[i][k] * stm[j][k];
            }
            out[i][j] = s;
        }
    }
    out
}

/// P = J · P₀ · Jᵀ, where J = [stm | ng] is 6×9 (stm 6×6, ng 6×3) and P₀ is 9×9.
fn covariance_9x9(stm: &[[f64; 6]; 6], ng: &[[f64; 3]; 6], p0: &[[f64; 9]; 9]) -> [[f64; 6]; 6] {
    let j = |i: usize, k: usize| -> f64 {
        if k < 6 {
            stm[i][k]
        } else {
            ng[i][k - 6]
        }
    };
    // tmp = J · P₀  (6×9)
    let mut tmp = [[0.0f64; 9]; 6];
    for i in 0..6 {
        for col in 0..9 {
            let mut s = 0.0;
            for k in 0..9 {
                s += j(i, k) * p0[k][col];
            }
            tmp[i][col] = s;
        }
    }
    // out = tmp · Jᵀ  (6×6)
    let mut out = [[0.0f64; 6]; 6];
    for i in 0..6 {
        for col in 0..6 {
            let mut s = 0.0;
            for k in 0..9 {
                s += tmp[i][k] * j(col, k);
            }
            out[i][col] = s;
        }
    }
    out
}

// ─── Free-function propagation ──────────────────────────────────────────────

/// Propagate a test particle from an initial heliocentric ecliptic J2000 orbit.
///
/// One-shot convenience: creates a fresh simulation, integrates, and drops on
/// return. For repeated propagation of many orbits that share the same force
/// model and variational-equation setup, prefer [`PropagatorPool`], which
/// amortizes the ~25 µs REBOUND/ASSIST setup across calls.
pub fn assist_propagate_single(
    data: &AssistData,
    orbit: &Orbit,
    target_epochs: &[f64],
    compute_stm: bool,
    integrator: &IntegratorConfig,
) -> Result<Vec<PropagatedState>> {
    if target_epochs.is_empty() {
        return Ok(vec![]);
    }
    data.verify_claimed_kernel_identity()?;

    let ephem = &data.ephem;
    let jd_ref = ephem.jd_ref();
    let t0 = mjd_to_assist_time(orbit.epoch, jd_ref);

    let has_nongrav = orbit.non_grav.is_some();
    let want_nongrav_partials = compute_stm && has_nongrav;

    let mut sim = Simulation::new()?;
    sim.set_t(t0);
    integrator.apply(&mut sim);
    let mut asim = AssistSim::new(sim, ephem)?;
    configure_forces(&mut asim, has_nongrav);

    let bary_state = ecl_orbit_to_bary_eq(&orbit.state, ephem, t0)?;
    let n_var = variational_count(compute_stm, has_nongrav);
    add_particles_and_variationals(&mut asim, &bary_state, n_var);

    if let Some(ng) = orbit.non_grav.as_ref() {
        apply_nongrav_scalars(&mut asim, ng);
        install_particle_params(&mut asim, ng, want_nongrav_partials);
    }

    run_integration(
        &mut asim,
        ephem,
        target_epochs,
        compute_stm,
        want_nongrav_partials,
    )
}

/// Propagate many orbits to a shared set of target epochs.
///
/// Shape: the returned `Vec<Vec<PropagatedState>>` is indexed
/// `[orbit_index][target_epoch_index]`.
///
/// `num_threads`: `None` = rayon's global pool, `Some(1)` = serial,
/// `Some(n)` = dedicated pool with `n` workers; `Some(0)` is an error.
pub fn assist_propagate(
    data: &AssistData,
    orbits: &[Orbit],
    target_epochs: &[f64],
    compute_stm: bool,
    num_threads: Option<usize>,
    integrator: &IntegratorConfig,
) -> Result<Vec<Vec<PropagatedState>>> {
    let op = |orbit: &Orbit| {
        assist_propagate_single(data, orbit, target_epochs, compute_stm, integrator)
    };
    map_with_threads(orbits, num_threads, op)
}

/// Dispatch a fallible per-item function across `items`:
/// `None` = rayon's global pool, `Some(1)` = serial, `Some(n)` = custom
/// pool with `n` workers. `Some(0)` is an error.
fn map_with_threads<T, U, F>(items: &[T], num_threads: Option<usize>, f: F) -> Result<Vec<U>>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> Result<U> + Send + Sync,
{
    use rayon::prelude::*;
    match num_threads {
        Some(0) => Err(Error::Other(
            "num_threads = Some(0) is not valid; use None for the default pool".into(),
        )),
        None => items.par_iter().map(&f).collect(),
        Some(1) => items.iter().map(&f).collect(),
        Some(n) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| Error::Other(format!("rayon pool build failed: {e}")))?;
            pool.install(|| items.par_iter().map(&f).collect())
        }
    }
}

/// N-body propagation of many same-epoch test particles sharing one Marsden
/// force model, integrated **together in one simulation**.
///
/// ASSIST stores the Marsden `g(r)` constants on the simulation, while A1/A2/A3
/// are per-particle. Callers must therefore partition by the canonical scalar
/// model before entering this function. Force-free rows may share the default
/// asteroid model because their per-particle coefficients are zero.
///
/// Unlike [`assist_propagate`], which builds one simulation per orbit, this
/// adds every orbit as a massless test particle to a single REBOUND/ASSIST
/// simulation and integrates the whole set jointly to each target epoch. That
/// amortizes simulation setup and per-step planetary ephemeris evaluation.
///
/// Because IAS15's adaptive step size is shared by all particles in a
/// simulation, results can differ from per-orbit [`assist_propagate_single`]
/// runs at the integrator-tolerance level; test particles are massless and do
/// not otherwise influence one another.
///
/// Shape: the returned `Vec<Vec<[f64; 6]>>` is indexed
/// `[orbit_index][target_epoch_index]`.
pub fn assist_propagate_orbits_same_epoch(
    data: &AssistData,
    orbits: &[Orbit],
    target_epochs: &[f64],
    integrator: &IntegratorConfig,
) -> Result<Vec<Vec<[f64; 6]>>> {
    if orbits.is_empty() {
        return Ok(Vec::new());
    }
    let epoch = orbits[0].epoch;
    if orbits.iter().any(|orbit| orbit.epoch != epoch) {
        return Err(Error::Other(
            "same-epoch ASSIST propagation received multiple input epochs".into(),
        ));
    }
    let expected_model = nongrav_model_key(orbits[0].non_grav.as_ref());
    if orbits
        .iter()
        .any(|orbit| nongrav_model_key(orbit.non_grav.as_ref()) != expected_model)
    {
        return Err(Error::Other(
            "same-epoch ASSIST propagation received multiple Marsden models".into(),
        ));
    }
    data.verify_claimed_kernel_identity()?;

    let ephem = &data.ephem;
    let jd_ref = ephem.jd_ref();
    let t0 = mjd_to_assist_time(epoch, jd_ref);
    let has_nongrav = orbits.iter().any(|orbit| orbit.non_grav.is_some());

    let mut sim = Simulation::new()?;
    sim.set_t(t0);
    integrator.apply(&mut sim);
    let mut asim = AssistSim::new(sim, ephem)?;
    configure_forces(&mut asim, has_nongrav);

    for orbit in orbits {
        let bary_eq = ecl_orbit_to_bary_eq(&orbit.state, ephem, t0)?;
        asim.sim_mut().add_test_particle(
            bary_eq[0], bary_eq[1], bary_eq[2], bary_eq[3], bary_eq[4], bary_eq[5],
        );
    }
    if let Some(model) = orbits.iter().find_map(|orbit| orbit.non_grav.as_ref()) {
        apply_nongrav_scalars(&mut asim, model);
        let mut particle_parameters = vec![0.0; 3 * orbits.len()];
        for (row, orbit) in orbits.iter().enumerate() {
            if let Some(parameters) = orbit.non_grav.as_ref() {
                particle_parameters[3 * row] = parameters.a1;
                particle_parameters[3 * row + 1] = parameters.a2;
                particle_parameters[3 * row + 2] = parameters.a3;
            }
        }
        asim.set_particle_params(particle_parameters);
    }

    let mut states_by_particle = vec![Vec::with_capacity(target_epochs.len()); orbits.len()];
    for &target_mjd in target_epochs {
        let t_target = mjd_to_assist_time(target_mjd, jd_ref);
        asim.integrate(t_target)?;
        let particles = asim.sim().particles();
        if particles.len() < orbits.len() {
            return Err(Error::Other("No particles after integration".into()));
        }
        let sun_t = ephem.get_body_state(ffi::ASSIST_BODY_SUN, t_target)?;
        for (row, p) in particles.iter().take(orbits.len()).enumerate() {
            let helio_eq = [
                p.x - sun_t.x,
                p.y - sun_t.y,
                p.z - sun_t.z,
                p.vx - sun_t.vx,
                p.vy - sun_t.vy,
                p.vz - sun_t.vz,
            ];
            states_by_particle[row].push(equatorial_to_ecliptic(&helio_eq));
        }
    }

    Ok(states_by_particle)
}

/// Gravity-only compatibility wrapper over [`assist_propagate_orbits_same_epoch`].
pub fn assist_propagate_states_same_epoch(
    data: &AssistData,
    states: &[[f64; 6]],
    epoch: f64,
    target_epochs: &[f64],
    integrator: &IntegratorConfig,
) -> Result<Vec<Vec<[f64; 6]>>> {
    let orbits = states
        .iter()
        .copied()
        .map(|state| Orbit::new(state, epoch))
        .collect::<Vec<_>>();
    assist_propagate_orbits_same_epoch(data, &orbits, target_epochs, integrator)
}

pub(crate) fn nongrav_model_key(parameters: Option<&NonGravParams>) -> [f64; 5] {
    let mut key = parameters.map_or([1.0, 0.0, 2.0, 5.093, 1.0], |parameters| {
        [
            parameters.alpha.unwrap_or(1.0),
            parameters.nk.unwrap_or(0.0),
            parameters.nm.unwrap_or(2.0),
            parameters.nn.unwrap_or(5.093),
            parameters.r0.unwrap_or(1.0),
        ]
    });
    if key[1] == 0.0 {
        key[3] = 5.093;
    }
    key
}

// ─── PropagatorPool: reusable simulation for many-orbit workloads ───────────

/// Configuration locked at [`PropagatorPool`] construction. A single pool can
/// only serve orbits whose variational dimension matches this config.
#[derive(Debug, Clone, Copy)]
pub struct PropagatorConfig {
    /// Compute the 6×6 state transition matrix via variational equations.
    pub compute_stm: bool,
    /// Whether orbits served by this pool carry non-gravitational parameters.
    pub has_nongrav: bool,
}

impl PropagatorConfig {
    /// Gravity-only orbits, no STM — the cheapest configuration.
    pub fn gravity_only() -> Self {
        Self {
            compute_stm: false,
            has_nongrav: false,
        }
    }

    /// Gravity-only orbits with STM (6×6).
    pub fn gravity_with_stm() -> Self {
        Self {
            compute_stm: true,
            has_nongrav: false,
        }
    }

    /// Non-grav orbits with STM (6×6) and parameter partials (6×3).
    pub fn nongrav_with_stm() -> Self {
        Self {
            compute_stm: true,
            has_nongrav: true,
        }
    }
}

/// Reusable propagator for many orbits sharing the same force model and
/// variational-equation configuration.
///
/// Construction attaches ASSIST to a REBOUND simulation, pre-adds the real
/// test particle + up to 9 variational particles, and (when `has_nongrav`)
/// installs a zeroed `particle_params` array of the right size. Each call
/// to [`propagate`](PropagatorPool::propagate) rewrites particle state in
/// place, resets the IAS15 integrator's scratch arrays, and re-runs the
/// integration loop — avoiding the ~25 µs per-orbit setup cost that
/// [`assist_propagate_single`] pays.
///
/// The pool is stateful and not thread-safe; `Ephemeris` is `Send + Sync`,
/// so build one pool per worker thread over a shared ephemeris.
pub struct PropagatorPool<'a> {
    asim: AssistSim,
    data: &'a AssistData,
    ephem: &'a Ephemeris,
    jd_ref: f64,
    config: PropagatorConfig,
    /// Cached variational count (0, 6, or 9) — derived from config.
    n_var: usize,
}

impl<'a> PropagatorPool<'a> {
    /// Construct a pool for a given force/variational configuration.
    pub fn new(
        data: &'a AssistData,
        config: PropagatorConfig,
        integrator: &IntegratorConfig,
    ) -> Result<Self> {
        data.verify_claimed_kernel_identity()?;
        let ephem = &data.ephem;
        let jd_ref = ephem.jd_ref();
        let mut sim = Simulation::new()?;
        sim.set_t(0.0);
        integrator.apply(&mut sim);
        let mut asim = AssistSim::new(sim, ephem)?;
        configure_forces(&mut asim, config.has_nongrav);

        let placeholder = [0.0f64; 6];
        let n_var = variational_count(config.compute_stm, config.has_nongrav);
        add_particles_and_variationals(&mut asim, &placeholder, n_var);

        if config.has_nongrav {
            // Install a zeroed params array of the correct total length; each
            // propagate() call updates the first three slots in place.
            let placeholder_ng = NonGravParams::new(0.0, 0.0, 0.0);
            install_particle_params(
                &mut asim,
                &placeholder_ng,
                config.compute_stm && config.has_nongrav,
            );
        }

        Ok(Self {
            asim,
            data,
            ephem,
            jd_ref,
            config,
            n_var,
        })
    }

    /// The configuration this pool was built with.
    pub fn config(&self) -> PropagatorConfig {
        self.config
    }

    /// Cumulative IAS15 step count across all propagations done by this pool.
    pub fn steps_done(&self) -> u64 {
        self.asim.sim().steps_done()
    }

    /// Propagate `orbit` to `target_epochs`, reusing the internal simulation.
    ///
    /// The orbit's non-grav flag must match the pool's `config.has_nongrav`.
    pub fn propagate(
        &mut self,
        orbit: &Orbit,
        target_epochs: &[f64],
    ) -> Result<Vec<PropagatedState>> {
        if target_epochs.is_empty() {
            return Ok(vec![]);
        }

        if orbit.non_grav.is_some() != self.config.has_nongrav {
            return Err(Error::Other(format!(
                "orbit non-grav flag ({}) does not match pool config \
                 has_nongrav ({}) — rebuild the pool with a matching config",
                orbit.non_grav.is_some(),
                self.config.has_nongrav,
            )));
        }
        self.data.verify_claimed_kernel_identity()?;

        let want_nongrav_partials = self.config.compute_stm && self.config.has_nongrav;
        let t0 = mjd_to_assist_time(orbit.epoch, self.jd_ref);

        // 1) Clear IAS15 scratch arrays so no compensated-sum or predictor
        //    state leaks from the previous orbit.
        self.asim.reset_integrator();

        // 2) Re-seed epoch and timestep. `dt = 0` would lock IAS15 into a
        //    zero-step infinite loop; use REBOUND's default of 0.001 days.
        self.asim.sim_mut().set_t(t0);
        self.asim.sim_mut().set_dt(0.001);

        // 3) Overwrite real particle + re-init variationals to their unit-
        //    perturbation ICs (STM identity basis, param variationals zero).
        let bary_state = ecl_orbit_to_bary_eq(&orbit.state, self.ephem, t0)?;
        overwrite_particles(&mut self.asim, &bary_state, self.n_var);

        // 4) Non-grav scalar params + in-place A1/A2/A3 update (no realloc).
        if let Some(ng) = orbit.non_grav.as_ref() {
            apply_nongrav_scalars(&mut self.asim, ng);
            self.asim.update_nongrav_coeffs(ng.a1, ng.a2, ng.a3)?;
        }

        run_integration(
            &mut self.asim,
            self.ephem,
            target_epochs,
            self.config.compute_stm,
            want_nongrav_partials,
        )
    }
}

// ─── Private helpers shared by assist_propagate_single and PropagatorPool ───

/// Heliocentric ecliptic J2000 → barycentric equatorial ICRF at ASSIST time `t`.
fn ecl_orbit_to_bary_eq(ecl_state: &[f64; 6], ephem: &Ephemeris, t: f64) -> Result<[f64; 6]> {
    let eq_state = ecliptic_to_equatorial(ecl_state);
    let sun = ephem.get_body_state(ffi::ASSIST_BODY_SUN, t)?;
    Ok([
        eq_state[0] + sun.x,
        eq_state[1] + sun.y,
        eq_state[2] + sun.z,
        eq_state[3] + sun.vx,
        eq_state[4] + sun.vy,
        eq_state[5] + sun.vz,
    ])
}

/// Set ASSIST force flags (default + optional non-grav).
fn configure_forces(asim: &mut AssistSim, has_nongrav: bool) {
    let mut forces = ffi::ASSIST_FORCES_DEFAULT;
    if has_nongrav {
        forces |= ffi::ASSIST_FORCE_NON_GRAVITATIONAL;
    }
    asim.set_forces(forces);
}

/// Variational particle count: 0, 6, or 9.
fn variational_count(compute_stm: bool, has_nongrav: bool) -> usize {
    match (compute_stm, has_nongrav) {
        (false, _) => 0,
        (true, false) => 6,
        (true, true) => 9,
    }
}

/// Apply the non-grav g(r) scalar parameters (α, nk, nm, nn, r0).
pub(crate) fn apply_nongrav_scalars(asim: &mut AssistSim, ng: &NonGravParams) {
    if let Some(v) = ng.alpha {
        asim.set_alpha(v);
    }
    if let Some(v) = ng.nk {
        asim.set_nk(v);
    }
    if let Some(v) = ng.nm {
        asim.set_nm(v);
    }
    if let Some(v) = ng.nn {
        asim.set_nn(v);
    }
    if let Some(v) = ng.r0 {
        asim.set_r0(v);
    }
}

/// Add the real test particle + `n_var` variational particles (0, 6, or 9)
/// to an empty simulation.
fn add_particles_and_variationals(asim: &mut AssistSim, bary_state: &[f64; 6], n_var: usize) {
    asim.sim_mut().add_test_particle(
        bary_state[0],
        bary_state[1],
        bary_state[2],
        bary_state[3],
        bary_state[4],
        bary_state[5],
    );
    for _ in 0..n_var {
        asim.sim_mut().add_variation_1st_order(0);
    }
    init_variational_state_perturbations(asim, n_var);
}

/// Overwrite particle[0] with `bary_state` and re-initialize variational
/// particles to their unit-perturbation / zero ICs. Used by
/// `PropagatorPool::propagate` between orbits.
fn overwrite_particles(asim: &mut AssistSim, bary_state: &[f64; 6], n_var: usize) {
    unsafe {
        let ptr = ffi::assist_rs_sim_get_particles(asim.raw_sim_ptr_mut());
        let p = &mut *ptr;
        p.x = bary_state[0];
        p.y = bary_state[1];
        p.z = bary_state[2];
        p.vx = bary_state[3];
        p.vy = bary_state[4];
        p.vz = bary_state[5];
        // ax/ay/az are set by the force evaluator on the first step.
    }
    init_variational_state_perturbations(asim, n_var);
}

/// Write the six identity-column state perturbations to particles[1..=6]
/// and zero the parameter variational particles (7..=9 if present).
fn init_variational_state_perturbations(asim: &mut AssistSim, n_var: usize) {
    debug_assert!(matches!(n_var, 0 | 6 | 9));
    if n_var == 0 {
        return;
    }
    unsafe {
        let ptr = ffi::assist_rs_sim_get_particles(asim.raw_sim_ptr_mut());
        for i in 0..n_var {
            *ptr.add(1 + i) = ffi::reb_particle::default();
        }
        // Six state-sensitivity columns: ∂/∂x₀, ∂/∂y₀, ... ∂/∂vz₀.
        for d in 0..6 {
            let p = &mut *ptr.add(1 + d);
            match d {
                0 => p.x = 1.0,
                1 => p.y = 1.0,
                2 => p.z = 1.0,
                3 => p.vx = 1.0,
                4 => p.vy = 1.0,
                5 => p.vz = 1.0,
                _ => unreachable!(),
            }
        }
    }
}

/// Build and install the `particle_params` array of length `3 * n_total`.
pub(crate) fn install_particle_params(
    asim: &mut AssistSim,
    ng: &NonGravParams,
    want_nongrav_partials: bool,
) {
    let n_total = asim.sim().n_particles();
    let mut params = vec![0.0f64; 3 * n_total];
    params[0] = ng.a1;
    params[1] = ng.a2;
    params[2] = ng.a3;
    if want_nongrav_partials {
        let n_real = 1usize;
        for k in 0..3 {
            params[3 * (n_real + 6 + k) + k] = 1.0;
        }
    }
    asim.set_particle_params(params);
}

/// Run the integration loop: step to each target epoch in order, then
/// extract helio-ecliptic state + optional STM / non-grav partials.
fn run_integration(
    asim: &mut AssistSim,
    ephem: &Ephemeris,
    target_epochs: &[f64],
    compute_stm: bool,
    want_nongrav_partials: bool,
) -> Result<Vec<PropagatedState>> {
    let jd_ref = ephem.jd_ref();
    let mut results = Vec::with_capacity(target_epochs.len());

    for &target_mjd in target_epochs {
        let t_target = mjd_to_assist_time(target_mjd, jd_ref);
        asim.integrate(t_target)?;

        let particles = asim.sim().particles();
        if particles.is_empty() {
            return Err(Error::Other("No particles after integration".into()));
        }
        let p = &particles[0];
        let bary_eq = [p.x, p.y, p.z, p.vx, p.vy, p.vz];

        // Convert barycentric equatorial → heliocentric ecliptic
        let sun_t = ephem.get_body_state(ffi::ASSIST_BODY_SUN, t_target)?;
        let helio_eq = [
            bary_eq[0] - sun_t.x,
            bary_eq[1] - sun_t.y,
            bary_eq[2] - sun_t.z,
            bary_eq[3] - sun_t.vx,
            bary_eq[4] - sun_t.vy,
            bary_eq[5] - sun_t.vz,
        ];
        let helio_ecl = equatorial_to_ecliptic(&helio_eq);

        let (stm, nongrav_partials) = if compute_stm {
            extract_stm_and_partials(particles, want_nongrav_partials)
        } else {
            (None, None)
        };

        results.push(PropagatedState {
            state: helio_ecl,
            epoch: target_mjd,
            stm,
            nongrav_partials,
        });
    }

    Ok(results)
}

/// `(stm, nongrav_partials)` as extracted from variational particles.
type StmAndPartials = (Option<[[f64; 6]; 6]>, Option<[[f64; 3]; 6]>);

/// Read variational particles out of `particles[1..]` and rotate the
/// columns from barycentric equatorial into heliocentric ecliptic.
fn extract_stm_and_partials(
    particles: &[ffi::reb_particle],
    want_nongrav_partials: bool,
) -> StmAndPartials {
    let n_real = 1usize;
    let mut stm_eq = [[0.0f64; 6]; 6];
    for (d, vp) in particles[n_real..n_real + 6].iter().enumerate() {
        stm_eq[0][d] = vp.x;
        stm_eq[1][d] = vp.y;
        stm_eq[2][d] = vp.z;
        stm_eq[3][d] = vp.vx;
        stm_eq[4][d] = vp.vy;
        stm_eq[5][d] = vp.vz;
    }
    let stm = Some(rotate_matrix_eq_to_ecl(&stm_eq));

    let nongrav = if want_nongrav_partials {
        let mut ng_eq = [[0.0f64; 3]; 6];
        for (k, vp) in particles[n_real + 6..n_real + 9].iter().enumerate() {
            ng_eq[0][k] = vp.x;
            ng_eq[1][k] = vp.y;
            ng_eq[2][k] = vp.z;
            ng_eq[3][k] = vp.vx;
            ng_eq[4][k] = vp.vy;
            ng_eq[5][k] = vp.vz;
        }
        let mut ng_ecl = [[0.0f64; 3]; 6];
        for k in 0..3 {
            let col = [
                ng_eq[0][k],
                ng_eq[1][k],
                ng_eq[2][k],
                ng_eq[3][k],
                ng_eq[4][k],
                ng_eq[5][k],
            ];
            let rotated = equatorial_to_ecliptic(&col);
            for r in 0..6 {
                ng_ecl[r][k] = rotated[r];
            }
        }
        Some(ng_ecl)
    } else {
        None
    };
    (stm, nongrav)
}

/// Convert MJD TDB to ASSIST time (days from jd_ref).
///
/// ASSIST time = JD - jd_ref, where JD = MJD + 2400000.5.
fn mjd_to_assist_time(mjd_tdb: f64, jd_ref: f64) -> f64 {
    (mjd_tdb + 2_400_000.5) - jd_ref
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires the reviewed ASSIST kernel pair"]
    fn construction_eager_region_change_fails_and_exact_retry_succeeds() {
        use std::io::{Read, Seek, Write};

        let source = PathBuf::from(std::env::var("ADAM_CORE_RS_ASSIST_PLANETS_PATH").unwrap());
        let asteroids = PathBuf::from(std::env::var("ADAM_CORE_RS_ASSIST_ASTEROIDS_PATH").unwrap());
        let copy = std::env::temp_dir().join(format!(
            "adam-assist-construction-race-{}.bsp",
            std::process::id()
        ));
        std::fs::copy(&source, &copy).unwrap();
        let mut contents = Vec::new();
        std::fs::File::open(&copy)
            .unwrap()
            .read_to_end(&mut contents)
            .unwrap();
        let needle = b"GM5           2.8253458252257917D-07";
        let offset = contents
            .windows(needle.len())
            .position(|window| window == needle)
            .unwrap()
            + 14;
        let original = contents[offset];

        let result = AssistData::from_paths_with_load_hook(&copy, &asteroids, || {
            let mut file = std::fs::OpenOptions::new().write(true).open(&copy).unwrap();
            file.seek(std::io::SeekFrom::Start(offset as u64)).unwrap();
            file.write_all(b"3").unwrap();
            file.sync_all().unwrap();
            Ok(())
        });
        let error = match result {
            Ok(_) => panic!("construction accepted changed eagerly parsed bytes"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("eagerly parsed content changed during ephemeris construction"),
            "{error}"
        );

        let mut file = std::fs::OpenOptions::new().write(true).open(&copy).unwrap();
        file.seek(std::io::SeekFrom::Start(offset as u64)).unwrap();
        file.write_all(&[original]).unwrap();
        file.sync_all().unwrap();
        let retry = AssistData::from_paths(&copy, &asteroids).unwrap();
        assert_eq!(
            retry.data_fingerprint().unwrap(),
            "f4ac12257d77c7e8dbafa0f24a0f029001b7144f0577bebab27a79c9aca10e69"
        );
        drop(retry);
        std::fs::remove_file(copy).unwrap();
    }

    #[test]
    #[ignore = "requires the reviewed ASSIST kernel pair"]
    fn direct_propagation_establishes_kernel_provenance_on_first_use() {
        let planets = std::env::var("ADAM_CORE_RS_ASSIST_PLANETS_PATH").unwrap();
        let asteroids = std::env::var("ADAM_CORE_RS_ASSIST_ASTEROIDS_PATH").unwrap();
        let data = AssistData::from_paths(planets, asteroids).unwrap();
        let kernel_files = data.kernel_files.as_ref().unwrap();
        assert!(kernel_files.identity.get().is_none());

        let orbit = Orbit::new([1.2, 0.1, 0.05, -0.002, 0.016, 0.001], 60_000.0);
        let propagated = assist_propagate_single(
            &data,
            &orbit,
            &[60_001.0],
            false,
            &IntegratorConfig::default(),
        )
        .unwrap();
        assert!(propagated
            .iter()
            .flat_map(|row| row.state)
            .all(f64::is_finite));
        assert!(kernel_files.identity.get().is_some());
    }

    #[test]
    fn kernel_pair_fingerprint_is_slot_aware_and_content_pinned() {
        let planets = FileIdentity {
            size: 119_799_808,
            sha256: "a4ce9bf9b3282becc9f4b2ac3cebe03a2ae7599981aabd7265fd8482fff7c4b5".to_string(),
        };
        let asteroids = FileIdentity {
            size: 645_727_232,
            sha256: "919d612ce3c72a78fc7158f9120156542d0f21e6b8b052e4c1339c759747fd90".to_string(),
        };
        assert_eq!(
            kernel_pair_fingerprint(&planets, &asteroids),
            "f4ac12257d77c7e8dbafa0f24a0f029001b7144f0577bebab27a79c9aca10e69"
        );
        assert_ne!(
            kernel_pair_fingerprint(&planets, &asteroids),
            kernel_pair_fingerprint(&asteroids, &planets)
        );
    }

    #[cfg(feature = "kernel-data")]
    #[test]
    fn kernel_identity_rejects_wrong_size_and_digest() {
        let path = std::env::temp_dir().join(format!(
            "adam-assist-kernel-identity-{}",
            std::process::id()
        ));
        std::fs::write(&path, b"kernel").unwrap();
        let identity = FileIdentity {
            size: 6,
            sha256: "6923dd1bc0460082c5d55a831908c24a282860b7f1cd6c2b79cf1bc8857c639c".to_string(),
        };
        verify_kernel_identity(
            &path,
            "test",
            &identity,
            6,
            "6923dd1bc0460082c5d55a831908c24a282860b7f1cd6c2b79cf1bc8857c639c",
        )
        .unwrap();
        let size_error = verify_kernel_identity(&path, "test", &identity, 7, "unused").unwrap_err();
        assert!(size_error.to_string().contains("size mismatch"));
        let digest_error =
            verify_kernel_identity(&path, "test", &identity, 6, &"0".repeat(64)).unwrap_err();
        assert!(digest_error.to_string().contains("SHA-256 mismatch"));
        std::fs::remove_file(path).unwrap();
    }

    fn temporary_kernel_pair(test_name: &str) -> (PathBuf, PathBuf) {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let nonce = NEXT.fetch_add(1, AtomicOrdering::SeqCst);
        let directory = std::env::temp_dir().join(format!(
            "adam-assist-{test_name}-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let planets = directory.join("planets.bsp");
        let asteroids = directory.join("asteroids.bsp");
        std::fs::write(&planets, b"planet-kernel-a").unwrap();
        std::fs::write(&asteroids, b"asteroid-kernel-a").unwrap();
        (planets, asteroids)
    }

    fn remove_temporary_kernel_pair(planets: &Path, asteroids: &Path) {
        let directory = planets.parent().unwrap();
        if planets.exists() {
            std::fs::remove_file(planets).unwrap();
        }
        if asteroids.exists() {
            std::fs::remove_file(asteroids).unwrap();
        }
        std::fs::remove_dir(directory).unwrap();
    }

    fn synthetic_daf_bytes() -> Vec<u8> {
        let mut bytes = vec![0_u8; 6 * DAF_RECORD_LENGTH];
        bytes[0..8].copy_from_slice(b"DAF/SPK ");
        bytes[8..12].copy_from_slice(&2_u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&6_u32.to_le_bytes());
        bytes[76..80].copy_from_slice(&3_u32.to_le_bytes());
        bytes[80..84].copy_from_slice(&5_u32.to_le_bytes());
        bytes[88..96].copy_from_slice(b"LTL-IEEE");
        let comment = b"synthetic-comment";
        bytes[DAF_RECORD_LENGTH..DAF_RECORD_LENGTH + comment.len()].copy_from_slice(comment);
        bytes[2 * DAF_RECORD_LENGTH..2 * DAF_RECORD_LENGTH + 8]
            .copy_from_slice(&5.0_f64.to_le_bytes());
        bytes[4 * DAF_RECORD_LENGTH..4 * DAF_RECORD_LENGTH + 8]
            .copy_from_slice(&0.0_f64.to_le_bytes());
        bytes[5 * DAF_RECORD_LENGTH..].fill(0x5a);
        bytes
    }

    fn temporary_daf_pair(test_name: &str) -> (PathBuf, PathBuf, Vec<u8>) {
        let (planets, asteroids) = temporary_kernel_pair(test_name);
        let bytes = synthetic_daf_bytes();
        std::fs::write(&planets, &bytes).unwrap();
        std::fs::write(&asteroids, &bytes).unwrap();
        (planets, asteroids, bytes)
    }

    #[test]
    fn eager_region_identity_covers_file_comments_and_complete_summary_chain() {
        let (planets, asteroids, original) = temporary_daf_pair("eager-regions");
        let file = KernelFile::open(&planets, "test").unwrap();
        let baseline = file.loaded_identity().unwrap();

        let mut changed = original.clone();
        changed[DAF_RECORD_LENGTH + 4] ^= 0xff;
        std::fs::write(&planets, &changed).unwrap();
        assert_ne!(file.loaded_identity().unwrap(), baseline);

        std::fs::write(&planets, &original).unwrap();
        let mut changed = original.clone();
        changed[4 * DAF_RECORD_LENGTH + 24] ^= 0xff;
        std::fs::write(&planets, &changed).unwrap();
        assert_ne!(file.loaded_identity().unwrap(), baseline);

        std::fs::write(&planets, &original).unwrap();
        let mut changed = original.clone();
        changed[5 * DAF_RECORD_LENGTH + 24] ^= 0xff;
        std::fs::write(&planets, &changed).unwrap();
        assert_eq!(file.loaded_identity().unwrap(), baseline);

        drop(file);
        remove_temporary_kernel_pair(&planets, &asteroids);
    }

    #[test]
    fn pretrust_eager_change_and_truncation_fail_without_caching_then_restore() {
        let (planets, asteroids, original) = temporary_daf_pair("eager-restore");
        let mut files = KernelFiles::open(&planets, &asteroids).unwrap();
        let loaded = files.observe_loaded_identity().unwrap();
        files.bind_loaded_identity(loaded);

        let mut changed = original.clone();
        changed[DAF_RECORD_LENGTH + 4] ^= 0xff;
        std::fs::write(&planets, &changed).unwrap();
        let error = files.identity().unwrap_err().to_string();
        assert!(error.contains("bytes ASSIST loaded"), "{error}");
        assert!(files.identity.get().is_none());

        std::fs::write(&planets, &original).unwrap();
        let fingerprint = files.identity().unwrap().fingerprint.clone();
        assert_eq!(fingerprint.len(), 64);

        drop(files);
        let mut files = KernelFiles::open(&planets, &asteroids).unwrap();
        let loaded = files.observe_loaded_identity().unwrap();
        files.bind_loaded_identity(loaded);
        std::fs::OpenOptions::new()
            .write(true)
            .open(&planets)
            .unwrap()
            .set_len((original.len() / 2) as u64)
            .unwrap();
        let error = files.identity().unwrap_err().to_string();
        assert!(error.contains("DAF") || error.contains("bytes ASSIST loaded"));
        assert!(files.identity.get().is_none());
        std::fs::write(&planets, &original).unwrap();
        assert_eq!(files.identity().unwrap().fingerprint.len(), 64);

        drop(files);
        remove_temporary_kernel_pair(&planets, &asteroids);
    }

    #[cfg(unix)]
    #[test]
    fn pretrust_inode_replacement_fails_then_original_inode_recovers() {
        let (planets, asteroids, _original_bytes) = temporary_daf_pair("eager-inode");
        let mut files = KernelFiles::open(&planets, &asteroids).unwrap();
        let loaded = files.observe_loaded_identity().unwrap();
        files.bind_loaded_identity(loaded);

        let original = planets.with_extension("loaded");
        let replacement = planets.with_extension("replacement");
        std::fs::rename(&planets, &original).unwrap();
        std::fs::copy(&original, &planets).unwrap();
        let error = files.identity().unwrap_err().to_string();
        assert!(error.contains("canonical path was replaced"), "{error}");
        assert!(files.identity.get().is_none());

        std::fs::rename(&planets, &replacement).unwrap();
        std::fs::rename(&original, &planets).unwrap();
        assert_eq!(files.identity().unwrap().fingerprint.len(), 64);

        drop(files);
        std::fs::remove_file(replacement).unwrap();
        remove_temporary_kernel_pair(&planets, &asteroids);
    }

    #[test]
    fn kernel_fingerprint_is_lazy_thread_safe_and_cached_per_provider() {
        let (planets, asteroids) = temporary_kernel_pair("fingerprint-cache");
        let files = std::sync::Arc::new(KernelFiles::open(&planets, &asteroids).unwrap());
        assert_eq!(files.compute_count(), 0);

        let workers = (0..8)
            .map(|_| {
                let files = std::sync::Arc::clone(&files);
                std::thread::spawn(move || files.identity().unwrap().fingerprint.clone())
            })
            .collect::<Vec<_>>();
        let fingerprints = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        assert!(fingerprints.windows(2).all(|pair| pair[0] == pair[1]));
        assert_eq!(files.compute_count(), 1);
        assert_eq!(files.identity().unwrap().fingerprint, fingerprints[0]);
        assert_eq!(files.compute_count(), 1);

        drop(files);
        remove_temporary_kernel_pair(&planets, &asteroids);
    }

    #[test]
    fn cached_kernel_fingerprint_rejects_tamper_retains_trust_and_recovers() {
        let (planets, asteroids) = temporary_kernel_pair("fingerprint-rewrite");
        let files = KernelFiles::open(&planets, &asteroids).unwrap();
        let fingerprint = files.identity().unwrap().fingerprint.clone();
        assert_eq!(files.compute_count(), 1);

        let original_mtime =
            filetime::FileTime::from_last_modification_time(&std::fs::metadata(&planets).unwrap());
        std::fs::write(&planets, b"planet-kernel-b").unwrap();
        filetime::set_file_mtime(&planets, original_mtime).unwrap();
        let error = files.identity().unwrap_err().to_string();
        assert!(error.contains("kernel bytes changed"), "{error}");
        assert_eq!(files.compute_count(), 2);

        // A failed verification must not replace the last trusted digest or
        // metadata. Restoring the exact bytes can therefore recover safely.
        std::fs::write(&planets, b"planet-kernel-a").unwrap();
        assert_eq!(files.identity().unwrap().fingerprint, fingerprint);
        assert_eq!(files.compute_count(), 3);

        drop(files);
        remove_temporary_kernel_pair(&planets, &asteroids);
    }

    #[test]
    fn cached_kernel_fingerprint_accepts_identical_path_replacement() {
        let (planets, asteroids) = temporary_kernel_pair("fingerprint-replace");
        let files = KernelFiles::open(&planets, &asteroids).unwrap();
        let fingerprint = files.identity().unwrap().fingerprint.clone();
        let original = planets.with_extension("original");
        std::fs::rename(&planets, &original).unwrap();
        std::fs::write(&planets, b"planet-kernel-a").unwrap();

        assert_eq!(files.identity().unwrap().fingerprint, fingerprint);
        assert_eq!(files.compute_count(), 2);

        std::fs::write(&planets, b"planet-kernel-b").unwrap();
        let error = files.identity().unwrap_err().to_string();
        assert!(
            error.contains("kernel bytes changed")
                || error.contains("path no longer identifies the retained ASSIST mapping"),
            "{error}"
        );
        assert_eq!(files.compute_count(), 3);

        std::fs::write(&planets, b"planet-kernel-a").unwrap();
        assert_eq!(files.identity().unwrap().fingerprint, fingerprint);
        assert_eq!(files.compute_count(), 4);

        drop(files);
        std::fs::remove_file(original).unwrap();
        remove_temporary_kernel_pair(&planets, &asteroids);
    }

    #[cfg(unix)]
    #[test]
    fn cached_kernel_fingerprint_rehashes_metadata_only_changes_once() {
        use std::os::unix::fs::PermissionsExt;

        let (planets, asteroids) = temporary_kernel_pair("fingerprint-metadata");
        let files = std::sync::Arc::new(KernelFiles::open(&planets, &asteroids).unwrap());
        let fingerprint = files.identity().unwrap().fingerprint.clone();
        assert_eq!(files.compute_count(), 1);

        let hard_link = planets.with_extension("hardlink");
        std::fs::hard_link(&planets, &hard_link).unwrap();
        assert_eq!(files.identity().unwrap().fingerprint, fingerprint);
        assert_eq!(files.compute_count(), 2);
        std::fs::remove_file(&hard_link).unwrap();
        assert_eq!(files.identity().unwrap().fingerprint, fingerprint);
        assert_eq!(files.compute_count(), 3);

        let original_mode = std::fs::metadata(&planets).unwrap().permissions().mode();
        let mut permissions = std::fs::metadata(&planets).unwrap().permissions();
        permissions.set_mode(original_mode & !0o222);
        std::fs::set_permissions(&planets, permissions).unwrap();
        assert_eq!(files.identity().unwrap().fingerprint, fingerprint);
        assert_eq!(files.compute_count(), 4);
        let mut permissions = std::fs::metadata(&planets).unwrap().permissions();
        permissions.set_mode(original_mode);
        std::fs::set_permissions(&planets, permissions).unwrap();
        assert_eq!(files.identity().unwrap().fingerprint, fingerprint);
        assert_eq!(files.compute_count(), 5);

        let original_mtime =
            filetime::FileTime::from_last_modification_time(&std::fs::metadata(&planets).unwrap());
        let touched = filetime::FileTime::from_unix_time(
            original_mtime.unix_seconds() + 10,
            original_mtime.nanoseconds(),
        );
        filetime::set_file_mtime(&planets, touched).unwrap();
        assert_eq!(files.identity().unwrap().fingerprint, fingerprint);
        assert_eq!(files.compute_count(), 6);

        let touched_again = filetime::FileTime::from_unix_time(
            original_mtime.unix_seconds() + 20,
            original_mtime.nanoseconds(),
        );
        filetime::set_file_mtime(&planets, touched_again).unwrap();
        let workers = (0..8)
            .map(|_| {
                let files = std::sync::Arc::clone(&files);
                std::thread::spawn(move || files.identity().unwrap().fingerprint.clone())
            })
            .collect::<Vec<_>>();
        for worker in workers {
            assert_eq!(worker.join().unwrap(), fingerprint);
        }
        assert_eq!(files.compute_count(), 7);

        drop(files);
        remove_temporary_kernel_pair(&planets, &asteroids);
    }

    #[cfg(unix)]
    #[test]
    fn first_kernel_use_accepts_benign_pretrust_metadata_changes() {
        use std::os::unix::fs::PermissionsExt;

        let (planets, asteroids) = temporary_kernel_pair("fingerprint-first-use-metadata");
        let files = KernelFiles::open(&planets, &asteroids).unwrap();
        assert_eq!(files.compute_count(), 0);

        let hard_link = planets.with_extension("hardlink");
        std::fs::hard_link(&planets, &hard_link).unwrap();
        let original_mode = std::fs::metadata(&planets).unwrap().permissions().mode();
        let mut permissions = std::fs::metadata(&planets).unwrap().permissions();
        permissions.set_mode(original_mode & !0o222);
        std::fs::set_permissions(&planets, permissions).unwrap();
        let original_mtime =
            filetime::FileTime::from_last_modification_time(&std::fs::metadata(&planets).unwrap());
        filetime::set_file_mtime(
            &planets,
            filetime::FileTime::from_unix_time(
                original_mtime.unix_seconds() + 10,
                original_mtime.nanoseconds(),
            ),
        )
        .unwrap();

        let fingerprint = files.identity().unwrap().fingerprint.clone();
        assert_eq!(fingerprint.len(), 64);
        assert_eq!(files.compute_count(), 1);

        let mut permissions = std::fs::metadata(&planets).unwrap().permissions();
        permissions.set_mode(original_mode);
        std::fs::set_permissions(&planets, permissions).unwrap();
        std::fs::remove_file(hard_link).unwrap();
        drop(files);
        remove_temporary_kernel_pair(&planets, &asteroids);
    }

    #[test]
    fn failed_first_identity_observation_is_retryable() {
        let (planets, asteroids) = temporary_kernel_pair("fingerprint-first-use-retry");
        let files = KernelFiles::open(&planets, &asteroids).unwrap();
        let moved = asteroids.with_extension("moved");
        std::fs::rename(&asteroids, &moved).unwrap();

        let error = files.identity().unwrap_err().to_string();
        assert!(
            error.contains("failed to re-inspect small-body kernel"),
            "{error}"
        );
        assert!(files.identity.get().is_none());
        assert_eq!(files.compute_count(), 0);

        std::fs::rename(&moved, &asteroids).unwrap();
        assert_eq!(files.identity().unwrap().fingerprint.len(), 64);
        assert_eq!(files.compute_count(), 1);

        drop(files);
        remove_temporary_kernel_pair(&planets, &asteroids);
    }

    #[test]
    fn metadata_race_during_first_hash_does_not_poison_retry() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let (planets, asteroids) = temporary_kernel_pair("fingerprint-first-use-race");
        std::fs::OpenOptions::new()
            .write(true)
            .open(&planets)
            .unwrap()
            .set_len(32 * 1024 * 1024)
            .unwrap();
        let files = KernelFiles::open(&planets, &asteroids).unwrap();
        let running = std::sync::Arc::new(AtomicBool::new(true));
        let changes = std::sync::Arc::new(AtomicUsize::new(0));
        let mutator = {
            let planets = planets.clone();
            let running = std::sync::Arc::clone(&running);
            let changes = std::sync::Arc::clone(&changes);
            std::thread::spawn(move || {
                while running.load(Ordering::SeqCst) {
                    let sequence = changes.fetch_add(1, AtomicOrdering::SeqCst) as i64;
                    filetime::set_file_mtime(
                        &planets,
                        filetime::FileTime::from_unix_time(2_000_000_000 + sequence, 0),
                    )
                    .unwrap();
                    std::thread::yield_now();
                }
            })
        };
        while changes.load(AtomicOrdering::SeqCst) < 10 {
            std::thread::yield_now();
        }

        let error = files.identity().unwrap_err().to_string();
        assert!(error.contains("did not remain stable"), "{error}");
        assert!(files.identity.get().is_none());
        running.store(false, Ordering::SeqCst);
        mutator.join().unwrap();

        assert_eq!(files.identity().unwrap().fingerprint.len(), 64);
        assert!(files.compute_count() >= 4);

        drop(files);
        remove_temporary_kernel_pair(&planets, &asteroids);
    }

    // ── frame rotation contracts (moved from assist-rs coordinates.rs) ─────

    #[test]
    fn obliquity_constants_are_consistent() {
        assert!((COS_EPS - OBLIQUITY_J2000.cos()).abs() <= 2e-16);
        assert!((SIN_EPS - OBLIQUITY_J2000.sin()).abs() <= 2e-16);
        assert!((COS_EPS * COS_EPS + SIN_EPS * SIN_EPS - 1.0).abs() <= 4e-16);
    }

    #[test]
    fn roundtrip_eq_ecl() {
        let state = [1.0, 0.5, -0.3, 0.01, 0.005, -0.003];
        let ecl = equatorial_to_ecliptic(&state);
        let back = ecliptic_to_equatorial(&ecl);
        for i in 0..6 {
            assert!((state[i] - back[i]).abs() < 1e-14, "element {i} mismatch");
        }
    }

    // ── covariance propagation contracts (moved from assist-rs propagate.rs)

    fn identity6() -> [[f64; 6]; 6] {
        let mut m = [[0.0; 6]; 6];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        m
    }

    fn identity9() -> [[f64; 9]; 9] {
        let mut m = [[0.0; 9]; 9];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        m
    }

    /// A non-trivial "STM-like" 6×6: entries chosen so both rotation and
    /// scaling happen, and it's not symmetric.
    fn sample_stm() -> [[f64; 6]; 6] {
        [
            [1.0, 0.0, 0.0, 30.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 30.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 30.0],
            [0.0001, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0001, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0001, 0.0, 0.0, 1.0],
        ]
    }

    fn max_abs_diff_6x6(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> f64 {
        let mut m: f64 = 0.0;
        for i in 0..6 {
            for j in 0..6 {
                m = m.max((a[i][j] - b[i][j]).abs());
            }
        }
        m
    }

    #[test]
    fn nongrav_model_key_canonicalizes_default_and_irrelevant_nn() {
        let default = NonGravParams::new(1.0e-9, -2.0e-10, 0.0);
        let explicit = NonGravParams {
            a1: 3.0e-9,
            a2: 0.0,
            a3: 0.0,
            alpha: Some(1.0),
            nk: Some(0.0),
            nm: Some(2.0),
            nn: Some(0.0),
            r0: Some(1.0),
        };
        assert_eq!(nongrav_model_key(None), nongrav_model_key(Some(&default)));
        assert_eq!(nongrav_model_key(None), nongrav_model_key(Some(&explicit)));
    }

    #[test]
    fn nongrav_model_key_separates_active_marsden_laws() {
        let default = NonGravParams::new(1.0e-9, 0.0, 0.0);
        let comet = NonGravParams {
            a1: 1.0e-9,
            a2: 0.0,
            a3: 0.0,
            alpha: Some(0.111_262_042_6),
            nk: Some(4.614_2),
            nm: Some(2.15),
            nn: Some(5.093),
            r0: Some(2.808),
        };
        assert_ne!(
            nongrav_model_key(Some(&default)),
            nongrav_model_key(Some(&comet))
        );
    }

    #[test]
    fn covariance_6x6_identity_p0() {
        let stm = sample_stm();
        let got = covariance_6x6(&stm, &identity6());
        let mut want = [[0.0f64; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                let mut s = 0.0;
                for k in 0..6 {
                    s += stm[i][k] * stm[j][k];
                }
                want[i][j] = s;
            }
        }
        assert!(max_abs_diff_6x6(&got, &want) < 1e-14);
        for i in 0..6 {
            for j in 0..6 {
                assert!((got[i][j] - got[j][i]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn covariance_6x6_zero_stm_is_zero() {
        let stm = [[0.0f64; 6]; 6];
        let got = covariance_6x6(&stm, &identity6());
        for row in &got {
            for value in row {
                assert_eq!(*value, 0.0);
            }
        }
    }

    #[test]
    fn propagate_covariance_method_wraps_helper() {
        let state = PropagatedState {
            state: [0.0; 6],
            epoch: 0.0,
            stm: Some(sample_stm()),
            nongrav_partials: None,
        };
        let p0 = identity6();
        let via_method = state.propagate_covariance(&p0).unwrap();
        let via_helper = covariance_6x6(state.stm.as_ref().unwrap(), &p0);
        assert!(max_abs_diff_6x6(&via_method, &via_helper) < 1e-14);
    }

    #[test]
    fn propagate_covariance_none_when_no_stm() {
        let state = PropagatedState {
            state: [0.0; 6],
            epoch: 0.0,
            stm: None,
            nongrav_partials: None,
        };
        assert!(state.propagate_covariance(&identity6()).is_none());
    }

    #[test]
    fn covariance_9x9_reduces_to_6x6_when_nongrav_block_is_zero() {
        let stm = sample_stm();
        let ng: [[f64; 3]; 6] = [
            [1e-3, 2e-3, -1e-3],
            [0.0, 1e-3, 0.0],
            [0.0, 0.0, 1e-3],
            [1e-5, 0.0, 0.0],
            [0.0, 1e-5, 0.0],
            [0.0, 0.0, 1e-5],
        ];
        let p_xx = {
            let mut m = [[0.0; 6]; 6];
            for (i, v) in [0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4].iter().enumerate() {
                m[i][i] = *v;
            }
            m
        };
        let mut p0_9 = [[0.0; 9]; 9];
        for i in 0..6 {
            for j in 0..6 {
                p0_9[i][j] = p_xx[i][j];
            }
        }

        let got = covariance_9x9(&stm, &ng, &p0_9);
        let want = covariance_6x6(&stm, &p_xx);
        assert!(
            max_abs_diff_6x6(&got, &want) < 1e-14,
            "9×9 with zero A-block should match 6×6 path; diff={:.3e}",
            max_abs_diff_6x6(&got, &want)
        );
    }

    #[test]
    fn covariance_9x9_picks_up_pure_nongrav_covariance() {
        let stm = sample_stm();
        let ng: [[f64; 3]; 6] = [
            [10.0, 5.0, 2.0],
            [3.0, 8.0, 1.0],
            [1.0, 2.0, 6.0],
            [0.1, 0.05, 0.02],
            [0.03, 0.08, 0.01],
            [0.01, 0.02, 0.06],
        ];
        let p_aa = [[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]];
        let mut p0_9 = [[0.0; 9]; 9];
        for i in 0..3 {
            for j in 0..3 {
                p0_9[6 + i][6 + j] = p_aa[i][j];
            }
        }

        let got = covariance_9x9(&stm, &ng, &p0_9);

        let mut tmp = [[0.0; 3]; 6];
        for i in 0..6 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += ng[i][k] * p_aa[k][j];
                }
                tmp[i][j] = s;
            }
        }
        let mut want = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += tmp[i][k] * ng[j][k];
                }
                want[i][j] = s;
            }
        }
        assert!(max_abs_diff_6x6(&got, &want) < 1e-12);
    }

    #[test]
    fn covariance_9x9_identity_p0_includes_both_blocks() {
        let stm = sample_stm();
        let ng: [[f64; 3]; 6] = [
            [1.0, 0.5, 0.2],
            [0.3, 0.8, 0.1],
            [0.1, 0.2, 0.6],
            [0.01, 0.005, 0.002],
            [0.003, 0.008, 0.001],
            [0.001, 0.002, 0.006],
        ];
        let got = covariance_9x9(&stm, &ng, &identity9());
        let state_part = covariance_6x6(&stm, &identity6());
        let mut ng_part = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += ng[i][k] * ng[j][k];
                }
                ng_part[i][j] = s;
            }
        }
        let mut want = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                want[i][j] = state_part[i][j] + ng_part[i][j];
            }
        }
        assert!(max_abs_diff_6x6(&got, &want) < 1e-12);
    }
}
