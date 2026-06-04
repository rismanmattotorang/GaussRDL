//! Local dataset cache management: status, download/materialize, delete, load.

use crate::download::download_file;
use crate::prepared;
use crate::registry::{DatasetInfo, DatasetSource};
use crate::Progress;
use gaussrdl_rdl::io::{load_database_csv, CsvDatabase};
use gaussrdl_rdl::{RdlError, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// State of a dataset in the local cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetStatus {
    /// Not present locally.
    NotDownloaded,
    /// Present and loadable (a `schema.json` exists).
    Ready,
    /// Present but incomplete/corrupt.
    Error(String),
}

/// Persisted alongside a downloaded dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Meta {
    id: String,
    official: bool,
    source_kind: String,
}

/// Manages a root cache directory containing one folder per dataset.
pub struct DatasetManager {
    root: PathBuf,
}

impl DatasetManager {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Default cache root: `$GAUSSRDL_DATA`, else `$HOME/.cache/gaussrdl/datasets`,
    /// else a temp directory.
    pub fn default_root() -> PathBuf {
        if let Ok(p) = std::env::var("GAUSSRDL_DATA") {
            return PathBuf::from(p);
        }
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(".cache/gaussrdl/datasets");
        }
        std::env::temp_dir().join("gaussrdl-datasets")
    }

    pub fn with_default_root() -> Self {
        Self::new(Self::default_root())
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Directory for a dataset id.
    pub fn dir(&self, id: &str) -> PathBuf {
        self.root.join(id)
    }

    /// Current status of a dataset.
    pub fn status(&self, info: &DatasetInfo) -> DatasetStatus {
        let dir = self.dir(&info.id);
        if !dir.exists() {
            return DatasetStatus::NotDownloaded;
        }
        if dir.join("schema.json").exists() {
            DatasetStatus::Ready
        } else {
            DatasetStatus::Error("missing schema.json".into())
        }
    }

    /// Ensure a dataset is present locally, materializing or downloading it as
    /// needed. Idempotent: a ready dataset is returned immediately.
    pub fn ensure(&self, info: &DatasetInfo, cb: &mut dyn FnMut(Progress)) -> Result<PathBuf> {
        let dir = self.dir(&info.id);
        if self.status(info) == DatasetStatus::Ready {
            cb(Progress::done("already present"));
            return Ok(dir);
        }
        std::fs::create_dir_all(&dir).map_err(|e| RdlError::Data(e.to_string()))?;

        match &info.source {
            DatasetSource::Prepared(scale) => {
                cb(Progress::msg("materializing prepared dataset…"));
                prepared::materialize(*scale, &dir)?;
            }
            DatasetSource::Remote { base_url, files } => {
                let n = files.len().max(1);
                for (i, f) in files.iter().enumerate() {
                    cb(Progress {
                        message: format!("downloading {f} ({}/{n})", i + 1, n = n),
                        fraction: Some(i as f32 / n as f32),
                    });
                    let url = format!("{base_url}{f}");
                    download_file(&url, &dir.join(f), cb)?;
                }
            }
        }

        let meta = Meta {
            id: info.id.clone(),
            official: info.official,
            source_kind: match info.source {
                DatasetSource::Prepared(_) => "prepared".into(),
                DatasetSource::Remote { .. } => "remote".into(),
            },
        };
        let _ = std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&meta).unwrap());
        cb(Progress::done("ready"));
        Ok(dir)
    }

    /// Delete a dataset from the cache.
    pub fn delete(&self, id: &str) -> Result<()> {
        let dir = self.dir(id);
        if dir.exists() {
            std::fs::remove_dir_all(&dir).map_err(|e| RdlError::Data(e.to_string()))?;
        }
        Ok(())
    }

    /// Load a ready dataset into a [`CsvDatabase`].
    pub fn load(&self, info: &DatasetInfo) -> Result<CsvDatabase> {
        if self.status(info) != DatasetStatus::Ready {
            return Err(RdlError::Data(format!(
                "dataset `{}` is not downloaded; call ensure() first",
                info.id
            )));
        }
        load_database_csv(self.dir(&info.id))
    }

    /// On-disk size of a dataset in bytes (0 if absent).
    pub fn size_bytes(&self, id: &str) -> u64 {
        fn walk(p: &Path) -> u64 {
            let mut total = 0;
            if let Ok(rd) = std::fs::read_dir(p) {
                for e in rd.flatten() {
                    let path = e.path();
                    if path.is_dir() {
                        total += walk(&path);
                    } else if let Ok(m) = e.metadata() {
                        total += m.len();
                    }
                }
            }
            total
        }
        walk(&self.dir(id))
    }
}
