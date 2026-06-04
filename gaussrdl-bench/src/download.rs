//! HTTP file download with progress reporting.
//!
//! Uses the blocking `reqwest` client and streams the body to disk so large
//! files do not need to fit in memory. Works against any reachable host; in
//! network-restricted environments it returns a clear, actionable error.

use crate::Progress;
use gaussrdl_rdl::{RdlError, Result};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Download `url` to `dest`, invoking `cb` with progress updates.
pub fn download_file(url: &str, dest: &Path, cb: &mut dyn FnMut(Progress)) -> Result<()> {
    cb(Progress::msg(format!("connecting to {url}")));
    let client = reqwest::blocking::Client::builder()
        .user_agent("gaussrdl-bench/0.3")
        .build()
        .map_err(|e| RdlError::Data(format!("http client: {e}")))?;
    let mut resp = client
        .get(url)
        .send()
        .map_err(|e| RdlError::Data(format!("request failed for {url}: {e}")))?;
    if !resp.status().is_success() {
        return Err(RdlError::Data(format!(
            "download of {url} failed: HTTP {} (the host may be blocked by this environment's network policy)",
            resp.status()
        )));
    }
    let total = resp.content_length();
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| RdlError::Data(e.to_string()))?;
    }
    let mut file = File::create(dest).map_err(|e| RdlError::Data(e.to_string()))?;
    let mut buf = [0u8; 64 * 1024];
    let mut downloaded: u64 = 0;
    loop {
        let n = resp.read(&mut buf).map_err(|e| RdlError::Data(format!("read body: {e}")))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| RdlError::Data(e.to_string()))?;
        downloaded += n as u64;
        let frac = total.map(|t| if t > 0 { downloaded as f32 / t as f32 } else { 0.0 });
        cb(Progress {
            message: format!("downloaded {} KB", downloaded / 1024),
            fraction: frac,
        });
    }
    file.flush().map_err(|e| RdlError::Data(e.to_string()))?;
    Ok(())
}
