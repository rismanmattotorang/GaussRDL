// src/utils/mod.rs
use candle_core::{Device, Result as CandleResult};
use crate::Result;
use std::path::PathBuf;
use std::fs;
use std::io::{self, Read, Write};
use reqwest;
use sha2::{Sha256, Digest};
use flate2::read::GzDecoder;
use tar::Archive;
use indicatif::{ProgressBar, ProgressStyle};

pub mod logger;
pub mod profiler;
pub mod checkpoint;

pub use logger::*;
pub use profiler::*;
pub use checkpoint::*;

pub fn get_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    if candle_core::utils::cuda_is_available() {
        return Ok(Device::new_cuda(0).map_err(crate::GaussRelgtError::from)?);
    }
    
    #[cfg(feature = "metal")]
    if candle_core::utils::metal_is_available() {
        return Ok(Device::new_metal(0).map_err(crate::GaussRelgtError::from)?);
    }
    
    Ok(Device::Cpu)
}

/// Downloads a file from a URL with progress bar
pub async fn download_file(url: &str, path: &PathBuf) -> Result<()> {
    let client = reqwest::Client::new();
    let response = client.get(url)
        .send()
        .await
        .map_err(|e| Error::download(e.to_string()))?;
        
    let total_size = response.content_length().unwrap_or(0);
    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
        .unwrap()
        .progress_chars("#>-"));
        
    let mut file = fs::File::create(path)
        .map_err(|e| Error::io(e))?;
        
    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| Error::download(e.to_string()))?;
        file.write_all(&chunk)
            .map_err(|e| Error::io(e))?;
        downloaded += chunk.len() as u64;
        pb.set_position(downloaded);
    }
    
    pb.finish_with_message("Download complete");
    Ok(())
}

/// Computes SHA256 hash of a file
pub fn compute_file_hash(path: &PathBuf) -> Result<String> {
    let mut file = fs::File::open(path)
        .map_err(|e| Error::io(e))?;
        
    let mut hasher = Sha256::new();
    let mut buffer = [0; 8192];
    
    loop {
        let count = file.read(&mut buffer)
            .map_err(|e| Error::io(e))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    
    Ok(format!("{:x}", hasher.finalize()))
}

/// Extracts a tar.gz file
pub fn extract_targz(src: &PathBuf, dst: &PathBuf) -> Result<()> {
    let file = fs::File::open(src)
        .map_err(|e| Error::io(e))?;
    let tar = GzDecoder::new(file);
    let mut archive = Archive::new(tar);
    
    archive.unpack(dst)
        .map_err(|e| Error::io(e))?;
        
    Ok(())
}

/// Creates directory if it doesn't exist
pub fn create_dir_if_not_exists(path: &PathBuf) -> Result<()> {
    if !path.exists() {
        fs::create_dir_all(path)
            .map_err(|e| Error::io(e))?;
    }
    Ok(())
}

/// Removes file if it exists
pub fn remove_file_if_exists(path: &PathBuf) -> Result<()> {
    if path.exists() {
        fs::remove_file(path)
            .map_err(|e| Error::io(e))?;
    }
    Ok(())
}

/// Removes directory if it exists
pub fn remove_dir_if_exists(path: &PathBuf) -> Result<()> {
    if path.exists() {
        fs::remove_dir_all(path)
            .map_err(|e| Error::io(e))?;
    }
    Ok(())
}

/// Gets cache directory
pub fn get_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("relbench")
}

/// Validates file hash
pub fn validate_file_hash(path: &PathBuf, expected_hash: &str) -> Result<bool> {
    let actual_hash = compute_file_hash(path)?;
    Ok(actual_hash == expected_hash)
}

/// Formats file size
pub fn format_size(size: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KB", "MB", "GB", "TB", "PB"];
    let mut size = size as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    format!("{:.2} {}", size, UNITS[unit_index])
}

/// Gets file size
pub fn get_file_size(path: &PathBuf) -> Result<u64> {
    let metadata = fs::metadata(path)
        .map_err(|e| Error::io(e))?;
    Ok(metadata.len())
}

/// Checks if a file exists
pub fn file_exists(path: &PathBuf) -> bool {
    path.exists() && path.is_file()
}

/// Checks if a directory exists
pub fn dir_exists(path: &PathBuf) -> bool {
    path.exists() && path.is_dir()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_file_hash() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, b"test data").unwrap();
        
        let hash = compute_file_hash(&file_path).unwrap();
        assert_eq!(hash.len(), 64);
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(1023), "1023.00 B");
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
    }

    #[test]
    fn test_file_operations() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        
        assert!(!file_exists(&file_path));
        fs::write(&file_path, b"test data").unwrap();
        assert!(file_exists(&file_path));
        
        remove_file_if_exists(&file_path).unwrap();
        assert!(!file_exists(&file_path));
    }
}