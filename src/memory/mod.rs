use std::fs::File;
use std::path::Path;
use memmap2::{Mmap, MmapOptions};
use crate::error::{Result, Error};

/// Memory mapped file reader
pub struct MmapReader {
    /// Memory mapped file
    mmap: Mmap,
}

impl MmapReader {
    /// Create a new memory mapped file reader
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Ok(Self { mmap })
    }
    
    /// Get a slice of the memory mapped file
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }
    
    /// Get the length of the memory mapped file
    pub fn len(&self) -> usize {
        self.mmap.len()
    }
    
    /// Check if the memory mapped file is empty
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }
}

/// Memory mapped file writer
pub struct MmapWriter {
    /// Memory mapped file
    mmap: Mmap,
}

impl MmapWriter {
    /// Create a new memory mapped file writer
    pub fn new<P: AsRef<Path>>(path: P, size: usize) -> Result<Self> {
        let file = File::create(path)?;
        file.set_len(size as u64)?;
        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        Ok(Self { mmap })
    }
    
    /// Write data to the memory mapped file
    pub fn write(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        if offset + data.len() > self.mmap.len() {
            return Err(Error::memory("Write would exceed file size"));
        }
        
        let slice = &mut self.mmap[offset..offset + data.len()];
        slice.copy_from_slice(data);
        Ok(())
    }
    
    /// Get the length of the memory mapped file
    pub fn len(&self) -> usize {
        self.mmap.len()
    }
    
    /// Check if the memory mapped file is empty
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }
    
    /// Flush changes to disk
    pub fn flush(&self) -> Result<()> {
        self.mmap.flush()?;
        Ok(())
    }
}

/// Memory cache
pub struct MemoryCache {
    /// Cache capacity in bytes
    capacity: usize,
    
    /// Current size in bytes
    size: usize,
    
    /// Cached items
    items: std::collections::HashMap<String, Vec<u8>>,
}

impl MemoryCache {
    /// Create a new memory cache
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            size: 0,
            items: std::collections::HashMap::new(),
        }
    }
    
    /// Get an item from the cache
    pub fn get(&self, key: &str) -> Option<&[u8]> {
        self.items.get(key).map(|v| v.as_slice())
    }
    
    /// Put an item in the cache
    pub fn put(&mut self, key: String, value: Vec<u8>) -> Result<()> {
        let value_size = value.len();
        
        // Check if we need to evict items
        while self.size + value_size > self.capacity {
            if let Some((k, v)) = self.items.iter().next() {
                let k = k.clone();
                let v_size = v.len();
                self.items.remove(&k);
                self.size -= v_size;
            } else {
                break;
            }
        }
        
        // Add new item
        if value_size <= self.capacity {
            self.items.insert(key, value);
            self.size += value_size;
            Ok(())
        } else {
            Err(Error::memory("Item too large for cache"))
        }
    }
    
    /// Remove an item from the cache
    pub fn remove(&mut self, key: &str) -> Option<Vec<u8>> {
        if let Some(value) = self.items.remove(key) {
            self.size -= value.len();
            Some(value)
        } else {
            None
        }
    }
    
    /// Clear the cache
    pub fn clear(&mut self) {
        self.items.clear();
        self.size = 0;
    }
    
    /// Get the current size of the cache
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Get the capacity of the cache
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    
    /// Get the number of items in the cache
    pub fn len(&self) -> usize {
        self.items.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_mmap_reader() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"hello world").unwrap();
        
        let reader = MmapReader::new(file.path()).unwrap();
        assert_eq!(reader.as_slice(), b"hello world");
        assert_eq!(reader.len(), 11);
        assert!(!reader.is_empty());
    }
    
    #[test]
    fn test_mmap_writer() {
        let file = NamedTempFile::new().unwrap();
        let mut writer = MmapWriter::new(file.path(), 11).unwrap();
        
        writer.write(0, b"hello world").unwrap();
        writer.flush().unwrap();
        
        let reader = MmapReader::new(file.path()).unwrap();
        assert_eq!(reader.as_slice(), b"hello world");
    }
    
    #[test]
    fn test_memory_cache() {
        let mut cache = MemoryCache::new(100);
        
        // Test put and get
        cache.put("key1".to_string(), b"value1".to_string().into_bytes()).unwrap();
        assert_eq!(cache.get("key1").unwrap(), b"value1");
        
        // Test eviction
        cache.put("key2".to_string(), vec![0; 90]).unwrap();
        assert!(cache.get("key1").is_none());
        assert!(cache.get("key2").is_some());
        
        // Test remove
        cache.remove("key2");
        assert!(cache.get("key2").is_none());
        assert_eq!(cache.size(), 0);
        
        // Test clear
        cache.put("key3".to_string(), b"value3".to_string().into_bytes()).unwrap();
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }
} 