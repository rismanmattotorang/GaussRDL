use std::sync::Arc;
use parking_lot::Mutex;
use crate::error::{Error, Result};
use std::collections::HashMap;

/// Resource type
#[derive(Debug)]
pub enum ResourceType {
    /// Memory mapped file
    Mmap(memmap2::Mmap),
    /// GPU memory
    Gpu(Vec<u8>),
    /// System memory
    Memory(Vec<u8>),
    /// File handle
    File(std::fs::File),
}

/// Resource handle
#[derive(Debug, Clone)]
pub struct ResourceHandle {
    id: usize,
    manager: Arc<ResourceManager>,
}

impl Drop for ResourceHandle {
    fn drop(&mut self) {
        self.manager.release_resource(self.id);
    }
}

/// Resource manager
#[derive(Debug)]
pub struct ResourceManager {
    resources: Mutex<HashMap<usize, ResourceType>>,
    next_id: Mutex<usize>,
}

impl ResourceManager {
    /// Create new resource manager
    pub fn new() -> Self {
        Self {
            resources: Mutex::new(HashMap::new()),
            next_id: Mutex::new(0),
        }
    }

    /// Allocate memory mapped file
    pub fn allocate_mmap(&self, path: &std::path::Path) -> Result<ResourceHandle> {
        let file = std::fs::File::open(path)
            .map_err(|e| Error::Io(e))?;
            
        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .map_err(|e| Error::Memory(e.to_string()))?
        };

        let id = {
            let mut next_id = self.next_id.lock();
            let id = *next_id;
            *next_id += 1;
            id
        };

        self.resources.lock().insert(id, ResourceType::Mmap(mmap));

        Ok(ResourceHandle {
            id,
            manager: Arc::new(self.clone()),
        })
    }

    /// Allocate GPU memory
    pub fn allocate_gpu(&self, size: usize) -> Result<ResourceHandle> {
        let memory = vec![0; size];

        let id = {
            let mut next_id = self.next_id.lock();
            let id = *next_id;
            *next_id += 1;
            id
        };

        self.resources.lock().insert(id, ResourceType::Gpu(memory));

        Ok(ResourceHandle {
            id,
            manager: Arc::new(self.clone()),
        })
    }

    /// Allocate system memory
    pub fn allocate_memory(&self, size: usize) -> Result<ResourceHandle> {
        let memory = vec![0; size];

        let id = {
            let mut next_id = self.next_id.lock();
            let id = *next_id;
            *next_id += 1;
            id
        };

        self.resources.lock().insert(id, ResourceType::Memory(memory));

        Ok(ResourceHandle {
            id,
            manager: Arc::new(self.clone()),
        })
    }

    /// Open file
    pub fn open_file(&self, path: &std::path::Path) -> Result<ResourceHandle> {
        let file = std::fs::File::open(path)
            .map_err(|e| Error::Io(e))?;

        let id = {
            let mut next_id = self.next_id.lock();
            let id = *next_id;
            *next_id += 1;
            id
        };

        self.resources.lock().insert(id, ResourceType::File(file));

        Ok(ResourceHandle {
            id,
            manager: Arc::new(self.clone()),
        })
    }

    /// Get resource
    pub fn get_resource(&self, id: usize) -> Option<ResourceType> {
        self.resources.lock().get(&id).cloned()
    }

    /// Release resource
    pub fn release_resource(&self, id: usize) {
        self.resources.lock().remove(&id);
    }
}

impl Clone for ResourceManager {
    fn clone(&self) -> Self {
        Self {
            resources: Mutex::new(self.resources.lock().clone()),
            next_id: Mutex::new(*self.next_id.lock()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_resource_lifecycle() {
        let manager = ResourceManager::new();

        // Test memory allocation
        let handle = manager.allocate_memory(1024).unwrap();
        assert!(manager.get_resource(handle.id).is_some());
        drop(handle);
        assert!(manager.get_resource(handle.id).is_none());

        // Test file handling
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "test data").unwrap();
        
        let handle = manager.open_file(file.path()).unwrap();
        assert!(manager.get_resource(handle.id).is_some());
        drop(handle);
        assert!(manager.get_resource(handle.id).is_none());

        // Test mmap
        let handle = manager.allocate_mmap(file.path()).unwrap();
        assert!(manager.get_resource(handle.id).is_some());
        drop(handle);
        assert!(manager.get_resource(handle.id).is_none());
    }
} 