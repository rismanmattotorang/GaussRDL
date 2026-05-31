use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use gaussrdl_core::{Error, Result};

/// Service provider trait
pub trait ServiceProvider: Send + Sync {
    /// Get service of type T
    fn get<T: Any + Send + Sync + Clone>(&self) -> Result<Arc<T>>;
    
    /// Register service of type T
    fn register<T: Any + Send + Sync>(&self, service: T) -> Result<()>;
    
    /// Register service factory
    fn register_factory<T, F>(&self, factory: F) -> Result<()>
    where
        T: Any + Send + Sync,
        F: Fn() -> Result<T> + Send + Sync + 'static;
}

/// Service collection
#[derive(Default)]
pub struct ServiceCollection {
    services: RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>,
    factories: RwLock<HashMap<TypeId, Box<dyn Fn() -> Result<Box<dyn Any + Send + Sync>> + Send + Sync>>>,
}

impl ServiceCollection {
    /// Create new service collection
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Build service provider
    pub fn build(self) -> Arc<Container> {
        Arc::new(Container {
            services: RwLock::new(self.services.into_inner()),
            factories: RwLock::new(self.factories.into_inner()),
        })
    }
    
    /// Register service
    pub fn register<T: Any + Send + Sync>(&self, service: T) -> &Self {
        let type_id = TypeId::of::<T>();
        self.services.write().insert(type_id, Box::new(service));
        self
    }
    
    /// Register service factory
    pub fn register_factory<T, F>(&self, factory: F) -> &Self
    where
        T: Any + Send + Sync,
        F: Fn() -> Result<T> + Send + Sync + 'static,
    {
        let type_id = TypeId::of::<T>();
        let factory = Box::new(move || {
            factory().map(|s| Box::new(s) as Box<dyn Any + Send + Sync>)
        });
        self.factories.write().insert(type_id, Box::new(factory));
        self
    }
}

/// Service container
pub struct Container {
    services: RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>,
    factories: RwLock<HashMap<TypeId, Box<dyn Fn() -> Result<Box<dyn Any + Send + Sync>> + Send + Sync>>>,
}

impl ServiceProvider for Container {
    fn get<T: Any + Send + Sync + Clone>(&self) -> Result<Arc<T>> {
        let type_id = TypeId::of::<T>();
        
        // Try to get existing service
        if let Some(service) = self.services.read().get(&type_id) {
            if let Some(service) = service.downcast_ref::<T>() {
                return Ok(Arc::new(service.clone()));
            }
        }
        
        // Try to create from factory
        if let Some(factory) = self.factories.read().get(&type_id) {
            let service = factory()?;
            if let Some(service) = service.downcast_ref::<T>() {
                let service = service.clone();
                self.services.write().insert(type_id, Box::new(service.clone()));
                return Ok(Arc::new(service));
            }
        }
        
        Err(Error::Resource {
            message: format!("Service not found: {:?}", type_id),
            resource_type: None,
            resource_id: None,
            backtrace: std::backtrace::Backtrace::capture(),
        })
    }
    
    fn register<T: Any + Send + Sync>(&self, service: T) -> Result<()> {
        let type_id = TypeId::of::<T>();
        self.services.write().insert(type_id, Box::new(service));
        Ok(())
    }
    
    fn register_factory<T, F>(&self, factory: F) -> Result<()>
    where
        T: Any + Send + Sync,
        F: Fn() -> Result<T> + Send + Sync + 'static,
    {
        let type_id = TypeId::of::<T>();
        let factory = Box::new(move || {
            factory().map(|s| Box::new(s) as Box<dyn Any + Send + Sync>)
        });
        self.factories.write().insert(type_id, Box::new(factory));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[derive(Clone)]
    struct TestService {
        value: i32,
    }
    
    #[test]
    fn test_service_provider() {
        let services = ServiceCollection::new();
        
        // Register service
        services.register(TestService { value: 42 });
        
        // Register factory
        services.register_factory(|| Ok(TestService { value: 24 }));
        
        let provider = services.build();
        
        // Get registered service
        let service: Arc<TestService> = provider.get().unwrap();
        assert_eq!(service.value, 42);
        
        // Get service from factory
        let service: Arc<TestService> = provider.get().unwrap();
        assert_eq!(service.value, 42); // Should return cached instance
    }
} 