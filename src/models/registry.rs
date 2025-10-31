//! Model registry for managing multiple embedding models
//!
//! This module provides a registry that can load, store, and manage
//! multiple embedding models based on the configuration.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::models::{EmbeddingResult, Embedding};

/// Model registry for managing multiple models
pub struct ModelRegistry {
    /// Loaded models
    models: RwLock<HashMap<String, Arc<dyn crate::models::model::EmbeddingModel>>>,
    /// Model information cache
    model_infos: RwLock<HashMap<String, crate::models::model::ModelInfo>>,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            model_infos: RwLock::new(HashMap::new()),
        }
    }

    /// Load models from configuration
    pub async fn load_from_config(
        &self,
        config: &crate::models::config::EmbeddingModelsConfig,
    ) -> EmbeddingResult<()> {
        for (model_name, model_config) in &config.models {
            if model_config.enabled {
                self.load_model(model_config).await?;
            }
        }
        Ok(())
    }

    /// Load a single model
    pub async fn load_model(
        &self,
        config: &crate::models::config::ModelConfig,
    ) -> EmbeddingResult<()> {
        let mut model = crate::models::model::ModelFactory::create_model(config);

        // Initialize the model
        model.initialize().await?;

        // Store model info
        let info = model.info().clone();
        self.model_infos.write().await.insert(config.name.clone(), info);

        // Store the model
        self.models.write().await.insert(
            config.name.clone(),
            Arc::from(model),
        );

        Ok(())
    }

    /// Get a model by name
    pub async fn get_model(&self, name: &str) -> Option<Arc<dyn crate::models::model::EmbeddingModel>> {
        self.models.read().await.get(name).cloned()
    }

    /// Get model information by name
    pub async fn get_model_info(&self, name: &str) -> Option<crate::models::model::ModelInfo> {
        self.model_infos.read().await.get(name).cloned()
    }

    /// List all loaded models
    pub async fn list_models(&self) -> Vec<String> {
        self.models.read().await.keys().cloned().collect()
    }

    /// List all model information
    pub async fn list_model_infos(&self) -> Vec<crate::models::model::ModelInfo> {
        self.model_infos.read().await.values().cloned().collect()
    }

    /// Check if a model is loaded
    pub async fn is_model_loaded(&self, name: &str) -> bool {
        self.models.read().await.contains_key(name)
    }

    /// Unload a model
    pub async fn unload_model(&self, name: &str) -> EmbeddingResult<()> {
        if let Some(model) = self.models.write().await.remove(name) {
            // The model will be dropped when the Arc is released
            self.model_infos.write().await.remove(name);
            Ok(())
        } else {
            Err(crate::EmbeddingError::ModelNotFound {
                model_name: name.to_string(),
            })
        }
    }

    /// Get the default model
    pub async fn get_default_model(
        &self,
        config: &crate::models::config::EmbeddingModelsConfig,
    ) -> Option<Arc<dyn crate::models::model::EmbeddingModel>> {
        self.get_model(&config.global.default_model).await
    }

    /// Get models by group
    pub async fn get_models_by_group(
        &self,
        group: &str,
        config: &crate::models::config::EmbeddingModelsConfig,
    ) -> Vec<Arc<dyn crate::models::model::EmbeddingModel>> {
        let model_names = match group {
            "general" => &config.model_groups.general,
            "multilingual" => &config.model_groups.multilingual,
            "high_dim" => &config.model_groups.high_dim,
            "gpu_models" => &config.model_groups.gpu_models,
            _ => return Vec::new(),
        };

        let mut models = Vec::new();
        for name in model_names {
            if let Some(model) = self.get_model(name).await {
                models.push(model);
            }
        }
        models
    }

    /// Get GPU-enabled models
    pub async fn get_gpu_models(&self) -> Vec<Arc<dyn crate::models::model::EmbeddingModel>> {
        let models = self.models.read().await;
        models
            .values()
            .filter(|model| model.supports_gpu())
            .cloned()
            .collect()
    }

    /// Shutdown all models
    pub async fn shutdown_all(&self) -> EmbeddingResult<()> {
        let mut models = self.models.write().await;
        models.clear();
        self.model_infos.write().await.clear();
        Ok(())
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_registry_creation() {
        let registry = ModelRegistry::new();
        let models = registry.list_models().await;
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn test_registry_operations() {
        let registry = ModelRegistry::new();

        // Initially empty
        assert!(registry.list_models().await.is_empty());

        // Check non-existent model
        assert!(!registry.is_model_loaded("test-model").await);
        assert!(registry.get_model("test-model").await.is_none());
    }
}
