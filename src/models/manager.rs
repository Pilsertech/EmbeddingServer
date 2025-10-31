//! Embedding Models Manager
//!
//! This module provides the main interface for managing embedding models,
//! including loading configurations, managing the registry, and providing
//! a unified API for embedding operations.

use std::path::Path;
use crate::models::{EmbeddingResult, Embedding};

/// Main manager for embedding models
pub struct EmbeddingModelsManager {
    /// Configuration
    config: crate::models::config::EmbeddingModelsConfig,
    /// Model registry
    registry: crate::models::registry::ModelRegistry,
}

impl EmbeddingModelsManager {
    /// Create a new manager with default configuration
    pub fn new() -> Self {
        Self {
            config: crate::models::config::EmbeddingModelsConfig::default(),
            registry: crate::models::registry::ModelRegistry::new(),
        }
    }

    /// Create a manager from configuration file
    pub fn from_config_file<P: AsRef<Path>>(config_path: P) -> EmbeddingResult<Self> {
        let config = crate::models::config::EmbeddingModelsConfig::from_file(config_path)?;
        config.validate()?;

        Ok(Self {
            config,
            registry: crate::models::registry::ModelRegistry::new(),
        })
    }

    /// Create a manager from configuration
    pub fn from_config(config: crate::models::config::EmbeddingModelsConfig) -> EmbeddingResult<Self> {
        config.validate()?;

        Ok(Self {
            config,
            registry: crate::models::registry::ModelRegistry::new(),
        })
    }

    /// Initialize the manager and load all enabled models
    pub async fn initialize(&mut self) -> EmbeddingResult<()> {
        self.registry.load_from_config(&self.config).await
    }

    /// Get the configuration
    pub fn config(&self) -> &crate::models::config::EmbeddingModelsConfig {
        &self.config
    }

    /// Get the model registry
    pub fn registry(&self) -> &crate::models::registry::ModelRegistry {
        &self.registry
    }

    /// Embed text using the default model
    pub async fn embed_text(&self, text: &str) -> EmbeddingResult<Embedding> {
        let model = self.registry.get_default_model(&self.config).await
            .ok_or_else(|| crate::EmbeddingError::ModelNotFound {
                model_name: self.config.global.default_model.clone(),
            })?;

        model.embed_text(text).await
    }

    /// Embed text using a specific model
    pub async fn embed_text_with_model(
        &self,
        text: &str,
        model_name: &str,
    ) -> EmbeddingResult<Embedding> {
        let model = self.registry.get_model(model_name).await
            .ok_or_else(|| crate::EmbeddingError::ModelNotFound {
                model_name: model_name.to_string(),
            })?;

        model.embed_text(text).await
    }

    /// Embed a batch of texts using the default model
    pub async fn embed_batch(&self, texts: &[String]) -> EmbeddingResult<Vec<Embedding>> {
        let model = self.registry.get_default_model(&self.config).await
            .ok_or_else(|| crate::EmbeddingError::ModelNotFound {
                model_name: self.config.global.default_model.clone(),
            })?;

        model.embed_batch(texts).await
    }

    /// Embed a batch of texts using a specific model
    pub async fn embed_batch_with_model(
        &self,
        texts: &[String],
        model_name: &str,
    ) -> EmbeddingResult<Vec<Embedding>> {
        let model = self.registry.get_model(model_name).await
            .ok_or_else(|| crate::EmbeddingError::ModelNotFound {
                model_name: model_name.to_string(),
            })?;

        model.embed_batch(texts).await
    }

    /// Get information about all loaded models
    pub async fn get_loaded_models_info(&self) -> Vec<crate::models::model::ModelInfo> {
        self.registry.list_model_infos().await
    }

    /// Get information about a specific model
    pub async fn get_model_info(&self, name: &str) -> Option<crate::models::model::ModelInfo> {
        self.registry.get_model_info(name).await
    }

    /// Check if a model is loaded
    pub async fn is_model_loaded(&self, name: &str) -> bool {
        self.registry.is_model_loaded(name).await
    }

    /// Load a specific model
    pub async fn load_model(&self, model_name: &str) -> EmbeddingResult<()> {
        if let Some(model_config) = self.config.get_model(model_name) {
            self.registry.load_model(model_config).await
        } else {
            Err(crate::EmbeddingError::ModelNotFound {
                model_name: model_name.to_string(),
            })
        }
    }

    /// Unload a specific model
    pub async fn unload_model(&self, model_name: &str) -> EmbeddingResult<()> {
        self.registry.unload_model(model_name).await
    }

    /// Get models by group
    pub async fn get_models_by_group(&self, group: &str) -> Vec<String> {
        match group {
            "general" => self.config.model_groups.general.clone(),
            "multilingual" => self.config.model_groups.multilingual.clone(),
            "high_dim" => self.config.model_groups.high_dim.clone(),
            "gpu_models" => self.config.model_groups.gpu_models.clone(),
            _ => Vec::new(),
        }
    }

    /// Get GPU-enabled models
    pub async fn get_gpu_models(&self) -> Vec<String> {
        let model_infos = self.registry.list_model_infos().await;
        model_infos.into_iter()
            .filter(|info| info.uses_gpu)
            .map(|info| info.name)
            .collect()
    }

    /// Reload configuration and update models
    pub async fn reload_config(&mut self, config_path: &Path) -> EmbeddingResult<()> {
        let new_config = crate::models::config::EmbeddingModelsConfig::from_file(config_path)?;
        new_config.validate()?;

        // Shutdown all current models
        self.registry.shutdown_all().await?;

        // Update configuration
        self.config = new_config;

        // Reload models
        self.initialize().await
    }

    /// Get performance metrics (if monitoring is enabled)
    pub async fn get_metrics(&self) -> Option<crate::models::manager::Metrics> {
        if self.config.monitoring.metrics_enabled {
            // In a real implementation, this would collect actual metrics
            Some(Metrics {
                total_requests: 0,
                average_latency_ms: 0.0,
                models_loaded: self.registry.list_models().await.len(),
            })
        } else {
            None
        }
    }

    /// Shutdown the manager and all models
    pub async fn shutdown(&self) -> EmbeddingResult<()> {
        self.registry.shutdown_all().await
    }
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct Metrics {
    pub total_requests: u64,
    pub average_latency_ms: f64,
    pub models_loaded: usize,
}

impl Default for EmbeddingModelsManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_manager_creation() {
        let manager = EmbeddingModelsManager::new();
        let models = manager.get_loaded_models_info().await;
        assert!(models.is_empty());
    }

    #[test]
    fn test_config_validation() {
        // Test with minimal valid config
        let config_str = r#"
            [global]
            default_model = "test-model"
            max_batch_size = 32
            cache_enabled = true
            cache_size_mb = 512
            init_timeout = 300
            inference_timeout = 60

            [models.test-model]
            name = "Test Model"
            description = "A test model"
            version = "1.0.0"
            enabled = true
            model_path = "test/model.onnx"
            tokenizer_path = "test/tokenizer.json"
            config_path = "test/config.json"
            max_sequence_length = 256
            embedding_dimension = 384
            pooling_mode = "mean"
            batch_size = 16
            use_gpu = false
            num_threads = 4
            onnx_runtime_path = "runtime"
            execution_provider = "CPU"
        "#;

        let config = crate::models::config::EmbeddingModelsConfig::from_str(config_str).unwrap();
        let manager = EmbeddingModelsManager::from_config(config);
        assert!(manager.is_ok());
    }
}
