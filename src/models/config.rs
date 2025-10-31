//! Configuration module for embedding models
//!
//! This module handles loading and parsing the embeddingmodels.toml configuration
//! and provides structured access to model settings.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Global configuration for embedding models
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingModelsConfig {
    /// Global settings
    pub global: GlobalConfig,
    /// Model-specific configurations
    pub models: HashMap<String, ModelConfig>,
    /// Model groups
    #[serde(default)]
    pub model_groups: ModelGroups,
    /// Monitoring settings
    #[serde(default)]
    pub monitoring: MonitoringConfig,
    /// Error handling settings
    #[serde(default)]
    pub error_handling: ErrorHandlingConfig,
    /// Development settings
    #[serde(default)]
    pub development: DevelopmentConfig,
}

impl Default for EmbeddingModelsConfig {
    fn default() -> Self {
        Self {
            global: GlobalConfig {
                default_model: "all-MiniLM-L6-v2".to_string(),
                max_batch_size: 32,
                cache_enabled: true,
                cache_size_mb: 512,
                init_timeout: 300,
                inference_timeout: 60,
            },
            models: HashMap::new(),
            model_groups: ModelGroups::default(),
            monitoring: MonitoringConfig::default(),
            error_handling: ErrorHandlingConfig::default(),
            development: DevelopmentConfig::default(),
        }
    }
}

/// Global settings
#[derive(Debug, Clone, Deserialize)]
pub struct GlobalConfig {
    /// Default model to use
    pub default_model: String,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Cache settings
    pub cache_enabled: bool,
    pub cache_size_mb: usize,
    /// Timeout settings (seconds)
    pub init_timeout: u64,
    pub inference_timeout: u64,
}

/// Configuration for a specific model
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// Model metadata
    pub name: String,
    pub description: String,
    pub version: String,
    pub enabled: bool,

    /// File paths (relative to EmbeddingModels directory)
    pub model_path: String,
    pub tokenizer_path: String,
    pub config_path: String,

    /// Model parameters
    pub max_sequence_length: usize,
    pub embedding_dimension: usize,
    pub pooling_mode: String,

    /// Performance settings
    pub batch_size: usize,
    pub use_gpu: bool,
    pub num_threads: usize,

    /// Runtime settings
    pub onnx_runtime_path: String,
    pub execution_provider: String,
}

/// Model groups for different use cases
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ModelGroups {
    pub general: Vec<String>,
    pub multilingual: Vec<String>,
    pub high_dim: Vec<String>,
    pub gpu_models: Vec<String>,
}

/// Monitoring configuration
#[derive(Debug, Clone, Deserialize, Default)]
pub struct MonitoringConfig {
    pub metrics_enabled: bool,
    pub log_inference_times: bool,
    pub track_usage: bool,
    pub metrics_interval: u64,
}

/// Error handling configuration
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ErrorHandlingConfig {
    pub max_retries: usize,
    pub retry_delay_ms: u64,
    pub circuit_breaker_enabled: bool,
    pub circuit_breaker_threshold: usize,
    pub circuit_breaker_timeout: u64,
}

/// Development configuration
#[derive(Debug, Clone, Deserialize, Default)]
pub struct DevelopmentConfig {
    pub debug_logging: bool,
    pub validate_outputs: bool,
    pub save_intermediates: bool,
}

impl EmbeddingModelsConfig {
    /// Load configuration from TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, crate::models::EmbeddingError> {
        let content = std::fs::read_to_string(path)?;
        let config: EmbeddingModelsConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// Load configuration from string
    pub fn from_str(content: &str) -> Result<Self, crate::models::EmbeddingError> {
        let config: EmbeddingModelsConfig = toml::from_str(content)?;
        Ok(config)
    }

    /// Get the default model configuration
    pub fn get_default_model(&self) -> Option<&ModelConfig> {
        self.models.values().find(|model| model.name == self.global.default_model)
    }

    /// Get all enabled models
    pub fn get_enabled_models(&self) -> Vec<&ModelConfig> {
        self.models
            .values()
            .filter(|model| model.enabled)
            .collect()
    }

    /// Get model by name
    pub fn get_model(&self, name: &str) -> Option<&ModelConfig> {
        self.models.values().find(|model| model.name == name)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), crate::models::EmbeddingError> {
        // Check that default model exists
        let default_model_exists = self.models.values().any(|model| model.name == self.global.default_model);
        if !default_model_exists {
            return Err(crate::models::EmbeddingError::ConfigError {
                message: format!("Default model '{}' not found in models", self.global.default_model),
            });
        }

        // Check that default model is enabled
        if let Some(default_model) = self.get_default_model() {
            if !default_model.enabled {
                return Err(crate::models::EmbeddingError::ConfigError {
                    message: format!("Default model '{}' is not enabled", self.global.default_model),
                });
            }
        }

        // Validate model groups reference existing models
        for group in [&self.model_groups.general, &self.model_groups.multilingual,
                     &self.model_groups.high_dim, &self.model_groups.gpu_models].iter() {
            for model_name in *group {
                if !self.models.contains_key(model_name) {
                    return Err(crate::models::EmbeddingError::ConfigError {
                        message: format!("Model '{}' in group not found in models", model_name),
                    });
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
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

        let config = EmbeddingModelsConfig::from_str(config_str).unwrap();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_default_model() {
        let config_str = r#"
            [global]
            default_model = "nonexistent-model"
            max_batch_size = 32
            cache_enabled = true
            cache_size_mb = 512
            init_timeout = 300
            inference_timeout = 60
        "#;

        let config = EmbeddingModelsConfig::from_str(config_str).unwrap();
        assert!(config.validate().is_err());
    }
}
