//! Model definitions and traits
//!
//! This module defines the core traits and structures for embedding models,
//! providing a unified interface for different model implementations.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Information about a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    /// Model description
    pub description: String,
    /// Model version
    pub version: String,
    /// Embedding dimension
    pub dimension: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Pooling mode
    pub pooling_mode: String,
    /// Whether the model uses GPU
    pub uses_gpu: bool,
    /// Model file path
    pub model_path: String,
    /// Tokenizer path
    pub tokenizer_path: String,
}

/// Core embedding model trait
#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    /// Get model information
    fn info(&self) -> &ModelInfo;

    /// Initialize the model
    async fn initialize(&mut self) -> crate::models::EmbeddingResult<()>;

    /// Check if the model is ready for inference
    async fn is_ready(&self) -> bool;

    /// Generate embeddings for a single text
    async fn embed_text(&self, text: &str) -> crate::models::EmbeddingResult<crate::models::Embedding>;

    /// Generate embeddings for a batch of texts
    async fn embed_batch(&self, texts: &[String]) -> crate::models::EmbeddingResult<Vec<crate::models::Embedding>>;

    /// Get the embedding dimension
    fn dimension(&self) -> usize {
        self.info().dimension
    }

    /// Check if model supports GPU
    fn supports_gpu(&self) -> bool {
        self.info().uses_gpu
    }

    /// Shutdown the model and free resources
    async fn shutdown(&mut self) -> crate::models::EmbeddingResult<()>;
}

/// ONNX-based embedding model implementation
pub mod onnx {
    use super::*;
    use crate::onnx;
    use std::path::Path;

    /// ONNX embedding model
    pub struct OnnxEmbeddingModel {
        info: ModelInfo,
        engine: Option<std::sync::Arc<tokio::sync::RwLock<crate::onnx::OnnxEmbeddingEngine>>>,
        config: crate::models::config::ModelConfig,
    }

    impl OnnxEmbeddingModel {
        /// Create a new ONNX embedding model
        pub fn new(config: crate::models::config::ModelConfig) -> Self {
            let info = ModelInfo {
                name: config.name.clone(),
                description: config.description.clone(),
                version: config.version.clone(),
                dimension: config.embedding_dimension,
                max_sequence_length: config.max_sequence_length,
                pooling_mode: config.pooling_mode.clone(),
                uses_gpu: config.use_gpu,
                model_path: config.model_path.clone(),
                tokenizer_path: config.tokenizer_path.clone(),
            };

            Self {
                info,
                engine: None,
                config,
            }
        }
    }

    #[async_trait]
    impl EmbeddingModel for OnnxEmbeddingModel {
        fn info(&self) -> &ModelInfo {
            &self.info
        }

        async fn initialize(&mut self) -> crate::models::EmbeddingResult<()> {
            // Initialize the ONNX engine
            let onnx_config = crate::onnx::OnnxConfig::default();
            let engine = crate::onnx::OnnxEmbeddingEngine::new(
                &self.config.model_path,
                &self.config.tokenizer_path,
                &onnx_config
            )?;

            self.engine = Some(std::sync::Arc::new(tokio::sync::RwLock::new(engine)));
            Ok(())
        }

        async fn is_ready(&self) -> bool {
            self.engine.is_some()
        }

        async fn embed_text(&self, text: &str) -> crate::models::EmbeddingResult<crate::models::Embedding> {
            if let Some(engine) = &self.engine {
                let mut engine = engine.write().await;
                let embeddings = engine.embed_texts(vec![text.to_string()]).await
                    .map_err(|e| crate::EmbeddingError::InferenceError {
                        model_name: self.info.name.clone(),
                        error: e.to_string(),
                    })?;
                
                if let Some(embedding) = embeddings.into_iter().next() {
                    Ok(embedding)
                } else {
                    Err(crate::EmbeddingError::InferenceError {
                        model_name: self.info.name.clone(),
                        error: "No embedding returned".to_string(),
                    })
                }
            } else {
                Err(crate::EmbeddingError::ModelNotFound {
                    model_name: self.info.name.clone(),
                })
            }
        }

        async fn embed_batch(&self, texts: &[String]) -> crate::models::EmbeddingResult<Vec<crate::models::Embedding>> {
            if let Some(engine) = &self.engine {
                let mut engine = engine.write().await;
                let embeddings = engine.embed_texts(texts.to_vec()).await
                    .map_err(|e| crate::EmbeddingError::InferenceError {
                        model_name: self.info.name.clone(),
                        error: e.to_string(),
                    })?;
                
                Ok(embeddings.into_iter().map(|e| e).collect())
            } else {
                Err(crate::EmbeddingError::ModelNotFound {
                    model_name: self.info.name.clone(),
                })
            }
        }

        async fn shutdown(&mut self) -> crate::models::EmbeddingResult<()> {
            if let Some(engine) = self.engine.take() {
                // The ONNX engine handles its own cleanup
                Ok(())
            } else {
                Ok(())
            }
        }
    }
}

/// Factory for creating embedding models
pub struct ModelFactory;

impl ModelFactory {
    /// Create a model from configuration
    pub fn create_model(config: &crate::models::config::ModelConfig) -> Box<dyn EmbeddingModel> {
        // For now, we only support ONNX models
        // In the future, this could support different model types
        Box::new(onnx::OnnxEmbeddingModel::new(config.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_model_info() {
        let info = ModelInfo {
            name: "test-model".to_string(),
            description: "Test model".to_string(),
            version: "1.0.0".to_string(),
            dimension: 384,
            max_sequence_length: 256,
            pooling_mode: "mean".to_string(),
            uses_gpu: false,
            model_path: "test/model.onnx".to_string(),
            tokenizer_path: "test/tokenizer.json".to_string(),
        };

        assert_eq!(info.name, "test-model");
        assert_eq!(info.dimension, 384);
        assert!(!info.uses_gpu);
    }
}
