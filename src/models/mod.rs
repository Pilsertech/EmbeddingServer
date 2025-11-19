// Copy all files from the parent directory's EmbeddingModels to this models directory
// These will be adapted for the standalone server

pub mod config;
pub mod manager;
pub mod model;
pub mod registry;

// Re-exports
pub use config::{EmbeddingModelsConfig, ModelConfig};
pub use manager::EmbeddingModelsManager;
pub use model::{EmbeddingModel, ModelInfo};
pub use registry::ModelRegistry;

/// Embedding vector type
pub type Embedding = Vec<f32>;

/// Result type for embedding models operations
pub type EmbeddingResult<T> = Result<T, EmbeddingError>;

/// Errors that can occur in embedding models operations
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    #[error("Model not found: {model_name}")]
    ModelNotFound { model_name: String },

    #[error("Model loading failed: {model_name} - {error}")]
    ModelLoadError { model_name: String, error: String },

    #[error("Model load failed: {error}")]
    ModelLoadFailed { error: String },

    #[error("Inference failed: {model_name} - {error}")]
    InferenceError { model_name: String, error: String },

    #[error("Embedding failed: {error}")]
    EmbeddingFailed { error: String },

    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    #[error("IO error: {error}")]
    IoError { error: std::io::Error },

    #[error("TOML parsing error: {error}")]
    TomlError { error: toml::de::Error },
}

impl From<std::io::Error> for EmbeddingError {
    fn from(error: std::io::Error) -> Self {
        EmbeddingError::IoError { error }
    }
}

impl From<toml::de::Error> for EmbeddingError {
    fn from(error: toml::de::Error) -> Self {
        EmbeddingError::TomlError { error }
    }
}

impl From<ort::Error> for EmbeddingError {
    fn from(error: ort::Error) -> Self {
        EmbeddingError::ModelLoadFailed { error: error.to_string() }
    }
}
