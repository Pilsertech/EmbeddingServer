//! Embedding Server Library
//!
//! Standalone TCP server for high-performance embedding generation

pub mod models;
pub mod onnx;
pub mod protocol;
pub mod server;

// Re-exports
pub use models::{EmbeddingModelsManager, EmbeddingError, Embedding};
pub use server::{EmbeddingServer, ServerConfig, start_hyper_http_server};
pub use protocol::{EmbedRequest, EmbedResponse};
