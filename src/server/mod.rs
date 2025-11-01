//! Server module

pub mod config;
pub mod server;
pub mod hyper_server;

pub use config::ServerConfig;
pub use server::EmbeddingServer;
pub use hyper_server::start_hyper_http_server;
