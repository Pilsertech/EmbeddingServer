//! TCP Embedding Server Configuration

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    pub network: NetworkConfig,
    pub performance: PerformanceConfig,
    pub embedding: EmbeddingConfig,
    pub monitoring: MonitoringConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NetworkConfig {
    pub bind_address: String,
    pub max_connections: usize,
    pub connection_timeout_secs: u64,
    pub read_timeout_secs: u64,
    pub write_timeout_secs: u64,
    pub keep_alive_interval_secs: u64,
    pub max_message_size: usize,
    pub buffer_size: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PerformanceConfig {
    pub worker_threads: usize,
    pub message_queue_size: usize,
    pub max_concurrent_tasks: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbeddingConfig {
    pub models_config: String,
    pub default_model: String,
    pub max_batch_size: usize,
    pub request_timeout_secs: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MonitoringConfig {
    pub enable_metrics: bool,
    pub metrics_interval_secs: u64,
    pub enable_detailed_logging: bool,
    pub log_level: String,
    pub enable_connection_stats: bool,
}

impl ServerConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: ServerConfig = toml::from_str(&content)?;
        Ok(config)
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            network: NetworkConfig {
                bind_address: "0.0.0.0:8787".to_string(),
                max_connections: 100,
                connection_timeout_secs: 30,
                read_timeout_secs: 30,
                write_timeout_secs: 30,
                keep_alive_interval_secs: 60,
                max_message_size: 5242880,
                buffer_size: 32768,
            },
            performance: PerformanceConfig {
                worker_threads: 4,
                message_queue_size: 5000,
                max_concurrent_tasks: 50,
            },
            embedding: EmbeddingConfig {
                models_config: "embeddingmodels.toml".to_string(),
                default_model: "All MiniLM L6 v2".to_string(),
                max_batch_size: 32,
                request_timeout_secs: 30,
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                metrics_interval_secs: 60,
                enable_detailed_logging: true,
                log_level: "info".to_string(),
                enable_connection_stats: true,
            },
        }
    }
}
