//! Embedding Server Main
//!
//! Entry point for the standalone embedding server
//! Runs TCP (OVNT protocol) and ULTRA-FAST Hyper HTTP servers concurrently

use embedding_server::{EmbeddingServer, ServerConfig, start_hyper_http_server};
use std::sync::Arc;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = ServerConfig::from_file("config.toml")?;
    
    // Initialize tracing
    let log_level = std::env::var("RUST_LOG")
        .unwrap_or_else(|_| {
            match config.monitoring.log_level.to_lowercase().as_str() {
                "trace" => "embedding_server=trace,trace".to_string(),
                "debug" => "embedding_server=debug,debug".to_string(),
                "info" => "embedding_server=info,info".to_string(),
                "warn" => "embedding_server=warn,warn".to_string(),
                "error" => "embedding_server=error,error".to_string(),
                _ => "embedding_server=info,info".to_string(),
            }
        });
    
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| log_level.into())
        )
        .with_target(false)
        .with_thread_ids(false)
        .with_line_number(false)
        .with_file(false)
        .without_time()
        .init();

    println!("🚀 Dual-Channel Embedding Server");
    println!("📊 Log Level: {}", config.monitoring.log_level);
    println!("===============================");

    // Create the server (which initializes embedding models)
    let mut tcp_server = EmbeddingServer::new(config.clone()).await?;

    println!("✅ Server created successfully!");
    println!("📡 TCP Server:  {}", config.network.bind_address);
    println!("🌐 HTTP Server: {}", config.network.http_bind_address);
    println!("📍 HTTP Endpoints:");
    println!("   POST http://{}/embed   - Generate embeddings", config.network.http_bind_address);
    println!("   GET  http://{}/health  - Health check", config.network.http_bind_address);
    println!("🛑 Press Ctrl+C to stop");

    // Get shared references for HTTP server
    let config_arc = Arc::new(config);
    let embedding_manager = tcp_server.get_embedding_manager();
    
    // Spawn ULTRA-FAST Hyper HTTP server in background
    let http_config = Arc::clone(&config_arc);
    let http_embedding_manager = Arc::clone(&embedding_manager);
    let http_handle = tokio::spawn(async move {
        if let Err(e) = start_hyper_http_server(http_config, http_embedding_manager).await {
            eprintln!("❌ HTTP server error: {}", e);
        }
    });

    // Start TCP server (blocks until shutdown)
    let tcp_result = tcp_server.start().await;
    
    // If TCP server exits, we should abort HTTP server too
    http_handle.abort();
    
    tcp_result
}
