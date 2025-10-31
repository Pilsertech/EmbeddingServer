//! Embedding Server Main
//!
//! Entry point for the standalone TCP embedding server

use embedding_server::{EmbeddingServer, ServerConfig};
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

    println!("🚀 Standalone TCP Embedding Server");
    println!("📊 Log Level: {}", config.monitoring.log_level);
    println!("===============================");

    // Create and start the server
    let mut server = EmbeddingServer::new(config).await?;

    println!("✅ Server created successfully!");
    println!("📡 Ready to accept embedding requests");
    println!("🛑 Press Ctrl+C to stop");

    // Start the server
    server.start().await?;

    Ok(())
}
