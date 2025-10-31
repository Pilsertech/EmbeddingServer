//! TCP Embedding Server
//!
//! High-performance TCP server for embedding generation using OVNT protocol

use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::models::EmbeddingModelsManager;
use crate::protocol::{
    deserialize_request, serialize_response, serialize_error,
    EmbedRequest, EmbedResponse, ErrorResponse, ProtocolMessage,
};
use crate::server::config::ServerConfig;

pub struct EmbeddingServer {
    config: Arc<ServerConfig>,
    embedding_manager: Arc<EmbeddingModelsManager>,
    listener: Option<TcpListener>,
    connection_limiter: Arc<Semaphore>,
    server_id: Uuid,
}

impl EmbeddingServer {
    /// Create a new embedding server
    pub async fn new(config: ServerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        info!("🚀 Initializing TCP Embedding Server");
        
        // Load embedding models
        let mut embedding_manager = EmbeddingModelsManager::from_config_file(&config.embedding.models_config)?;
        embedding_manager.initialize().await?;
        info!("✅ Embedding models loaded successfully");
        
        // Create TCP listener
        let listener = TcpListener::bind(&config.network.bind_address).await?;
        info!("📡 Server bound to {}", config.network.bind_address);
        
        // Create connection limiter
        let connection_limiter = Arc::new(Semaphore::new(config.network.max_connections));
        
        let server_id = Uuid::new_v4();
        info!("🆔 Server ID: {}", server_id);
        
        Ok(Self {
            config: Arc::new(config),
            embedding_manager: Arc::new(embedding_manager),
            listener: Some(listener),
            connection_limiter,
            server_id,
        })
    }
    
    /// Start the server
    pub async fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("🚀 Starting TCP Embedding Server");
        info!("📡 Listening on {}", self.config.network.bind_address);
        
        let listener = self.listener.take().ok_or("Server already running")?;
        let config = Arc::clone(&self.config);
        let embedding_manager = Arc::clone(&self.embedding_manager);
        let connection_limiter = Arc::clone(&self.connection_limiter);
        let server_id = self.server_id;
        
        loop {
            match listener.accept().await {
                Ok((stream, addr)) => {
                    debug!("✅ New connection from {}", addr);
                    
                    // Acquire connection permit
                    let permit = match connection_limiter.clone().try_acquire_owned() {
                        Ok(permit) => permit,
                        Err(_) => {
                            warn!("⚠️  Connection limit reached, rejecting {}", addr);
                            continue;
                        }
                    };
                    
                    let config = Arc::clone(&config);
                    let embedding_manager = Arc::clone(&embedding_manager);
                    
                    // Handle connection in background
                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_connection(
                            stream,
                            addr,
                            config,
                            embedding_manager,
                            server_id,
                        ).await {
                            error!("❌ Connection handler error for {}: {}", addr, e);
                        }
                        drop(permit); // Release connection slot
                    });
                }
                Err(e) => {
                    error!("❌ Accept error: {}", e);
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
            }
        }
    }
    
    /// Handle individual connection
    async fn handle_connection(
        mut stream: TcpStream,
        addr: std::net::SocketAddr,
        config: Arc<ServerConfig>,
        embedding_manager: Arc<EmbeddingModelsManager>,
        server_id: Uuid,
    ) -> Result<(), Box<dyn std::error::Error>> {
        debug!("🔌 Connection handler started for {}", addr);
        
        loop {
            // Read OVNT protocol message
            let request_msg = match ProtocolMessage::read_from_stream(&mut stream).await {
                Ok(msg) => msg,
                Err(e) => {
                    if e.kind() == std::io::ErrorKind::UnexpectedEof {
                        debug!("🔌 Client {} disconnected", addr);
                        break;
                    }
                    error!("❌ Protocol error from {}: {}", addr, e);
                    break;
                }
            };
            
            debug!("📨 Received message from {} (ID: {})", addr, request_msg.message_id);
            
            // Deserialize request
            let embed_request: EmbedRequest = match deserialize_request(&request_msg.payload) {
                Ok(req) => req,
                Err(e) => {
                    error!("❌ Failed to deserialize request: {}", e);
                    let error_response = ErrorResponse {
                        error: format!("Invalid request format: {}", e),
                    };
                    let error_payload = serialize_error(&error_response)?;
                    let response_msg = ProtocolMessage::new(
                        server_id,
                        Some(request_msg.sender_id),
                        error_payload,
                    );
                    response_msg.write_to_stream(&mut stream).await?;
                    continue;
                }
            };
            
            debug!("🔤 Embedding request for text length: {}", embed_request.text.len());
            
            // Generate embedding
            let embedding_result = if let Some(model_name) = &embed_request.model {
                embedding_manager.embed_text_with_model(&embed_request.text, model_name).await
            } else {
                embedding_manager.embed_text(&embed_request.text).await
            };
            
            // Prepare response
            let response_payload = match embedding_result {
                Ok(embedding) => {
                    debug!("✅ Generated embedding with {} dimensions", embedding.len());
                    let response = EmbedResponse::new(embedding);
                    serialize_response(&response)?
                }
                Err(e) => {
                    error!("❌ Embedding generation failed: {:?}", e);
                    let error_response = ErrorResponse {
                        error: format!("Embedding failed: {:?}", e),
                    };
                    serialize_error(&error_response)?
                }
            };
            
            // Send response
            let response_msg = ProtocolMessage::new(
                server_id,
                Some(request_msg.sender_id),
                response_payload,
            );
            
            response_msg.write_to_stream(&mut stream).await?;
            debug!("📤 Response sent to {}", addr);
        }
        
        Ok(())
    }
}
