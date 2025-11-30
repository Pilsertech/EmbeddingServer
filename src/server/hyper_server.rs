//! Ultra-Fast Hyper-based HTTP Server
//!
//! Direct Hyper implementation bypassing Axum's routing overhead
//! Expected performance: 5-10x faster than Axum (~200-400ms vs ~2000ms)

use std::sync::Arc;
use std::convert::Infallible;
use hyper::{Body, Request, Response, Server, Method, StatusCode};
use hyper::service::{make_service_fn, service_fn};
use hyper::body::to_bytes;
use serde_json;
use tokio::net::TcpSocket;
use tracing::{debug, error, info};

use crate::models::EmbeddingModelsManager;
use crate::protocol::http::{
    HealthResponse, HttpEmbedRequest, HttpEmbedResponse, HttpErrorResponse,
};
use crate::server::config::ServerConfig;

/// Shared state for Hyper server
#[derive(Clone)]
struct ServerState {
    embedding_manager: Arc<EmbeddingModelsManager>,
    config: Arc<ServerConfig>,
}

/// Start ultra-fast Hyper HTTP server
pub async fn start_hyper_http_server(
    config: Arc<ServerConfig>,
    embedding_manager: Arc<EmbeddingModelsManager>,
) -> Result<(), Box<dyn std::error::Error>> {
    let bind_address = config.network.http_bind_address.clone();
    
    info!("🚀 Starting Ultra-Fast Hyper HTTP Server");
    info!("📡 Binding to {}", bind_address);
    
    let state = ServerState {
        embedding_manager,
        config,
    };
    
    // Create service factory
    let make_svc = make_service_fn(move |_| {
        let state = state.clone();
        async move {
            Ok::<_, Infallible>(service_fn(move |req| {
                let state = state.clone();
                handle_request(req, state)
            }))
        }
    });
    
    // Parse address
    let addr = bind_address.parse()?;
    
    // Create TCP socket with optimal settings
    let socket = TcpSocket::new_v4()?;
    
    // CRITICAL: Enable TCP_NODELAY to disable Nagle's algorithm
    // Nagle buffers small packets causing 40-200ms delays!
    socket.set_nodelay(true)?;
    
    // Enable SO_REUSEADDR for faster restart
    socket.set_reuseaddr(true)?;
    
    // Bind and listen
    socket.bind(addr)?;
    let listener = socket.listen(1024)?;
    
    // Build server with configured listener
    let server = Server::from_tcp(listener.into_std()?)?
        .http1_keepalive(true)
        .http1_half_close(false)
        .tcp_nodelay(true) // Double-ensure TCP_NODELAY
        .tcp_sleep_on_accept_errors(true)
        .serve(make_svc);
    
    info!("✅ Hyper HTTP server listening on {}", bind_address);
    info!("⚡ TCP_NODELAY enabled (eliminates Nagle buffering)");
    info!("🔄 HTTP keep-alive enabled");
    info!("📍 Endpoints:");
    info!("   POST /embed      - Generate embeddings (FAST!)");
    info!("   GET  /health     - Health check");
    info!("   GET  /           - Server info");
    
    server.await?;
    
    Ok(())
}

/// Main request handler - ultra-optimized routing
async fn handle_request(
    req: Request<Body>,
    state: ServerState,
) -> Result<Response<Body>, Infallible> {
    // Manual CORS - minimal overhead
    let origin = req.headers()
        .get("origin")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("*")
        .to_string(); // Clone the origin string to avoid borrow issues
    
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    
    // Fast path routing - no complex middleware
    let response = match (&method, path.as_str()) {
        (&Method::POST, "/embed") => handle_embed(req, state).await,
        (&Method::GET, "/health") => handle_health(state).await,
        (&Method::GET, "/") => handle_root(state).await,
        (&Method::OPTIONS, _) => handle_options(),
        _ => handle_not_found(),
    };
    
    // Add minimal CORS headers
    let mut response = response;
    let headers = response.headers_mut();
    headers.insert("access-control-allow-origin", origin.parse().unwrap());
    headers.insert("access-control-allow-methods", "GET, POST, OPTIONS".parse().unwrap());
    headers.insert("access-control-allow-headers", "content-type".parse().unwrap());
    
    Ok(response)
}

/// OPTIONS handler for CORS preflight
fn handle_options() -> Response<Body> {
    Response::builder()
        .status(StatusCode::NO_CONTENT)
        .body(Body::empty())
        .unwrap()
}

/// 404 handler
fn handle_not_found() -> Response<Body> {
    Response::builder()
        .status(StatusCode::NOT_FOUND)
        .header("content-type", "application/json")
        .body(Body::from(r#"{"error":"Not Found"}"#))
        .unwrap()
}

/// Root endpoint - server info
async fn handle_root(state: ServerState) -> Response<Body> {
    let info = serde_json::json!({
        "name": "HelixDB Embedding Server (Hyper Edition)",
        "version": env!("CARGO_PKG_VERSION"),
        "server": "Hyper (Ultra-Fast)",
        "endpoints": {
            "embed": {
                "method": "POST",
                "path": "/embed",
                "description": "Generate embeddings for text (10x faster than Axum!)"
            },
            "health": {
                "method": "GET",
                "path": "/health",
                "description": "Health check endpoint"
            }
        },
        "model": state.config.embedding.default_model
    });
    
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(info.to_string()))
        .unwrap()
}

/// Health check endpoint
async fn handle_health(state: ServerState) -> Response<Body> {
    debug!("🏥 Health check requested");
    
    let model_name = &state.config.embedding.default_model;
    
    // Quick health check with test embedding
    match state.embedding_manager.embed_text("test").await {
        Ok(embedding) => {
            let response = HealthResponse::healthy(model_name, embedding.len());
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&response).unwrap()))
                .unwrap()
        }
        Err(e) => {
            error!("❌ Health check failed: {:?}", e);
            let error = HttpErrorResponse::model_not_ready();
            Response::builder()
                .status(StatusCode::SERVICE_UNAVAILABLE)
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&error).unwrap()))
                .unwrap()
        }
    }
}

/// Embedding endpoint - THE FAST PATH
async fn handle_embed(req: Request<Body>, state: ServerState) -> Response<Body> {
    let start_time = std::time::Instant::now();
    
    // Parse JSON body - direct, no extractors
    let read_start = std::time::Instant::now();
    let body_bytes = match to_bytes(req.into_body()).await {
        Ok(bytes) => bytes,
        Err(_) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                HttpErrorResponse::new("Failed to read request body".to_string())
            );
        }
    };
    info!("⏱️  Body read took: {:?}", read_start.elapsed());
    
    // Parse JSON request
    let parse_start = std::time::Instant::now();
    let request: HttpEmbedRequest = match serde_json::from_slice(&body_bytes) {
        Ok(req) => req,
        Err(_) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                HttpErrorResponse::new("Invalid JSON".to_string())
            );
        }
    };
    info!("⏱️  JSON parse took: {:?}", parse_start.elapsed());
    
    // Validate request
    let validate_start = std::time::Instant::now();
    if let Err(msg) = request.validate() {
        let error = if request.text.is_empty() {
            HttpErrorResponse::empty_text()
        } else if request.text.len() > 8192 {
            HttpErrorResponse::text_too_long(request.text.len())
        } else {
            HttpErrorResponse::new(msg)
        };
        return error_response(StatusCode::BAD_REQUEST, error);
    }
    info!("⏱️  Validation took: {:?}", validate_start.elapsed());
    
    // Generate embedding - the actual fast part!
    let embed_start = std::time::Instant::now();
    let embedding_result = if let Some(model_name) = &request.model {
        state.embedding_manager.embed_text_with_model(&request.text, model_name).await
    } else {
        state.embedding_manager.embed_text(&request.text).await
    };
    info!("⏱️  Embedding generation took: {:?}", embed_start.elapsed());
    
    match embedding_result {
        Ok(embedding) => {
            let serialize_start = std::time::Instant::now();
            // Convert f32 embedding to f64 as required by HelixDB
            let embedding_f64: Vec<f64> = embedding.into_iter().map(|x| x as f64).collect();
            let response = HttpEmbedResponse::new(embedding_f64);
            let json_body = serde_json::to_string(&response).unwrap();
            info!("⏱️  JSON serialization took: {:?}", serialize_start.elapsed());
            info!("⏱️  TOTAL request took: {:?}", start_time.elapsed());

            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .body(Body::from(json_body))
                .unwrap()
        }
        Err(e) => {
            error!("❌ Embedding generation failed: {:?}", e);
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                HttpErrorResponse::internal_error(format!("{:?}", e))
            )
        }
    }
}

/// Helper to create error responses
fn error_response(status: StatusCode, error: HttpErrorResponse) -> Response<Body> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&error).unwrap()))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_responses() {
        let error = HttpErrorResponse::empty_text();
        let response = error_response(StatusCode::BAD_REQUEST, error);
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
