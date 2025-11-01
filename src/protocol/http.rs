//! HTTP REST API Protocol
//!
//! This module implements the HTTP REST API protocol for HelixDB compatibility.
//! HelixDB expects:
//! - Endpoint: POST /embed
//! - Port: 8699
//! - Request body: {"text": "...", "chunk_style": "recursive", "chunk_size": 100}
//! - Response body: {"embedding": [0.1, 0.2, 0.3, ...]}

use serde::{Deserialize, Serialize};

/// HTTP Embedding Request - HelixDB Format
/// 
/// HelixDB sends these fields:
/// - text: The text to embed (required)
/// - chunk_style: Text chunking style (required by HelixDB, set to "recursive")
/// - chunk_size: Size of text chunks (required by HelixDB, set to 100)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpEmbedRequest {
    /// Text to embed
    pub text: String,
    
    /// Chunking style (required by HelixDB)
    #[serde(default = "default_chunk_style")]
    pub chunk_style: String,
    
    /// Chunk size (required by HelixDB)
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    
    /// Optional model name (extension for multi-model support)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

fn default_chunk_style() -> String {
    "recursive".to_string()
}

fn default_chunk_size() -> usize {
    100
}

impl HttpEmbedRequest {
    /// Validate the request
    pub fn validate(&self) -> Result<(), String> {
        if self.text.is_empty() {
            return Err("Text field cannot be empty".to_string());
        }
        
        if self.text.len() > 8192 {
            return Err(format!(
                "Text exceeds maximum length of 8192 characters (got {})",
                self.text.len()
            ));
        }
        
        Ok(())
    }
}

/// HTTP Embedding Response - HelixDB Format
/// 
/// HelixDB expects: {"embedding": [0.1, 0.2, 0.3, ...]}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpEmbedResponse {
    /// The embedding vector
    pub embedding: Vec<f32>,
}

impl HttpEmbedResponse {
    /// Create a new response
    pub fn new(embedding: Vec<f32>) -> Self {
        Self { embedding }
    }
}

/// HTTP Error Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpErrorResponse {
    /// Error message
    pub error: String,
    
    /// Error code (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    
    /// Additional details (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

impl HttpErrorResponse {
    /// Create a new error response
    pub fn new(error: impl Into<String>) -> Self {
        Self {
            error: error.into(),
            code: None,
            details: None,
        }
    }
    
    /// Create error with code
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }
    
    /// Create error with details
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }
    
    /// Create missing fields error
    pub fn missing_fields(fields: &[&str]) -> Self {
        Self {
            error: format!("Missing required fields: {}", fields.join(", ")),
            code: Some("MISSING_REQUIRED_FIELDS".to_string()),
            details: None,
        }
    }
    
    /// Create empty text error
    pub fn empty_text() -> Self {
        Self {
            error: "Text field cannot be empty".to_string(),
            code: Some("EMPTY_TEXT".to_string()),
            details: None,
        }
    }
    
    /// Create text too long error
    pub fn text_too_long(length: usize) -> Self {
        Self {
            error: format!("Text exceeds maximum length of 8192 characters (got {})", length),
            code: Some("TEXT_TOO_LONG".to_string()),
            details: None,
        }
    }
    
    /// Create model not ready error
    pub fn model_not_ready() -> Self {
        Self {
            error: "Embedding model is still loading, please try again later".to_string(),
            code: Some("MODEL_NOT_READY".to_string()),
            details: None,
        }
    }
    
    /// Create internal error
    pub fn internal_error(details: impl Into<String>) -> Self {
        Self {
            error: "Internal server error occurred during embedding generation".to_string(),
            code: Some("INTERNAL_ERROR".to_string()),
            details: Some(details.into()),
        }
    }
}

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
    pub version: String,
    pub embedding_dimension: usize,
}

impl HealthResponse {
    pub fn healthy(model: impl Into<String>, dimension: usize) -> Self {
        Self {
            status: "healthy".to_string(),
            model: model.into(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            embedding_dimension: dimension,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_embed_request_validation() {
        // Valid request
        let req = HttpEmbedRequest {
            text: "Hello world".to_string(),
            chunk_style: "recursive".to_string(),
            chunk_size: 100,
            model: None,
        };
        assert!(req.validate().is_ok());

        // Empty text
        let req = HttpEmbedRequest {
            text: "".to_string(),
            chunk_style: "recursive".to_string(),
            chunk_size: 100,
            model: None,
        };
        assert!(req.validate().is_err());

        // Text too long
        let req = HttpEmbedRequest {
            text: "x".repeat(9000),
            chunk_style: "recursive".to_string(),
            chunk_size: 100,
            model: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_http_embed_response() {
        let embedding = vec![0.1, 0.2, 0.3];
        let response = HttpEmbedResponse::new(embedding.clone());
        assert_eq!(response.embedding, embedding);
    }

    #[test]
    fn test_error_response() {
        let err = HttpErrorResponse::new("Test error")
            .with_code("TEST_ERROR")
            .with_details("Additional details");
        
        assert_eq!(err.error, "Test error");
        assert_eq!(err.code, Some("TEST_ERROR".to_string()));
        assert_eq!(err.details, Some("Additional details".to_string()));
    }
}
