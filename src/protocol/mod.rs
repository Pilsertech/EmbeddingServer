//! OVNT Protocol Implementation for Embedding Server
//!
//! This module implements the same OVNT (TCP) protocol used by the main server
//! for fast, efficient communication over the network.
//!
//! Protocol Format:
//! - Magic bytes (4): [0x4F, 0x56, 0x4E, 0x54] = "OVNT"
//! - Version (1): 0x01
//! - Message type (1): 4 = Data
//! - Length (4): u32 little-endian
//! - Sender ID (16): UUID
//! - Target ID option (17): 1 byte tag + 16 bytes UUID
//! - Message ID (16): UUID
//! - Payload: MessagePack serialized data

pub mod http;

use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use uuid::Uuid;

/// OVNT Protocol magic bytes
pub const MAGIC_BYTES: [u8; 4] = [0x4F, 0x56, 0x4E, 0x54]; // "OVNT"

/// Protocol version
pub const VERSION: u8 = 0x01;

/// Message type for data
pub const MSG_TYPE_DATA: u8 = 4;

/// Embedding request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedRequest {
    /// Text to embed
    pub text: String,
    /// Optional model name (uses default if None)
    pub model: Option<String>,
}

/// Embedding response message - SIMPLE MODE
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbedResponse {
    /// Direct array format: [0.1, 0.2, ...]
    DirectArray(Vec<f32>),
    /// Wrapped format: {"embedding": [...]}
    Wrapped { embedding: Vec<f32> },
    /// Alternative wrapped: {"vector": [...]}
    VectorWrapped { vector: Vec<f32> },
}

impl EmbedResponse {
    /// Create response from embedding vector
    pub fn new(embedding: Vec<f32>) -> Self {
        EmbedResponse::DirectArray(embedding)
    }

    /// Get the embedding vector regardless of format
    pub fn get_embedding(&self) -> &Vec<f32> {
        match self {
            EmbedResponse::DirectArray(v) => v,
            EmbedResponse::Wrapped { embedding } => embedding,
            EmbedResponse::VectorWrapped { vector } => vector,
        }
    }
}

/// Error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// Protocol message envelope
#[derive(Debug)]
pub struct ProtocolMessage {
    pub sender_id: Uuid,
    pub target_id: Option<Uuid>,
    pub message_id: Uuid,
    pub payload: Vec<u8>,
}

impl ProtocolMessage {
    /// Create a new protocol message
    pub fn new(sender_id: Uuid, target_id: Option<Uuid>, payload: Vec<u8>) -> Self {
        Self {
            sender_id,
            target_id,
            message_id: Uuid::new_v4(),
            payload,
        }
    }

    /// Write message to TCP stream
    pub async fn write_to_stream(&self, stream: &mut TcpStream) -> io::Result<()> {
        // Magic bytes
        stream.write_all(&MAGIC_BYTES).await?;
        
        // Version
        stream.write_u8(VERSION).await?;
        
        // Message type
        stream.write_u8(MSG_TYPE_DATA).await?;
        
        // Length (payload size)
        stream.write_u32_le(self.payload.len() as u32).await?;
        
        // Sender ID
        stream.write_all(self.sender_id.as_bytes()).await?;
        
        // Target ID (optional)
        if let Some(target) = self.target_id {
            stream.write_u8(1).await?; // Some tag
            stream.write_all(target.as_bytes()).await?;
        } else {
            stream.write_u8(0).await?; // None tag
        }
        
        // Message ID
        stream.write_all(self.message_id.as_bytes()).await?;
        
        // Payload
        stream.write_all(&self.payload).await?;
        
        stream.flush().await?;
        Ok(())
    }

    /// Read message from TCP stream
    pub async fn read_from_stream(stream: &mut TcpStream) -> io::Result<Self> {
        // Read magic bytes
        let mut magic = [0u8; 4];
        stream.read_exact(&mut magic).await?;
        if magic != MAGIC_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid magic bytes: {:?}", magic),
            ));
        }
        
        // Read version
        let version = stream.read_u8().await?;
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported version: {}", version),
            ));
        }
        
        // Read message type
        let _msg_type = stream.read_u8().await?;
        
        // Read length
        let length = stream.read_u32_le().await?;
        
        // Read sender ID
        let mut sender_bytes = [0u8; 16];
        stream.read_exact(&mut sender_bytes).await?;
        let sender_id = Uuid::from_bytes(sender_bytes);
        
        // Read target ID (optional)
        let target_tag = stream.read_u8().await?;
        let target_id = if target_tag == 1 {
            let mut target_bytes = [0u8; 16];
            stream.read_exact(&mut target_bytes).await?;
            Some(Uuid::from_bytes(target_bytes))
        } else {
            None
        };
        
        // Read message ID
        let mut message_bytes = [0u8; 16];
        stream.read_exact(&mut message_bytes).await?;
        let message_id = Uuid::from_bytes(message_bytes);
        
        // Read payload
        let mut payload = vec![0u8; length as usize];
        stream.read_exact(&mut payload).await?;
        
        Ok(Self {
            sender_id,
            target_id,
            message_id,
            payload,
        })
    }
}

/// Serialize request to MessagePack
pub fn serialize_request(request: &EmbedRequest) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    rmp_serde::to_vec(request)
}

/// Deserialize request from MessagePack
pub fn deserialize_request(data: &[u8]) -> Result<EmbedRequest, rmp_serde::decode::Error> {
    rmp_serde::from_slice(data)
}

/// Serialize response to MessagePack
pub fn serialize_response(response: &EmbedResponse) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    rmp_serde::to_vec(response)
}

/// Deserialize response from MessagePack
pub fn deserialize_response(data: &[u8]) -> Result<EmbedResponse, rmp_serde::decode::Error> {
    rmp_serde::from_slice(data)
}

/// Serialize error to MessagePack
pub fn serialize_error(error: &ErrorResponse) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    rmp_serde::to_vec(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_request_serialization() {
        let request = EmbedRequest {
            text: "Hello, world!".to_string(),
            model: Some("All MiniLM L6 v2".to_string()),
        };

        let serialized = serialize_request(&request).unwrap();
        let deserialized = deserialize_request(&serialized).unwrap();

        assert_eq!(request.text, deserialized.text);
        assert_eq!(request.model, deserialized.model);
    }

    #[test]
    fn test_embed_response_formats() {
        let embedding = vec![0.1, 0.2, 0.3];
        
        // Test direct array
        let resp1 = EmbedResponse::DirectArray(embedding.clone());
        assert_eq!(resp1.get_embedding(), &embedding);
        
        // Test wrapped
        let resp2 = EmbedResponse::Wrapped { embedding: embedding.clone() };
        assert_eq!(resp2.get_embedding(), &embedding);
        
        // Test vector wrapped
        let resp3 = EmbedResponse::VectorWrapped { vector: embedding.clone() };
        assert_eq!(resp3.get_embedding(), &embedding);
    }
}
