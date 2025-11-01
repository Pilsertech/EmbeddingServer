# 🚀 EmbeddingServer - Dual-Channel Embedding Server

## What You Now Have

A **standalone, dual-channel embedding server** that:
- 🔌 Serves embeddings over **TCP** (OVNT protocol) for maximum speed
- 🌐 Serves embeddings over **HTTP REST API** (HelixDB compatible on port 8699)
- ⚡ Uses the same OVNT protocol as your main server for TCP
- 🎯 Compatible with HelixDB's local embedding requirements
- 📦 Can be deployed independently
- 🔗 Used by `helix_mcp_server` for embedding generation

## Directory Structure

```
TCPMemoryServer/
├── EmbeddingServer/              ← New standalone server
│   ├── Cargo.toml
│   ├── config.toml              ← Server configuration
│   ├── embeddingmodels.toml     ← Model configuration
│   └── src/
│       ├── main.rs              ← Entry point
│       ├── lib.rs
│       ├── protocol/            ← OVNT protocol
│       ├── server/              ← TCP server
│       ├── models/              ← Model management
│       └── onnx/                ← ONNX engine
│
├── helix_mcp_server/            ← Uses embedding client
│   └── src/
│       └── embedding_client.rs   ← TCP client
│
└── all-MiniLM-L6-v2/            ← Shared model files
    ├── model.onnx
    └── tokenizer.json
```

## Quick Start

### Step 1: Build EmbeddingServer

```powershell
cd EmbeddingServer
cargo build --release
```

### Step 2: Configure Model Paths

Edit `EmbeddingServer/embeddingmodels.toml`:

```toml
[models.all-MiniLM-L6-v2]
model_path = "../all-MiniLM-L6-v2/model.onnx"
tokenizer_path = "../all-MiniLM-L6-v2/tokenizer.json"
onnx_runtime_path = "../onnxruntime-win-x64-1.22.0"
```

### Step 3: Start EmbeddingServer

```powershell
cd EmbeddingServer
cargo run --release
```

Expected output:
```
🚀 Dual-Channel Embedding Server
📊 Log Level: info
===============================
🚀 Initializing TCP Embedding Server
✅ Embedding models loaded successfully
📡 Server bound to 0.0.0.0:8787
🆔 Server ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
✅ Server created successfully!
📡 TCP Server:  0.0.0.0:8787
🌐 HTTP Server: 0.0.0.0:8699
� HTTP Endpoints:
   POST http://0.0.0.0:8699/embed   - Generate embeddings
   GET  http://0.0.0.0:8699/health  - Health check
🛑 Press Ctrl+C to stop
```

### Step 4: Update helix_mcp_server Config

Edit `helix_mcp_server/mcpconfig.toml`:

```toml
[embedding]
provider = "local"
# Use TCP address (not HTTP URL)
local_api_url = "127.0.0.1:8787"
dimensions = 384
```

### Step 5: Update helix_mcp_server Code

In your embedding logic, use the TCP client:

```rust
use crate::embedding_client::EmbeddingClient;

// Initialize once (at startup)
let embedding_client = Arc::new(EmbeddingClient::new(
    config.embedding.local_api_url.clone(),
    30  // timeout seconds
));

// Use in tools
let embedding = embedding_client
    .embed_text(&query_text)
    .await?;
```

### Step 6: Start helix_mcp_server

```powershell
cd helix_mcp_server
cargo run --release
```

## Testing

### Test HTTP REST API (HelixDB Compatible)

The server provides a Python test script:

```powershell
# Install dependencies
pip install requests

# Run test suite
python test_http_embedding_api.py
```

Or test manually with curl:

```powershell
# Health check
curl http://localhost:8699/health

# Generate embedding (HelixDB format)
curl -X POST http://localhost:8699/embed `
  -H "Content-Type: application/json" `
  -d '{\"text\": \"Hello world\", \"chunk_style\": \"recursive\", \"chunk_size\": 100}'

# Server info
curl http://localhost:8699/
```

### Test EmbeddingServer via TCP

Create a test client:

```rust
use embedding_server::{EmbedRequest, protocol::*};
use tokio::net::TcpStream;

#[tokio::main]
async fn main() {
    let mut stream = TcpStream::connect("127.0.0.1:8787").await.unwrap();
    
    let request = EmbedRequest {
        text: "Hello, world!".to_string(),
        model: None,
    };
    
    let payload = rmp_serde::to_vec(&request).unwrap();
    let msg = ProtocolMessage::new(Uuid::new_v4(), None, payload);
    
    msg.write_to_stream(&mut stream).await.unwrap();
    
    let response = ProtocolMessage::read_from_stream(&mut stream).await.unwrap();
    let embedding: Vec<f32> = rmp_serde::from_slice(&response.payload).unwrap();
    
    println!("Embedding dimension: {}", embedding.len());
    println!("First 5 values: {:?}", &embedding[0..5]);
}
```

### Test from helix_mcp_server

```rust
#[tokio::test]
async fn test_embedding_client() {
    let client = EmbeddingClient::new("127.0.0.1:8787".to_string(), 30);
    
    // Test connection
    client.test_connection().await.unwrap();
    
    // Generate embedding
    let embedding = client.embed_text("Test text").await.unwrap();
    assert_eq!(embedding.len(), 384);
}
```

## Configuration

### EmbeddingServer (config.toml)

```toml
[network]
bind_address = "0.0.0.0:8787"         # TCP server (OVNT protocol)
http_bind_address = "0.0.0.0:8699"    # HTTP server (REST API - HelixDB compatible)
max_connections = 100                  # Concurrent connections
connection_timeout_secs = 30

[performance]
worker_threads = 4              # CPU threads
max_concurrent_tasks = 50       # Concurrent embeddings

[embedding]
models_config = "embeddingmodels.toml"
default_model = "All MiniLM L6 v2"
max_batch_size = 32

[monitoring]
log_level = "info"              # Change to "debug" for troubleshooting
```

### Model Configuration (embeddingmodels.toml)

```toml
[global]
default_model = "All MiniLM L6 v2"
max_batch_size = 32

[models.all-MiniLM-L6-v2]
name = "All MiniLM L6 v2"
enabled = true
model_path = "../all-MiniLM-L6-v2/model.onnx"
tokenizer_path = "../all-MiniLM-L6-v2/tokenizer.json"
embedding_dimension = 384
use_gpu = false
num_threads = 4
```

## Protocol Details

### TCP Protocol (OVNT)

**Request Format:**
```rust
EmbedRequest {
    text: String,           // Text to embed
    model: Option<String>   // Optional model name
}
```

**Response Format (3 supported formats):**
```rust
// 1. Direct array (default)
[0.1, 0.2, ..., 0.384]

// 2. Wrapped
{"embedding": [0.1, 0.2, ...]}

// 3. Alternative
{"vector": [0.1, 0.2, ...]}
```

All are automatically handled by the client.

### HTTP Protocol (REST API)

**Endpoints:**

| Method | Path      | Description           |
|--------|-----------|-----------------------|
| POST   | /embed    | Generate embeddings   |
| GET    | /health   | Health check          |
| GET    | /         | Server information    |

**POST /embed Request (HelixDB Format):**
```json
{
  "text": "Text to embed",
  "chunk_style": "recursive",
  "chunk_size": 100,
  "model": "All MiniLM L6 v2"  // Optional
}
```

**POST /embed Response:**
```json
{
  "embedding": [0.1, 0.2, 0.3, ..., 0.384]
}
```

**GET /health Response:**
```json
{
  "status": "healthy",
  "model": "All MiniLM L6 v2",
  "version": "0.1.0",
  "embedding_dimension": 384
}
```

**Error Response:**
```json
{
  "error": "Error message",
  "code": "ERROR_CODE",
  "details": "Additional details"
}
```

**Error Codes:**
- `EMPTY_TEXT` - Text field is empty
- `TEXT_TOO_LONG` - Text exceeds 8192 characters
- `INVALID_REQUEST` - Malformed request
- `MODEL_NOT_READY` - Model still loading
- `INTERNAL_ERROR` - Server error

## Performance

### TCP Channel (OVNT Protocol)
| Metric | Value |
|--------|-------|
| Protocol Overhead | ~0.5ms |
| Local Network Latency | 1-5ms |
| Embedding Generation | 10-50ms (CPU) |
| **Total** | **11-55ms per request** |

### HTTP Channel (REST API)
| Metric | Value |
|--------|-------|
| Protocol Overhead | ~2-5ms |
| Local Network Latency | 1-5ms |
| Embedding Generation | 10-50ms (CPU) |
| **Total** | **13-60ms per request** |

**Recommendation:** Use TCP for internal high-performance needs, HTTP for HelixDB compatibility and external integrations.

## Deployment Scenarios

### Development (Same Machine)
```
EmbeddingServer:
  - TCP:  127.0.0.1:8787
  - HTTP: 127.0.0.1:8699

helix_mcp_server: connects to 127.0.0.1:8787 (TCP)
HelixDB: connects to http://127.0.0.1:8699 (HTTP)
```

### Production (Separate Machines)
```
EmbeddingServer:
  - TCP:  192.168.1.100:8787
  - HTTP: 192.168.1.100:8699

helix_mcp_server: connects to 192.168.1.100:8787 (TCP)
HelixDB: connects to http://192.168.1.100:8699 (HTTP)
```

Update firewall to allow:
- TCP port 8787 (OVNT protocol)
- TCP port 8699 (HTTP REST API)

### Docker Deployment

```yaml
version: '3.8'
services:
  embedding-server:
    build: ./EmbeddingServer
    ports:
      - "8787:8787"  # TCP (OVNT)
      - "8699:8699"  # HTTP (REST)
    volumes:
      - ./all-MiniLM-L6-v2:/app/models
      - ./onnxruntime-win-x64-1.22.0:/app/runtime

  helix-mcp-server:
    build: ./helix_mcp_server
    environment:
      EMBEDDING_SERVER: "embedding-server:8787"
    depends_on:
      - embedding-server

  helixdb:
    build: ./helixdb
    environment:
      EMBEDDING_SERVICE_URL: "http://embedding-server:8699"
    depends_on:
      - embedding-server
```

## Troubleshooting

### Server Won't Start

**Error**: "Address already in use"
- Another process is using port 8787
- Solution: Change port in `config.toml` or kill the other process

**Error**: "Model files not found"
- Check `model_path` and `tokenizer_path` in `embeddingmodels.toml`
- Ensure paths are correct (relative or absolute)

### Client Can't Connect

**Error**: "Connection refused"
- Ensure EmbeddingServer is running
- Check firewall settings
- Verify server address in client config

**Error**: "Connection timeout"
- Server might be overloaded
- Increase timeout in client
- Check network connectivity

### Protocol Errors

**Error**: "Invalid magic bytes"
- Client and server version mismatch
- Ensure both use same OVNT protocol version

**Error**: "Deserialization failed"
- MessagePack format issue
- Check request/response structures match

## Next Steps

1. ✅ **Test the server** - Start EmbeddingServer, verify it loads models
2. ✅ **Update MCP server** - Integrate TCP client in your embedding logic
3. ✅ **Test integration** - Run end-to-end tests
4. ✅ **Monitor performance** - Check logs, measure latency
5. ✅ **Deploy** - Move to production environment

## Channel Comparison

| Feature | HTTP REST | TCP (OVNT) |
|---------|-----------|------------|
| Protocol Overhead | ~2-5ms | ~0.5-1ms |
| Serialization | JSON | MessagePack (binary) |
| Connection | HTTP/1.1 | Direct TCP |
| Type Safety | Runtime | Compile-time |
| Use Case | HelixDB, external integrations | Internal, high-performance |
| Port | 8699 | 8787 |
| Compatibility | Standard HTTP clients | Custom OVNT protocol |

**Both channels run simultaneously** - use the one that fits your needs!

## HelixDB Integration

For HelixDB to use this local embedding server:

1. **Configure HelixDB** with `embedding_model: "local"` in `config.hx.json`
2. **HelixDB expects** the HTTP server on port 8699 (already configured)
3. **Endpoint**: `POST http://localhost:8699/embed`
4. **Request format**: HelixDB automatically sends `text`, `chunk_style`, and `chunk_size`
5. **Response format**: Server returns `{"embedding": [...]}`

See `LOCAL_EMBEDDING_SETUP.md` for detailed HelixDB integration guide.

## Support

- **HTTP API Spec**: See `LOCAL_EMBEDDING_SETUP.md`
- **TCP Client Guide**: See `helix_mcp_server/TCP_EMBEDDING_CLIENT_GUIDE.md`
- **Documentation**: See `EMBEDDING_SEPARATION_PLAN.md`
- **Test Script**: Run `test_http_embedding_api.py`
- **Issues**: Check logs with `log_level = "debug"`

---

**Your embedding model is now running as a dual-channel server supporting both TCP and HTTP!** 🎉
#   E m b e d d i n g S e r v e r 
 
 #   E m b e d d i n g S e r v e r 
 
 