# Embedding Server

A high-performance, standalone embedding server implementing dual-channel architecture for text embeddings.

## Overview

The Embedding Server is a production-ready Rust service that provides text embeddings through both TCP and HTTP interfaces. It uses the ONNX Runtime for efficient inference with the all-MiniLM-L6-v2 model, delivering 384-dimensional embeddings.

**Key Features:**
- **Dual-Channel Architecture**: TCP (binary protocol) for performance-critical applications, HTTP REST API for general integration
- **Pure Rust Implementation**: No Python dependencies; fully compiled and optimized
- **High Performance**: Asynchronous I/O with Tokio, optimized ONNX inference, connection pooling
- **HelixDB Compatible**: HTTP endpoint (port 8699) matches HelixDB's local embedding API specification
- **Configurable**: Flexible model and server configuration via TOML files
- **Production-Ready**: Connection limiting, error handling, structured logging with tracing

## Architecture

### Core Components

- **TCP Server** (`src/server/server.rs`): OVNT binary protocol listener on port 8787, handles embedding requests asynchronously
- **HTTP Server** (`src/protocol/http.rs`): Hyper-based REST API on port 8699 for HelixDB integration
- **ONNX Engine** (`src/onnx/onnx_engine.rs`): ONNX Runtime wrapper for tokenization and embedding inference
- **Models Manager** (`src/models/manager.rs`): Model lifecycle management, registry, and loading from configuration
- **Protocol** (`src/protocol/`): Binary message serialization (MessagePack) and HTTP REST formats

### Configuration Files

- `config.toml`: Server network settings, logging, resource limits
- `embeddingmodels.toml`: Model definitions, ONNX runtime paths, tokenizer locations

## Getting Started

### Prerequisites
- Rust 1.70+
- ONNX Runtime libraries (included: Linux x64 and Windows x64)
- Model files (`all-MiniLM-L6-v2/model.onnx`, `all-MiniLM-L6-v2/tokenizer.json`)

### Build

```bash
cd EmbeddingServer
cargo build --release
```

### Configure

Update `embeddingmodels.toml` with paths to your model and ONNX runtime:

```toml
[models.all-MiniLM-L6-v2]
model_path = "../all-MiniLM-L6-v2/model.onnx"
tokenizer_path = "../all-MiniLM-L6-v2/tokenizer.json"
onnx_runtime_path = "../onnxruntime-linux-x64-1.22.0"  # or Windows path
```

### Run

```bash
cargo run --release
```

Server will start on:
- **TCP**: `0.0.0.0:8787`
- **HTTP**: `0.0.0.0:8699`

## API Reference

### HTTP Endpoint

```
POST http://localhost:8699/embed
```

**Request:**
```json
{
  "text": "The text to embed",
  "chunk_style": "recursive",
  "chunk_size": 100,
  "model": "all-MiniLM-L6-v2"
}
```

**Response:**
```json
{
  "embedding": [0.123, -0.456, ..., 0.789],
  "dimensions": 384,
  "model": "all-MiniLM-L6-v2"
}
```

**Health Check:**
```
GET http://localhost:8699/health
```

### TCP Protocol

Binary protocol using MessagePack serialization for embedding requests. Used by clients requiring low-latency, high-throughput embedding generation.

## Integration

### With HelixDB

HelixDB automatically uses the local HTTP endpoint at `http://localhost:8699/embed` for embedding operations when configured for local embeddings.

### Custom Applications

- **Rust clients**: Use binary TCP protocol with MessagePack deserialization
- **Other languages**: Use HTTP REST API endpoint

## Configuration

Edit `config.toml` to customize server behavior:

```toml
[network]
bind_address = "0.0.0.0:8787"      # TCP server address
http_bind_address = "0.0.0.0:8699" # HTTP server address
max_connections = 100
connection_timeout_secs = 30

[monitoring]
log_level = "info"  # Change to "debug" for detailed logs
```

Edit `embeddingmodels.toml` to manage embedding models:

```toml
[models.all-MiniLM-L6-v2]
model_path = "../all-MiniLM-L6-v2/model.onnx"
tokenizer_path = "../all-MiniLM-L6-v2/tokenizer.json"
onnx_runtime_path = "../onnxruntime-linux-x64-1.22.0"
```

## Performance

- **Embedding latency**: 10-50ms per request (CPU-based inference)
- **Throughput**: 100+ concurrent connections (configurable)
- **Memory**: ~500MB resident (model + runtime)
- **TCP overhead**: <1ms per message
- **HTTP overhead**: 2-5ms per request

## Troubleshooting

**Server won't start**: Check ports 8787 and 8699 are available. Modify `config.toml` to use different ports.

**Model not found**: Verify paths in `embeddingmodels.toml` are correct and models exist at specified locations.

**Connection refused**: Ensure the server is running and firewall allows access to configured ports.

**Out of memory**: Reduce `max_connections` in `config.toml` to lower concurrent request handling.

## License

Embedded ONNX Runtime components maintain their original licenses. See respective LICENSE files in runtime directories.
