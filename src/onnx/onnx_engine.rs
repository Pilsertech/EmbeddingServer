//! # ONNX Embedding Engine
//!
//! Pure Rust implementation of embedding generation using ONNX Runtime
//! and the all-MiniLM-L6-v2 model for fast, lightweight inference.
//!
//! ## Features
//!
//! - No Python dependencies - pure Rust implementation
//! - ONNX Runtime for optimized inference on CPU/GPU
//! - Batch processing support for multiple texts
//! - Memory-efficient with proper resource management
//! - 384-dimensional embeddings from all-MiniLM-L6-v2
//!
//! ## Usage
//!
//! ```rust
//! use crate::onnx_embedder::OnnxEmbeddingEngine;
//!
//! let engine = OnnxEmbeddingEngine::new("path/to/model.onnx", "path/to/tokenizer.json")?;
//! let embeddings = engine.embed_texts(vec!["Hello world".to_string()]).await?;
//! ```
//! - 384-dimensional embeddings from all-MiniLM-L6-v2
//! - Async/await support for non-blocking operations

use crate::models::EmbeddingError;
use ndarray::{ArrayViewD};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use tokenizers::Tokenizer;
use tracing::{debug, info, instrument};

/// Configuration for ONNX Runtime
#[derive(Debug, Clone)]
pub struct OnnxConfig {
    /// Path to ONNX Runtime library (DLL/so/dylib)
    pub library_path: String,
    /// ONNX Runtime version
    pub version: String,
    /// Enable profiling
    pub enable_profiling: bool,
    /// Enable memory optimization
    pub enable_memory_optimization: bool,
    /// Thread pool size for inference
    pub thread_pool_size: usize,
}

impl Default for OnnxConfig {
    fn default() -> Self {
        // Use absolute path to the ONNX runtime library
        let runtime_path = std::env::current_dir()
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join("onnxruntime-win-x64-1.22.0")
            .join("lib")
            .join("onnxruntime.dll");
        
        Self {
            library_path: runtime_path.to_string_lossy().to_string(),
            version: "1.22.0".to_string(),
            enable_profiling: false,
            enable_memory_optimization: true,
            thread_pool_size: 4,
        }
    }
}

/// ONNX-based embedding engine for generating text embeddings
#[cfg(feature = "onnx")]
#[derive(Debug)]
pub struct OnnxEmbeddingEngine {
    /// ONNX Runtime session for model inference
    session: Session,
    /// HuggingFace tokenizer for text preprocessing
    tokenizer: Tokenizer,
    /// ONNX Runtime configuration
    _config: OnnxConfig,
    /// Configuration for device and performance settings
    device: String,
    /// Batch size for processing
    batch_size: usize,
    /// Maximum sequence length
    max_seq_length: usize,
}

#[cfg(feature = "onnx")]
impl OnnxEmbeddingEngine {
    /// Create a new ONNX embedding engine
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file (model.onnx)
    /// * `tokenizer_path` - Path to the tokenizer configuration file (tokenizer.json)
    /// * `onnx_config` - ONNX Runtime configuration
    ///
    /// # Returns
    /// A new OnnxEmbeddingEngine instance or an EmbeddingError
    ///
    /// # Example
    /// ```rust
    /// let engine = OnnxEmbeddingEngine::new(
    ///     "ml_models/onnx/all-MiniLM-L6-v2/model.onnx",
    ///     "ml_models/onnx/all-MiniLM-L6-v2/tokenizer.json",
    ///     &onnx_config
    /// )?;
    /// ```
    pub fn new(model_path: &str, tokenizer_path: &str, onnx_config: &OnnxConfig) -> Result<Self, EmbeddingError> {
        Self::new_with_config(model_path, tokenizer_path, onnx_config, "cpu", 32, 512)
    }

    /// Create a new ONNX embedding engine with custom configuration
    pub fn new_with_config(
        model_path: &str,
        tokenizer_path: &str,
        onnx_config: &OnnxConfig,
        device: &str,
        batch_size: usize,
        max_seq_length: usize,
    ) -> Result<Self, EmbeddingError> {
        info!("Initializing ONNX embedding engine with model: {} (ONNX Runtime v{})", 
              model_path, onnx_config.version);

        // Set environment variable for ONNX Runtime library path if specified
        if !onnx_config.library_path.is_empty() {
            unsafe {
                std::env::set_var("ORT_DYLIB_PATH", &onnx_config.library_path);
            }
            debug!("Set ORT_DYLIB_PATH to: {}", onnx_config.library_path);
        }

        // Configure session based on device
        let session = if device == "cuda" {
            Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(1)? // CUDA doesn't benefit from multiple threads
                .commit_from_file(model_path)
        } else {
            Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(onnx_config.thread_pool_size)?
                .commit_from_file(model_path)
        }
        .map_err(|e| EmbeddingError::ModelLoadFailed {
            error: format!("Failed to load ONNX model: {}", e),
        })?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| EmbeddingError::ModelLoadFailed {
                error: format!("Failed to load tokenizer: {}", e),
            })?;

        info!("ONNX embedding engine initialized successfully with {} threads", onnx_config.thread_pool_size);
        Ok(Self {
            session,
            tokenizer,
            _config: onnx_config.clone(),
            device: device.to_string(),
            batch_size,
            max_seq_length,
        })
    }

    /// Generate embeddings for a batch of texts
    ///
    /// # Arguments
    /// * `texts` - Vector of text strings to embed
    ///
    /// # Returns
    /// Vector of embeddings (one per input text) or an EmbeddingError
    ///
    /// Each embedding is a 384-dimensional vector of f32 values.
    ///
    /// # Example
    /// ```rust
    /// let texts = vec!["Hello world".to_string(), "How are you?".to_string()];
    /// let embeddings = engine.embed_texts(texts).await?;
    /// assert_eq!(embeddings.len(), 2);
    /// assert_eq!(embeddings[0].len(), 384);
    /// ```
    #[instrument(skip(self), fields(text_count = texts.len()))]
    pub async fn embed_texts(&mut self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Err(EmbeddingError::InvalidInput {
                message: "Cannot embed empty text list".to_string(),
            });
        }

        debug!("Generating embeddings for {} texts", texts.len());

        let mut embeddings = Vec::with_capacity(texts.len());

        for text in &texts {
            // Tokenize the text
            let encoding = self.tokenizer.encode(text.as_str(), true)
                .map_err(|e| EmbeddingError::EmbeddingFailed {
                    error: format!("Tokenization failed: {}", e),
                })?;

            let input_ids = encoding.get_ids();
            let attention_mask = encoding.get_attention_mask();

            // Convert to tensors using v2.x API - Create 2D tensors [batch_size=1, seq_len]
            let input_ids_vec: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
            let attention_mask_vec: Vec<i64> = attention_mask.iter().map(|&x| x as i64).collect();

            // Create token_type_ids (all zeros for single sequence)
            let token_type_ids_vec: Vec<i64> = vec![0i64; input_ids_vec.len()];

            // Create ONNX tensors with proper 2D shape [1, seq_len] for single sequence
            let input_ids_tensor = Tensor::from_array(([1i64, input_ids_vec.len() as i64], input_ids_vec))
                .map_err(|e| EmbeddingError::EmbeddingFailed {
                    error: format!("Failed to create input_ids tensor: {}", e),
                })?;

            let attention_mask_tensor = Tensor::from_array(([1i64, attention_mask_vec.len() as i64], attention_mask_vec))
                .map_err(|e| EmbeddingError::EmbeddingFailed {
                    error: format!("Failed to create attention_mask tensor: {}", e),
                })?;

            let token_type_ids_tensor = Tensor::from_array(([1i64, token_type_ids_vec.len() as i64], token_type_ids_vec))
                .map_err(|e| EmbeddingError::EmbeddingFailed {
                    error: format!("Failed to create token_type_ids tensor: {}", e),
                })?;

            // Run inference using ort v2.x API
            let outputs = self.session.run(vec![
                ("input_ids", input_ids_tensor),
                ("attention_mask", attention_mask_tensor),
                ("token_type_ids", token_type_ids_tensor),
            ])
            .map_err(|e| EmbeddingError::EmbeddingFailed {
                error: format!("ONNX inference failed: {}", e),
            })?;

            // Extract the output tensor (last_hidden_state) using v2.x API
            let (shape, data) = outputs["last_hidden_state"]
                .try_extract_tensor::<f32>()
                .map_err(|e| EmbeddingError::EmbeddingFailed {
                    error: format!("Failed to extract output tensor: {}", e),
                })?;

            // Convert to ndarray for processing
            let dims: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
            let output_array = ndarray::ArrayView::from_shape(dims.as_slice(), data)
                .map_err(|e| EmbeddingError::EmbeddingFailed {
                    error: format!("Failed to create output array view: {:?}", e),
                })?;

            // Apply mean pooling over the sequence dimension (excluding padding tokens)
            let embedding = Self::mean_pooling(&output_array, attention_mask)?;

            // Normalize the embedding (L2 normalization)
            let normalized_embedding = Self::normalize_embedding(&embedding)?;

            embeddings.push(normalized_embedding);
        }

        debug!("Successfully generated {} embeddings", embeddings.len());
        Ok(embeddings)
    }

    /// Apply mean pooling to the token embeddings
    ///
    /// # Arguments
    /// * `output_tensor` - Output tensor from the model [batch_size, seq_len, hidden_size]
    /// * `attention_mask` - Attention mask indicating which tokens are real (1) vs padding (0)
    ///
    /// # Returns
    /// Mean-pooled embedding vector
    fn mean_pooling(output_tensor: &ArrayViewD<f32>, attention_mask: &[u32]) -> Result<Vec<f32>, EmbeddingError> {
        let shape = output_tensor.shape();
        if shape.len() != 3 {
            return Err(EmbeddingError::EmbeddingFailed {
                error: format!("Expected 3D output tensor, got {}D", shape.len()),
            });
        }

        let seq_len = shape[1];
        let hidden_size = shape[2];

        if attention_mask.len() != seq_len {
            return Err(EmbeddingError::EmbeddingFailed {
                error: format!("Attention mask length {} doesn't match sequence length {}", attention_mask.len(), seq_len),
            });
        }

        let mut pooled = vec![0.0f32; hidden_size];
        let mut valid_tokens = 0;

        // Sum embeddings for valid tokens (attention_mask == 1)
        for seq_idx in 0..seq_len {
            if attention_mask[seq_idx] == 1 {
                for hidden_idx in 0..hidden_size {
                    pooled[hidden_idx] += output_tensor[[0, seq_idx, hidden_idx]];
                }
                valid_tokens += 1;
            }
        }

        if valid_tokens == 0 {
            return Err(EmbeddingError::EmbeddingFailed {
                error: "No valid tokens found in attention mask".to_string(),
            });
        }

        // Compute mean
        for val in &mut pooled {
            *val /= valid_tokens as f32;
        }

        Ok(pooled)
    }

    /// Normalize embedding using L2 normalization
    ///
    /// # Arguments
    /// * `embedding` - Input embedding vector
    ///
    /// # Returns
    /// L2-normalized embedding vector
    fn normalize_embedding(embedding: &[f32]) -> Result<Vec<f32>, EmbeddingError> {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm == 0.0 {
            return Err(EmbeddingError::EmbeddingFailed {
                error: "Cannot normalize zero vector".to_string(),
            });
        }

        Ok(embedding.iter().map(|x| x / norm).collect())
    }

    /// Get information about the loaded model
    ///
    /// # Returns
    /// Model information including input/output names and shapes
    pub fn get_model_info(&self) -> Result<ModelInfo, EmbeddingError> {
        let inputs = &self.session.inputs;
        let outputs = &self.session.outputs;

        let input_names = inputs.iter().map(|i| i.name.clone()).collect();
        let output_names = outputs.iter().map(|o| o.name.clone()).collect();

        Ok(ModelInfo {
            input_names,
            output_names,
            embedding_dimension: 384, // all-MiniLM-L6-v2 produces 384-dim embeddings
        })
    }

    /// Handle configuration changes for ONNX engine settings
    ///
    /// # Arguments
    /// * `key` - Configuration key that changed
    /// * `value` - New value as JSON string
    ///
    /// # Returns
    /// Result indicating success or failure
    pub async fn handle_config_change(&mut self, key: &str, value: &str) -> Result<(), EmbeddingError> {
        match key {
            "embedding.onnx.device" => {
                let device: String = serde_json::from_str(value)
                    .map_err(|e| EmbeddingError::EmbeddingFailed {
                        error: format!("Invalid device config: {}", e),
                    })?;
                self.update_device(device).await?;
            }
            "embedding.onnx.batch_size" => {
                let batch_size: usize = serde_json::from_str(value)
                    .map_err(|e| EmbeddingError::EmbeddingFailed {
                        error: format!("Invalid batch_size config: {}", e),
                    })?;
                self.update_batch_size(batch_size).await?;
            }
            "embedding.onnx.max_seq_length" => {
                let max_seq_len: usize = serde_json::from_str(value)
                    .map_err(|e| EmbeddingError::EmbeddingFailed {
                        error: format!("Invalid max_seq_length config: {}", e),
                    })?;
                self.update_max_sequence_length(max_seq_len).await?;
            }
            _ => {
                debug!("OnnxEmbeddingEngine ignoring config change for key: {}", key);
            }
        }
        Ok(())
    }

    /// Update the device setting (cpu/cuda)
    async fn update_device(&mut self, device: String) -> Result<(), EmbeddingError> {
        if device != self.device {
            info!("Updating ONNX engine device from {} to {}", self.device, device);
            self.device = device;
            // Note: In a real implementation, you might need to recreate the session
            // For now, we just update the setting
        }
        Ok(())
    }

    /// Update the batch size setting
    async fn update_batch_size(&mut self, batch_size: usize) -> Result<(), EmbeddingError> {
        if batch_size != self.batch_size {
            info!("Updating ONNX engine batch size from {} to {}", self.batch_size, batch_size);
            self.batch_size = batch_size;
        }
        Ok(())
    }

    /// Update the maximum sequence length setting
    async fn update_max_sequence_length(&mut self, max_seq_length: usize) -> Result<(), EmbeddingError> {
        if max_seq_length != self.max_seq_length {
            info!("Updating ONNX engine max sequence length from {} to {}", self.max_seq_length, max_seq_length);
            self.max_seq_length = max_seq_length;
        }
        Ok(())
    }
}

/// Information about the loaded ONNX model
#[cfg(feature = "onnx")]
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Names of input tensors
    pub input_names: Vec<String>,
    /// Names of output tensors
    pub output_names: Vec<String>,
    /// Dimension of output embeddings
    pub embedding_dimension: usize,
}
