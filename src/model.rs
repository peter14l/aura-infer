use crate::tensor::Tensor;

/// Root Mean Square Normalization (RMSNorm).
/// Used in modern LLMs like Llama-3 instead of standard LayerNorm for speed.
pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        // Initialize weights to 1.0 (in reality, loaded from a model file)
        let weight = Tensor::new(vec![1.0; dim], vec![dim]);
        RMSNorm { weight, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let size = x.data.len();
        let mut out = Tensor::zeros(vec![size]);

        // Calculate Sum of Squares
        let mut ss = 0.0;
        for i in 0..size {
            ss += x.data[i] * x.data[i];
        }
        ss /= size as f32;
        ss += self.eps;
        let inv_std = 1.0 / ss.sqrt();

        // Normalize and scale
        for i in 0..size {
            out.data[i] = (x.data[i] * inv_std) * self.weight.data[i];
        }

        out
    }
}

/// Scaled Dot-Product Self-Attention mechanism.
/// The heart of the Transformer. It calculates how much each word should "pay attention" to other words.
pub struct SelfAttention {
    pub dim: usize,
    // Linear projection weights
    pub wq: Tensor, // Query
    pub wk: Tensor, // Key
    pub wv: Tensor, // Value
    pub wo: Tensor, // Output projection
}

impl SelfAttention {
    pub fn new(dim: usize) -> Self {
        SelfAttention {
            dim,
            wq: Tensor::new(vec![0.1; dim * dim], vec![dim, dim]),
            wk: Tensor::new(vec![0.1; dim * dim], vec![dim, dim]),
            wv: Tensor::new(vec![0.1; dim * dim], vec![dim, dim]),
            wo: Tensor::new(vec![0.1; dim * dim], vec![dim, dim]),
        }
    }

    /// Forward pass for a single token (simplified for PoC).
    /// In a real engine, this processes sequences and uses RoPE (Rotary Positional Embeddings).
    pub fn forward(&self, x: &Tensor, prev_keys: &Tensor, prev_values: &Tensor) -> Tensor {
        // 1. Calculate Query, Key, Value vectors for the current token
        let q = x.matmul(&self.wq);
        let k = x.matmul(&self.wk);
        let v = x.matmul(&self.wv);

        // 2. Calculate Attention Scores: Q dot K^T / sqrt(dim)
        // (Assuming a 1-token query against previous context).
        // For this bare-metal PoC, we will simulate a score vector.
        let mut scores = Tensor::zeros(vec![prev_keys.shape[0] + 1]);
        
        // ... (Dot product loops over sequence length would go here) ...
        
        // Scale by 1 / sqrt(dim)
        let scale = 1.0 / (self.dim as f32).sqrt();
        scores.scale_(scale);

        // 3. Apply Softmax to get attention weights (probabilities)
        let probs = scores.softmax();

        // 4. Multiply probabilities by Values to get context vector
        let context = Tensor::zeros(vec![self.dim]); // Simulated result
        
        // 5. Final output projection
        context.matmul(&self.wo)
    }
}

/// A basic skeleton of a Transformer Block.
pub struct TransformerBlock {
    pub attn_norm: RMSNorm,
    pub attention: SelfAttention,
}

impl TransformerBlock {
    pub fn new(dim: usize) -> Self {
        TransformerBlock {
            attn_norm: RMSNorm::new(dim, 1e-5),
            attention: SelfAttention::new(dim),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // 1. Normalize the input (Pre-Norm architecture)
        let normalized = self.attn_norm.forward(x);
        
        // 2. Self-Attention (Using dummy previous keys/values for now)
        let dummy_k = Tensor::zeros(vec![1, self.attention.dim]);
        let dummy_v = Tensor::zeros(vec![1, self.attention.dim]);
        let mut attn_out = self.attention.forward(&normalized, &dummy_k, &dummy_v);
        
        // 3. Residual Connection (Add original input back)
        attn_out.add_(x);
        
        attn_out
    }
}
