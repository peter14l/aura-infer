# Aura-Infer: Bare-Metal LLM Inference Engine

![Status](https://img.shields.io/badge/Status-Beta-orange) ![Rust](https://img.shields.io/badge/Language-Rust-black)

Aura-Infer is an ultra-lightweight, high-performance LLM inference engine written entirely in memory-safe Rust. It is designed to load and run AI models locally without the massive overhead of Python, PyTorch, or complex dependencies. 

Built to answer the question: *Can we run Llama-3 purely on bare-metal CPU with zero bloat?*

## Architecture

This engine implements the absolute foundational mathematics of modern AI:

1. **Bare-Metal Tensor Math (`tensor.rs`)**
   * Custom `Tensor` structs mapping directly to continuous memory.
   * `O(N^2)` Matrix Multiplication (`matmul`) built from scratch.
   * Custom `softmax` with numerical stability optimizations.

2. **The Transformer Block (`model.rs`)**
   * **Scaled Dot-Product Self-Attention**: The algorithm that makes neural networks understand context. Calculates $Q$, $K$, and $V$ matrices and contextual weights.
   * **RMSNorm**: Root Mean Square Normalization for significantly faster layer processing (Standard in Llama architectures).

3. **Binary Model Loading (`loader.rs`)**
   * Raw byte-buffer reading to convert Little-Endian disk bytes directly into Float32 Tensors, bypassing JSON parsing. Designed to eventually interface directly with `.safetensors` via memory mapping (mmap).

## Usage
(Under Construction: Requires a compatible `.bin` weights file).

```rust
use aura_infer::loader::ModelLoader;
use aura_infer::model::TransformerBlock;

fn main() {
    let loader = ModelLoader::new("model.safetensors");
    // Load Weights...
    // Run Inference...
}
```

## Why Rust?
LLM inference is memory-bound. Python introduces global interpreter locks and massive RAM overhead. Rust provides C-level performance with strict memory safety, meaning zero segfaults during 40-gigabyte matrix multiplications.

## License
MIT
