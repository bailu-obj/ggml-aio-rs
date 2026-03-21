# llama-cpp-rs

Rust 2024 bindings for `llama.cpp`, built on top of `ggml-aio-sys`.

This crate keeps the API close to upstream `llama.cpp` concepts (`backend`, `model`, `context`, `batch`) while still providing safer ownership boundaries in Rust.

## Quick start

```rust
use llama_cpp_rs::context::params::LlamaContextParams;
use llama_cpp_rs::llama_backend::LlamaBackend;
use llama_cpp_rs::model::params::LlamaModelParams;
use llama_cpp_rs::model::LlamaModel;

let backend = LlamaBackend::init()?;
let model = LlamaModel::load_from_file(
    &backend,
    "model.gguf",
    &LlamaModelParams::default(),
)?;
let mut ctx = model.new_context(&backend, LlamaContextParams::default())?;
let _n_ctx = ctx.n_ctx();
# Ok::<(), llama_cpp_rs::LlamaCppError>(())
```

You can also run:

```bash
cargo run -p llama-cpp-rs --example llama_simple
```

Set `LLAMA_MODEL_PATH=/absolute/path/to/model.gguf` first.

## Feature flags

- `cuda`: enables CUDA backend through `ggml-aio-sys`.
- `hipblas`: enables ROCm/HIP backend.
- `metal`: enables Metal backend.
- `vulkan`: enables Vulkan backend.
- `openmp`: enables OpenMP where supported.
- `sampler`: enables sampler helpers.
- `mtmd`: enables multimodal wrappers.

## Mapping from `llama_cpp_2`

If you are coming from `llama_cpp_2`, the core flow is the same:

- initialize backend once
- load model
- create context from model
- decode + sample

This crate intentionally follows a similar shape so most code ports with minor module path updates.
