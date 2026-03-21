# whisper-cpp-ggml

Rust 2024 bindings for `whisper.cpp` built on top of `ggml-aio-sys`.

## Quick start

```rust
use whisper_cpp_ggml::{WhisperContext, WhisperContextParameters};

let params = WhisperContextParameters::default();
let ctx = WhisperContext::new_with_params("ggml-model.bin", params)?;
let _ = ctx.n_vocab();
# Ok::<(), whisper_cpp_ggml::WhisperError>(())
```

Run the simple example with:

```bash
cargo run -p whisper-cpp-ggml --example whisper_simple
```

Set `WHISPER_MODEL_PATH=/absolute/path/to/model.bin` first.
