# sense-voice-cpp-rs

Rust 2024 bindings for `sense-voice.cpp` built on top of `ggml-aio-sys`.

## Quick start

```rust
use sense_voice_cpp_rs::{SenseVoiceContext, SenseVoiceContextParameters};

let params = SenseVoiceContextParameters::default();
let _ctx = SenseVoiceContext::new_with_params("sense-voice.bin", params)?;
# Ok::<(), sense_voice_cpp_rs::SenseVoiceError>(())
```

Run the simple example with:

```bash
cargo run -p sense-voice-cpp-rs --example sense_voice_simple
```

Set `SENSE_VOICE_MODEL_PATH=/absolute/path/to/model.bin` first.
