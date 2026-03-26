# qwen3-asr-cpp-rs

Rust 2024 bindings for `qwen3-asr.cpp` built on top of `ggml-aio-sys`.

## Quick start

```rust
use qwen3_asr_cpp_rs::{Qwen3Asr, Qwen3AsrParams};

let mut asr = Qwen3Asr::new("qwen3-asr-0.6b-f16.gguf")?;
let params = Qwen3AsrParams::default().n_threads(4).max_tokens(1024);
let result = asr.transcribe_file("audio.wav", &params)?;
println!("{}", result.text);
# Ok::<(), qwen3_asr_cpp_rs::Qwen3AsrError>(())
```

Run the simple example with:

```bash
cargo run -p qwen3-asr-cpp-rs --example qwen3_asr_simple
```

Set these env vars first:

- `QWEN3_ASR_MODEL_PATH=/absolute/path/to/qwen3-asr-0.6b-f16.gguf`
- `QWEN3_ASR_AUDIO_PATH=/absolute/path/to/audio.wav` (16 kHz mono WAV)
