//! Basic synthesis: `QWEN3_TTS_MODELS=/path/to/models cargo run -p qwen3-tts-cpp-rs --example qwen3_tts_simple`
//!
//! The model directory must contain `qwen3-tts-0.6b-f16.gguf` (or `qwen3-tts-0.6b-q8_0.gguf`)
//! and `qwen3-tts-tokenizer-f16.gguf`.
//!
//! On macOS, keep `models/coreml/code_predictor.mlpackage` (as in upstream) so the CoreML code
//! predictor runs; use **`QWEN3_TTS_USE_COREML=0`** only to force the slower GGML predictor.
//!
//! Defaults match the upstream CLI (`qwen3-tts-cli`): `temperature=0.9`, `max_audio_tokens=4096`,
//! etc. (see `qwen3_tts_default_params` in the C API).
//!
//! **Do not force `temperature(0.0)` unless you mean it:** greedy decoding often fails to hit the
//! codec EOS token as early as the default sampler, so generation runs out the full
//! `max_audio_tokens` cap → **much longer audio and far more work** than the stock C++ binary.
//!
//! Optional: `QWEN3_TTS_MAX_AUDIO_TOKENS=<n>` caps frames for quick tests.

use std::env;
use std::path::PathBuf;

use hound::{SampleFormat, WavSpec, WavWriter};
use qwen3_tts_cpp_rs::{Qwen3Tts, Qwen3TtsParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models = env::var("QWEN3_TTS_MODELS")
        .unwrap_or_else(|_| "/Users/neko/projects/qwen3-tts.cpp/models".to_string());
    let out_path = env::var("QWEN3_TTS_OUT").unwrap_or_else(|_| "qwen3_tts_out.wav".to_string());
    let text = env::var("QWEN3_TTS_TEXT")
        .unwrap_or_else(|_| "Hello from qwen3-tts via ggml-aio-rs.".to_string());

    let mut params = Qwen3TtsParams::default();
    if let Ok(s) = env::var("QWEN3_TTS_MAX_AUDIO_TOKENS") {
        if let Ok(n) = s.parse::<i32>() {
            params = params.max_audio_tokens(n);
            eprintln!("qwen3_tts_simple: max_audio_tokens={n} (override from env)");
        }
    }

    let mut tts = Qwen3Tts::new(&models, 4)?;
    let audio = tts.synthesize(&text, &params)?;

    let spec = WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate as u32,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let path = PathBuf::from(&out_path);
    let mut writer = WavWriter::create(path, spec)?;
    for &s in &audio.samples {
        let x = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(x)?;
    }
    writer.finalize()?;

    println!(
        "Wrote {} samples at {} Hz to {}",
        audio.samples.len(),
        audio.sample_rate,
        out_path
    );
    Ok(())
}
