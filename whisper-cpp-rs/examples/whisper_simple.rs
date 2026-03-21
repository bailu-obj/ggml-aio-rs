#![allow(missing_docs)]

use std::env;

use whisper_cpp_ggml::{WhisperContext, WhisperContextParameters};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = match env::var("WHISPER_MODEL_PATH") {
        Ok(path) => path,
        Err(_) => {
            eprintln!("Set WHISPER_MODEL_PATH to run this example.");
            return Ok(());
        }
    };

    let params = WhisperContextParameters::default();
    let context = WhisperContext::new_with_params(&model_path, params)?;

    println!(
        "Loaded whisper model successfully. n_vocab={}, n_audio_ctx={}",
        context.n_vocab(),
        context.n_audio_ctx()
    );
    Ok(())
}
