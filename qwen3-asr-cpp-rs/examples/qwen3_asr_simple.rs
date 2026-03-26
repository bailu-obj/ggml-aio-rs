use std::env;

use qwen3_asr_cpp_rs::{Qwen3Asr, Qwen3AsrParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = match env::var("QWEN3_ASR_MODEL_PATH") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("Set QWEN3_ASR_MODEL_PATH to run this example.");
            return Ok(());
        }
    };
    let audio_path = match env::var("QWEN3_ASR_AUDIO_PATH") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("Set QWEN3_ASR_AUDIO_PATH (16kHz mono WAV) to run this example.");
            return Ok(());
        }
    };

    let mut asr = Qwen3Asr::new(&model_path)?;
    let params = Qwen3AsrParams::default().n_threads(4).max_tokens(1024);
    let result = asr.transcribe_file(&audio_path, &params)?;

    println!("{}", result.text);
    Ok(())
}
