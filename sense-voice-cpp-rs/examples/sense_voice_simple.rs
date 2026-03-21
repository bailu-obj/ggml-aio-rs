#![allow(missing_docs)]

use std::env;

use sense_voice_cpp_rs::{SenseVoiceContext, SenseVoiceContextParameters};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = match env::var("SENSE_VOICE_MODEL_PATH") {
        Ok(path) => path,
        Err(_) => {
            eprintln!("Set SENSE_VOICE_MODEL_PATH to run this example.");
            return Ok(());
        }
    };

    let params = SenseVoiceContextParameters::default();
    let _context = SenseVoiceContext::new_with_params(&model_path, params)?;

    println!("Loaded SenseVoice model successfully.");
    Ok(())
}
