#![allow(missing_docs)]

use std::env;

use llama_cpp_rs::context::params::LlamaContextParams;
use llama_cpp_rs::llama_backend::LlamaBackend;
use llama_cpp_rs::model::LlamaModel;
use llama_cpp_rs::model::params::LlamaModelParams;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = match env::var("LLAMA_MODEL_PATH") {
        Ok(path) => path,
        Err(_) => {
            eprintln!("Set LLAMA_MODEL_PATH to run this example.");
            return Ok(());
        }
    };

    let backend = LlamaBackend::init()?;
    let model = LlamaModel::load_from_file(&backend, model_path, &LlamaModelParams::default())?;
    let context = model.new_context(&backend, LlamaContextParams::default())?;

    println!(
        "Loaded model successfully. n_ctx={}, n_batch={}",
        context.n_ctx(),
        context.n_batch()
    );
    Ok(())
}
