use thiserror::Error;

#[derive(Debug, Error)]
pub enum Qwen3AsrError {
    #[error("input string contains interior NUL byte: {0}")]
    Nul(#[from] std::ffi::NulError),
    #[error("failed to initialize qwen3-asr context: {message}")]
    InitFailed { message: String },
    #[error("qwen3-asr operation failed: {message}")]
    OperationFailed { message: String },
}
