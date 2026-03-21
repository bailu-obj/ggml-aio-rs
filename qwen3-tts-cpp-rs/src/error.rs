use std::ffi::NulError;

/// Errors from the Qwen3-TTS C API wrapper.
#[derive(Debug, Clone)]
pub enum Qwen3TtsError {
    /// Failed to load models from the given directory.
    InitFailed {
        /// Detail from the loader or a fallback message.
        message: String,
    },
    /// Synthesis or embedding extraction returned no audio / failure.
    OperationFailed {
        /// Detail from the engine or a fallback message.
        message: String,
    },
    /// A null byte was present in an input string.
    NullByteInString {
        /// Byte index of the NUL.
        idx: usize,
    },
}

impl From<NulError> for Qwen3TtsError {
    fn from(e: NulError) -> Self {
        Self::NullByteInString {
            idx: e.nul_position(),
        }
    }
}

impl std::fmt::Display for Qwen3TtsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Qwen3TtsError::InitFailed { message } => {
                write!(f, "failed to load Qwen3-TTS models: {message}")
            }
            Qwen3TtsError::OperationFailed { message } => {
                write!(f, "qwen3-tts operation failed: {message}")
            }
            Qwen3TtsError::NullByteInString { idx } => {
                write!(f, "null byte in input string at index {idx}")
            }
        }
    }
}

impl std::error::Error for Qwen3TtsError {}
