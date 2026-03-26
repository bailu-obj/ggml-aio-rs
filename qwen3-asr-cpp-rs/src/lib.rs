//! Safe Rust wrapper around the `qwen3-asr.cpp` C API exposed via [`ggml_aio_sys`].
//!
//! Point [`Qwen3Asr::new`] at a `qwen3-asr-*.gguf` model and call [`Qwen3Asr::transcribe_file`]
//! with a 16 kHz mono WAV path.

mod error;

use std::ffi::{CStr, CString};
use std::ptr;

pub use error::Qwen3AsrError;

#[derive(Debug, Clone)]
pub struct Qwen3AsrParams {
    pub max_tokens: i32,
    pub n_threads: i32,
    pub print_progress: bool,
    pub print_timing: bool,
    pub language: Option<String>,
}

impl Qwen3AsrParams {
    fn from_native_defaults() -> Self {
        let mut p = std::mem::MaybeUninit::<ggml_aio_sys::Qwen3AsrParams>::uninit();
        unsafe {
            ggml_aio_sys::qwen3_asr_default_params(p.as_mut_ptr());
            let p = p.assume_init();
            Self {
                max_tokens: p.max_tokens,
                n_threads: p.n_threads,
                print_progress: p.print_progress,
                print_timing: p.print_timing,
                language: None,
            }
        }
    }

    pub fn max_tokens(mut self, v: i32) -> Self {
        self.max_tokens = v;
        self
    }

    pub fn n_threads(mut self, v: i32) -> Self {
        self.n_threads = v;
        self
    }

    pub fn print_progress(mut self, v: bool) -> Self {
        self.print_progress = v;
        self
    }

    pub fn print_timing(mut self, v: bool) -> Self {
        self.print_timing = v;
        self
    }

    pub fn language(mut self, v: impl Into<String>) -> Self {
        self.language = Some(v.into());
        self
    }
}

impl Default for Qwen3AsrParams {
    fn default() -> Self {
        Self::from_native_defaults()
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3AsrResult {
    pub text: String,
    pub language: String,
    pub tokens: Vec<i32>,
    pub success: bool,
    pub error_msg: String,
    pub t_load_ms: i64,
    pub t_mel_ms: i64,
    pub t_encode_ms: i64,
    pub t_decode_ms: i64,
    pub t_total_ms: i64,
}

pub struct Qwen3Asr {
    ptr: *mut ggml_aio_sys::Qwen3Asr,
}

impl Drop for Qwen3Asr {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ggml_aio_sys::qwen3_asr_destroy(self.ptr);
            }
        }
        self.ptr = ptr::null_mut();
    }
}

impl Qwen3Asr {
    pub fn new(model_path: &str) -> Result<Self, Qwen3AsrError> {
        let model_path = CString::new(model_path)?;
        let ptr = unsafe { ggml_aio_sys::qwen3_asr_create(model_path.as_ptr()) };
        if ptr.is_null() {
            return Err(Qwen3AsrError::InitFailed {
                message: "qwen3_asr_create returned null".to_string(),
            });
        }
        let ok = unsafe { ggml_aio_sys::qwen3_asr_is_loaded(ptr) };
        if !ok {
            let message = unsafe {
                let p = ggml_aio_sys::qwen3_asr_get_error(ptr);
                if p.is_null() {
                    "unknown initialization error".to_string()
                } else {
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            };
            unsafe {
                ggml_aio_sys::qwen3_asr_destroy(ptr);
            }
            return Err(Qwen3AsrError::InitFailed { message });
        }
        Ok(Self { ptr })
    }

    pub fn transcribe_file(
        &mut self,
        audio_path: &str,
        params: &Qwen3AsrParams,
    ) -> Result<Qwen3AsrResult, Qwen3AsrError> {
        let audio_path = CString::new(audio_path)?;
        let language_c = match &params.language {
            Some(s) => Some(CString::new(s.as_str())?),
            None => None,
        };
        let native_params = ggml_aio_sys::Qwen3AsrParams {
            max_tokens: params.max_tokens,
            n_threads: params.n_threads,
            print_progress: params.print_progress,
            print_timing: params.print_timing,
            language: language_c.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
        };
        let result = unsafe {
            ggml_aio_sys::qwen3_asr_transcribe_file(self.ptr, audio_path.as_ptr(), &native_params)
        };
        self.convert_result(result)
    }

    fn convert_result(
        &mut self,
        result: *mut ggml_aio_sys::Qwen3AsrResult,
    ) -> Result<Qwen3AsrResult, Qwen3AsrError> {
        if result.is_null() {
            let message = unsafe {
                let p = ggml_aio_sys::qwen3_asr_get_error(self.ptr);
                if p.is_null() {
                    "null transcription result".to_string()
                } else {
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            };
            return Err(Qwen3AsrError::OperationFailed { message });
        }

        unsafe {
            let r = &*result;
            let text = if r.text.is_null() {
                String::new()
            } else {
                CStr::from_ptr(r.text).to_string_lossy().into_owned()
            };
            let language = if r.language.is_null() {
                String::new()
            } else {
                CStr::from_ptr(r.language).to_string_lossy().into_owned()
            };
            let error_msg = if r.error_msg.is_null() {
                String::new()
            } else {
                CStr::from_ptr(r.error_msg).to_string_lossy().into_owned()
            };
            let tokens = if r.tokens.is_null() || r.n_tokens <= 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(r.tokens, r.n_tokens as usize).to_vec()
            };
            let out = Qwen3AsrResult {
                text,
                language,
                tokens,
                success: r.success,
                error_msg,
                t_load_ms: r.t_load_ms,
                t_mel_ms: r.t_mel_ms,
                t_encode_ms: r.t_encode_ms,
                t_decode_ms: r.t_decode_ms,
                t_total_ms: r.t_total_ms,
            };
            ggml_aio_sys::qwen3_asr_free_result(result);
            if out.success {
                Ok(out)
            } else {
                Err(Qwen3AsrError::OperationFailed {
                    message: if out.error_msg.is_empty() {
                        "transcription failed".to_string()
                    } else {
                        out.error_msg.clone()
                    },
                })
            }
        }
    }
}

unsafe impl Send for Qwen3Asr {}
unsafe impl Sync for Qwen3Asr {}
