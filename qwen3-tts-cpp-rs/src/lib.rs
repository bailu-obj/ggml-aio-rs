//! Safe Rust wrapper around the `qwen3-tts.cpp` C API exposed via [`ggml_aio_sys`].
//!
//! Point [`Qwen3Tts::new`] at a directory containing `qwen3-tts-0.6b-f16.gguf` (or
//! `qwen3-tts-0.6b-q8_0.gguf`) and `qwen3-tts-tokenizer-f16.gguf`.
//!
//! ## Long runs and audio length
//!
//! Generation stops at the codec **EOS** token or when **`max_audio_tokens`** frames are reached
//! (default **4096**, same as upstream). Using **`temperature = 0`** (greedy) often keeps the
//! model away from EOS until the cap, so you get **near-max-length output** and **much more CPU
//! time** than the default **`temperature = 0.9`** used by the upstream CLI. Prefer the default
//! params unless you know you need greedy decoding.
//!
//! On **macOS**, when `models/coreml/code_predictor.mlpackage` is present, the native CoreML
//! bridge matches upstream `qwen3-tts-cli` performance. Set **`QWEN3_TTS_USE_COREML=0`** only if
//! you need the pure-GGML code predictor (e.g. debugging). On non-Apple targets the CoreML stub
//! is used instead (much slower at full `max_audio_tokens`).
//!
//! Cap frames via [`Qwen3TtsParams::max_audio_tokens`] for quick tests on slow paths.

mod error;

use std::ffi::{CStr, CString};
use std::ptr;

pub use error::Qwen3TtsError;

/// Synthesized PCM audio (mono `f32`, typically 24 kHz).
#[derive(Debug, Clone)]
pub struct Qwen3TtsAudio {
    /// Sample values in `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
    /// Sample rate in Hz (library uses 24_000).
    pub sample_rate: i32,
}

/// Generation parameters (mirrors `Qwen3TtsParams` in the C header).
#[derive(Debug, Clone, Copy)]
pub struct Qwen3TtsParams {
    inner: ggml_aio_sys::Qwen3TtsParams,
}

impl Default for Qwen3TtsParams {
    fn default() -> Self {
        let mut inner = std::mem::MaybeUninit::<ggml_aio_sys::Qwen3TtsParams>::uninit();
        unsafe {
            ggml_aio_sys::qwen3_tts_default_params(inner.as_mut_ptr());
            Self {
                inner: inner.assume_init(),
            }
        }
    }
}

impl Qwen3TtsParams {
    /// Maximum audio tokens to generate.
    pub fn max_audio_tokens(mut self, v: i32) -> Self {
        self.inner.max_audio_tokens = v;
        self
    }
    /// Sampling temperature (`0` = greedy).
    pub fn temperature(mut self, v: f32) -> Self {
        self.inner.temperature = v;
        self
    }
    /// Nucleus sampling `top_p`.
    pub fn top_p(mut self, v: f32) -> Self {
        self.inner.top_p = v;
        self
    }
    /// Top-k (`0` = disabled).
    pub fn top_k(mut self, v: i32) -> Self {
        self.inner.top_k = v;
        self
    }
    /// Thread count for GGML compute.
    pub fn n_threads(mut self, v: i32) -> Self {
        self.inner.n_threads = v;
        self
    }
    /// Repetition penalty for codebook-0 tokens.
    pub fn repetition_penalty(mut self, v: f32) -> Self {
        self.inner.repetition_penalty = v;
        self
    }
    /// Codec language id (e.g. `2050` = English).
    pub fn language_id(mut self, v: i32) -> Self {
        self.inner.language_id = v;
        self
    }
}

/// TTS engine: load models once, then synthesize.
pub struct Qwen3Tts {
    ptr: *mut ggml_aio_sys::Qwen3Tts,
}

impl Drop for Qwen3Tts {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ggml_aio_sys::qwen3_tts_destroy(self.ptr);
            }
        }
        self.ptr = ptr::null_mut();
    }
}

impl Qwen3Tts {
    /// Load models from `model_dir` (must contain the expected `.gguf` files).
    pub fn new(model_dir: &str, n_threads: i32) -> Result<Self, Qwen3TtsError> {
        let dir = CString::new(model_dir)?;
        let ptr = unsafe { ggml_aio_sys::qwen3_tts_create(dir.as_ptr(), n_threads) };
        if ptr.is_null() {
            let message = unsafe {
                let p = ggml_aio_sys::qwen3_tts_last_init_error();
                if p.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            };
            return Err(Qwen3TtsError::InitFailed {
                message: if message.is_empty() {
                    "qwen3_tts_create returned null (no error message; see stderr)".into()
                } else {
                    message
                },
            });
        }
        Ok(Self { ptr })
    }

    fn last_error_message(&self) -> String {
        unsafe {
            let p = ggml_aio_sys::qwen3_tts_get_error(self.ptr);
            if p.is_null() {
                return String::new();
            }
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }

    /// Whether models are loaded.
    pub fn is_loaded(&self) -> bool {
        unsafe { ggml_aio_sys::qwen3_tts_is_loaded(self.ptr) != 0 }
    }

    /// Nominal output sample rate (24_000).
    pub fn sample_rate(&self) -> i32 {
        unsafe { ggml_aio_sys::qwen3_tts_sample_rate(self.ptr) }
    }

    /// Basic text-to-speech.
    pub fn synthesize(
        &mut self,
        text: &str,
        params: &Qwen3TtsParams,
    ) -> Result<Qwen3TtsAudio, Qwen3TtsError> {
        let text = CString::new(text)?;
        let audio = unsafe {
            ggml_aio_sys::qwen3_tts_synthesize(self.ptr, text.as_ptr(), &params.inner)
        };
        self.audio_from_ptr(audio)
    }

    /// Voice cloning from a reference WAV path (24 kHz mono recommended).
    pub fn synthesize_with_voice_file(
        &mut self,
        text: &str,
        reference_wav: &str,
        params: &Qwen3TtsParams,
    ) -> Result<Qwen3TtsAudio, Qwen3TtsError> {
        let text = CString::new(text)?;
        let path = CString::new(reference_wav)?;
        let audio = unsafe {
            ggml_aio_sys::qwen3_tts_synthesize_with_voice_file(
                self.ptr,
                text.as_ptr(),
                path.as_ptr(),
                &params.inner,
            )
        };
        self.audio_from_ptr(audio)
    }

    /// Voice cloning from raw PCM samples (24 kHz mono, `f32` in `[-1, 1]`).
    pub fn synthesize_with_voice_samples(
        &mut self,
        text: &str,
        ref_samples: &[f32],
        params: &Qwen3TtsParams,
    ) -> Result<Qwen3TtsAudio, Qwen3TtsError> {
        let text = CString::new(text)?;
        let audio = unsafe {
            ggml_aio_sys::qwen3_tts_synthesize_with_voice_samples(
                self.ptr,
                text.as_ptr(),
                ref_samples.as_ptr(),
                ref_samples.len().try_into().unwrap_or(i32::MAX),
                &params.inner,
            )
        };
        self.audio_from_ptr(audio)
    }

    /// Speaker embedding from a WAV file; returns float count written.
    pub fn extract_embedding_file(
        &mut self,
        reference_wav: &str,
        out: &mut [f32],
    ) -> Result<i32, Qwen3TtsError> {
        let path = CString::new(reference_wav)?;
        let n = unsafe {
            ggml_aio_sys::qwen3_tts_extract_embedding_file(
                self.ptr,
                path.as_ptr(),
                out.as_mut_ptr(),
                out.len().try_into().unwrap_or(i32::MAX),
            )
        };
        if n < 0 {
            return Err(Qwen3TtsError::OperationFailed {
                message: self.last_error_message(),
            });
        }
        Ok(n)
    }

    /// Synthesize using a precomputed embedding from [`Self::extract_embedding_file`].
    pub fn synthesize_with_embedding(
        &mut self,
        text: &str,
        embedding: &[f32],
        params: &Qwen3TtsParams,
    ) -> Result<Qwen3TtsAudio, Qwen3TtsError> {
        let text = CString::new(text)?;
        let audio = unsafe {
            ggml_aio_sys::qwen3_tts_synthesize_with_embedding(
                self.ptr,
                text.as_ptr(),
                embedding.as_ptr(),
                embedding.len().try_into().unwrap_or(i32::MAX),
                &params.inner,
            )
        };
        self.audio_from_ptr(audio)
    }

    fn audio_from_ptr(
        &mut self,
        audio: *mut ggml_aio_sys::Qwen3TtsAudio,
    ) -> Result<Qwen3TtsAudio, Qwen3TtsError> {
        if audio.is_null() {
            return Err(Qwen3TtsError::OperationFailed {
                message: self.last_error_message(),
            });
        }
        unsafe {
            let n = (*audio).n_samples as usize;
            let sr = (*audio).sample_rate;
            let slice = std::slice::from_raw_parts((*audio).samples, n);
            let samples = slice.to_vec();
            ggml_aio_sys::qwen3_tts_free_audio(audio);
            Ok(Qwen3TtsAudio {
                samples,
                sample_rate: sr,
            })
        }
    }
}

// The underlying library uses GGML backends (Metal, etc.); treat like other crates in this repo.
unsafe impl Send for Qwen3Tts {}
unsafe impl Sync for Qwen3Tts {}
