use std::{
    ffi::{CStr, CString, c_int},
    ptr::null_mut,
    str::FromStr,
    thread,
};

use ggml_aio_sys::{
    sense_voice_full_params, sense_voice_full_params__bindgen_ty_1,
    sense_voice_full_params__bindgen_ty_2,
};

use crate::error::SenseVoiceError;

pub mod error;

// following implementations are safe
// see https://github.com/ggerganov/whisper.cpp/issues/32#issuecomment-1272790388
unsafe impl Send for SenseVoiceContext {}
unsafe impl Sync for SenseVoiceContext {}

pub struct SenseVoiceContextParameters {
    /// Use GPU if available.
    pub use_gpu: bool,

    pub use_itn: bool,
    /// Enable flash attention, default false
    ///
    /// **Warning** Can't be used with DTW. DTW will be disabled if flash_attn is true
    pub flash_attn: bool,
    /// GPU device id, default 0
    pub gpu_device: c_int,
}
impl SenseVoiceContextParameters {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn use_gpu(&mut self, use_gpu: bool) -> &mut Self {
        self.use_gpu = use_gpu;
        self
    }
    pub fn flash_attn(&mut self, flash_attn: bool) -> &mut Self {
        self.flash_attn = flash_attn;
        self
    }
    pub fn gpu_device(&mut self, gpu_device: c_int) -> &mut Self {
        self.gpu_device = gpu_device;
        self
    }

    fn to_c_struct(&self) -> ggml_aio_sys::sense_voice_context_params {
        ggml_aio_sys::sense_voice_context_params {
            use_gpu: self.use_gpu,
            use_itn: self.use_itn,
            flash_attn: self.flash_attn,
            gpu_device: self.gpu_device,
            cb_eval: None,
            cb_eval_user_data: std::ptr::null_mut(),
        }
    }
}

#[derive(Debug)]
pub struct SenseVoiceContext {
    pub(crate) ctx: *mut ggml_aio_sys::sense_voice_context,
}

impl SenseVoiceContext {
    /// Create a new SenseVoiceContext from a file, with parameters.
    ///
    /// # Arguments
    /// * path: The path to the model file.
    /// * parameters: A parameter struct containing the parameters to use.
    ///
    /// # Returns
    /// Ok(Self) on success, Err(SenseVoiceError) on failure.
    ///
    /// # C++ equivalent
    /// `struct whisper_context * sense_voice_small_init_from_file_with_params(const char * path_model, struct whisper_context_params params);`
    pub fn new_with_params(
        path: &str,
        parameters: SenseVoiceContextParameters,
    ) -> Result<Self, SenseVoiceError> {
        let path_cstr = CString::new(path)?;
        let ctx = unsafe {
            ggml_aio_sys::sense_voice_small_init_from_file_with_params(
                path_cstr.as_ptr(),
                parameters.to_c_struct(),
            )
        };
        if ctx.is_null() {
            Err(SenseVoiceError::InitError)
        } else {
            Ok(Self { ctx })
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum SenseVoiceDecodingStrategy {
    SamplingGreedy,
    SamplingBeamSearch,
}

#[derive(Clone)]
pub struct SenseVoiceFullParams {
    pub strategy: SenseVoiceDecodingStrategy,
    pub n_threads: i32,
    pub language: String,
    pub n_max_text_ctx: i32,
    pub offset_ms: i32,
    pub duration_ms: i32,
    pub no_timestamps: bool,
    pub single_segment: bool,
    pub print_progress: bool,
    pub print_timestamps: bool,
    pub debug_mode: bool,
    pub audio_ctx: i32,
    pub greedy: GreedyParams,
    pub beam_search: BeamSearchParams,
}

#[derive(Clone)]
pub struct GreedyParams {
    pub best_of: i32,
}

#[derive(Clone)]
pub struct BeamSearchParams {
    pub beam_size: i32,
}

impl SenseVoiceFullParams {
    pub fn builder(strategy: SenseVoiceDecodingStrategy) -> SenseVoiceFullParamsBuilder {
        SenseVoiceFullParamsBuilder::new(strategy)
    }

    pub fn default_params(strategy: SenseVoiceDecodingStrategy) -> Self {
        SenseVoiceFullParamsBuilder::new(strategy).build()
    }

    pub fn to_c_struct(&self) -> sense_voice_full_params {
        let c_language =
            CString::new(self.language.as_str()).expect("Failed to convert language to C string");

        let c_strategy = self.strategy as u32;

        let c_struct = sense_voice_full_params {
            strategy: c_strategy,
            n_threads: self.n_threads,
            language: c_language.as_ptr(),
            n_max_text_ctx: self.n_max_text_ctx,
            offset_ms: self.offset_ms,
            duration_ms: self.offset_ms,
            no_timestamps: self.no_timestamps,
            single_segment: self.single_segment,
            print_progress: self.print_progress,
            print_timestamps: self.print_timestamps,
            debug_mode: self.debug_mode,
            audio_ctx: self.audio_ctx,
            greedy: sense_voice_full_params__bindgen_ty_1 {
                best_of: self.greedy.best_of,
            },
            beam_search: sense_voice_full_params__bindgen_ty_2 {
                beam_size: self.beam_search.beam_size,
            },
            progress_callback: None,
            progress_callback_user_data: null_mut(),
        };

        // Return both the C struct and the CString to keep it alive
        c_struct
    }
}

pub struct SenseVoiceFullParamsBuilder {
    params: SenseVoiceFullParams,
}

impl SenseVoiceFullParamsBuilder {
    pub fn new(strategy: SenseVoiceDecodingStrategy) -> Self {
        let mut params = SenseVoiceFullParams {
            strategy,
            n_threads: std::cmp::min(
                4,
                thread::available_parallelism().map_or(4, |n| n.get() as i32),
            ),
            language: "auto".to_string(),
            n_max_text_ctx: 16384,
            offset_ms: 0,
            duration_ms: 0,
            no_timestamps: false,
            single_segment: true,
            print_progress: true,
            print_timestamps: true,
            debug_mode: false,
            audio_ctx: 0,
            greedy: GreedyParams { best_of: -1 },
            beam_search: BeamSearchParams { beam_size: -1 },
        };

        // Set strategy-specific defaults
        match strategy {
            SenseVoiceDecodingStrategy::SamplingGreedy => {
                params.greedy.best_of = 5;
            }
            SenseVoiceDecodingStrategy::SamplingBeamSearch => {
                params.beam_search.beam_size = 5;
            }
        }

        Self { params }
    }

    pub fn n_threads(mut self, n_threads: i32) -> Self {
        self.params.n_threads = n_threads;
        self
    }

    pub fn language(mut self, language: &str) -> Self {
        self.params.language = language.to_string();
        self
    }

    pub fn n_max_text_ctx(mut self, n_max_text_ctx: i32) -> Self {
        self.params.n_max_text_ctx = n_max_text_ctx;
        self
    }

    pub fn offset_ms(mut self, offset_ms: i32) -> Self {
        self.params.offset_ms = offset_ms;
        self
    }

    pub fn duration_ms(mut self, duration_ms: i32) -> Self {
        self.params.duration_ms = duration_ms;
        self
    }

    pub fn no_timestamps(mut self, no_timestamps: bool) -> Self {
        self.params.no_timestamps = no_timestamps;
        self
    }

    pub fn single_segment(mut self, single_segment: bool) -> Self {
        self.params.single_segment = single_segment;
        self
    }

    pub fn print_progress(mut self, print_progress: bool) -> Self {
        self.params.print_progress = print_progress;
        self
    }

    pub fn print_timestamps(mut self, print_timestamps: bool) -> Self {
        self.params.print_timestamps = print_timestamps;
        self
    }

    pub fn debug_mode(mut self, debug_mode: bool) -> Self {
        self.params.debug_mode = debug_mode;
        self
    }

    pub fn audio_ctx(mut self, audio_ctx: i32) -> Self {
        self.params.audio_ctx = audio_ctx;
        self
    }

    pub fn greedy_best_of(mut self, best_of: i32) -> Self {
        self.params.greedy.best_of = best_of;
        self
    }

    pub fn beam_search_beam_size(mut self, beam_size: i32) -> Self {
        self.params.beam_search.beam_size = beam_size;
        self
    }
    pub fn build(self) -> SenseVoiceFullParams {
        self.params
    }
}

pub fn get_speech_prob(ctx: &mut SenseVoiceContext, data: &[f64]) -> f32 {
    if data.is_empty() {
        return -1.0f32;
    }
    let ret = unsafe {
        ggml_aio_sys::sense_voice_get_speech_prob(ctx.ctx, data.as_ptr(), data.len() as c_int, 8)
    };
    ret
}

pub fn full_parallel(
    ctx: &mut SenseVoiceContext,
    params: SenseVoiceFullParams,
    data: &[f64],
) -> Result<c_int, SenseVoiceError> {
    if data.is_empty() {
        // can randomly trigger segmentation faults if we don't check this
        return Err(SenseVoiceError::NoSamples);
    }

    let ret = unsafe {
        ggml_aio_sys::sense_voice_full_parallel(
            ctx.ctx,
            &params.to_c_struct(),
            data.as_ptr(),
            data.len() as c_int,
            8,
        )
    };
    if ret == -1 {
        Err(SenseVoiceError::UnableToCalculateSpectrogram)
    } else if ret == 7 {
        Err(SenseVoiceError::FailedToEncode)
    } else if ret == 8 {
        Err(SenseVoiceError::FailedToDecode)
    } else if ret == 0 {
        Ok(ret)
    } else {
        Err(SenseVoiceError::GenericError(ret))
    }
}

#[allow(clippy::derivable_impls)] // this impl cannot be derived
impl Default for SenseVoiceContextParameters {
    fn default() -> Self {
        Self {
            use_gpu: cfg!(feature = "_gpu"),
            use_itn: false,
            flash_attn: false,
            gpu_device: 0,
        }
    }
}

pub fn full_get_text(
    ctx: &mut SenseVoiceContext,
    need_prefix: bool,
) -> Result<String, SenseVoiceError> {
    let ret = unsafe { ggml_aio_sys::sense_voice_full_get_text(ctx.ctx, need_prefix) };
    if ret.is_null() {
        return Err(SenseVoiceError::NullPointer);
    }
    unsafe { Ok(String::from_str(CStr::from_ptr(ret).to_str().unwrap()).unwrap()) }
}

pub fn reset_ctx_state(ctx: &mut SenseVoiceContext) {
    unsafe { ggml_aio_sys::sense_voice_reset_ctx_state(ctx.ctx) };
}
