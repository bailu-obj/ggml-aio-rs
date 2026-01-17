//
// Created by lovemefan on 2024/7/19.
//

#ifndef SENSEVOICE_CPP_SENSE_VOICE_H
#define SENSEVOICE_CPP_SENSE_VOICE_H

#include <ggml-cpu.h>
#include <gguf.h>


#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __GNUC__
#define SENSE_VOICE_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#define SENSE_VOICE_DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#define SENSE_VOICE_DEPRECATED(func, hint) func
#endif

#ifdef SENSE_VOICE_SHARED
#ifdef _WIN32
#ifdef SENSE_VOICE_BUILD
#define SENSE_VOICE_API __declspec(dllexport)
#else
#define SENSE_VOICE_API __declspec(dllimport)
#endif
#else
#define SENSE_VOICE_API __attribute__((visibility("default")))
#endif
#else
#define SENSE_VOICE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct sense_voice_context;
struct sense_voice_state;

struct sense_voice_context_params {
    bool use_gpu;
    bool use_itn;
    bool flash_attn;
    int gpu_device;// CUDA device
    ggml_backend_sched_eval_callback cb_eval;
    void *cb_eval_user_data;
};


// Progress callback
typedef void (*sense_voice_progress_callback)(struct sense_voice_context *ctx,
                                              struct sense_voice_state *state,
                                              int progress, void *user_data);

// Available sampling strategies
enum sense_voice_decoding_strategy {
    SENSE_VOICE_SAMPLING_GREEDY,
    SENSE_VOICE_SAMPLING_BEAM_SEARCH,
};

struct sense_voice_full_params {
    enum sense_voice_decoding_strategy strategy;
    int n_threads;
    const char *language;
    int n_max_text_ctx;// max tokens to use from past text as prompt for the
                       // decoder
    int offset_ms;     // start offset in ms
    int duration_ms;   // audio duration to process in ms

    bool no_timestamps;   // do not generate timestamps
    bool single_segment;  // force single segment output (useful for streaming)
    bool print_progress;  // print progress information
    bool print_timestamps;// print timestamps for each text segment when
                          // printing realtime

    bool debug_mode;// enable debug_mode provides extra info (eg. Dump log_mel)
    int audio_ctx;

    struct {
        int best_of;
    } greedy;

    struct {
        int beam_size;
    } beam_search;

    // called on each progress update
    sense_voice_progress_callback progress_callback;
    void *progress_callback_user_data;
};


SENSE_VOICE_API int sense_voice_lang_id(const char *lang);
SENSE_VOICE_API const char *sense_voice_lang_str(int id);
SENSE_VOICE_API struct sense_voice_context_params sense_voice_context_default_params();
SENSE_VOICE_API struct sense_voice_context *sense_voice_small_init_from_file_with_params(const char *path_model, struct sense_voice_context_params params);
SENSE_VOICE_API struct sense_voice_context *sense_voice_small_init_from_file_with_params_no_state(const char *path_model, struct sense_voice_context_params params);
SENSE_VOICE_API struct sense_voice_context *sense_voice_init_with_params_no_state(const char *path_model, struct sense_voice_context_params params);
SENSE_VOICE_API int sense_voice_full_parallel(struct sense_voice_context *ctx,
                                              const struct sense_voice_full_params *params,
                                              const double *samples,
                                              int n_samples,
                                              int n_processors);
SENSE_VOICE_API const char *sense_voice_full_get_text(struct sense_voice_context *ctx, bool need_prefix);
SENSE_VOICE_API void sense_voice_reset_ctx_state(struct sense_voice_context *ctx);
#ifdef __cplusplus
}
#endif

#endif//SENSEVOICE_CPP_SENSE_VOICE_H
