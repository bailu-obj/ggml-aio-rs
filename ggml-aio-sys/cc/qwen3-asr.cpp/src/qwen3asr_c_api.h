#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Qwen3Asr Qwen3Asr;

typedef struct Qwen3AsrParams {
    int32_t max_tokens;
    int32_t n_threads;
    bool print_progress;
    bool print_timing;
    const char * language;
} Qwen3AsrParams;

typedef struct Qwen3AsrResult {
    const char * text;
    const char * language;
    int32_t * tokens;
    int32_t n_tokens;
    bool success;
    const char * error_msg;
    int64_t t_load_ms;
    int64_t t_mel_ms;
    int64_t t_encode_ms;
    int64_t t_decode_ms;
    int64_t t_total_ms;
} Qwen3AsrResult;

void qwen3_asr_default_params(Qwen3AsrParams * out_params);
Qwen3Asr * qwen3_asr_create(const char * model_path);
void qwen3_asr_destroy(Qwen3Asr * asr);
bool qwen3_asr_is_loaded(const Qwen3Asr * asr);
const char * qwen3_asr_get_error(const Qwen3Asr * asr);
Qwen3AsrResult * qwen3_asr_transcribe_file(
    Qwen3Asr * asr,
    const char * audio_path,
    const Qwen3AsrParams * params);
Qwen3AsrResult * qwen3_asr_transcribe_samples(
    Qwen3Asr * asr,
    const float * samples,
    int32_t n_samples,
    const Qwen3AsrParams * params);
void qwen3_asr_free_result(Qwen3AsrResult * result);

#ifdef __cplusplus
}
#endif
