#include "qwen3asr_c_api.h"

#include "../include/qwen3_asr.h"

#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

struct Qwen3Asr {
    qwen3_asr::Qwen3ASR inner;
    std::string last_error;
};

static char * c_strdup(const std::string & s) {
    char * p = static_cast<char *>(std::malloc(s.size() + 1));
    if (!p) {
        return nullptr;
    }
    std::memcpy(p, s.c_str(), s.size() + 1);
    return p;
}

static Qwen3AsrResult * alloc_result() {
    Qwen3AsrResult * out = static_cast<Qwen3AsrResult *>(std::calloc(1, sizeof(Qwen3AsrResult)));
    return out;
}

static Qwen3AsrResult * make_error_result(const std::string & msg) {
    Qwen3AsrResult * out = alloc_result();
    if (!out) {
        return nullptr;
    }
    out->success = false;
    out->error_msg = c_strdup(msg);
    if (!out->error_msg) {
        std::free(out);
        return nullptr;
    }
    return out;
}

static qwen3_asr::transcribe_params to_cpp_params(const Qwen3AsrParams * params) {
    qwen3_asr::transcribe_params p;
    if (!params) {
        return p;
    }
    p.max_tokens = params->max_tokens;
    p.n_threads = params->n_threads;
    p.print_progress = params->print_progress;
    p.print_timing = params->print_timing;
    if (params->language) {
        p.language = params->language;
    }
    return p;
}

static Qwen3AsrResult * convert_result(const qwen3_asr::transcribe_result & src) {
    Qwen3AsrResult * out = alloc_result();
    if (!out) {
        return nullptr;
    }

    out->success = src.success;
    out->t_load_ms = src.t_load_ms;
    out->t_mel_ms = src.t_mel_ms;
    out->t_encode_ms = src.t_encode_ms;
    out->t_decode_ms = src.t_decode_ms;
    out->t_total_ms = src.t_total_ms;
    out->n_tokens = static_cast<int32_t>(src.tokens.size());

    out->text = c_strdup(src.text);
    out->language = c_strdup(src.language);
    out->error_msg = c_strdup(src.error_msg);

    if (!src.tokens.empty()) {
        size_t bytes = src.tokens.size() * sizeof(int32_t);
        out->tokens = static_cast<int32_t *>(std::malloc(bytes));
        if (!out->tokens) {
            qwen3_asr_free_result(out);
            return nullptr;
        }
        std::memcpy(out->tokens, src.tokens.data(), bytes);
    }

    return out;
}

void qwen3_asr_default_params(Qwen3AsrParams * out_params) {
    if (!out_params) {
        return;
    }
    out_params->max_tokens = 1024;
    out_params->n_threads = 4;
    out_params->print_progress = false;
    out_params->print_timing = true;
    out_params->language = nullptr;
}

Qwen3Asr * qwen3_asr_create(const char * model_path) {
    if (!model_path) {
        return nullptr;
    }
    Qwen3Asr * ctx = new (std::nothrow) Qwen3Asr();
    if (!ctx) {
        return nullptr;
    }
    if (!ctx->inner.load_model(model_path)) {
        ctx->last_error = ctx->inner.get_error();
        return ctx;
    }
    ctx->last_error.clear();
    return ctx;
}

void qwen3_asr_destroy(Qwen3Asr * asr) {
    delete asr;
}

bool qwen3_asr_is_loaded(const Qwen3Asr * asr) {
    return asr ? asr->inner.is_loaded() : false;
}

const char * qwen3_asr_get_error(const Qwen3Asr * asr) {
    if (!asr) {
        return "";
    }
    if (!asr->last_error.empty()) {
        return asr->last_error.c_str();
    }
    return asr->inner.get_error().c_str();
}

Qwen3AsrResult * qwen3_asr_transcribe_file(
    Qwen3Asr * asr,
    const char * audio_path,
    const Qwen3AsrParams * params) {
    if (!asr) {
        return make_error_result("qwen3_asr_transcribe_file: null context");
    }
    if (!audio_path) {
        return make_error_result("qwen3_asr_transcribe_file: null audio_path");
    }

    auto result = asr->inner.transcribe(audio_path, to_cpp_params(params));
    asr->last_error = result.error_msg;
    Qwen3AsrResult * out = convert_result(result);
    if (!out) {
        return make_error_result("qwen3_asr_transcribe_file: allocation failure");
    }
    return out;
}

Qwen3AsrResult * qwen3_asr_transcribe_samples(
    Qwen3Asr * asr,
    const float * samples,
    int32_t n_samples,
    const Qwen3AsrParams * params) {
    if (!asr) {
        return make_error_result("qwen3_asr_transcribe_samples: null context");
    }
    if (!samples || n_samples <= 0) {
        return make_error_result("qwen3_asr_transcribe_samples: invalid samples");
    }

    auto result = asr->inner.transcribe(samples, n_samples, to_cpp_params(params));
    asr->last_error = result.error_msg;
    Qwen3AsrResult * out = convert_result(result);
    if (!out) {
        return make_error_result("qwen3_asr_transcribe_samples: allocation failure");
    }
    return out;
}

void qwen3_asr_free_result(Qwen3AsrResult * result) {
    if (!result) {
        return;
    }
    std::free(const_cast<char *>(result->text));
    std::free(const_cast<char *>(result->language));
    std::free(result->tokens);
    std::free(const_cast<char *>(result->error_msg));
    std::free(result);
}
