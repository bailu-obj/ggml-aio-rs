# Qwen3-TTS — integrated with workspace GGML (no vendored ggml submodule).
# Upstream: https://github.com/bailu-obj/qwen3-tts.cpp

cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(QWEN3_TTS_TIMING "Enable detailed timing instrumentation" OFF)
if(QWEN3_TTS_TIMING)
    add_compile_definitions(QWEN3_TTS_TIMING)
endif()

find_package(Threads REQUIRED)

set(QWEN3_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src")

set(QWEN3TTS_SOURCES
    ${QWEN3_SRC}/gguf_loader.cpp
    ${QWEN3_SRC}/text_tokenizer.cpp
    ${QWEN3_SRC}/tts_transformer.cpp
    ${QWEN3_SRC}/audio_tokenizer_encoder.cpp
    ${QWEN3_SRC}/audio_tokenizer_decoder.cpp
    ${QWEN3_SRC}/qwen3_tts.cpp
    ${QWEN3_SRC}/qwen3tts_c_api.cpp
)

if(APPLE)
    list(APPEND QWEN3TTS_SOURCES ${QWEN3_SRC}/coreml_code_predictor.mm)
    set_source_files_properties(${QWEN3_SRC}/coreml_code_predictor.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
    list(APPEND QWEN3TTS_SOURCES ${QWEN3_SRC}/coreml_code_predictor_stub.cpp)
endif()

add_library(qwen3tts STATIC ${QWEN3TTS_SOURCES})
target_include_directories(qwen3tts PUBLIC ${QWEN3_SRC})
target_link_libraries(qwen3tts PUBLIC ggml ggml-base ggml-cpu Threads::Threads)
if(TARGET ggml-metal)
    target_link_libraries(qwen3tts PUBLIC ggml-metal)
endif()
if(APPLE)
    target_link_libraries(qwen3tts PUBLIC
        "-framework Foundation"
        "-framework CoreML"
    )
endif()
