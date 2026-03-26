cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

set(QWEN3_ASR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(QWEN3_ASR_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(QWEN3ASR_SOURCES
    ${QWEN3_ASR_SRC}/audio_encoder.cpp
    ${QWEN3_ASR_SRC}/audio_injection.cpp
    ${QWEN3_ASR_SRC}/gguf_loader.cpp
    ${QWEN3_ASR_SRC}/mel_spectrogram.cpp
    ${QWEN3_ASR_SRC}/text_decoder.cpp
    ${QWEN3_ASR_SRC}/qwen3_asr.cpp
    ${QWEN3_ASR_SRC}/qwen3asr_c_api.cpp
)

add_library(qwen3asr STATIC ${QWEN3ASR_SOURCES})
target_include_directories(qwen3asr PUBLIC
    ${QWEN3_ASR_SRC}
    ${QWEN3_ASR_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src
)
target_link_libraries(qwen3asr PUBLIC ggml ggml-base ggml-cpu Threads::Threads)

if(TARGET ggml-metal)
    target_link_libraries(qwen3asr PUBLIC ggml-metal)
endif()

if(APPLE)
    target_link_libraries(qwen3asr PUBLIC "-framework Accelerate")
endif()
