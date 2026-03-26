cmake_minimum_required(VERSION 3.20) # for add_link_options and implicit target directories.

# Objective-C++ must be enabled here (not in a late add_subdirectory) so CMAKE_OBJCXX_* is
# defined before ggml-metal and qwen3-tts CoreML sources are processed.
if(APPLE)
    project(ggml-aio-sys LANGUAGES C CXX OBJC OBJCXX)
else()
    project(ggml-aio-sys LANGUAGES C CXX)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ggml)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/whisper.cpp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/sense-voice.cpp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/qwen3-tts.cpp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/qwen3-asr.cpp)

# todo
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp)