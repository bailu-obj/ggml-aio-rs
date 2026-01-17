cmake_minimum_required(VERSION 3.20) # for add_link_options and implicit target directories.

project(ggml-aio-sys)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ggml)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/whisper.cpp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/sense-voice.cpp)

# todo
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp)