# Standalone checks for the parts of the plugin that don't need OBS.
#   cmake -S tests -B build-tests -DONNXRUNTIME_DIR=<extracted onnxruntime release> && cmake --build build-tests
cmake_minimum_required(VERSION 3.28)
project(eyeliner-tests CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(ONNXRUNTIME_DIR "" CACHE PATH "An extracted ONNX Runtime release (only its headers are used)")

add_executable(track-video track-video.cpp ../src/face-tracker.cpp ../src/onnx-runtime.cpp)
target_include_directories(track-video PRIVATE ../src "${ONNXRUNTIME_DIR}/include")
target_link_libraries(track-video PRIVATE ${CMAKE_DL_LIBS})

add_executable(render-video render-video.cpp ../src/eyeliner-shape.cpp ../src/face-tracker.cpp ../src/onnx-runtime.cpp)
target_include_directories(render-video PRIVATE ../src "${ONNXRUNTIME_DIR}/include")
target_link_libraries(render-video PRIVATE ${CMAKE_DL_LIBS})
