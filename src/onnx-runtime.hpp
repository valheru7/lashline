#pragma once

// ONNX Runtime is loaded at run time from a full path instead of being linked. Another OBS
// plugin may ship its own onnxruntime library of a different version, and loading ours by
// path keeps the two from clashing.

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>

#include <string>

namespace eyeliner {

// Loads the library at `library_path` and initializes the ONNX Runtime C++ API.
// Returns an error message, or an empty string on success. Safe to call more than once.
std::string load_onnx_runtime(const std::string &library_path);

} // namespace eyeliner
