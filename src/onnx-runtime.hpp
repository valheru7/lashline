#pragma once

// In OBS, ONNX Runtime is loaded at run time from a full path instead of being linked. Another
// OBS plugin may ship its own onnxruntime library of a different version, and loading ours by
// path keeps the two from clashing. The iPhone app links it normally (LASHLINE_LINK_ONNXRUNTIME),
// from the onnxruntime framework.

#ifdef LASHLINE_LINK_ONNXRUNTIME
#include <onnxruntime/onnxruntime_cxx_api.h>
#else
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#endif

#include <string>

namespace eyeliner {

// Loads the library at `library_path` and initializes the ONNX Runtime C++ API.
// Returns an error message, or an empty string on success. Safe to call more than once.
std::string load_onnx_runtime(const std::string &library_path);

} // namespace eyeliner
