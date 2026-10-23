#include "onnx-runtime.hpp"

#include <mutex>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace eyeliner {

namespace {

using GetApiBase = const OrtApiBase *(ORT_API_CALL *)();

GetApiBase find_get_api_base(const std::string &library_path, std::string &error)
{
#ifdef _WIN32
	std::wstring wide_path(MultiByteToWideChar(CP_UTF8, 0, library_path.c_str(), -1, nullptr, 0), L'\0');
	MultiByteToWideChar(CP_UTF8, 0, library_path.c_str(), -1, wide_path.data(), (int)wide_path.size());
	// LOAD_WITH_ALTERED_SEARCH_PATH makes the DLL's own folder the place its dependencies
	// are found, so the onnxruntime DLLs we ship are the ones used.
	HMODULE library = LoadLibraryExW(wide_path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
	if (!library) {
		error = "Couldn't load " + library_path + " (error " + std::to_string(GetLastError()) + ")";
		return nullptr;
	}
	return reinterpret_cast<GetApiBase>(GetProcAddress(library, "OrtGetApiBase"));
#else
	void *library = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
	if (!library) {
		error = "Couldn't load " + library_path + ": " + dlerror();
		return nullptr;
	}
	return reinterpret_cast<GetApiBase>(dlsym(library, "OrtGetApiBase"));
#endif
}

} // namespace

std::string load_onnx_runtime(const std::string &library_path)
{
	static std::mutex mutex;
	static bool loaded = false;
	std::lock_guard lock(mutex);
	if (loaded)
		return "";

	std::string error;
	GetApiBase get_api_base = find_get_api_base(library_path, error);
	if (!get_api_base)
		return error.empty() ? library_path + " has no OrtGetApiBase" : error;

	const OrtApi *api = get_api_base()->GetApi(ORT_API_VERSION);
	if (!api)
		return std::string(
			       "The ONNX Runtime library is older than this plugin expects (it doesn't support API version ") +
		       std::to_string(ORT_API_VERSION) + ")";

	Ort::InitApi(api);
	loaded = true;
	return "";
}

} // namespace eyeliner
