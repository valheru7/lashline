# Downloads the official ONNX Runtime release for this platform and ships its shared library in
# the plugin's data folder (onnxruntime/), where src/onnx-runtime.cpp loads it by full path.
# Only the headers are used at build time; nothing links against it.

include_guard(GLOBAL)

set(ONNXRUNTIME_VERSION 1.30.0)
set(_ort_base_url "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}")

if(OS_WINDOWS)
  set(_ort_package "onnxruntime-win-x64-${ONNXRUNTIME_VERSION}")
  set(_ort_archive "${_ort_package}.zip")
  set(_ort_sha256 c6ba983baf5681af108599675d2a89c2d145512d02de28aed0bff177cd0ba949)
  set(_ort_libraries lib/onnxruntime.dll lib/onnxruntime_providers_shared.dll)
elseif(OS_MACOS)
  set(_ort_package "onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}")
  set(_ort_archive "${_ort_package}.tgz")
  set(_ort_sha256 6ebb5062a934537c352937821f9fe9718e7de1a2db1122a93dd363ffd53a7012)
  set(_ort_libraries "lib/libonnxruntime.${ONNXRUNTIME_VERSION}.dylib")
elseif(OS_LINUX)
  set(_ort_package "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")
  set(_ort_archive "${_ort_package}.tgz")
  set(_ort_sha256 a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd)
  set(_ort_libraries "lib/libonnxruntime.so.${ONNXRUNTIME_VERSION}")
else()
  message(FATAL_ERROR "No ONNX Runtime release is set up for this platform")
endif()

set(_ort_download "${CMAKE_BINARY_DIR}/deps/${_ort_archive}")
set(ONNXRUNTIME_DIR "${CMAKE_BINARY_DIR}/deps/${_ort_package}")

if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
  message(STATUS "Downloading ONNX Runtime ${ONNXRUNTIME_VERSION}")
  file(
    DOWNLOAD "${_ort_base_url}/${_ort_archive}" "${_ort_download}"
    EXPECTED_HASH SHA256=${_ort_sha256}
    SHOW_PROGRESS
  )
  file(ARCHIVE_EXTRACT INPUT "${_ort_download}" DESTINATION "${CMAKE_BINARY_DIR}/deps")
  file(REMOVE "${_ort_download}")
endif()

# The file name src/onnx-runtime.cpp loads, inside the data folder.
list(GET _ort_libraries 0 _ort_main_library)
cmake_path(GET _ort_main_library FILENAME ONNXRUNTIME_LIBRARY_NAME)

# Adds the headers to `target` and ships the libraries with it.
function(target_add_onnxruntime target)
  target_include_directories(${target} SYSTEM PRIVATE "${ONNXRUNTIME_DIR}/include")
  target_compile_definitions(${target} PRIVATE ONNXRUNTIME_LIBRARY_NAME="${ONNXRUNTIME_LIBRARY_NAME}")

  foreach(library IN LISTS _ort_libraries)
    set(library_path "${ONNXRUNTIME_DIR}/${library}")
    if(OS_MACOS)
      target_sources(${target} PRIVATE "${library_path}")
      set_property(SOURCE "${library_path}" PROPERTY MACOSX_PACKAGE_LOCATION "Resources/onnxruntime")
    else()
      if(OS_WINDOWS)
        set(data_destination "${target}/data")
      else()
        set(data_destination "${CMAKE_INSTALL_DATAROOTDIR}/obs/obs-plugins/${target}")
      endif()
      install(FILES "${library_path}" DESTINATION "${data_destination}/onnxruntime")
      add_custom_command(
        TARGET ${target}
        POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/rundir/$<CONFIG>/${target}/onnxruntime"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${library_path}" "${CMAKE_CURRENT_BINARY_DIR}/rundir/$<CONFIG>/${target}/onnxruntime"
        VERBATIM
      )
    endif()
  endforeach()
endfunction()
