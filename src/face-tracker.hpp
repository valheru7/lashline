#pragma once

// Finds the 478 face-mesh points in a video frame, using MediaPipe's face detector and face
// landmark models (converted to ONNX). This is a port of prototype/tracker.py; keep the two
// in step, since compare_tracker.py checks the Python one against MediaPipe itself.
//
// 1. When there's no face yet, the detector (128x128) finds one in the whole frame.
// 2. The face's region is cut out, rotated upright using the eyes, and fed to the landmark
//    model (256x256).
// 3. On the next frame the region comes from those points, so the detector only runs again
//    when the face is lost.

#include "onnx-runtime.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace eyeliner {

struct Point {
	float x = 0;
	float y = 0;
};

// An RGBA image in memory. The tracker only reads it.
struct ImageView {
	const uint8_t *pixels = nullptr;
	int width = 0;
	int height = 0;
	int stride = 0; // bytes per row
};

// A rotated square in image pixels.
struct Region {
	Point center;
	float size = 0;
	float rotation = 0; // radians
};

constexpr int FACE_MESH_POINT_COUNT = 478;
using FaceMesh = std::array<Point, FACE_MESH_POINT_COUNT>;

class FaceTracker {
public:
	// Throws Ort::Exception when the models can't be loaded.
	FaceTracker(const std::string &detector_model_path, const std::string &landmark_model_path);

	// The face's points in image pixels, or nothing when there's no face.
	std::optional<FaceMesh> track(const ImageView &image);

private:
	std::optional<Region> detect(const ImageView &image);
	std::optional<FaceMesh> landmarks(const ImageView &image, const Region &region);

	Ort::Env env;
	Ort::Session detector{nullptr};
	Ort::Session landmarker{nullptr};
	std::optional<Region> region;
	std::vector<float> detector_input;
	std::vector<float> landmark_input;
};

} // namespace eyeliner
