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
//
// 4. Each face's points are smoothed over time (a One Euro filter on every point), since the
//    landmark model's points wobble by a pixel or so from frame to frame even when the face is
//    still. The regions in step 3 come from the unsmoothed points.
//
// track_all() follows several faces the same way, one region each, and also runs the
// detector every few frames while it has room for more, to pick up someone stepping in.
// (The Python prototype only does one face, and doesn't smooth: compare it with smoothing off.)

#include "one-euro-filter.hpp"
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

// The byte order of a 4-byte pixel. OBS hands us RGBA; iPhone cameras give BGRA.
enum class PixelOrder { RGBA, BGRA };

// An image in memory, 4 bytes per pixel. The tracker only reads it.
struct ImageView {
	const uint8_t *pixels = nullptr;
	int width = 0;
	int height = 0;
	int stride = 0; // bytes per row
	PixelOrder order = PixelOrder::RGBA;
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

	// The face's points in image pixels, or nothing when there's no face. `time` is when the
	// frame was captured, in seconds from any starting point; smoothing goes by it.
	std::optional<FaceMesh> track(const ImageView &image, double time);

	// Up to `max_faces` faces' points in image pixels, in the order they were found.
	std::vector<FaceMesh> track_all(const ImageView &image, int max_faces, double time);

	// Smoothing is on unless this turns it off.
	void set_smoothing(bool enabled) { smoothing = enabled; }

	// Forgets every face, so the next frame starts afresh: for when the video jumps to
	// somewhere else, like a different camera.
	void reset() { faces.clear(); }

private:
	struct PointFilter {
		OneEuroFilter x;
		OneEuroFilter y;
	};

	struct FollowedFace {
		Region region; // where to look in the next frame
		// Each point's smoothing, which belongs to this face alone.
		std::array<PointFilter, FACE_MESH_POINT_COUNT> filters;
		std::optional<double> last_time; // when the last points were smoothed

		// `points` smoothed with this face's earlier ones. `face_size` is in pixels.
		FaceMesh smooth(const FaceMesh &points, float face_size, double time);
	};

	// Every face the detector finds, most confident first.
	std::vector<Region> detect(const ImageView &image);
	std::optional<FaceMesh> landmarks(const ImageView &image, const Region &region);

	Ort::Env env;
	Ort::Session detector{nullptr};
	Ort::Session landmarker{nullptr};
	std::vector<FollowedFace> faces; // one per face being followed
	bool smoothing = true;
	int frames_since_detection = 0;
	std::vector<float> detector_input;
	std::vector<float> landmark_input;
};

} // namespace eyeliner
