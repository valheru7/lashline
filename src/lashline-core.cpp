#include "include/lashline-core.h"

#include "eyeliner-shape.hpp"
#include "face-tracker.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>

using namespace eyeliner;

// The C structs are copied straight into the C++ ones, so their layouts must match.
static_assert(sizeof(LashlinePoint) == sizeof(Point));
static_assert(sizeof(LashlineVertex) == sizeof(Vertex));
static_assert(LASHLINE_FACE_POINT_COUNT == FACE_MESH_POINT_COUNT);

struct LashlineTracker {
	FaceTracker tracker;
};

namespace {

Look look_from(const LashlineLook &look)
{
	return {look.inner_thickness,   look.outer_thickness, look.wing_length,
		look.wing_lift_degrees, look.softness,        look.lashes};
}

} // namespace

LashlineLook lashline_default_look(void)
{
	Look look;
	return {look.inner_thickness,   look.outer_thickness, look.wing_length,
		look.wing_lift_degrees, look.softness,        look.lashes};
}

LashlineTracker *lashline_tracker_create(const char *detector_model_path, const char *landmark_model_path, char *error,
					 size_t error_size)
{
	try {
		return new LashlineTracker{FaceTracker(detector_model_path, landmark_model_path)};
	} catch (const std::exception &exception) {
		std::snprintf(error, error_size, "%s", exception.what());
		return nullptr;
	}
}

void lashline_tracker_destroy(LashlineTracker *tracker)
{
	delete tracker;
}

bool lashline_track_bgra(LashlineTracker *tracker, const uint8_t *pixels, int width, int height, int stride,
			 LashlinePoint *points)
{
	ImageView image{pixels, width, height, stride, PixelOrder::BGRA};
	std::optional<FaceMesh> face = tracker->tracker.track(image);
	if (!face)
		return false;
	std::copy(face->begin(), face->end(), reinterpret_cast<Point *>(points));
	return true;
}

int lashline_track_faces_bgra(LashlineTracker *tracker, const uint8_t *pixels, int width, int height, int stride,
			      int max_faces, LashlinePoint *points)
{
	ImageView image{pixels, width, height, stride, PixelOrder::BGRA};
	std::vector<FaceMesh> faces = tracker->tracker.track_all(image, max_faces);
	Point *out = reinterpret_cast<Point *>(points);
	for (const FaceMesh &face : faces)
		out = std::copy(face.begin(), face.end(), out);
	return int(faces.size());
}

size_t lashline_liner_triangles(const LashlinePoint *points, const LashlineLook *look, LashlineVertex *vertices,
				size_t capacity)
{
	FaceMesh face;
	std::copy_n(reinterpret_cast<const Point *>(points), face.size(), face.begin());
	std::vector<Vertex> triangles = eyeliner_triangles(face, look_from(*look));
	std::copy_n(triangles.begin(), std::min(capacity, triangles.size()), reinterpret_cast<Vertex *>(vertices));
	return triangles.size();
}
