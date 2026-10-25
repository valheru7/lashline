#pragma once

// A small C interface to Lashline's face tracker and liner shape, for the iPhone app. Swift
// calls C directly, while the C++ types underneath (ONNX Runtime sessions, std::optional)
// don't cross over cleanly.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { LASHLINE_FACE_POINT_COUNT = 478 };

typedef struct LashlineTracker LashlineTracker;

typedef struct {
	float x;
	float y;
} LashlinePoint;

// Sizes are fractions of the eye's width. See Look in eyeliner-shape.hpp.
typedef struct {
	float inner_thickness;
	float outer_thickness;
	float wing_length;
	float wing_lift_degrees;
	float softness;
	bool lashes;
} LashlineLook;

typedef struct {
	LashlinePoint position;
	float alpha; // 1 inside the liner, fading to 0 at its soft edge
} LashlineVertex;

LashlineLook lashline_default_look(void);

// Returns NULL and writes a message to `error` when the models can't be loaded.
LashlineTracker *lashline_tracker_create(const char *detector_model_path, const char *landmark_model_path, char *error,
					 size_t error_size);
void lashline_tracker_destroy(LashlineTracker *tracker);

// Tracks a face in a BGRA image (the iPhone camera's format). When there is one, fills
// `points` (LASHLINE_FACE_POINT_COUNT of them, in image pixels) and returns true.
bool lashline_track_bgra(LashlineTracker *tracker, const uint8_t *pixels, int width, int height, int stride,
			 LashlinePoint *points);

// Tracks up to `max_faces` faces in a BGRA image. Fills `points` with
// LASHLINE_FACE_POINT_COUNT points per face, one face after another (so it needs room for
// max_faces * LASHLINE_FACE_POINT_COUNT), and returns how many faces there are.
int lashline_track_faces_bgra(LashlineTracker *tracker, const uint8_t *pixels, int width, int height, int stride,
			      int max_faces, LashlinePoint *points);

// Writes the liner's triangles (three vertices each, in the same space as `points`) into
// `vertices` and returns how many vertices there are. If that's more than `capacity`, only
// `capacity` were written; call again with a bigger buffer.
size_t lashline_liner_triangles(const LashlinePoint *points, const LashlineLook *look, LashlineVertex *vertices,
				size_t capacity);

#ifdef __cplusplus
}
#endif
