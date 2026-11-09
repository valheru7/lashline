#pragma once

// Turns the tracked eyelids into triangles to draw: a liner band along each upper lash line,
// thin at the inner corner and thicker at the outer corner, with an optional wing and lashes.
// Also the mesh for the colored contacts: each iris where it shows between the lids.
// A port of prototype/eyeliner.py's shape, except that soft edges come from a fringe of
// vertices that fade to transparent instead of from blurring a mask.

#include "face-tracker.hpp"

#include <vector>

namespace eyeliner {

struct Look {
	// Sizes are fractions of the eye's width, so the look holds steady as you lean in or out.
	float inner_thickness = 0.015f;
	float outer_thickness = 0.08f;
	float wing_length = 0.32f; // past the outer corner; 0 means no wing
	float wing_lift_degrees = 22.0f;
	float softness = 0.012f;
	bool lashes = true;
	// Multipliers on the standard lash's length and on its thickness: 1 is the standard lash,
	// 0.5 half its size, 2 double. Scaling the two together keeps the lashes' shape.
	float lash_length_scale = 1.0f;
	float lash_thickness_scale = 1.0f;
	// Colored contacts: each iris is tinted toward a color the renderer supplies. Intensity is
	// how strongly (0 to 1), and glow is a soft light over and just around the iris (0 is none).
	bool eye_color = false;
	float eye_color_intensity = 0.7f;
	float eye_glow = 0.0f;
};

struct Vertex {
	Point position;
	float alpha; // 1 inside the liner, fading to 0 at its soft edge
};

// Triangles (three vertices each) for both eyes, in the same pixel space as `face`.
std::vector<Vertex> eyeliner_triangles(const FaceMesh &face, const Look &look);

// A point of the colored-contacts mesh. The renderer works out the color per pixel, since the
// iris is only a few pixels across on a stream and a per-vertex color would look blocky:
//
//   r = length(iris)
//   tint:  mix the frame toward the iris color by  tint * IRIS_RING(r),  where IRIS_RING is
//          smoothstep(0.28, 0.45, r) * (1 - smoothstep(0.85, 1.0, r)): the pupil stays dark
//          and the iris's outer edge is soft.
//   glow:  screen the iris color over that, by  glow * (1 - smoothstep(0.6, 1.8, r)).
//
// The OBS filter's data/effects/iris.effect, any other host's shader and tests/render-video.cpp
// all do this; keep them in step.
struct IrisVertex {
	Point position;
	Point iris; // relative to the iris: its center is (0, 0) and its edge is 1 away
	float tint; // how strongly to tint here, before the ring above (0 to 1)
	float glow; // how strongly to glow here, before the falloff above (0 to 1)
};

// Triangles (three vertices each) covering both irises where they show between the lids, in the
// same pixel space as `face`. Empty when `look.eye_color` is off or no iris shows.
std::vector<IrisVertex> iris_triangles(const FaceMesh &face, const Look &look);

} // namespace eyeliner
