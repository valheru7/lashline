#pragma once

// Turns the tracked eyelids into triangles to draw: a liner band along each upper lash line,
// thin at the inner corner and thicker at the outer corner, with an optional wing and lashes.
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
};

struct Vertex {
	Point position;
	float alpha; // 1 inside the liner, fading to 0 at its soft edge
};

// Triangles (three vertices each) for both eyes, in the same pixel space as `face`.
std::vector<Vertex> eyeliner_triangles(const FaceMesh &face, const Look &look);

} // namespace eyeliner
