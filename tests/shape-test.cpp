// Checks of the liner shape on a made-up face, no camera or models needed:
//
//   cmake --build build-tests --target shape-test && build-tests/shape-test
//
// Prints each failed check and exits with 1 if any failed.

#include "eyeliner-shape.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

using namespace eyeliner;

namespace {

int failures = 0;

void check(bool passed, const char *what)
{
	if (passed)
		return;
	std::printf("FAILED: %s\n", what);
	failures++;
}

// The eyes' corners sit on this line, in image pixels (y runs down).
constexpr float CORNER_Y = 5000;
// Big eyes, so the lashes' one-pixel soft fringe is too small to throw the sizes off.
constexpr float EYE_WIDTH = 4000;

// A face looking straight at the camera with its eyes open. Only the points the liner shape
// reads are placed; the rest stay at (0, 0).
FaceMesh straight_on_face()
{
	FaceMesh face{};
	constexpr std::array<int, 9> right_lid{133, 173, 157, 158, 159, 160, 161, 246, 33};
	constexpr std::array<int, 9> left_lid{362, 398, 384, 385, 386, 387, 388, 466, 263};
	constexpr float middle = 10000, inner_corner_gap = 1000, arch = 1200;
	for (size_t i = 0; i < right_lid.size(); i++) {
		// Inner corner to outer corner, arching up toward the forehead.
		float t = float(i) / (right_lid.size() - 1);
		float out = inner_corner_gap + EYE_WIDTH * t, up = arch * std::sin(3.14159265f * t);
		face[right_lid[i]] = {middle - out, CORNER_Y - up};
		face[left_lid[i]] = {middle + out, CORNER_Y - up};
	}
	// The lower lids, well below the upper ones: wide open.
	face[145] = {middle - inner_corner_gap - EYE_WIDTH / 2, CORNER_Y + 600};
	face[374] = {middle + inner_corner_gap + EYE_WIDTH / 2, CORNER_Y + 600};
	face[10] = {middle, 0};                // forehead
	face[152] = {middle, CORNER_Y + 9000}; // chin
	// The nose, down the middle from between the eyes.
	constexpr std::array<int, 7> nose{168, 6, 197, 195, 5, 4, 1};
	for (size_t i = 0; i < nose.size(); i++)
		face[nose[i]] = {middle, CORNER_Y - 200 + 700 * float(i)};
	return face;
}

// The right eye's lashes. Each eye's liner comes first, then its lashes, and the face is the
// same on both sides, so they're what lashes add to the first half of the triangles.
std::vector<Vertex> right_eye_lashes(const FaceMesh &face, Look look)
{
	look.lashes = false;
	size_t liner = eyeliner_triangles(face, look).size() / 2;
	look.lashes = true;
	std::vector<Vertex> both = eyeliner_triangles(face, look);
	size_t lashes = both.size() / 2 - liner;
	return {both.begin() + liner, both.begin() + liner + lashes};
}

// How much of the image the lashes cover, in square pixels.
float area(const std::vector<Vertex> &triangles)
{
	float total = 0;
	for (size_t i = 0; i + 2 < triangles.size(); i += 3) {
		Point a = triangles[i].position, b = triangles[i + 1].position, c = triangles[i + 2].position;
		total += std::abs((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) / 2;
	}
	return total;
}

// How far the lashes reach above the eye's corners.
float reach(const std::vector<Vertex> &triangles)
{
	float highest = CORNER_Y;
	for (const Vertex &vertex : triangles)
		highest = std::min(highest, vertex.position.y);
	return CORNER_Y - highest;
}

bool near(float value, float expected, float tolerance)
{
	return std::abs(value - expected) <= tolerance * std::abs(expected);
}

void lash_size_scales_the_lashes()
{
	const FaceMesh face = straight_on_face();
	Look standard;
	standard.softness = 0; // the smallest fringe, so sizes are mostly the lashes themselves
	check(standard.lash_length_scale == 1 && standard.lash_thickness_scale == 1,
	      "a new look has the standard lash");

	const std::vector<Vertex> lashes = right_eye_lashes(face, standard);
	check(!lashes.empty(), "the made-up face has lashes");

	Look longer = standard;
	longer.lash_length_scale = 2;
	const std::vector<Vertex> long_lashes = right_eye_lashes(face, longer);
	check(long_lashes.size() == lashes.size(), "longer lashes are the same number of triangles");
	check(near(area(long_lashes), 2 * area(lashes), 0.05f), "twice the length covers twice the area");
	// The lashes stand on a lid arching 1200 pixels above the corners, so their reach above
	// the corners grows by the lashes' own height, not double.
	check(reach(long_lashes) > reach(lashes) + 0.1f * EYE_WIDTH, "longer lashes reach higher");

	Look thicker = standard;
	thicker.lash_thickness_scale = 2;
	const std::vector<Vertex> thick_lashes = right_eye_lashes(face, thicker);
	check(near(area(thick_lashes), 2 * area(lashes), 0.05f), "twice the thickness covers twice the area");
	check(near(reach(thick_lashes), reach(lashes), 0.01f), "thicker lashes are no longer");

	Look smaller = standard;
	smaller.lash_length_scale = 0.5f;
	smaller.lash_thickness_scale = 0.5f;
	check(near(area(right_eye_lashes(face, smaller)), area(lashes) / 4, 0.1f),
	      "half the length and thickness covers a quarter of the area");
}

// `point` turned by `degrees` about `center`.
Point turned(Point point, Point center, float degrees)
{
	float angle = degrees * 3.14159265f / 180, cos = std::cos(angle), sin = std::sin(angle);
	float x = point.x - center.x, y = point.y - center.y;
	return {center.x + x * cos - y * sin, center.y + x * sin + y * cos};
}

// The liner goes by the face's own directions, never the image's: a face lying on its side or
// upside down in the frame gets the same liner, turned with it.
void a_turned_face_gets_the_same_liner_turned()
{
	const FaceMesh face = straight_on_face();
	const Look look;
	const std::vector<Vertex> upright = eyeliner_triangles(face, look);
	const Point center{10000, 5000};
	for (float degrees : {45.0f, 90.0f, 135.0f, 180.0f, 225.0f, 270.0f}) {
		FaceMesh turned_face;
		for (size_t i = 0; i < face.size(); i++)
			turned_face[i] = turned(face[i], center, degrees);
		const std::vector<Vertex> triangles = eyeliner_triangles(turned_face, look);

		bool same = triangles.size() == upright.size();
		for (size_t i = 0; same && i < triangles.size(); i++) {
			Point expected = turned(upright[i].position, center, degrees);
			Point position = triangles[i].position;
			float miss = std::hypot(position.x - expected.x, position.y - expected.y);
			// Half a pixel on an eye 4000 pixels wide: only float rounding.
			same = miss < 0.5f && std::abs(triangles[i].alpha - upright[i].alpha) < 1e-3f;
		}
		char what[80];
		std::snprintf(what, sizeof(what), "a face turned %.0f degrees gets the same liner, turned", degrees);
		check(same, what);
	}
}

} // namespace

int main()
{
	lash_size_scales_the_lashes();
	a_turned_face_gets_the_same_liner_turned();
	if (failures == 0)
		std::printf("All shape checks passed.\n");
	return failures == 0 ? 0 : 1;
}
