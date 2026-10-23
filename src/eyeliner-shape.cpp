#include "eyeliner-shape.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace eyeliner {

namespace {

// Upper-lid points of the face mesh, from the inner corner to the outer corner. "Right" and
// "left" are the person's own eyes.
constexpr std::array<int, 9> RIGHT_UPPER_LID{133, 173, 157, 158, 159, 160, 161, 246, 33};
constexpr std::array<int, 9> LEFT_UPPER_LID{362, 398, 384, 385, 386, 387, 388, 466, 263};

constexpr int CURVE_SAMPLES_PER_SEGMENT = 8;
constexpr int WING_SAMPLES = 6;
constexpr int LASH_COUNT = 7;
// The liner starts a hair below the lid's edge so no skin shows between it and the lashes.
constexpr float BELOW_LASH_LINE = 0.008f;
constexpr float PI = 3.14159265f;
// When the head turns, the far eye looks narrower than the near one, and once it's partly
// behind the nose its mesh points are guesses: lashes drawn there land on the nose. So the
// far eye's lashes fade out as its width drops from FADE_START to FADE_END of the near eye's,
// and its liner (which hugs the lid more tightly) fades a little later.
constexpr float LASH_FADE_START = 0.6f;
constexpr float LASH_FADE_END = 0.42f;
constexpr float LINER_FADE_START = 0.55f;
constexpr float LINER_FADE_END = 0.4f;

Point operator+(Point a, Point b)
{
	return {a.x + b.x, a.y + b.y};
}
Point operator-(Point a, Point b)
{
	return {a.x - b.x, a.y - b.y};
}
Point operator*(Point a, float s)
{
	return {a.x * s, a.y * s};
}
float dot(Point a, Point b)
{
	return a.x * b.x + a.y * b.y;
}
float length(Point a)
{
	return std::sqrt(dot(a, a));
}
Point lerp(Point a, Point b, float t)
{
	return a + (b - a) * t;
}

Point normalized(Point a)
{
	float l = length(a);
	return l > 1e-6f ? a * (1.0f / l) : Point{0, 0};
}

// Rotates `along` toward `up` by `angle`.
Point turn(Point along, Point up, float angle)
{
	return along * std::cos(angle) + up * std::sin(angle);
}

// A smooth curve through `points` (Catmull-Rom), so the liner doesn't show the mesh's corners.
std::vector<Point> catmull_rom(const std::vector<Point> &points)
{
	std::vector<Point> padded;
	padded.push_back(points[0] * 2 - points[1]);
	padded.insert(padded.end(), points.begin(), points.end());
	padded.push_back(points.back() * 2 - points[points.size() - 2]);

	std::vector<Point> curve;
	for (size_t i = 1; i + 2 < padded.size(); i++) {
		Point p0 = padded[i - 1], p1 = padded[i], p2 = padded[i + 1], p3 = padded[i + 2];
		for (int step = 0; step < CURVE_SAMPLES_PER_SEGMENT; step++) {
			float t = float(step) / CURVE_SAMPLES_PER_SEGMENT, t2 = t * t, t3 = t2 * t;
			curve.push_back((p1 * 2 + (p2 - p0) * t + (p0 * 2 - p1 * 5 + p2 * 4 - p3) * t2 +
					 (p3 - p0 + (p1 - p2) * 3) * t3) *
					0.5f);
		}
	}
	curve.push_back(points.back());
	return curve;
}

// A smoothed upper lid, from the inner to the outer corner, with directions along it.
struct LidCurve {
	std::vector<Point> points;
	std::vector<Point> tangents; // toward the outer corner
	std::vector<Point> normals;  // up, away from the eyeball
	float eye_width;
};

LidCurve lid_curve(const FaceMesh &face, const std::array<int, 9> &indices)
{
	std::vector<Point> lid;
	for (int index : indices)
		lid.push_back(face[index]);

	LidCurve curve;
	curve.points = catmull_rom(lid);
	curve.eye_width = length(lid.back() - lid.front());
	size_t count = curve.points.size();
	for (size_t i = 0; i < count; i++) {
		Point before = curve.points[i == 0 ? 0 : i - 1];
		Point after = curve.points[i + 1 == count ? i : i + 1];
		Point tangent = normalized(after - before);
		curve.tangents.push_back(tangent);
		curve.normals.push_back({tangent.y, -tangent.x});
	}
	// The upper lid arches above the line between the corners; that's the "up" side.
	Point lid_middle{0, 0};
	for (Point p : lid)
		lid_middle = lid_middle + p * (1.0f / lid.size());
	Point arch = lid_middle - (lid.front() + lid.back()) * 0.5f;
	if (dot(curve.normals[count / 2], arch) < 0)
		for (Point &normal : curve.normals)
			normal = normal * -1;
	return curve;
}

// One cross-section of a stroke: its two edges at one point along it.
struct Section {
	Point low;  // the lash side
	Point high; // the far side
};

// Adds a stroke as triangles: a core of `opacity` between consecutive sections, plus a fringe
// that fades from the edge to transparent `feather` pixels further out, all the way round.
void add_stroke(const std::vector<Section> &sections, float feather, float opacity, std::vector<Vertex> &out)
{
	auto triangle = [&](Vertex a, Vertex b, Vertex c) {
		out.push_back(a);
		out.push_back(b);
		out.push_back(c);
	};
	auto quad = [&](Vertex a, Vertex b, Vertex c, Vertex d) { // a-b along one edge, d-c along the other
		triangle(a, b, c);
		triangle(a, c, d);
	};

	size_t count = sections.size();
	std::vector<Point> along(count), across(count);
	for (size_t i = 0; i < count; i++) {
		Point before = lerp(sections[i == 0 ? 0 : i - 1].low, sections[i == 0 ? 0 : i - 1].high, 0.5f);
		Point after =
			lerp(sections[i + 1 == count ? i : i + 1].low, sections[i + 1 == count ? i : i + 1].high, 0.5f);
		along[i] = normalized(after - before);
		Point width = sections[i].high - sections[i].low;
		// Where the stroke tapers to a point, fall back to the direction across it.
		Point side{-along[i].y, along[i].x};
		across[i] = length(width) > 1e-3f ? normalized(width)
						  : (dot(side, across[i == 0 ? 0 : i - 1]) < 0 ? side * -1 : side);
	}

	for (size_t i = 0; i + 1 < count; i++) {
		Vertex low_a{sections[i].low, opacity}, high_a{sections[i].high, opacity};
		Vertex low_b{sections[i + 1].low, opacity}, high_b{sections[i + 1].high, opacity};
		Vertex fringe_low_a{sections[i].low - across[i] * feather, 0};
		Vertex fringe_low_b{sections[i + 1].low - across[i + 1] * feather, 0};
		Vertex fringe_high_a{sections[i].high + across[i] * feather, 0};
		Vertex fringe_high_b{sections[i + 1].high + across[i + 1] * feather, 0};
		quad(low_a, low_b, high_b, high_a);
		quad(fringe_low_a, fringe_low_b, low_b, low_a);
		quad(high_a, high_b, fringe_high_b, fringe_high_a);
	}

	// Soft caps at both ends.
	for (auto [index, outward] : {std::pair{size_t(0), along[0] * -1}, std::pair{count - 1, along[count - 1]}}) {
		const Section &end = sections[index];
		Point fringe_low = end.low - across[index] * feather, fringe_high = end.high + across[index] * feather;
		quad({end.low, opacity}, {end.high, opacity}, {fringe_high + outward * feather, 0},
		     {fringe_low + outward * feather, 0});
		triangle({end.low, opacity}, {fringe_low + outward * feather, 0}, {fringe_low, 0});
		triangle({end.high, opacity}, {fringe_high, 0}, {fringe_high + outward * feather, 0});
	}
}

std::vector<Section> liner_sections(const LidCurve &lid, const Look &look)
{
	std::vector<Section> sections;
	size_t count = lid.points.size();
	for (size_t i = 0; i < count; i++) {
		// Thin at the inner corner, growing toward the outer corner; it eases in so the inner
		// third stays fine.
		float t = float(i) / (count - 1);
		float thickness =
			look.inner_thickness + (look.outer_thickness - look.inner_thickness) * std::pow(t, 1.6f);
		sections.push_back({lid.points[i] - lid.normals[i] * (BELOW_LASH_LINE * lid.eye_width),
				    lid.points[i] + lid.normals[i] * (thickness * lid.eye_width)});
	}

	if (look.wing_length > 0) {
		// The wing continues the lid's outer direction, tipped up, and tapers to a point.
		Point direction = turn(lid.tangents.back(), lid.normals.back(), look.wing_lift_degrees * PI / 180);
		Point tip = lid.points.back() + direction * (look.wing_length * lid.eye_width);
		Section corner = sections.back();
		for (int i = 1; i <= WING_SAMPLES; i++) {
			float t = float(i) / WING_SAMPLES;
			sections.push_back({lerp(corner.low, tip, t), lerp(corner.high, tip, t)});
		}
	}
	return sections;
}

// A few thin, tapered lashes on the outer half of the lid, fanning out toward the corner.
std::vector<std::vector<Section>> lash_strokes(const LidCurve &lid)
{
	std::vector<std::vector<Section>> lashes;
	for (int i = 0; i < LASH_COUNT; i++) {
		float t = 0.4f + 0.58f * i / (LASH_COUNT - 1);
		float progress = (t - 0.4f) / 0.58f;
		size_t index = size_t(std::round(t * (lid.points.size() - 1)));
		Point root = lid.points[index], up = lid.normals[index], along = lid.tangents[index];
		// Lashes near the middle point up; toward the outer corner they sweep outward.
		float sweep = (15 + 45 * progress) * PI / 180;
		float lash_length = (0.13f + 0.07f * progress) * lid.eye_width;
		float half_width = 0.012f * lid.eye_width;
		// A slight curl: the tip bends further outward than the base.
		Point base_direction = turn(up, along, sweep);
		Point tip_direction = turn(up, along, sweep + 0.35f);
		Point middle = root + base_direction * (lash_length * 0.55f);
		Point tip = middle + tip_direction * (lash_length * 0.45f);
		lashes.push_back({
			{root - along * half_width, root + along * half_width},
			{middle - along * (half_width * 0.5f), middle + along * (half_width * 0.5f)},
			{tip, tip},
		});
	}
	return lashes;
}

// 1 while `ratio` is at or above `start`, falling smoothly to 0 at `end`.
float fade(float ratio, float start, float end)
{
	float t = std::clamp((ratio - end) / (start - end), 0.0f, 1.0f);
	return t * t * (3 - 2 * t);
}

} // namespace

std::vector<Vertex> eyeliner_triangles(const FaceMesh &face, const Look &look)
{
	const std::array<LidCurve, 2> lids{lid_curve(face, RIGHT_UPPER_LID), lid_curve(face, LEFT_UPPER_LID)};
	const float widest = std::max(lids[0].eye_width, lids[1].eye_width);

	std::vector<Vertex> triangles;
	for (const LidCurve &lid : lids) {
		float width_ratio = widest > 0 ? lid.eye_width / widest : 1.0f;
		// At least a pixel of fringe, so edges are antialiased even with no softness.
		float feather = std::max(1.0f, 2 * look.softness * lid.eye_width);

		float liner_opacity = fade(width_ratio, LINER_FADE_START, LINER_FADE_END);
		if (liner_opacity > 0)
			add_stroke(liner_sections(lid, look), feather, liner_opacity, triangles);

		float lash_opacity = fade(width_ratio, LASH_FADE_START, LASH_FADE_END);
		if (look.lashes && lash_opacity > 0)
			for (const auto &lash : lash_strokes(lid))
				add_stroke(lash, std::max(1.0f, feather * 0.5f), lash_opacity, triangles);
	}
	return triangles;
}

} // namespace eyeliner
