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

// The top of the forehead and the bottom of the chin, which give the face's "up".
constexpr int FOREHEAD = 10;
constexpr int CHIN = 152;

constexpr int CURVE_SAMPLES_PER_SEGMENT = 8;
constexpr int WING_SAMPLES = 6;
constexpr int LASH_COUNT = 7;
// The tracked lid line can be off by a few percent of the eye's width (the landmark model sees
// each eye only about 35 pixels across), which showed as a strip of skin between the liner and
// the lashes. Snapping to the lash line (face-tracker.cpp) closes most of that, but not where
// the lash line is faint or the lid has no contrast. So the liner also reaches down into the
// lash line, and its lash side fades out over a wide soft edge: a small miss overlaps the
// lashes instead of leaving a gap. Sizes are fractions of the eye's width.
constexpr float BELOW_LASH_LINE = 0.02f;
constexpr float LASH_SIDE_SOFTNESS = 0.035f;
// The inner corner is the point the tracker places least surely, so the liner fades in over
// this much of the lid rather than starting at a hard tip.
constexpr float INNER_CORNER_FADE = 0.18f;
constexpr float PI = 3.14159265f;
// When the head turns, the far eye looks narrower than the near one, and once it's partly
// behind the nose its mesh points are guesses: lashes drawn there land on the nose. So the
// far eye's lashes fade out as its width drops from FADE_START to FADE_END of the near eye's,
// and its liner (which hugs the lid more tightly) fades a little later.
constexpr float LASH_FADE_START = 0.6f;
constexpr float LASH_FADE_END = 0.42f;
constexpr float LINER_FADE_START = 0.55f;
constexpr float LINER_FADE_END = 0.4f;
// Points down the middle of the nose, from between the eyes to the tip. When the head turns
// (especially tipped back), the nose covers the far eye's inner corner first, and the mesh's
// guesses for a half-hidden eye put liner on the nose. So any part of the liner that lands
// over the nose is hidden, fading over NOSE_EDGE_SOFTNESS of an eye width, and the far eye's
// lashes, then liner, fade as the nose closes in on its inner corner. Clearance of the inner corner, in
// eye widths: about 0.15 for a side glance at the monitor, 0.05 to -0.01 in the bad frames
// of her test video.
constexpr std::array<int, 7> NOSE_MIDLINE{168, 6, 197, 195, 5, 4, 1};
constexpr float LASH_CORNER_FADE_START = 0.12f;
constexpr float LASH_CORNER_FADE_END = 0.0f;
constexpr float LINER_CORNER_FADE_START = 0.11f;
constexpr float LINER_CORNER_FADE_END = 0.03f;
constexpr float NOSE_EDGE_SOFTNESS = 0.08f;
// The middle of each eye's upper and lower lid, for how open the eye is: the gap between them
// in eye widths, 0.2 to 0.45 open, about 0.1 looking down and under 0.05 closed. The mesh's
// guess for a closing lid runs straighter than the real one, so anything drawn on it sits off
// the lid like a strip lash peeling away. Closed, real lashes also point down over the cheek,
// and drawn ones standing up out of the lid look like spikes. So as the eye closes the lashes
// fade and the wing shrinks, then the liner fades too; a blink is quick enough that it isn't
// missed.
constexpr int RIGHT_LID_TOP = 159;
constexpr int RIGHT_LID_BOTTOM = 145;
constexpr int LEFT_LID_TOP = 386;
constexpr int LEFT_LID_BOTTOM = 374;
constexpr float LASH_OPEN_FADE_START = 0.15f;
constexpr float LASH_OPEN_FADE_END = 0.06f;
constexpr float WING_OPEN_FADE_START = 0.15f;
constexpr float WING_OPEN_FADE_END = 0.06f;
constexpr float LINER_OPEN_FADE_START = 0.08f;
constexpr float LINER_OPEN_FADE_END = 0.03f;

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
	Point axis;                  // from the inner to the outer corner
	Point axis_up;               // perpendicular to `axis`, toward the forehead
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
	// "Up" is the side toward the forehead. It can't come from the lid's own arch: a closed
	// lid is nearly straight and often sags below the line between the corners, which turned
	// the liner, wing and lashes upside down for the length of every blink.
	Point face_up = face[FOREHEAD] - face[CHIN];
	curve.axis = normalized(lid.back() - lid.front());
	curve.axis_up = {curve.axis.y, -curve.axis.x};
	if (dot(curve.axis_up, face_up) < 0)
		curve.axis_up = curve.axis_up * -1;
	// The normals all sit on the same side of the curve (it always runs inner to outer
	// corner), so one check decides them all.
	if (dot(curve.normals[count / 2], curve.axis_up) < 0)
		for (Point &normal : curve.normals)
			normal = normal * -1;
	return curve;
}

// 1 while `ratio` is at or above `start`, falling smoothly to 0 at `end`.
float fade(float ratio, float start, float end)
{
	float t = std::clamp((ratio - end) / (start - end), 0.0f, 1.0f);
	return t * t * (3 - 2 * t);
}

// One cross-section of a stroke: its two edges at one point along it.
struct Section {
	Point low;         // the lash side
	Point high;        // the far side
	float alpha = 1.0f; // the share of the stroke's opacity here
};

// Adds a stroke as triangles: a core of `opacity` (times each section's `alpha`) between
// consecutive sections, plus a fringe
// that fades from the edge to transparent `feather` pixels further out, all the way round
// (`low_feather` on the lash side).
void add_stroke(const std::vector<Section> &sections, float feather, float low_feather, float opacity,
		std::vector<Vertex> &out)
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
		float alpha_a = opacity * sections[i].alpha, alpha_b = opacity * sections[i + 1].alpha;
		Vertex low_a{sections[i].low, alpha_a}, high_a{sections[i].high, alpha_a};
		Vertex low_b{sections[i + 1].low, alpha_b}, high_b{sections[i + 1].high, alpha_b};
		Vertex fringe_low_a{sections[i].low - across[i] * low_feather, 0};
		Vertex fringe_low_b{sections[i + 1].low - across[i + 1] * low_feather, 0};
		Vertex fringe_high_a{sections[i].high + across[i] * feather, 0};
		Vertex fringe_high_b{sections[i + 1].high + across[i + 1] * feather, 0};
		quad(low_a, low_b, high_b, high_a);
		quad(fringe_low_a, fringe_low_b, low_b, low_a);
		quad(high_a, high_b, fringe_high_b, fringe_high_a);
	}

	// Soft caps at both ends.
	for (auto [index, outward] : {std::pair{size_t(0), along[0] * -1}, std::pair{count - 1, along[count - 1]}}) {
		const Section &end = sections[index];
		float alpha = opacity * end.alpha;
		Point fringe_low = end.low - across[index] * low_feather, fringe_high = end.high + across[index] * feather;
		quad({end.low, alpha}, {end.high, alpha}, {fringe_high + outward * feather, 0},
		     {fringe_low + outward * feather, 0});
		triangle({end.low, alpha}, {fringe_low + outward * feather, 0}, {fringe_low, 0});
		triangle({end.high, alpha}, {fringe_high, 0}, {fringe_high + outward * feather, 0});
	}
}

// `wing_scale` shrinks the wing from its full length (1) to nothing (0).
std::vector<Section> liner_sections(const LidCurve &lid, const Look &look, float wing_scale)
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
				    lid.points[i] + lid.normals[i] * (thickness * lid.eye_width),
				    fade(t, INNER_CORNER_FADE, 0.0f)});
	}

	float wing_length = look.wing_length * wing_scale;
	if (wing_length > 0) {
		// The wing points up from the eye's own line (inner corner to outer corner) and tapers
		// to a point. Measuring the lift from the lid's direction instead aims it down, since
		// the lid already curves down into the outer corner.
		Point direction = turn(lid.axis, lid.axis_up, look.wing_lift_degrees * PI / 180);
		Point corner = lid.points.back();
		Point tip = corner + direction * (wing_length * lid.eye_width);
		// The wing's base spans across the wing, not across the lid, so the wing keeps its
		// width however far it turns from the lid. The gap between the lid's last section and
		// this one fills in as the wedge at the corner.
		Point across{-direction.y, direction.x};
		if (dot(across, lid.axis_up) < 0)
			across = across * -1;
		// The base narrows with the wing too, so a shrinking wing doesn't leave a full-size
		// wedge at the corner that then vanishes all at once.
		Section base{corner - across * (BELOW_LASH_LINE * lid.eye_width),
			     corner + across * (look.outer_thickness * wing_scale * lid.eye_width)};
		for (int i = 0; i <= WING_SAMPLES; i++) {
			float t = float(i) / WING_SAMPLES;
			sections.push_back({lerp(base.low, tip, t), lerp(base.high, tip, t)});
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
		// Directions come from the eye's line, not the lid's local curve, which turns down
		// into the outer corner and would point the outer lashes down with it.
		Point root = lid.points[index], up = lid.axis_up, along = lid.axis;
		// Lashes near the middle point up; toward the outer corner they sweep outward.
		float sweep = (10 + 40 * progress) * PI / 180;
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

// The middle of the nose as a line down the face, so a point can be checked against the part
// of the nose at its own height (the nose tip sticks out past the inner corners on any turn,
// but it's lower down, so it doesn't cover them unless the head is also tipped back).
// Positions run along the line between the eyes' outer corners, from the person's right eye
// toward their left: the right eye's liner belongs before the nose, the left eye's after it.
class NoseLine {
public:
	explicit NoseLine(const FaceMesh &face) : across(normalized(face[263] - face[33])), up{across.y, -across.x}
	{
		for (size_t i = 0; i < NOSE_MIDLINE.size(); i++)
			points[i] = {dot(face[NOSE_MIDLINE[i]], across), dot(face[NOSE_MIDLINE[i]], up)};
	}

	// How far `point` is clear of the nose, toward the given eye's side (negative: over it).
	// Points above or below the nose are always clear.
	float clearance(Point point, bool left_eye) const
	{
		float position = dot(point, across), height = dot(point, up);
		float clearance = INFINITY;
		for (size_t i = 0; i + 1 < points.size(); i++) {
			Point a = points[i], b = points[i + 1];
			if (height < std::min(a.y, b.y) || height > std::max(a.y, b.y) || a.y == b.y)
				continue;
			float nose = a.x + (b.x - a.x) * (height - a.y) / (b.y - a.y);
			clearance = std::min(clearance, left_eye ? position - nose : nose - position);
		}
		return clearance;
	}

private:
	Point across;
	Point up;
	std::array<Point, NOSE_MIDLINE.size()> points; // (position across, height) of each midline point
};

} // namespace

std::vector<Vertex> eyeliner_triangles(const FaceMesh &face, const Look &look)
{
	const std::array<LidCurve, 2> lids{lid_curve(face, RIGHT_UPPER_LID), lid_curve(face, LEFT_UPPER_LID)};
	const float widest = std::max(lids[0].eye_width, lids[1].eye_width);
	const NoseLine nose(face);

	std::vector<Vertex> triangles;
	for (const LidCurve &lid : lids) {
		const bool left_eye = &lid == &lids[1];
		const size_t first_vertex = triangles.size();
		// At least a pixel of fringe, so edges are antialiased even with no softness.
		float feather = std::max(1.0f, 2 * look.softness * lid.eye_width);

		// Once the nose closes in on this eye's inner corner, its mesh points are guesses; the
		// lashes, which stick out furthest, go first, then the liner.
		float width_ratio = widest > 0 ? lid.eye_width / widest : 1.0f;
		float corner_clearance = widest > 0 ? nose.clearance(lid.points.front(), left_eye) / widest : 1.0f;

		// Measured toward the forehead, so an upper lid snapped to just below the lower one
		// (see face-tracker.cpp) reads as closed, not as opening again.
		Point lid_gap = face[left_eye ? LEFT_LID_TOP : RIGHT_LID_TOP] -
				face[left_eye ? LEFT_LID_BOTTOM : RIGHT_LID_BOTTOM];
		float openness = lid.eye_width > 0 ? std::max(0.0f, dot(lid_gap, lid.axis_up)) / lid.eye_width : 0.0f;

		float liner_opacity = fade(width_ratio, LINER_FADE_START, LINER_FADE_END) *
				      fade(corner_clearance, LINER_CORNER_FADE_START, LINER_CORNER_FADE_END) *
				      fade(openness, LINER_OPEN_FADE_START, LINER_OPEN_FADE_END);
		float wing_scale = fade(openness, WING_OPEN_FADE_START, WING_OPEN_FADE_END);
		if (liner_opacity > 0)
			add_stroke(liner_sections(lid, look, wing_scale), feather,
				   feather + LASH_SIDE_SOFTNESS * lid.eye_width, liner_opacity, triangles);
		float lash_opacity = fade(width_ratio, LASH_FADE_START, LASH_FADE_END) *
				     fade(corner_clearance, LASH_CORNER_FADE_START, LASH_CORNER_FADE_END) *
				     fade(openness, LASH_OPEN_FADE_START, LASH_OPEN_FADE_END);
		if (look.lashes && lash_opacity > 0)
			for (const auto &lash : lash_strokes(lid))
				add_stroke(lash, std::max(1.0f, feather * 0.5f), std::max(1.0f, feather * 0.5f),
					   lash_opacity, triangles);

		// And whatever still lands over the nose is hidden, with a soft edge.
		float softness = NOSE_EDGE_SOFTNESS * widest;
		for (size_t i = first_vertex; i < triangles.size(); i++)
			triangles[i].alpha *= fade(nose.clearance(triangles[i].position, left_eye), softness, 0.0f);
	}
	return triangles;
}

} // namespace eyeliner
