// Draws the eyeliner onto raw RGBA frames from stdin and writes them to stdout, with a small
// software rasterizer standing in for OBS's GPU drawing. Lets the shape be checked without OBS:
//
//   ffmpeg -i clip.mp4 -f rawvideo -pix_fmt rgba - | render-video W H ORT_LIBRARY MODELS_DIR [lashes [R,G,B]] |
//     ffmpeg -f rawvideo -pix_fmt rgba -s WxH -r 60 -i - out.mp4
//
// LASHLINE_EYE_COLOR=R,G,B turns on the colored contacts in that color, with
// LASHLINE_EYE_COLOR_INTENSITY and LASHLINE_EYE_GLOW (0 to 1) to change them.

#include "eyeliner-shape.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace eyeliner;

namespace {

float COLOR[3] = {20, 16, 18}; // soft black, RGB; a 6th argument "R,G,B" changes it
constexpr float OPACITY = 0.85f;
// The frames' rate, which the tracker's smoothing goes by; the output above is written at 60.
constexpr double FRAMES_PER_SECOND = 60;

// Blends one triangle over the image, interpolating alpha across it like the GPU does.
void fill_triangle(std::vector<uint8_t> &image, int width, int height, const Vertex &a, const Vertex &b,
		   const Vertex &c)
{
	float area = (b.position.x - a.position.x) * (c.position.y - a.position.y) -
		     (c.position.x - a.position.x) * (b.position.y - a.position.y);
	if (std::abs(area) < 1e-6f)
		return;
	int x0 = std::max(0, int(std::floor(std::min({a.position.x, b.position.x, c.position.x}))));
	int x1 = std::min(width - 1, int(std::ceil(std::max({a.position.x, b.position.x, c.position.x}))));
	int y0 = std::max(0, int(std::floor(std::min({a.position.y, b.position.y, c.position.y}))));
	int y1 = std::min(height - 1, int(std::ceil(std::max({a.position.y, b.position.y, c.position.y}))));
	for (int y = y0; y <= y1; y++) {
		for (int x = x0; x <= x1; x++) {
			float px = x + 0.5f, py = y + 0.5f;
			float wa = ((b.position.x - px) * (c.position.y - py) -
				    (c.position.x - px) * (b.position.y - py)) /
				   area;
			float wb = ((c.position.x - px) * (a.position.y - py) -
				    (a.position.x - px) * (c.position.y - py)) /
				   area;
			float wc = 1 - wa - wb;
			if (wa < 0 || wb < 0 || wc < 0)
				continue;
			float alpha = (wa * a.alpha + wb * b.alpha + wc * c.alpha) * OPACITY;
			uint8_t *pixel = &image[(size_t(y) * width + x) * 4];
			for (int channel = 0; channel < 3; channel++)
				pixel[channel] =
					uint8_t(std::lround(pixel[channel] * (1 - alpha) + COLOR[channel] * alpha));
		}
	}
}

// The colored contacts, the same math as data/effects/iris.effect (see IrisVertex). Colors
// are 0 to 1, in gamma space like the GPU's.
struct Color {
	float r, g, b;
};

Color operator+(Color c, float v)
{
	return {c.r + v, c.g + v, c.b + v};
}

float luminance(Color c)
{
	return 0.3f * c.r + 0.59f * c.g + 0.11f * c.b;
}

float smoothstep(float edge0, float edge1, float x)
{
	float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
	return t * t * (3 - 2 * t);
}

// `color` at luminance `light`, keeping its hue, and its saturation as far as it fits.
Color with_luminance(Color color, float light)
{
	Color c = color + (light - luminance(color));
	float l = luminance(c), lowest = std::min({c.r, c.g, c.b}), highest = std::max({c.r, c.g, c.b});
	auto squeeze = [&](float scale) {
		c = {l + (c.r - l) * scale, l + (c.g - l) * scale, l + (c.b - l) * scale};
	};
	if (lowest < 0)
		squeeze(l / (l - lowest));
	if (highest > 1)
		squeeze((1 - l) / (highest - l));
	return c;
}

// The iris's own light and dark, lifted toward the new color's lightness (colored contacts
// over dark eyes read lighter), but never darkened, so catchlights stay.
Color tinted(Color frame, Color color)
{
	float light = luminance(frame);
	return with_luminance(color, std::max(light, light + (luminance(color) - light) * 0.25f));
}

float iris_ring(float r)
{
	return smoothstep(0.28f, 0.45f, r) * (1 - smoothstep(0.85f, 1.0f, r));
}

float glow_falloff(float r)
{
	return 1 - smoothstep(0.6f, 1.8f, r);
}

enum class IrisPass { Tint, Glow };

// Draws one iris triangle, reading `original` (the frame before any drawing, as the GPU reads
// the frame's texture) and writing `image`.
void fill_iris_triangle(std::vector<uint8_t> &image, const std::vector<uint8_t> &original, int width, int height,
			const IrisVertex &a, const IrisVertex &b, const IrisVertex &c, Color color, IrisPass pass)
{
	float area = (b.position.x - a.position.x) * (c.position.y - a.position.y) -
		     (c.position.x - a.position.x) * (b.position.y - a.position.y);
	if (std::abs(area) < 1e-6f)
		return;
	int x0 = std::max(0, int(std::floor(std::min({a.position.x, b.position.x, c.position.x}))));
	int x1 = std::min(width - 1, int(std::ceil(std::max({a.position.x, b.position.x, c.position.x}))));
	int y0 = std::max(0, int(std::floor(std::min({a.position.y, b.position.y, c.position.y}))));
	int y1 = std::min(height - 1, int(std::ceil(std::max({a.position.y, b.position.y, c.position.y}))));
	for (int y = y0; y <= y1; y++) {
		for (int x = x0; x <= x1; x++) {
			float px = x + 0.5f, py = y + 0.5f;
			float wa = ((b.position.x - px) * (c.position.y - py) -
				    (c.position.x - px) * (b.position.y - py)) /
				   area;
			float wb = ((c.position.x - px) * (a.position.y - py) -
				    (a.position.x - px) * (c.position.y - py)) /
				   area;
			float wc = 1 - wa - wb;
			if (wa < 0 || wb < 0 || wc < 0)
				continue;
			float r = std::hypot(wa * a.iris.x + wb * b.iris.x + wc * c.iris.x,
					     wa * a.iris.y + wb * b.iris.y + wc * c.iris.y);
			size_t offset = (size_t(y) * width + x) * 4;
			uint8_t *pixel = &image[offset];
			Color out;
			if (pass == IrisPass::Tint) {
				const uint8_t *source = &original[offset];
				Color frame{source[0] / 255.0f, source[1] / 255.0f, source[2] / 255.0f};
				Color target = tinted(frame, color);
				float alpha = (wa * a.tint + wb * b.tint + wc * c.tint) * iris_ring(r);
				out = {pixel[0] / 255.0f * (1 - alpha) + target.r * alpha,
				       pixel[1] / 255.0f * (1 - alpha) + target.g * alpha,
				       pixel[2] / 255.0f * (1 - alpha) + target.b * alpha};
			} else {
				// Screen: lightens, and never past white.
				float strength = (wa * a.glow + wb * b.glow + wc * c.glow) * glow_falloff(r);
				Color light{color.r * strength, color.g * strength, color.b * strength};
				out = {light.r + pixel[0] / 255.0f * (1 - light.r),
				       light.g + pixel[1] / 255.0f * (1 - light.g),
				       light.b + pixel[2] / 255.0f * (1 - light.b)};
			}
			pixel[0] = uint8_t(std::lround(std::clamp(out.r, 0.0f, 1.0f) * 255));
			pixel[1] = uint8_t(std::lround(std::clamp(out.g, 0.0f, 1.0f) * 255));
			pixel[2] = uint8_t(std::lround(std::clamp(out.b, 0.0f, 1.0f) * 255));
		}
	}
}

} // namespace

int main(int argc, char **argv)
{
	if (argc < 5) {
		std::cerr << "usage: render-video WIDTH HEIGHT ORT_LIBRARY MODELS_DIR [lashes]\n";
		return 2;
	}
	int width = std::atoi(argv[1]), height = std::atoi(argv[2]);
	std::string error = load_onnx_runtime(argv[3]);
	if (!error.empty()) {
		std::cerr << error << "\n";
		return 1;
	}
	std::string models = argv[4];
	FaceTracker tracker(models + "/face_detector.onnx", models + "/face_landmarks_detector.onnx");
	// LASHLINE_NO_LID_SNAP=1 shows the landmark model's own lid line, for comparing.
	tracker.set_lid_snapping(std::getenv("LASHLINE_NO_LID_SNAP") == nullptr);
	Look look;
	look.lashes = argc > 5 && std::strcmp(argv[5], "lashes") == 0;
	if (argc > 6)
		std::sscanf(argv[6], "%f,%f,%f", &COLOR[0], &COLOR[1], &COLOR[2]);
	Color eye_color{};
	if (const char *setting = std::getenv("LASHLINE_EYE_COLOR")) {
		look.eye_color = std::sscanf(setting, "%f,%f,%f", &eye_color.r, &eye_color.g, &eye_color.b) == 3;
		eye_color = {eye_color.r / 255, eye_color.g / 255, eye_color.b / 255};
	}
	if (const char *setting = std::getenv("LASHLINE_EYE_COLOR_INTENSITY"))
		look.eye_color_intensity = float(std::atof(setting));
	if (const char *setting = std::getenv("LASHLINE_EYE_GLOW"))
		look.eye_glow = float(std::atof(setting));

	std::vector<uint8_t> frame(size_t(width) * height * 4), original;
	double iris_seconds = 0;
	int frames = 0;
	for (int index = 0; std::fread(frame.data(), 1, frame.size(), stdin) == frame.size(); index++) {
		frames++;
		if (auto face = tracker.track({frame.data(), width, height, width * 4}, index / FRAMES_PER_SECOND)) {
			// The irises go first, so the liner draws over any glow on the lid.
			auto start = std::chrono::steady_clock::now();
			std::vector<IrisVertex> iris = iris_triangles(*face, look);
			iris_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
			original = frame;
			for (IrisPass pass : {IrisPass::Tint, IrisPass::Glow})
				for (size_t i = 0; i + 2 < iris.size(); i += 3)
					fill_iris_triangle(frame, original, width, height, iris[i], iris[i + 1],
							   iris[i + 2], eye_color, pass);

			std::vector<Vertex> triangles = eyeliner_triangles(*face, look);
			for (size_t i = 0; i + 2 < triangles.size(); i += 3)
				fill_triangle(frame, width, height, triangles[i], triangles[i + 1], triangles[i + 2]);
		}
		std::fwrite(frame.data(), 1, frame.size(), stdout);
	}
	if (look.eye_color && frames > 0)
		std::cerr << "iris mesh: " << iris_seconds * 1000 / frames << " ms per frame\n";
}
