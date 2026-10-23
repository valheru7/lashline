// Draws the eyeliner onto raw RGBA frames from stdin and writes them to stdout, with a small
// software rasterizer standing in for OBS's GPU drawing. Lets the shape be checked without OBS:
//
//   ffmpeg -i clip.mp4 -f rawvideo -pix_fmt rgba - | render-video W H ORT_LIBRARY MODELS_DIR [lashes] |
//     ffmpeg -f rawvideo -pix_fmt rgba -s WxH -r 60 -i - out.mp4

#include "eyeliner-shape.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>

using namespace eyeliner;

namespace {

constexpr float COLOR[3] = {20, 16, 18}; // soft black, RGB
constexpr float OPACITY = 0.85f;

// Blends one triangle over the image, interpolating alpha across it like the GPU does.
void fill_triangle(std::vector<uint8_t> &image, int width, int height, const Vertex &a, const Vertex &b, const Vertex &c)
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
			float wa = ((b.position.x - px) * (c.position.y - py) - (c.position.x - px) * (b.position.y - py)) / area;
			float wb = ((c.position.x - px) * (a.position.y - py) - (a.position.x - px) * (c.position.y - py)) / area;
			float wc = 1 - wa - wb;
			if (wa < 0 || wb < 0 || wc < 0)
				continue;
			float alpha = (wa * a.alpha + wb * b.alpha + wc * c.alpha) * OPACITY;
			uint8_t *pixel = &image[(size_t(y) * width + x) * 4];
			for (int channel = 0; channel < 3; channel++)
				pixel[channel] = uint8_t(std::lround(pixel[channel] * (1 - alpha) + COLOR[channel] * alpha));
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
	Look look;
	look.lashes = argc > 5 && std::strcmp(argv[5], "lashes") == 0;

	std::vector<uint8_t> frame(size_t(width) * height * 4);
	while (std::fread(frame.data(), 1, frame.size(), stdin) == frame.size()) {
		if (auto face = tracker.track({frame.data(), width, height, width * 4})) {
			std::vector<Vertex> triangles = eyeliner_triangles(*face, look);
			for (size_t i = 0; i + 2 < triangles.size(); i += 3)
				fill_triangle(frame, width, height, triangles[i], triangles[i + 1], triangles[i + 2]);
		}
		std::fwrite(frame.data(), 1, frame.size(), stdout);
	}
}
