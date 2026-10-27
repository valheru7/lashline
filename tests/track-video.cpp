// Runs the face tracker on raw RGBA frames from stdin and prints every frame's 478 points as
// one line of "x y" pairs ("none" when there's no face). Used to check the C++ tracker against
// prototype/tracker.py:
//
//   ffmpeg -i clip.mp4 -f rawvideo -pix_fmt rgba - | track-video WIDTH HEIGHT ORT_LIBRARY MODELS_DIR [FPS]
//
// The points come out unsmoothed, as the prototype's do, unless the video's frame rate is given:
// then they're smoothed over time the way the plugin and the apps draw them.

#include "face-tracker.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace eyeliner;

int main(int argc, char **argv)
{
	if (argc != 5 && argc != 6) {
		std::cerr << "usage: track-video WIDTH HEIGHT ORT_LIBRARY MODELS_DIR [FPS]\n";
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
	double frames_per_second = argc == 6 ? std::atof(argv[5]) : 0;
	tracker.set_smoothing(frames_per_second > 0);

	std::vector<uint8_t> frame(size_t(width) * height * 4);
	double total_ms = 0;
	int frames = 0;
	while (std::fread(frame.data(), 1, frame.size(), stdin) == frame.size()) {
		auto start = std::chrono::steady_clock::now();
		double time = frames_per_second > 0 ? frames / frames_per_second : 0;
		auto points = tracker.track({frame.data(), width, height, width * 4}, time);
		total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		frames++;
		if (!points) {
			std::printf("none\n");
			continue;
		}
		for (Point p : *points)
			std::printf("%.3f %.3f ", p.x, p.y);
		std::printf("\n");
	}
	std::cerr << frames << " frames, " << total_ms / frames << " ms per frame\n";
}
