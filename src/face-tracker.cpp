#include "face-tracker.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>

namespace eyeliner {

namespace {

constexpr int DETECTOR_SIZE = 128;
constexpr int LANDMARK_SIZE = 256;
constexpr int DETECTOR_ANCHOR_COUNT = 896;
constexpr int DETECTOR_VALUES_PER_ANCHOR = 16; // box (4) + 6 keypoints (12)
constexpr float MIN_DETECTION_SCORE = 0.5f;
constexpr float MIN_FACE_PRESENCE = 0.5f;
// The region fed to the landmark model is this much larger than the face.
constexpr float ROI_SCALE = 1.5f;
// Detector keypoints 0 and 1 are the eyes; mesh points 33 and 263 are the outer eye corners.
constexpr int DETECTOR_RIGHT_EYE = 0;
constexpr int DETECTOR_LEFT_EYE = 1;
constexpr int MESH_RIGHT_EYE = 33;
constexpr int MESH_LEFT_EYE = 263;

// A 2x3 transform: x' = a*x + b*y + c, y' = d*x + e*y + f.
struct Affine {
	float a, b, c, d, e, f;

	Point apply(Point p) const { return {a * p.x + b * p.y + c, d * p.x + e * p.y + f}; }

	Affine inverse() const
	{
		float determinant = a * e - b * d;
		float ia = e / determinant, ib = -b / determinant;
		float id = -d / determinant, ie = a / determinant;
		return {ia, ib, -(ia * c + ib * f), id, ie, -(id * c + ie * f)};
	}
};

float sigmoid(float x)
{
	x = std::clamp(x, -100.0f, 100.0f);
	return 1.0f / (1.0f + std::exp(-x));
}

// Anchor centers for the short-range BlazeFace model: 16x16 cells with 2 anchors each, then
// 8x8 cells with 6 each.
std::vector<Point> detector_anchors()
{
	std::vector<Point> anchors;
	for (auto [grid, per_cell] : {std::pair{16, 2}, std::pair{8, 6}})
		for (int y = 0; y < grid; y++)
			for (int x = 0; x < grid; x++)
				for (int i = 0; i < per_cell; i++)
					anchors.push_back({(x + 0.5f) / grid, (y + 0.5f) / grid});
	return anchors;
}

// Fills `tensor` (size x size x 3, channels last) by sampling `image` at
// `to_image(output pixel)`, bilinearly, with black outside the image. Each channel is mapped
// from 0..255 to `low`..`high`. Matches OpenCV's warpAffine with INTER_LINEAR.
void sample_into(const ImageView &image, const Affine &to_image, int size, float low, float high,
		 std::vector<float> &tensor)
{
	tensor.resize(size_t(size) * size * 3);
	float scale = (high - low) / 255.0f;
	float *out = tensor.data();
	for (int y = 0; y < size; y++) {
		for (int x = 0; x < size; x++) {
			Point source = to_image.apply({float(x), float(y)});
			int x0 = int(std::floor(source.x)), y0 = int(std::floor(source.y));
			float fx = source.x - x0, fy = source.y - y0;
			float rgb[3] = {0, 0, 0};
			for (int corner = 0; corner < 4; corner++) {
				int cx = x0 + (corner & 1), cy = y0 + (corner >> 1);
				if (cx < 0 || cy < 0 || cx >= image.width || cy >= image.height)
					continue;
				float weight = ((corner & 1) ? fx : 1 - fx) * ((corner >> 1) ? fy : 1 - fy);
				const uint8_t *pixel = image.pixels + size_t(cy) * image.stride + size_t(cx) * 4;
				for (int channel = 0; channel < 3; channel++)
					rgb[channel] += weight * pixel[channel];
			}
			for (float value : rgb)
				*out++ = low + value * scale;
		}
	}
}

float eye_rotation(Point right_eye, Point left_eye)
{
	return std::atan2(left_eye.y - right_eye.y, left_eye.x - right_eye.x);
}

Region region_from_landmarks(const FaceMesh &points)
{
	float rotation = eye_rotation(points[MESH_RIGHT_EYE], points[MESH_LEFT_EYE]);
	float cos = std::cos(rotation), sin = std::sin(rotation);
	// The face's extent measured along the rotated axes.
	float low_x = INFINITY, low_y = INFINITY, high_x = -INFINITY, high_y = -INFINITY;
	for (Point p : points) {
		float upright_x = p.x * cos + p.y * sin;
		float upright_y = -p.x * sin + p.y * cos;
		low_x = std::min(low_x, upright_x);
		high_x = std::max(high_x, upright_x);
		low_y = std::min(low_y, upright_y);
		high_y = std::max(high_y, upright_y);
	}
	float middle_x = (low_x + high_x) / 2, middle_y = (low_y + high_y) / 2;
	Point center{middle_x * cos - middle_y * sin, middle_x * sin + middle_y * cos};
	return {center, std::max(high_x - low_x, high_y - low_y) * ROI_SCALE, rotation};
}

// The transform from crop pixels to image pixels.
Affine region_to_image(const Region &region, int crop_size)
{
	float scale = region.size / crop_size;
	float cos = std::cos(region.rotation) * scale, sin = std::sin(region.rotation) * scale;
	float half = crop_size / 2.0f;
	return {cos, -sin, region.center.x - cos * half + sin * half,
		sin, cos,  region.center.y - sin * half - cos * half};
}

Ort::Session open_session(Ort::Env &env, const std::string &path)
{
	Ort::SessionOptions options;
	options.SetIntraOpNumThreads(2);
	options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
	return Ort::Session(env, std::filesystem::u8path(path).wstring().c_str(), options);
#else
	return Ort::Session(env, path.c_str(), options);
#endif
}

Ort::Value tensor_for(std::vector<float> &data, int size)
{
	static const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	const std::array<int64_t, 4> shape{1, size, size, 3};
	return Ort::Value::CreateTensor<float>(memory, data.data(), data.size(), shape.data(), shape.size());
}

} // namespace

FaceTracker::FaceTracker(const std::string &detector_model_path, const std::string &landmark_model_path)
	: env(ORT_LOGGING_LEVEL_WARNING, "lashline")
{
	detector = open_session(env, detector_model_path);
	landmarker = open_session(env, landmark_model_path);
}

std::optional<FaceMesh> FaceTracker::track(const ImageView &image)
{
	if (!region) {
		region = detect(image);
		if (!region)
			return std::nullopt;
	}
	std::optional<FaceMesh> points = landmarks(image, *region);
	region = points ? std::optional(region_from_landmarks(*points)) : std::nullopt;
	return points;
}

std::optional<Region> FaceTracker::detect(const ImageView &image)
{
	static const std::vector<Point> anchors = detector_anchors();

	// Letterbox the frame into a square, keeping its aspect ratio.
	float scale = float(DETECTOR_SIZE) / std::max(image.width, image.height);
	float pad_x = (DETECTOR_SIZE - image.width * scale) / 2;
	float pad_y = (DETECTOR_SIZE - image.height * scale) / 2;
	Affine to_square{scale, 0, pad_x, 0, scale, pad_y};
	sample_into(image, to_square.inverse(), DETECTOR_SIZE, -1.0f, 1.0f, detector_input);

	const char *input_names[] = {"input"};
	const char *output_names[] = {"regressors", "classificators"};
	Ort::Value input = tensor_for(detector_input, DETECTOR_SIZE);
	auto outputs = detector.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 2);
	const float *regressors = outputs[0].GetTensorData<float>();
	const float *scores = outputs[1].GetTensorData<float>();

	int best = int(std::max_element(scores, scores + DETECTOR_ANCHOR_COUNT) - scores);
	if (sigmoid(scores[best]) < MIN_DETECTION_SCORE)
		return std::nullopt;

	const float *raw = regressors + best * DETECTOR_VALUES_PER_ANCHOR;
	Point anchor = anchors[best];
	// Detector output is in 128-pixel units relative to the anchor; undo the letterbox.
	auto to_image = [&](float x, float y) {
		Point square{x + anchor.x * DETECTOR_SIZE, y + anchor.y * DETECTOR_SIZE};
		return Point{(square.x - pad_x) / scale, (square.y - pad_y) / scale};
	};
	Point center = to_image(raw[0], raw[1]);
	float box_size = std::max(raw[2], raw[3]) / scale;
	Point right_eye = to_image(raw[4 + 2 * DETECTOR_RIGHT_EYE], raw[5 + 2 * DETECTOR_RIGHT_EYE]);
	Point left_eye = to_image(raw[4 + 2 * DETECTOR_LEFT_EYE], raw[5 + 2 * DETECTOR_LEFT_EYE]);
	return Region{center, box_size * ROI_SCALE, eye_rotation(right_eye, left_eye)};
}

std::optional<FaceMesh> FaceTracker::landmarks(const ImageView &image, const Region &face)
{
	Affine to_image = region_to_image(face, LANDMARK_SIZE);
	sample_into(image, to_image, LANDMARK_SIZE, 0.0f, 1.0f, landmark_input);

	const char *input_names[] = {"input_12"};
	const char *output_names[] = {"Identity", "Identity_1"};
	Ort::Value input = tensor_for(landmark_input, LANDMARK_SIZE);
	auto outputs = landmarker.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 2);
	if (sigmoid(outputs[1].GetTensorData<float>()[0]) < MIN_FACE_PRESENCE)
		return std::nullopt;

	const float *crop_points = outputs[0].GetTensorData<float>(); // x, y, z per point
	FaceMesh points;
	for (int i = 0; i < FACE_MESH_POINT_COUNT; i++)
		points[i] = to_image.apply({crop_points[i * 3], crop_points[i * 3 + 1]});
	return points;
}

} // namespace eyeliner
