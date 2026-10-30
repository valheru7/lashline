#include "face-tracker.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <utility>

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
// While following fewer faces than asked for, look for new ones this often (in frames).
constexpr int REDETECT_INTERVAL = 10;
// Detector keypoints 0 and 1 are the eyes; mesh points 33 and 263 are the outer eye corners.
constexpr int DETECTOR_RIGHT_EYE = 0;
constexpr int DETECTOR_LEFT_EYE = 1;
constexpr int MESH_RIGHT_EYE = 33;
constexpr int MESH_LEFT_EYE = 263;
// How each point is smoothed over time; see one-euro-filter.hpp. Speeds are in face sizes per
// second, so the same settings work for a face near the camera or far from it. The speed cutoff
// is higher than the paper's usual 1 Hz so a head turn is noticed within a frame or two instead
// of trailing at its start. Tuned on stream footage: still eyes jitter 3.5 to 6 times less,
// the liner trails a head turn by under half a frame, and it still closes with a blink, since a
// blinking lid moves fast enough to get through almost unsmoothed.
constexpr OneEuroSettings POINT_SMOOTHING{0.5f, 40.0f, 4.0f};
// A face whose frames come further apart than this (seconds) starts its smoothing over, rather
// than blending in where it was before the video paused or skipped.
constexpr double MAX_SMOOTHING_GAP = 0.5;

// Snapping the upper lids to the lash line. The landmark model sees each eye only about 35
// pixels across, so its lid line can sit several percent of an eye width off the real one,
// most of all when the eyes are lowered. The lash line is the darkest line along the lid, so
// each point between the corners moves to the darkest spot near it, across the lid, in the
// frame itself. Sizes are fractions of the eye's width.
// Upper-lid points between the corners, inner to outer; the corners stay where the model put them.
constexpr std::array<int, 7> RIGHT_UPPER_LID_MIDDLE{173, 157, 158, 159, 160, 161, 246};
constexpr std::array<int, 7> LEFT_UPPER_LID_MIDDLE{398, 384, 385, 386, 387, 388, 466};
constexpr std::array<std::pair<int, int>, 2> EYE_CORNERS{{{133, 33}, {362, 263}}};
constexpr int FOREHEAD = 10;
constexpr int CHIN = 152;
// How far either way across the lid to look.
constexpr float SNAP_RANGE = 0.1f;
// How wide a strip along the lid each sample averages, so single lashes and noise don't count.
constexpr float SNAP_STRIP = 0.04f;
// Darker spots further from the model's guess must be darker by this much (0..255 brightness at
// the edge of the range), so the iris or a crease far off doesn't win over the lash line.
constexpr float SNAP_DISTANCE_COST = 25.0f;
// A lid with less contrast than this (darkest against brightest along the search) has no clear
// lash line to snap to, so it's left as it is.
constexpr float SNAP_MIN_CONTRAST = 20.0f;

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
			if (image.order == PixelOrder::BGRA)
				std::swap(rgb[0], rgb[2]);
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

// Brightness (0..255) at a point, bilinearly; nothing outside the image.
std::optional<float> brightness(const ImageView &image, Point at)
{
	int x0 = int(std::floor(at.x)), y0 = int(std::floor(at.y));
	if (x0 < 0 || y0 < 0 || x0 + 1 >= image.width || y0 + 1 >= image.height)
		return std::nullopt;
	float fx = at.x - x0, fy = at.y - y0, total = 0;
	for (int corner = 0; corner < 4; corner++) {
		int cx = x0 + (corner & 1), cy = y0 + (corner >> 1);
		const uint8_t *pixel = image.pixels + size_t(cy) * image.stride + size_t(cx) * 4;
		// Green carries most of the brightness and is in the same place in RGBA and BGRA.
		float weight = ((corner & 1) ? fx : 1 - fx) * ((corner >> 1) ? fy : 1 - fy);
		total += weight * pixel[1];
	}
	return total;
}

// Moves each eye's upper-lid points onto the darkest line near them (see SNAP_RANGE).
void snap_upper_lids(FaceMesh &points, const ImageView &image)
{
	Point up{points[FOREHEAD].x - points[CHIN].x, points[FOREHEAD].y - points[CHIN].y};
	const std::array<const std::array<int, 7> *, 2> lids{&RIGHT_UPPER_LID_MIDDLE, &LEFT_UPPER_LID_MIDDLE};
	for (size_t eye = 0; eye < 2; eye++) {
		Point inner = points[EYE_CORNERS[eye].first], outer = points[EYE_CORNERS[eye].second];
		float eye_width = std::hypot(outer.x - inner.x, outer.y - inner.y);
		if (eye_width < 4)
			continue; // too small to see a lash line
		Point along{(outer.x - inner.x) / eye_width, (outer.y - inner.y) / eye_width};
		Point across{along.y, -along.x}; // toward the forehead
		if (across.x * up.x + across.y * up.y < 0)
			across = {-across.x, -across.y};

		float range = SNAP_RANGE * eye_width, half_strip = SNAP_STRIP * eye_width / 2;
		int steps = std::max(4, int(range * 2)); // about half-pixel steps
		std::array<float, 7> offsets{};
		for (size_t i = 0; i < offsets.size(); i++) {
			Point point = points[(*lids[eye])[i]];
			float best_score = INFINITY, darkest = INFINITY, brightest = -INFINITY;
			for (int step = -steps; step <= steps; step++) {
				float offset = range * step / steps;
				float total = 0;
				int samples = 0;
				for (float side : {-half_strip, 0.0f, half_strip}) {
					Point at{point.x + across.x * offset + along.x * side,
						 point.y + across.y * offset + along.y * side};
					if (std::optional<float> value = brightness(image, at)) {
						total += *value;
						samples++;
					}
				}
				if (samples == 0)
					continue;
				float value = total / samples;
				darkest = std::min(darkest, value);
				brightest = std::max(brightest, value);
				float distance = offset / range;
				float score = value + SNAP_DISTANCE_COST * distance * distance;
				if (score < best_score) {
					best_score = score;
					offsets[i] = offset;
				}
			}
			if (brightest - darkest < SNAP_MIN_CONTRAST)
				offsets[i] = 0;
		}

		// One point snapping onto something else (a stray lash, the iris) would kink the line,
		// so each point takes the middle of its own offset and its neighbours'. The two ends have
		// only one neighbour and keep their own.
		std::array<float, 7> settled = offsets;
		for (size_t i = 1; i + 1 < offsets.size(); i++) {
			std::array<float, 3> nearby{offsets[i - 1], offsets[i], offsets[i + 1]};
			std::sort(nearby.begin(), nearby.end());
			settled[i] = nearby[1];
		}
		for (size_t i = 0; i < settled.size(); i++) {
			Point &point = points[(*lids[eye])[i]];
			point.x += across.x * settled[i];
			point.y += across.y * settled[i];
		}
	}
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

// Whether two regions are on the same face: their centers are closer than 40% of the larger
// one. Side-by-side faces in a selfie are about a face width (two thirds of a region) apart.
bool same_face(const Region &a, const Region &b)
{
	float dx = a.center.x - b.center.x, dy = a.center.y - b.center.y;
	return std::hypot(dx, dy) < 0.4f * std::max(a.size, b.size);
}

bool same_face_as_any(const Region &region, const std::vector<Region> &others)
{
	return std::any_of(others.begin(), others.end(), [&](const Region &other) { return same_face(region, other); });
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

// ONNX Runtime's builds for Apple platforms and Linux send usage telemetry to Microsoft
// (mobile.events.data.microsoft.com). Lashline promises nothing leaves the device, so it's
// turned off twice: the variable stops the telemetry system from starting with the runtime,
// and the call covers anything after.
Ort::Env make_env()
{
#ifdef _WIN32
	_putenv_s("ORT_DISABLE_TELEMETRY", "1");
#else
	setenv("ORT_DISABLE_TELEMETRY", "1", 1);
#endif
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "lashline");
	env.DisableTelemetryEvents();
	return env;
}

Ort::Value tensor_for(std::vector<float> &data, int size)
{
	static const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	const std::array<int64_t, 4> shape{1, size, size, 3};
	return Ort::Value::CreateTensor<float>(memory, data.data(), data.size(), shape.data(), shape.size());
}

} // namespace

FaceTracker::FaceTracker(const std::string &detector_model_path, const std::string &landmark_model_path)
	: env(make_env())
{
	detector = open_session(env, detector_model_path);
	landmarker = open_session(env, landmark_model_path);
}

std::optional<FaceMesh> FaceTracker::track(const ImageView &image, double time)
{
	std::vector<FaceMesh> found = track_all(image, 1, time);
	return found.empty() ? std::nullopt : std::optional(found.front());
}

std::vector<FaceMesh> FaceTracker::track_all(const ImageView &image, int max_faces, double time)
{
	if (int(faces.size()) > max_faces)
		faces.resize(size_t(max_faces));

	auto already_followed = [](const std::vector<FollowedFace> &followed, const Region &region) {
		return std::any_of(followed.begin(), followed.end(),
				   [&](const FollowedFace &face) { return same_face(region, face.region); });
	};

	// Look for faces whenever there are none, and every so often when there's room for more.
	frames_since_detection++;
	if (faces.empty() || (int(faces.size()) < max_faces && frames_since_detection >= REDETECT_INTERVAL)) {
		frames_since_detection = 0;
		for (const Region &found : detect(image)) {
			if (int(faces.size()) >= max_faces)
				break;
			if (!already_followed(faces, found))
				faces.push_back(FollowedFace{found});
		}
	}

	std::vector<FaceMesh> found_points;
	std::vector<FollowedFace> followed;
	for (FollowedFace &face : faces) {
		std::optional<FaceMesh> points = landmarks(image, face.region);
		if (!points)
			continue; // lost this one, and its smoothing with it
		// The next region comes from the points as the model gave them, as in MediaPipe.
		Region next = region_from_landmarks(*points);
		// Two regions can drift onto the same face; keep the first.
		if (already_followed(followed, next))
			continue;
		face.region = next;
		if (lid_snapping)
			snap_upper_lids(*points, image);
		found_points.push_back(smoothing ? face.smooth(*points, next.size / ROI_SCALE, time) : *points);
		followed.push_back(std::move(face));
	}
	faces = std::move(followed);
	return found_points;
}

FaceMesh FaceTracker::FollowedFace::smooth(const FaceMesh &points, float face_size, double time)
{
	// Speeds are divided by the face's size, so a face with no size (points the model put all in
	// one spot) would make them infinite. Pass such points through and start over next frame.
	if (!(face_size > 0)) {
		last_time.reset();
		return points;
	}

	float seconds = last_time ? float(time - *last_time) : 0;
	// A new face, a gap, or time going backwards: start from these points as they are.
	if (!last_time || seconds <= 0 || seconds > MAX_SMOOTHING_GAP)
		filters = {};
	last_time = time;

	FaceMesh smoothed;
	for (size_t i = 0; i < points.size(); i++) {
		smoothed[i].x = filters[i].x.apply(points[i].x, seconds, face_size, POINT_SMOOTHING);
		smoothed[i].y = filters[i].y.apply(points[i].y, seconds, face_size, POINT_SMOOTHING);
	}
	return smoothed;
}

std::vector<Region> FaceTracker::detect(const ImageView &image)
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

	// Neighboring anchors fire on the same face, so take the most confident first and skip
	// later ones on a face already taken.
	std::vector<int> confident;
	for (int i = 0; i < DETECTOR_ANCHOR_COUNT; i++)
		if (sigmoid(scores[i]) >= MIN_DETECTION_SCORE)
			confident.push_back(i);
	std::sort(confident.begin(), confident.end(), [&](int a, int b) { return scores[a] > scores[b]; });

	std::vector<Region> faces;
	for (int index : confident) {
		const float *raw = regressors + index * DETECTOR_VALUES_PER_ANCHOR;
		Point anchor = anchors[index];
		// Detector output is in 128-pixel units relative to the anchor; undo the letterbox.
		auto to_image = [&](float x, float y) {
			Point square{x + anchor.x * DETECTOR_SIZE, y + anchor.y * DETECTOR_SIZE};
			return Point{(square.x - pad_x) / scale, (square.y - pad_y) / scale};
		};
		Point center = to_image(raw[0], raw[1]);
		float box_size = std::max(raw[2], raw[3]) / scale;
		Point right_eye = to_image(raw[4 + 2 * DETECTOR_RIGHT_EYE], raw[5 + 2 * DETECTOR_RIGHT_EYE]);
		Point left_eye = to_image(raw[4 + 2 * DETECTOR_LEFT_EYE], raw[5 + 2 * DETECTOR_LEFT_EYE]);
		Region face{center, box_size * ROI_SCALE, eye_rotation(right_eye, left_eye)};
		if (!same_face_as_any(face, faces))
			faces.push_back(face);
	}
	return faces;
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
