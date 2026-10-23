// The OBS filter.
//
// Face tracking takes a few milliseconds and the frame has to come back from the GPU first, so
// a frame's face points are only ready a couple of frames after the frame itself. Drawing the
// liner from those late points onto the current frame left it behind whenever the head moved
// (during a turn, on the bridge of the nose). So the filter holds the video back instead:
//
//   1. Each new frame is copied into a small ring of textures, and a downscaled copy is read
//      back (one frame later, so the GPU is never waited on) and handed to the tracker thread.
//   2. What OBS shows is the frame from DISPLAY_DELAY frames ago, with the liner drawn from
//      that same frame's face points.
//
// The video runs DISPLAY_DELAY frames late, the way OBS's own Render Delay filter delays it,
// and OBS's render thread never waits on the GPU or on the tracker.

#include "eyeliner-shape.hpp"
#include "face-tracker.hpp"

#include <obs-module.h>
#include <plugin-support.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>

using namespace eyeliner;

namespace {

// Frames between capturing a frame and showing it. The face points are read back one frame
// after capture and tracked in a few milliseconds, so they're ready by the frame after that.
constexpr uint64_t DISPLAY_DELAY = 2;
// Frames kept: the ones waiting to be shown, plus the one being captured.
constexpr size_t FRAME_SLOTS = DISPLAY_DELAY + 2;
// Results kept, so a slightly late lookup still finds its frame.
constexpr size_t RESULT_HISTORY = 8;
// Frames the tracker may have queued; it normally keeps up, and this caps memory if it can't.
constexpr size_t MAX_QUEUED_FRAMES = 3;
// The tracker sees the frame at most this wide. A face filling a third of a 1080p webcam
// frame is still ~200 pixels across, plenty for the 256x256 landmark model.
constexpr uint32_t MAX_CAPTURE_WIDTH = 960;
// Both eyes with lashes come to about 3,200 vertices.
constexpr uint32_t VERTEX_CAPACITY = 8192;

constexpr const char *SETTING_COLOR = "color";
constexpr const char *SETTING_OPACITY = "opacity";
constexpr const char *SETTING_THICKNESS = "thickness";
constexpr const char *SETTING_WING_LENGTH = "wing_length";
constexpr const char *SETTING_WING_ANGLE = "wing_angle";
constexpr const char *SETTING_SOFTNESS = "softness";
constexpr const char *SETTING_LASHES = "lashes";

// The points of a face, with the size of the frame they were found in.
struct TrackedFace {
	FaceMesh points;
	uint32_t frame_width;
	uint32_t frame_height;
};

// What the tracker found in one frame. `frame_id` 0 means no frame has been tracked yet.
struct TrackingResult {
	uint64_t frame_id = 0;
	std::optional<TrackedFace> face;
};

// Runs the face tracker on its own thread, one frame at a time, in the order they arrive.
class TrackerThread {
public:
	TrackerThread() : thread([this] { run(); }) {}

	~TrackerThread()
	{
		{
			std::lock_guard lock(mutex);
			stopping = true;
		}
		wake.notify_one();
		thread.join();
	}

	void submit(uint64_t frame_id, const uint8_t *pixels, uint32_t linesize, uint32_t width, uint32_t height)
	{
		Frame frame{frame_id, width, height, std::vector<uint8_t>(size_t(width) * height * 4)};
		for (uint32_t row = 0; row < height; row++)
			std::copy_n(pixels + size_t(row) * linesize, width * 4, &frame.pixels[size_t(row) * width * 4]);
		{
			std::lock_guard lock(mutex);
			if (queue.size() == MAX_QUEUED_FRAMES)
				queue.pop_front();
			queue.push_back(std::move(frame));
		}
		wake.notify_one();
	}

	// The result for `frame_id`, or if that frame wasn't tracked, for the newest frame before it.
	TrackingResult result_for(uint64_t frame_id)
	{
		std::lock_guard lock(mutex);
		TrackingResult best;
		for (const TrackingResult &result : results)
			if (result.frame_id <= frame_id && result.frame_id > best.frame_id)
				best = result;
		return best;
	}

private:
	struct Frame {
		uint64_t id;
		uint32_t width;
		uint32_t height;
		std::vector<uint8_t> pixels;
	};

	void run()
	{
		std::unique_ptr<FaceTracker> tracker = load_tracker();
		while (true) {
			Frame frame;
			{
				std::unique_lock lock(mutex);
				wake.wait(lock, [this] { return stopping || !queue.empty(); });
				if (stopping)
					return;
				frame = std::move(queue.front());
				queue.pop_front();
			}
			if (!tracker)
				continue;

			std::optional<FaceMesh> points;
			try {
				points = tracker->track({frame.pixels.data(), int(frame.width), int(frame.height),
							 int(frame.width * 4)});
			} catch (const Ort::Exception &error) {
				obs_log(LOG_ERROR, "Face tracking failed: %s", error.what());
			}
			std::lock_guard lock(mutex);
			TrackingResult &slot = results[frame.id % RESULT_HISTORY];
			slot.frame_id = frame.id;
			slot.face = points ? std::optional(TrackedFace{*points, frame.width, frame.height})
					   : std::nullopt;
		}
	}

	static std::unique_ptr<FaceTracker> load_tracker()
	{
		std::string library = module_file(std::string("onnxruntime/") + ONNXRUNTIME_LIBRARY_NAME);
		std::string detector = module_file("models/face_detector.onnx");
		std::string landmarker = module_file("models/face_landmarks_detector.onnx");
		if (library.empty() || detector.empty() || landmarker.empty()) {
			obs_log(LOG_ERROR, "The plugin's data files are missing; reinstall it");
			return nullptr;
		}
		std::string error = load_onnx_runtime(library);
		if (!error.empty()) {
			obs_log(LOG_ERROR, "%s", error.c_str());
			return nullptr;
		}
		try {
			return std::make_unique<FaceTracker>(detector, landmarker);
		} catch (const Ort::Exception &exception) {
			obs_log(LOG_ERROR, "Couldn't load the face models: %s", exception.what());
			return nullptr;
		}
	}

	static std::string module_file(const std::string &name)
	{
		char *path = obs_module_file(name.c_str());
		std::string result = path ? path : "";
		bfree(path);
		return result;
	}

	std::mutex mutex;
	std::condition_variable wake;
	bool stopping = false;
	std::deque<Frame> queue;
	std::array<TrackingResult, RESULT_HISTORY> results;
	std::thread thread; // last, so it starts after everything it uses exists
};

struct Settings {
	Look look;
	vec4 color; // RGB from the color picker, A from the opacity slider
};

Settings read_settings(obs_data_t *data)
{
	Settings settings;
	settings.look.outer_thickness = float(obs_data_get_double(data, SETTING_THICKNESS)) / 100;
	settings.look.wing_length = float(obs_data_get_double(data, SETTING_WING_LENGTH)) / 100;
	settings.look.wing_lift_degrees = float(obs_data_get_double(data, SETTING_WING_ANGLE));
	settings.look.softness = float(obs_data_get_double(data, SETTING_SOFTNESS)) / 100;
	settings.look.lashes = obs_data_get_bool(data, SETTING_LASHES);
	// OBS stores colors as 0xAABBGGRR.
	uint32_t color = uint32_t(obs_data_get_int(data, SETTING_COLOR));
	float opacity = float(obs_data_get_int(data, SETTING_OPACITY)) / 100;
	vec4_set(&settings.color, (color & 0xff) / 255.0f, ((color >> 8) & 0xff) / 255.0f,
		 ((color >> 16) & 0xff) / 255.0f, opacity);
	return settings;
}

// One captured frame, waiting to be shown.
struct FrameSlot {
	gs_texrender_t *texture = nullptr;
	uint64_t frame_id = 0;
};

class EyelinerFilter {
public:
	EyelinerFilter(obs_source_t *context, obs_data_t *data) : context(context) { update(data); }

	~EyelinerFilter()
	{
		obs_enter_graphics();
		for (FrameSlot &slot : slots)
			gs_texrender_destroy(slot.texture);
		gs_texrender_destroy(capture);
		gs_stagesurface_destroy(stage);
		gs_vertexbuffer_destroy(vertices);
		obs_leave_graphics();
	}

	void update(obs_data_t *data)
	{
		std::lock_guard lock(settings_mutex);
		settings = read_settings(data);
	}

	void render()
	{
		obs_source_t *target = obs_filter_get_target(context);
		uint32_t width = target ? obs_source_get_base_width(target) : 0;
		uint32_t height = target ? obs_source_get_base_height(target) : 0;
		if (width == 0 || height == 0) {
			obs_source_skip_video_filter(context);
			return;
		}

		// OBS can render a source more than once per frame (preview, projectors); capture once.
		uint64_t video_time = obs_get_video_frame_time();
		if (video_time != last_video_time) {
			last_video_time = video_time;
			capture_frame(target, width, height);
		}

		uint64_t shown_id = latest_frame_id > DISPLAY_DELAY ? latest_frame_id - DISPLAY_DELAY : 1;
		const FrameSlot *shown = &slots[shown_id % FRAME_SLOTS];
		if (shown->frame_id != shown_id) // just started: nothing that old yet
			shown = &slots[latest_frame_id % FRAME_SLOTS];
		gs_texture_t *texture = gs_texrender_get_texture(shown->texture);
		if (!texture) {
			obs_source_skip_video_filter(context);
			return;
		}

		draw_texture(texture, width, height);
		TrackingResult result = tracker.result_for(shown->frame_id);
		if (result.face)
			draw_liner(*result.face, width, height);
	}

private:
	// Copies the frame beneath this filter into the next slot and starts the tracker's copy.
	void capture_frame(obs_source_t *target, uint32_t width, uint32_t height)
	{
		latest_frame_id++;
		FrameSlot &slot = slots[latest_frame_id % FRAME_SLOTS];
		if (!slot.texture)
			slot.texture = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
		gs_texrender_reset(slot.texture);

		gs_blend_state_push();
		gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
		if (gs_texrender_begin_with_color_space(slot.texture, width, height, GS_CS_SRGB)) {
			vec4 clear_color;
			vec4_zero(&clear_color);
			gs_clear(GS_CLEAR_COLOR, &clear_color, 0.0f, 0);
			gs_ortho(0.0f, float(width), 0.0f, float(height), -100.0f, 100.0f);
			// The same way OBS's own GPU delay filter renders what's beneath it.
			obs_source_t *parent = obs_filter_get_parent(context);
			uint32_t parent_flags = obs_source_get_output_flags(target);
			bool custom_draw = (parent_flags & OBS_SOURCE_CUSTOM_DRAW) != 0;
			bool async = (parent_flags & OBS_SOURCE_ASYNC) != 0;
			if (target == parent && !custom_draw && !async)
				obs_source_default_render(target);
			else
				obs_source_video_render(target);
			gs_texrender_end(slot.texture);
		}
		gs_blend_state_pop();
		slot.frame_id = latest_frame_id;

		send_to_tracker(slot, width, height);
	}

	// Reads back last frame's downscaled copy for the tracker, then starts this frame's.
	void send_to_tracker(const FrameSlot &slot, uint32_t width, uint32_t height)
	{
		if (stage_frame_id != 0) {
			uint8_t *pixels;
			uint32_t linesize;
			if (gs_stagesurface_map(stage, &pixels, &linesize)) {
				tracker.submit(stage_frame_id, pixels, linesize, gs_stagesurface_get_width(stage),
					       gs_stagesurface_get_height(stage));
				gs_stagesurface_unmap(stage);
			}
			stage_frame_id = 0;
		}

		gs_texture_t *texture = gs_texrender_get_texture(slot.texture);
		if (!texture)
			return;
		uint32_t capture_width = std::min(width, MAX_CAPTURE_WIDTH);
		uint32_t capture_height = std::max(1u, uint32_t(uint64_t(height) * capture_width / width));
		if (!capture)
			capture = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
		if (!stage || gs_stagesurface_get_width(stage) != capture_width ||
		    gs_stagesurface_get_height(stage) != capture_height) {
			gs_stagesurface_destroy(stage);
			stage = gs_stagesurface_create(capture_width, capture_height, GS_RGBA);
		}

		gs_texrender_reset(capture);
		gs_blend_state_push();
		gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
		if (gs_texrender_begin(capture, capture_width, capture_height)) {
			// The full-size frame into the smaller texture, which scales it down. The bytes are
			// copied as they are (no sRGB conversion): the face models expect ordinary RGB.
			gs_ortho(0.0f, float(width), 0.0f, float(height), -100.0f, 100.0f);
			const bool previous_srgb = gs_framebuffer_srgb_enabled();
			gs_enable_framebuffer_srgb(false);
			gs_effect_t *effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
			gs_effect_set_texture(gs_effect_get_param_by_name(effect, "image"), texture);
			while (gs_effect_loop(effect, "Draw"))
				gs_draw_sprite(texture, 0, width, height);
			gs_enable_framebuffer_srgb(previous_srgb);
			gs_texrender_end(capture);
			gs_stage_texture(stage, gs_texrender_get_texture(capture));
			stage_frame_id = slot.frame_id;
		}
		gs_blend_state_pop();
	}

	static void draw_texture(gs_texture_t *texture, uint32_t width, uint32_t height)
	{
		gs_effect_t *effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
		const bool previous_srgb = gs_framebuffer_srgb_enabled();
		gs_enable_framebuffer_srgb(true);
		gs_effect_set_texture_srgb(gs_effect_get_param_by_name(effect, "image"), texture);
		while (gs_effect_loop(effect, "Draw"))
			gs_draw_sprite(texture, 0, width, height);
		gs_enable_framebuffer_srgb(previous_srgb);
	}

	void draw_liner(TrackedFace face, uint32_t width, uint32_t height)
	{
		Settings current;
		{
			std::lock_guard lock(settings_mutex);
			current = settings;
		}

		// The tracker saw a downscaled frame; bring its points up to the source's size.
		float scale_x = float(width) / face.frame_width, scale_y = float(height) / face.frame_height;
		for (Point &point : face.points) {
			point.x *= scale_x;
			point.y *= scale_y;
		}
		std::vector<Vertex> triangles = eyeliner_triangles(face.points, current.look);
		uint32_t count = std::min(uint32_t(triangles.size()), VERTEX_CAPACITY) / 3 * 3;
		if (count == 0)
			return;

		if (!vertices)
			vertices = create_vertex_buffer();
		gs_vb_data *data = gs_vertexbuffer_get_data(vertices);
		for (uint32_t i = 0; i < count; i++) {
			vec3_set(&data->points[i], triangles[i].position.x, triangles[i].position.y, 0);
			// White, with the fade in alpha; the effect multiplies in the liner's color.
			uint32_t alpha = uint32_t(std::clamp(triangles[i].alpha, 0.0f, 1.0f) * 255 + 0.5f);
			data->colors[i] = (alpha << 24) | 0x00ffffff;
		}
		gs_vertexbuffer_flush(vertices);

		gs_effect_t *solid = obs_get_base_effect(OBS_EFFECT_SOLID);
		gs_effect_set_vec4(gs_effect_get_param_by_name(solid, "color"), &current.color);

		// Blend in gamma space, the way makeup filters (and the prototype) look.
		const bool previous_srgb = gs_framebuffer_srgb_enabled();
		gs_enable_framebuffer_srgb(false);
		gs_blend_state_push();
		gs_enable_blending(true);
		gs_blend_function_separate(GS_BLEND_SRCALPHA, GS_BLEND_INVSRCALPHA, GS_BLEND_ZERO, GS_BLEND_ONE);

		gs_load_vertexbuffer(vertices);
		gs_load_indexbuffer(nullptr);
		while (gs_effect_loop(solid, "SolidColored"))
			gs_draw(GS_TRIS, 0, count);

		gs_blend_state_pop();
		gs_enable_framebuffer_srgb(previous_srgb);
	}

	static gs_vertbuffer_t *create_vertex_buffer()
	{
		gs_vb_data *data = gs_vbdata_create();
		data->num = VERTEX_CAPACITY;
		data->points = static_cast<vec3 *>(bzalloc(sizeof(vec3) * VERTEX_CAPACITY));
		data->colors = static_cast<uint32_t *>(bzalloc(sizeof(uint32_t) * VERTEX_CAPACITY));
		return gs_vertexbuffer_create(data, GS_DYNAMIC);
	}

	obs_source_t *context;
	TrackerThread tracker;
	std::mutex settings_mutex;
	Settings settings;

	// Used only on OBS's render thread.
	std::array<FrameSlot, FRAME_SLOTS> slots;
	uint64_t latest_frame_id = 0;
	uint64_t last_video_time = 0;
	gs_texrender_t *capture = nullptr;
	gs_stagesurf_t *stage = nullptr;
	uint64_t stage_frame_id = 0; // the frame waiting in `stage`, or 0
	gs_vertbuffer_t *vertices = nullptr;
};

const char *filter_name(void *)
{
	return obs_module_text("Lashline");
}

void *filter_create(obs_data_t *settings, obs_source_t *context)
{
	return new EyelinerFilter(context, settings);
}

void filter_destroy(void *data)
{
	delete static_cast<EyelinerFilter *>(data);
}

void filter_update(void *data, obs_data_t *settings)
{
	static_cast<EyelinerFilter *>(data)->update(settings);
}

void filter_render(void *data, gs_effect_t *)
{
	static_cast<EyelinerFilter *>(data)->render();
}

void filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_int(settings, SETTING_COLOR, 0xff141012); // a soft black
	obs_data_set_default_int(settings, SETTING_OPACITY, 85);
	obs_data_set_default_double(settings, SETTING_THICKNESS, 8.0);
	obs_data_set_default_double(settings, SETTING_WING_LENGTH, 32.0);
	obs_data_set_default_double(settings, SETTING_WING_ANGLE, 22.0);
	obs_data_set_default_double(settings, SETTING_SOFTNESS, 1.2);
	obs_data_set_default_bool(settings, SETTING_LASHES, true);
}

obs_properties_t *filter_properties(void *)
{
	obs_properties_t *properties = obs_properties_create();
	obs_properties_add_color(properties, SETTING_COLOR, obs_module_text("Color"));
	obs_properties_add_int_slider(properties, SETTING_OPACITY, obs_module_text("Opacity"), 0, 100, 1);
	obs_properties_add_float_slider(properties, SETTING_THICKNESS, obs_module_text("Thickness"), 2.0, 20.0, 0.5);
	obs_properties_add_float_slider(properties, SETTING_WING_LENGTH, obs_module_text("WingLength"), 0.0, 60.0, 1.0);
	obs_properties_add_float_slider(properties, SETTING_WING_ANGLE, obs_module_text("WingAngle"), 0.0, 45.0, 1.0);
	obs_properties_add_float_slider(properties, SETTING_SOFTNESS, obs_module_text("Softness"), 0.0, 5.0, 0.1);
	obs_properties_add_bool(properties, SETTING_LASHES, obs_module_text("Lashes"));
	return properties;
}

obs_source_info make_filter_info()
{
	obs_source_info info = {};
	info.id = "lashline_filter";
	info.type = OBS_SOURCE_TYPE_FILTER;
	info.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_SRGB;
	info.get_name = filter_name;
	info.create = filter_create;
	info.destroy = filter_destroy;
	info.update = filter_update;
	info.get_defaults = filter_defaults;
	info.get_properties = filter_properties;
	info.video_render = filter_render;
	return info;
}

} // namespace

obs_source_info eyeliner_filter = make_filter_info();
