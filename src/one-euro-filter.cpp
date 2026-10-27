#include "one-euro-filter.hpp"

#include <cmath>

namespace eyeliner {

namespace {

constexpr float PI = 3.14159265f;

// How far an exponential low-pass filter with this cutoff moves toward a new value that came
// `seconds` after the last one: near 0 holds the old value, 1 takes the new one.
float smoothing_factor(float cutoff, float seconds)
{
	float time_constant = 1 / (2 * PI * cutoff);
	return 1 / (1 + time_constant / seconds);
}

} // namespace

float OneEuroFilter::apply(float value, float seconds, float scale, const OneEuroSettings &settings)
{
	if (!started) {
		started = true;
		previous_value = value;
		smoothed_value = value;
		smoothed_speed = 0;
		return value;
	}

	float speed = (value - previous_value) / seconds / scale;
	smoothed_speed += smoothing_factor(settings.speed_cutoff, seconds) * (speed - smoothed_speed);
	float cutoff = settings.min_cutoff + settings.speed_coefficient * std::abs(smoothed_speed);
	smoothed_value += smoothing_factor(cutoff, seconds) * (value - smoothed_value);
	previous_value = value;
	return smoothed_value;
}

} // namespace eyeliner
