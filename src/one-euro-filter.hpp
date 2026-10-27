#pragma once

// The One Euro filter (Casiez, Roussel and Vogel, "1€ Filter: A Simple Speed-based Low-pass
// Filter for Noisy Input in Interactive Systems", CHI 2012), for one value that changes over
// time.
//
// It's a low-pass filter whose cutoff frequency goes up with the value's speed. A value that's
// holding still is smoothed a lot, which hides noise; a value that's moving fast is hardly
// smoothed at all, so it doesn't trail behind. The face tracker runs one on each coordinate of
// each face point, the way MediaPipe's own face pipeline does.

namespace eyeliner {

struct OneEuroSettings {
	// The cutoff (Hz) when the value is still. Lower is steadier but slower to catch up.
	float min_cutoff;
	// How much the cutoff goes up per unit of speed (the paper's beta). Higher lags less when
	// moving fast, but lets more noise through when moving slowly.
	float speed_coefficient;
	// The cutoff (Hz) for the speed itself, which is noisier than the value.
	float speed_cutoff;
};

class OneEuroFilter {
public:
	// The smoothed value. `seconds` is the time since the previous value. Speed is measured in
	// `scale`s per second, so a caller can make it relative to something's size; the value
	// itself isn't scaled. The first value comes back as it is.
	float apply(float value, float seconds, float scale, const OneEuroSettings &settings);

private:
	bool started = false;
	float previous_value = 0; // as it came in
	float smoothed_value = 0;
	float smoothed_speed = 0;
};

} // namespace eyeliner
