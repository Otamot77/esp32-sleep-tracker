#include "Max30102Sensor.h"

#include <esp_timer.h>

#include "Config.h"

bool Max30102Sensor::begin(TwoWire& wire) {
    if (!sensor_.begin(wire, I2C_SPEED_FAST)) {
        return false;
    }

    sensor_.setup(
        Config::PPG_LED_BRIGHTNESS,
        Config::PPG_SAMPLE_AVERAGE,
        Config::PPG_LED_MODE,
        Config::PPG_SAMPLE_RATE_HZ,
        Config::PPG_PULSE_WIDTH_US,
        Config::PPG_ADC_RANGE
    );

    sensor_.setPulseAmplitudeGreen(0);
    sensor_.setPulseAmplitudeRed(Config::PPG_LED_BRIGHTNESS);
    sensor_.setPulseAmplitudeIR(Config::PPG_LED_BRIGHTNESS);
    sensor_.clearFIFO();

    return true;
}

uint8_t Max30102Sensor::update() {
    sensor_.check();

    const uint8_t count = sensor_.available();

    if (count > 0 && timestampSamplesRemaining_ == 0) {
        const uint64_t nowUs =
            static_cast<uint64_t>(esp_timer_get_time());

        nextFifoTimestampUs_ =
            nowUs -
            static_cast<uint64_t>(count - 1) *
                Config::PPG_SAMPLE_PERIOD_US;

        timestampSamplesRemaining_ = count;
    }

    return count;
}

bool Max30102Sensor::read(PpgSample& sample) {
    if (sensor_.available() == 0) {
        return false;
    }

    sample.timestampUs = nextFifoTimestampUs_;
    sample.red = sensor_.getFIFORed();
    sample.ir = sensor_.getFIFOIR();

    sensor_.nextSample();

    nextFifoTimestampUs_ += Config::PPG_SAMPLE_PERIOD_US;

    if (timestampSamplesRemaining_ > 0) {
        --timestampSamplesRemaining_;
    }

    return true;
}
