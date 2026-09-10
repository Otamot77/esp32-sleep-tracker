#pragma once

#include <Arduino.h>
#include <Wire.h>
#include <MAX30105.h>

#include "SampleTypes.h"

class Max30102Sensor {
public:
    bool begin(TwoWire& wire);
    uint8_t update();
    bool read(PpgSample& sample);

private:
    MAX30105 sensor_;

    uint64_t nextFifoTimestampUs_ = 0;
    uint8_t timestampSamplesRemaining_ = 0;
};
