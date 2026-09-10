#pragma once

#include <Arduino.h>

struct PpgSample {
    uint64_t timestampUs = 0;
    uint32_t red = 0;
    uint32_t ir = 0;
};

struct ImuSample {
    uint64_t timestampUs = 0;

    // Adafruit MPU6050 returns acceleration in m/s^2.
    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

    // Angular velocity is returned in rad/s.
    float gx = 0.0f;
    float gy = 0.0f;
    float gz = 0.0f;

    float temperatureC = 0.0f;
};
