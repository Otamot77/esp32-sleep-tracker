#pragma once

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include "SampleTypes.h"

class Mpu6050Sensor {
public:
    bool begin(TwoWire& wire);
    bool read(ImuSample& sample);

private:
    Adafruit_MPU6050 sensor_;
};
