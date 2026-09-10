#include "Mpu6050Sensor.h"

#include <esp_timer.h>

#include "Config.h"

bool Mpu6050Sensor::begin(TwoWire& wire) {
    if (!sensor_.begin(Config::MPU6050_I2C_ADDRESS, &wire)) {
        return false;
    }

    // Good initial ranges for wrist movement during sleep.
    sensor_.setAccelerometerRange(MPU6050_RANGE_4_G);
    sensor_.setGyroRange(MPU6050_RANGE_500_DEG);
    sensor_.setFilterBandwidth(MPU6050_BAND_21_HZ);

    return true;
}

bool Mpu6050Sensor::read(ImuSample& sample) {
    sensors_event_t accel;
    sensors_event_t gyro;
    sensors_event_t temperature;

    sensor_.getEvent(&accel, &gyro, &temperature);

    sample.timestampUs = static_cast<uint64_t>(esp_timer_get_time());

    sample.ax = accel.acceleration.x;
    sample.ay = accel.acceleration.y;
    sample.az = accel.acceleration.z;

    sample.gx = gyro.gyro.x;
    sample.gy = gyro.gyro.y;
    sample.gz = gyro.gyro.z;

    sample.temperatureC = temperature.temperature;

    return true;
}
