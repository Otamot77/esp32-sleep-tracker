#include <Arduino.h>
#include <Wire.h>
#include <esp_system.h>
#include <esp_timer.h>

#include "Config.h"
#include "Max30102Sensor.h"
#include "Mpu6050Sensor.h"
#include "NetworkTransport.h"
#include "PacketProtocol.h"

namespace {

Max30102Sensor ppg;
Mpu6050Sensor imu;
NetworkTransport network;

uint64_t nextImuUs = 0;
uint32_t sequence = 0;
uint64_t bootId = 0;

uint64_t makeBootId() {
    return
        (static_cast<uint64_t>(esp_random()) << 32) |
        static_cast<uint64_t>(esp_random());
}

void halt(const char* message) {
    Serial.println(message);

    while (true) {
        delay(1000);
    }
}

}  // namespace

void setup() {
    Serial.begin(Config::SERIAL_BAUD);
    delay(1000);

    bootId = makeBootId();

    Serial.println();
    Serial.println("# sleep_tracker");
    Serial.printf(
        "# boot_id=%016llX\n",
        static_cast<unsigned long long>(bootId)
    );

    Wire.begin(SDA, SCL);
    Wire.setClock(Config::I2C_CLOCK_HZ);

    if (!ppg.begin(Wire)) {
        halt("ERROR: MAX30102 failed");
    }

    if (!imu.begin(Wire)) {
        halt("ERROR: MPU6050 failed");
    }

    nextImuUs =
        static_cast<uint64_t>(esp_timer_get_time()) +
        Config::IMU_SAMPLE_PERIOD_US;

    network.begin(bootId);
}

void loop() {
    // Service both sensors before doing network maintenance.
    ppg.update();

    PpgSample ppgSample;

    while (ppg.read(ppgSample)) {
        network.enqueue(
            PacketProtocol::encodePpg(
                ppgSample,
                sequence++
            )
        );
    }

    const uint64_t nowUs =
        static_cast<uint64_t>(esp_timer_get_time());

    if (nowUs >= nextImuUs) {
        const uint64_t latenessUs = nowUs - nextImuUs;
        ImuSample imuSample;

        if (imu.read(imuSample)) {
            network.enqueue(
                PacketProtocol::encodeImu(
                    imuSample,
                    sequence++
                )
            );
        }

        // Keep normal samples anchored to the ESP32's monotonic timeline.
        // If the loop falls far behind, skip the missed slots instead of
        // taking a burst of catch-up samples.
        nextImuUs += Config::IMU_SAMPLE_PERIOD_US;

        if (
            latenessUs >=
            5ULL * Config::IMU_SAMPLE_PERIOD_US
        ) {
            nextImuUs =
                nowUs + Config::IMU_SAMPLE_PERIOD_US;
        }
    }

    network.maintain();

    delay(1);
}
