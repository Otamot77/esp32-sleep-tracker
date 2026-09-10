#pragma once

#include <Arduino.h>

#ifdef SDA
#undef SDA
#endif

#ifdef SCL
#undef SCL
#endif

#define SDA 32
#define SCL 33

namespace Config {

constexpr uint32_t SERIAL_BAUD = 115200;
constexpr uint32_t I2C_CLOCK_HZ = 400000;

// MAX30102
constexpr uint8_t MAX30102_I2C_ADDRESS = 0x57;
constexpr uint16_t PPG_SAMPLE_RATE_HZ = 100;
constexpr uint8_t PPG_LED_BRIGHTNESS = 0x1F;
constexpr uint8_t PPG_SAMPLE_AVERAGE = 1;
constexpr uint32_t PPG_SAMPLE_PERIOD_US =
    1000000UL / PPG_SAMPLE_RATE_HZ;
constexpr uint8_t PPG_LED_MODE = 2;
constexpr uint16_t PPG_PULSE_WIDTH_US = 411;
constexpr uint16_t PPG_ADC_RANGE = 4096;

// MPU-6050
constexpr uint8_t MPU6050_I2C_ADDRESS = 0x68;
constexpr uint16_t IMU_SAMPLE_RATE_HZ = 50;
constexpr uint32_t IMU_SAMPLE_PERIOD_US =
    1000000UL / IMU_SAMPLE_RATE_HZ;

// Network reliability
constexpr uint32_t WIFI_RETRY_MS = 5000;
constexpr uint32_t SERVER_RETRY_MS = 2000;
constexpr uint32_t CLOCK_SYNC_INTERVAL_MS = 60000;
constexpr uint32_t TELEMETRY_INTERVAL_MS = 60000;

// EncodedPacket is at most 64 bytes. 768 queued packets is a short,
// RAM-only outage cushion (roughly five seconds at 150 packets/s).
// Long offline recording would require non-volatile storage hardware.
constexpr size_t OUTBOUND_QUEUE_PACKETS = 768;

// Stay below a typical Ethernet/TCP MSS so one flush is efficient.
constexpr size_t NETWORK_BATCH_BYTES = 1400;

// Normal streaming may tolerate a tiny amount of latency. Waiting 100 ms
// lets several sensor packets share one TCP write, reducing per-write overhead.
constexpr uint32_t NETWORK_FLUSH_INTERVAL_MS = 100;
constexpr size_t NETWORK_FLUSH_FORCE_PACKETS = 64;

}  // namespace Config
