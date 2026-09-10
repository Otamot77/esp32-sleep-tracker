#pragma once

#include <Arduino.h>

#include "SampleTypes.h"

enum class PacketType : uint8_t {
    Ppg = 1,
    Imu = 2,
    Hello = 3,
    Sync = 4,
    Telemetry = 5
};

struct EncodedPacket {
    uint8_t bytes[64];
    size_t length = 0;
};

namespace PacketProtocol {

EncodedPacket encodePpg(const PpgSample& sample, uint32_t sequence);
EncodedPacket encodeImu(const ImuSample& sample, uint32_t sequence);

EncodedPacket encodeHello(
    uint64_t deviceTimeUs,
    uint64_t bootId
);

EncodedPacket encodeSync(
    uint64_t deviceTimeUs
);

EncodedPacket encodeTelemetry(
    uint64_t deviceTimeUs,
    uint32_t queueDepth,
    uint32_t queueHighWatermark,
    uint32_t droppedPackets,
    uint32_t reconnectCount,
    uint32_t sendFailures
);

}  // namespace PacketProtocol
