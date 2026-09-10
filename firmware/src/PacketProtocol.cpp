#include "PacketProtocol.h"

#include <cstring>

namespace {

constexpr uint8_t VERSION = 1;

void writeU16LE(uint8_t* b, size_t& o, uint16_t v) {
    b[o++] = static_cast<uint8_t>(v & 0xFF);
    b[o++] = static_cast<uint8_t>((v >> 8) & 0xFF);
}

void writeU32LE(uint8_t* b, size_t& o, uint32_t v) {
    for (int shift = 0; shift < 32; shift += 8) {
        b[o++] = static_cast<uint8_t>((v >> shift) & 0xFF);
    }
}

void writeU64LE(uint8_t* b, size_t& o, uint64_t v) {
    for (int shift = 0; shift < 64; shift += 8) {
        b[o++] = static_cast<uint8_t>((v >> shift) & 0xFF);
    }
}

void writeF32LE(uint8_t* b, size_t& o, float v) {
    uint32_t raw = 0;
    memcpy(&raw, &v, sizeof(raw));
    writeU32LE(b, o, raw);
}

uint32_t crc32(const uint8_t* data, size_t n) {
    uint32_t crc = 0xFFFFFFFFUL;

    for (size_t i = 0; i < n; ++i) {
        crc ^= data[i];

        for (int bit = 0; bit < 8; ++bit) {
            crc =
                (crc >> 1) ^
                (0xEDB88320UL &
                 static_cast<uint32_t>(
                     -(static_cast<int32_t>(crc & 1U))
                 ));
        }
    }

    return ~crc;
}

void writeHeader(
    EncodedPacket& packet,
    PacketType type,
    uint16_t payloadLength,
    uint32_t sequence,
    uint64_t deviceTimeUs
) {
    packet.bytes[packet.length++] = 'S';
    packet.bytes[packet.length++] = 'T';
    packet.bytes[packet.length++] = VERSION;
    packet.bytes[packet.length++] = static_cast<uint8_t>(type);

    writeU16LE(packet.bytes, packet.length, payloadLength);
    writeU32LE(packet.bytes, packet.length, sequence);
    writeU64LE(packet.bytes, packet.length, deviceTimeUs);
}

void finish(EncodedPacket& packet) {
    const uint32_t checksum = crc32(packet.bytes, packet.length);
    writeU32LE(packet.bytes, packet.length, checksum);
}

}  // namespace

namespace PacketProtocol {

EncodedPacket encodePpg(
    const PpgSample& sample,
    uint32_t sequence
) {
    EncodedPacket packet;
    writeHeader(
        packet,
        PacketType::Ppg,
        8,
        sequence,
        sample.timestampUs
    );

    writeU32LE(packet.bytes, packet.length, sample.red);
    writeU32LE(packet.bytes, packet.length, sample.ir);

    finish(packet);
    return packet;
}

EncodedPacket encodeImu(
    const ImuSample& sample,
    uint32_t sequence
) {
    EncodedPacket packet;
    writeHeader(
        packet,
        PacketType::Imu,
        28,
        sequence,
        sample.timestampUs
    );

    writeF32LE(packet.bytes, packet.length, sample.ax);
    writeF32LE(packet.bytes, packet.length, sample.ay);
    writeF32LE(packet.bytes, packet.length, sample.az);

    writeF32LE(packet.bytes, packet.length, sample.gx);
    writeF32LE(packet.bytes, packet.length, sample.gy);
    writeF32LE(packet.bytes, packet.length, sample.gz);

    writeF32LE(
        packet.bytes,
        packet.length,
        sample.temperatureC
    );

    finish(packet);
    return packet;
}

EncodedPacket encodeHello(
    uint64_t deviceTimeUs,
    uint64_t bootId
) {
    EncodedPacket packet;
    writeHeader(
        packet,
        PacketType::Hello,
        8,
        0,
        deviceTimeUs
    );

    writeU64LE(packet.bytes, packet.length, bootId);
    finish(packet);
    return packet;
}

EncodedPacket encodeSync(
    uint64_t deviceTimeUs
) {
    EncodedPacket packet;
    writeHeader(
        packet,
        PacketType::Sync,
        0,
        0,
        deviceTimeUs
    );

    finish(packet);
    return packet;
}

EncodedPacket encodeTelemetry(
    uint64_t deviceTimeUs,
    uint32_t queueDepth,
    uint32_t queueHighWatermark,
    uint32_t droppedPackets,
    uint32_t reconnectCount,
    uint32_t sendFailures
) {
    EncodedPacket packet;
    writeHeader(
        packet,
        PacketType::Telemetry,
        20,
        0,
        deviceTimeUs
    );

    writeU32LE(packet.bytes, packet.length, queueDepth);
    writeU32LE(
        packet.bytes,
        packet.length,
        queueHighWatermark
    );
    writeU32LE(packet.bytes, packet.length, droppedPackets);
    writeU32LE(packet.bytes, packet.length, reconnectCount);
    writeU32LE(packet.bytes, packet.length, sendFailures);

    finish(packet);
    return packet;
}

}  // namespace PacketProtocol
