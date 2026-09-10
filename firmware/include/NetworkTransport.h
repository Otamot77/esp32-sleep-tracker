#pragma once

#include <Arduino.h>
#include <WiFi.h>

#include "PacketProtocol.h"
#include "PacketRingBuffer.h"

class NetworkTransport {
public:
    void begin(uint64_t bootId);
    void maintain();

    // Samples enter the RAM queue before the loop services Wi-Fi/TCP.
    // If the queue is full, the oldest packet is discarded and a
    // sequence-number gap makes that loss visible at the backend.
    bool enqueue(const EncodedPacket& packet);

    bool connected();

    size_t queueDepth() const;
    size_t queueHighWatermark() const;
    uint32_t droppedPackets() const;
    uint32_t reconnectCount() const;
    uint32_t sendFailures() const;

private:
    WiFiClient client_;
    PacketRingBuffer queue_;

    uint64_t bootId_ = 0;

    uint32_t lastWifiAttemptMs_ = 0;
    uint32_t lastServerAttemptMs_ = 0;
    uint32_t lastSyncMs_ = 0;
    uint32_t lastTelemetryMs_ = 0;
    uint32_t lastFlushMs_ = 0;

    uint32_t reconnectCount_ = 0;
    uint32_t sendFailures_ = 0;

    bool hasConnectedBefore_ = false;

    void connectWifi();
    void connectServer();
    void flushQueuedPackets();

    bool writeControlPacket(const EncodedPacket& packet);
    void sendHello();
    void maybeSendSync();
    void maybeQueueTelemetry();
};
