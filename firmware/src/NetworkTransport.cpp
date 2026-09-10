#include "NetworkTransport.h"

#include <cstring>
#include <esp_timer.h>
#include <esp_wifi.h>

#include "Config.h"
#include "Secrets.h"

void NetworkTransport::begin(uint64_t bootId) {
    bootId_ = bootId;

    WiFi.mode(WIFI_STA);

    // Reduce Wi-Fi power while keeping the station connected.
    esp_wifi_set_ps(WIFI_PS_MIN_MODEM);

    connectWifi();
}

bool NetworkTransport::connected() {
    return
        WiFi.status() == WL_CONNECTED &&
        client_.connected();
}

void NetworkTransport::connectWifi() {
    if (WiFi.status() == WL_CONNECTED) {
        return;
    }

    Serial.printf(
        "# WiFi connecting SSID=%s\n",
        Secrets::WIFI_SSID
    );

    WiFi.begin(
        Secrets::WIFI_SSID,
        Secrets::WIFI_PASSWORD
    );

    lastWifiAttemptMs_ = millis();
}

void NetworkTransport::sendHello() {
    const EncodedPacket hello =
        PacketProtocol::encodeHello(
            static_cast<uint64_t>(esp_timer_get_time()),
            bootId_
        );

    if (!writeControlPacket(hello)) {
        client_.stop();
    }
}

void NetworkTransport::connectServer() {
    if (
        WiFi.status() != WL_CONNECTED ||
        client_.connected()
    ) {
        return;
    }

    Serial.printf(
        "# server connecting %s:%u\n",
        Secrets::SERVER_HOST,
        Secrets::SERVER_PORT
    );

    if (
        client_.connect(
            Secrets::SERVER_HOST,
            Secrets::SERVER_PORT
        )
    ) {
        client_.setNoDelay(true);

        if (hasConnectedBefore_) {
            ++reconnectCount_;
        }

        hasConnectedBefore_ = true;

        Serial.printf(
            "# server connected boot=%016llX queued=%u\n",
            static_cast<unsigned long long>(bootId_),
            static_cast<unsigned>(queue_.size())
        );

        // Establish a new clock/session anchor before draining backlog.
        sendHello();

        lastSyncMs_ = millis();
        lastTelemetryMs_ = millis();
        lastFlushMs_ = millis();
    } else {
        Serial.println("# server connection failed");
    }

    lastServerAttemptMs_ = millis();
}

bool NetworkTransport::writeControlPacket(
    const EncodedPacket& packet
) {
    if (!connected()) {
        return false;
    }

    const size_t written =
        client_.write(packet.bytes, packet.length);

    if (written != packet.length) {
        ++sendFailures_;
        return false;
    }

    return true;
}

bool NetworkTransport::enqueue(
    const EncodedPacket& packet
) {
    return queue_.push(packet);
}

void NetworkTransport::flushQueuedPackets() {
    if (!connected() || queue_.empty()) {
        return;
    }

    uint8_t batch[Config::NETWORK_BATCH_BYTES];
    size_t batchBytes = 0;
    size_t packetCount = 0;

    while (packetCount < queue_.size()) {
        const EncodedPacket& packet =
            queue_.at(packetCount);

        if (
            batchBytes + packet.length >
            sizeof(batch)
        ) {
            break;
        }

        memcpy(
            batch + batchBytes,
            packet.bytes,
            packet.length
        );

        batchBytes += packet.length;
        ++packetCount;
    }

    if (packetCount == 0) {
        return;
    }

    const size_t written =
        client_.write(batch, batchBytes);

    if (written == batchBytes) {
        queue_.pop(packetCount);
        return;
    }

    ++sendFailures_;
    client_.stop();
}

void NetworkTransport::maybeSendSync() {
    if (!connected() || !queue_.empty()) {
        return;
    }

    const uint32_t nowMs = millis();

    if (
        nowMs - lastSyncMs_ <
        Config::CLOCK_SYNC_INTERVAL_MS
    ) {
        return;
    }

    const EncodedPacket sync =
        PacketProtocol::encodeSync(
            static_cast<uint64_t>(esp_timer_get_time())
        );

    if (writeControlPacket(sync)) {
        lastSyncMs_ = nowMs;
    } else {
        client_.stop();
    }
}

void NetworkTransport::maybeQueueTelemetry() {
    if (!connected() || queue_.full()) {
        return;
    }

    const uint32_t nowMs = millis();

    if (
        nowMs - lastTelemetryMs_ <
        Config::TELEMETRY_INTERVAL_MS
    ) {
        return;
    }

    // Put telemetry behind the samples already waiting to be sent. This
    // preserves stream order and records the real pre-flush queue depth.
    const EncodedPacket telemetry =
        PacketProtocol::encodeTelemetry(
            static_cast<uint64_t>(esp_timer_get_time()),
            static_cast<uint32_t>(queue_.size()),
            static_cast<uint32_t>(
                queue_.highWatermark()
            ),
            queue_.droppedPackets(),
            reconnectCount_,
            sendFailures_
        );

    queue_.push(telemetry);
    lastTelemetryMs_ = nowMs;
}

void NetworkTransport::maintain() {
    const uint32_t nowMs = millis();

    if (WiFi.status() != WL_CONNECTED) {
        if (
            nowMs - lastWifiAttemptMs_ >=
            Config::WIFI_RETRY_MS
        ) {
            WiFi.disconnect();
            connectWifi();
        }

        return;
    }

    if (!client_.connected()) {
        if (
            nowMs - lastServerAttemptMs_ >=
            Config::SERVER_RETRY_MS
        ) {
            connectServer();
        }

        return;
    }

    // Queue telemetry before a scheduled flush so queue_depth is useful and
    // the control packet cannot overtake older sensor packets.
    maybeQueueTelemetry();

    // Batch normal traffic briefly; flush immediately when backlog grows.
    if (
        queue_.size() >= Config::NETWORK_FLUSH_FORCE_PACKETS ||
        nowMs - lastFlushMs_ >= Config::NETWORK_FLUSH_INTERVAL_MS
    ) {
        flushQueuedPackets();
        lastFlushMs_ = nowMs;
    }

    // A SYNC is a useful clock anchor only after backlog is drained.
    maybeSendSync();
}

size_t NetworkTransport::queueDepth() const {
    return queue_.size();
}

size_t NetworkTransport::queueHighWatermark() const {
    return queue_.highWatermark();
}

uint32_t NetworkTransport::droppedPackets() const {
    return queue_.droppedPackets();
}

uint32_t NetworkTransport::reconnectCount() const {
    return reconnectCount_;
}

uint32_t NetworkTransport::sendFailures() const {
    return sendFailures_;
}
