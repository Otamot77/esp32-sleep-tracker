#pragma once

#include <Arduino.h>

#include "Config.h"
#include "PacketProtocol.h"

class PacketRingBuffer {
public:
    bool push(const EncodedPacket& packet) {
        bool droppedOldest = false;

        if (count_ == Config::OUTBOUND_QUEUE_PACKETS) {
            tail_ = (tail_ + 1) % Config::OUTBOUND_QUEUE_PACKETS;
            --count_;
            ++droppedPackets_;
            droppedOldest = true;
        }

        packets_[head_] = packet;
        head_ = (head_ + 1) % Config::OUTBOUND_QUEUE_PACKETS;
        ++count_;

        if (count_ > highWatermark_) {
            highWatermark_ = count_;
        }

        return !droppedOldest;
    }

    const EncodedPacket& at(size_t index) const {
        const size_t physical =
            (tail_ + index) % Config::OUTBOUND_QUEUE_PACKETS;
        return packets_[physical];
    }

    void pop(size_t amount = 1) {
        if (amount > count_) {
            amount = count_;
        }

        tail_ =
            (tail_ + amount) % Config::OUTBOUND_QUEUE_PACKETS;
        count_ -= amount;
    }

    size_t size() const {
        return count_;
    }

    bool empty() const {
        return count_ == 0;
    }

    bool full() const {
        return count_ == Config::OUTBOUND_QUEUE_PACKETS;
    }

    uint32_t droppedPackets() const {
        return droppedPackets_;
    }

    size_t highWatermark() const {
        return highWatermark_;
    }

private:
    EncodedPacket packets_[Config::OUTBOUND_QUEUE_PACKETS];

    size_t head_ = 0;
    size_t tail_ = 0;
    size_t count_ = 0;
    size_t highWatermark_ = 0;

    uint32_t droppedPackets_ = 0;
};
