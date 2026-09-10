from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct
import zlib


MAGIC = b"ST"
VERSION = 1

HEADER_FORMAT = "<2sBBHIQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
CRC_SIZE = 4


class PacketType(IntEnum):
    PPG = 1
    IMU = 2
    HELLO = 3
    SYNC = 4
    TELEMETRY = 5


@dataclass(frozen=True)
class PpgPacket:
    sequence: int
    device_time_us: int
    red: int
    ir: int


@dataclass(frozen=True)
class ImuPacket:
    sequence: int
    device_time_us: int
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    temperature_c: float


@dataclass(frozen=True)
class HelloPacket:
    device_time_us: int
    boot_id: int


@dataclass(frozen=True)
class SyncPacket:
    device_time_us: int


@dataclass(frozen=True)
class TelemetryPacket:
    device_time_us: int
    queue_depth: int
    queue_high_watermark: int
    dropped_packets: int
    reconnect_count: int
    send_failures: int


DecodedPacket = (
    PpgPacket
    | ImuPacket
    | HelloPacket
    | SyncPacket
    | TelemetryPacket
)


class ProtocolError(Exception):
    pass


class PacketParser:
    """
    Converts the arbitrary TCP byte stream back into protocol packets.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[DecodedPacket]:
        self._buffer.extend(data)
        decoded: list[DecodedPacket] = []

        while True:
            packet = self._try_parse_one()

            if packet is None:
                break

            decoded.append(packet)

        return decoded

    def _try_parse_one(self) -> DecodedPacket | None:
        while len(self._buffer) >= 2 and self._buffer[:2] != MAGIC:
            del self._buffer[0]

        if len(self._buffer) < HEADER_SIZE:
            return None

        (
            magic,
            version,
            packet_type_raw,
            payload_length,
            sequence,
            device_time_us,
        ) = struct.unpack_from(
            HEADER_FORMAT,
            self._buffer,
            0,
        )

        if magic != MAGIC:
            return None

        if version != VERSION:
            raise ProtocolError(
                f"Unsupported protocol version: {version}"
            )

        total_size = HEADER_SIZE + payload_length + CRC_SIZE

        if total_size > 1024:
            raise ProtocolError(
                f"Unreasonable packet size: {total_size}"
            )

        if len(self._buffer) < total_size:
            return None

        packet_bytes = bytes(self._buffer[:total_size])
        del self._buffer[:total_size]

        expected_crc = struct.unpack_from(
            "<I",
            packet_bytes,
            total_size - CRC_SIZE,
        )[0]

        actual_crc = zlib.crc32(
            packet_bytes[:-CRC_SIZE]
        ) & 0xFFFFFFFF

        if actual_crc != expected_crc:
            raise ProtocolError("CRC mismatch")

        payload = packet_bytes[
            HEADER_SIZE : HEADER_SIZE + payload_length
        ]

        try:
            packet_type = PacketType(packet_type_raw)
        except ValueError as exc:
            raise ProtocolError(
                f"Unknown packet type: {packet_type_raw}"
            ) from exc

        if packet_type is PacketType.PPG:
            if payload_length != 8:
                raise ProtocolError("Invalid PPG payload length")

            red, ir = struct.unpack("<II", payload)

            return PpgPacket(
                sequence=sequence,
                device_time_us=device_time_us,
                red=red,
                ir=ir,
            )

        if packet_type is PacketType.IMU:
            if payload_length != 28:
                raise ProtocolError("Invalid IMU payload length")

            values = struct.unpack("<7f", payload)

            return ImuPacket(
                sequence=sequence,
                device_time_us=device_time_us,
                ax=values[0],
                ay=values[1],
                az=values[2],
                gx=values[3],
                gy=values[4],
                gz=values[5],
                temperature_c=values[6],
            )

        if packet_type is PacketType.HELLO:
            if payload_length != 8:
                raise ProtocolError("Invalid HELLO payload length")

            (boot_id,) = struct.unpack("<Q", payload)

            return HelloPacket(
                device_time_us=device_time_us,
                boot_id=boot_id,
            )

        if packet_type is PacketType.SYNC:
            if payload_length != 0:
                raise ProtocolError("Invalid SYNC payload length")

            return SyncPacket(
                device_time_us=device_time_us,
            )

        if packet_type is PacketType.TELEMETRY:
            if payload_length != 20:
                raise ProtocolError(
                    "Invalid TELEMETRY payload length"
                )

            values = struct.unpack("<5I", payload)

            return TelemetryPacket(
                device_time_us=device_time_us,
                queue_depth=values[0],
                queue_high_watermark=values[1],
                dropped_packets=values[2],
                reconnect_count=values[3],
                send_failures=values[4],
            )

        raise ProtocolError("Unhandled packet type")


def _build_packet(
    packet_type: PacketType,
    sequence: int,
    device_time_us: int,
    payload: bytes,
) -> bytes:
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        VERSION,
        int(packet_type),
        len(payload),
        sequence,
        device_time_us,
    )

    body = header + payload
    checksum = zlib.crc32(body) & 0xFFFFFFFF

    return body + struct.pack("<I", checksum)


def encode_ppg(
    sequence: int,
    device_time_us: int,
    red: int,
    ir: int,
) -> bytes:
    return _build_packet(
        PacketType.PPG,
        sequence,
        device_time_us,
        struct.pack("<II", red, ir),
    )


def encode_imu(
    sequence: int,
    device_time_us: int,
    ax: float,
    ay: float,
    az: float,
    gx: float,
    gy: float,
    gz: float,
    temperature_c: float,
) -> bytes:
    payload = struct.pack(
        "<7f",
        ax,
        ay,
        az,
        gx,
        gy,
        gz,
        temperature_c,
    )

    return _build_packet(
        PacketType.IMU,
        sequence,
        device_time_us,
        payload,
    )


def encode_hello(
    device_time_us: int,
    boot_id: int,
) -> bytes:
    return _build_packet(
        PacketType.HELLO,
        0,
        device_time_us,
        struct.pack("<Q", boot_id),
    )


def encode_sync(
    device_time_us: int,
) -> bytes:
    return _build_packet(
        PacketType.SYNC,
        0,
        device_time_us,
        b"",
    )


def encode_telemetry(
    device_time_us: int,
    queue_depth: int,
    queue_high_watermark: int,
    dropped_packets: int,
    reconnect_count: int,
    send_failures: int,
) -> bytes:
    return _build_packet(
        PacketType.TELEMETRY,
        0,
        device_time_us,
        struct.pack(
            "<5I",
            queue_depth,
            queue_high_watermark,
            dropped_packets,
            reconnect_count,
            send_failures,
        ),
    )
