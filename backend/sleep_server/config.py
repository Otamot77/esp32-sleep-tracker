from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BACKEND_DIR / ".env")


def _optional_float(name: str) -> float | None:
    raw = os.getenv(name)

    if raw is None or raw.strip() == "":
        return None

    return float(raw)


def _local_path(
    env_name: str,
    default_relative_to_backend: str,
) -> str:
    raw = os.getenv(
        env_name,
        default_relative_to_backend,
    )

    path = Path(raw)

    if not path.is_absolute():
        path = BACKEND_DIR / path

    return str(path.resolve())


@dataclass(frozen=True)
class Settings:
    tcp_host: str = os.getenv(
        "SLEEP_TCP_HOST",
        "0.0.0.0",
    )
    tcp_port: int = int(
        os.getenv("SLEEP_TCP_PORT", "9000")
    )

    influx_url: str = os.getenv(
        "INFLUX_URL",
        "http://localhost:8086",
    )
    influx_token: str = os.getenv(
        "INFLUXDB_TOKEN",
        "change-me-super-secret-token",
    )
    influx_org: str = os.getenv(
        "INFLUXDB_ORG",
        "sleep-lab",
    )
    influx_bucket: str = os.getenv(
        "INFLUXDB_BUCKET",
        "sleep_raw",
    )

    device_id: str = os.getenv(
        "DEVICE_ID",
        "esp32-wrist-01",
    )

    history_path: str = _local_path(
        "SLEEP_HISTORY_PATH",
        "data/night_history.json",
    )

    recording_directory: str = _local_path(
        "SLEEP_RECORDING_DIRECTORY",
        "data/recordings",
    )

    spo2_cal_a: float | None = _optional_float(
        "SPO2_CAL_A"
    )
    spo2_cal_b: float | None = _optional_float(
        "SPO2_CAL_B"
    )
    spo2_cal_c: float | None = _optional_float(
        "SPO2_CAL_C"
    )

    reconnect_grace_seconds: float = float(
        os.getenv(
            "SLEEP_RECONNECT_GRACE_SECONDS",
            "15",
        )
    )


settings = Settings()
