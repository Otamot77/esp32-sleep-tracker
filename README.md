# ESP32 Sleep Tracker

An ongoing wrist-worn sleep tracker prototype using an ESP32, MAX30102 PPG
sensor, and MPU-6050 IMU. The project covers the path from sensor acquisition
to experimental sleep-stage and recovery estimates.

## How it works

`MAX30102 + MPU-6050 -> ESP32 -> Wi-Fi/TCP -> Python -> .stbin / InfluxDB`

The ESP32 reads both sensors over I2C, timestamps the samples, creates binary
packets, and stores outgoing data in a RAM ring buffer during short network
interruptions. The MAX30102 FIFO is sampled at 100 Hz. The IMU targets 50 Hz,
with its schedule anchored to the ESP32 monotonic clock to avoid accumulated
loop-timing drift.

Packets include sequence numbers and CRC checks. HELLO, SYNC, and TELEMETRY
packets identify each boot session, map device time to server time, and report
basic device health.

The Python backend parses the TCP stream and saves the original packet bytes in
`.stbin` files for replay. It can also write decoded data and results to
InfluxDB. Processing currently includes:

- PPG filtering, beat detection, heart rate, RMSSD, and SDNN
- acceleration and motion features from the IMU
- signal-quality and sample-coverage checks
- 30-second feature windows
- heuristic Wake, Light, Deep, and REM probabilities with Viterbi smoothing
- 120-second respiration and red/IR ratio-of-ratios estimates
- a recovery score using a rolling personal baseline

## Current status

The firmware builds, the backend pipeline runs, and automated tests cover the
main packet, signal-processing, staging, and recovery paths. `run_demo.py`
checks the full software pipeline with generated sensor data.

Physical and overnight testing are still incomplete. Wrist signal quality,
sensor timing, battery life, long-session reliability, and sleep-stage accuracy
have not yet been validated. SpO2 percentage is disabled until suitable
reference data is available for calibration. Sleep-stage and recovery outputs
are experimental and are not medical measurements.

## Development context

I built this as a learning-focused project to understand the full path from embedded sensing to backend analysis. I used AI tools throughout the project as a learning and development aid, including for explanations, planning, implementation guidance, debugging, and code review. I checked suggestions against technical documentation and tests, and focused on understanding the system’s design decisions, tradeoffs, and limitations.

## Running the checks

The backend uses Python 3.11 or newer:

```powershell
cd backend
python -m pip install -r requirements.txt
python -m pytest
cd ..
python run_demo.py
```

The ESP32 firmware uses PlatformIO:

```powershell
Copy-Item firmware\include\Secrets.h.example firmware\include\Secrets.h
pio run -d firmware
```

`Secrets.h` is ignored by Git so Wi-Fi details are not committed.

## Repository layout

- `firmware/` - ESP32 sensor collection, packet creation, buffering, and networking
- `backend/` - Python server, processing pipeline, scoring, and tests
- `docs/references.md` - main technical references used during development
- `run_demo.py` - synthetic end-to-end software demo

The main technical sources used are listed in
[docs/references.md](docs/references.md).
