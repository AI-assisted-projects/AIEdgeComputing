"""Course-wide constants.

Everything that must agree between Python (training) and C++ (the Arduino
sketches) lives here. If you change a value, re-run notebook 05 so the new
value is written into ``arduino/gesture_classifier/model_data.h``, and update
the matching ``#define`` at the top of ``arduino/gesture_capture/gesture_capture.ino``.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"                 # recorded gestures go in data/recorded/
RECORDED_DIR = DATA_DIR / "recorded"
ARDUINO_DIR = ROOT / "arduino"
CLASSIFIER_SKETCH_DIR = ARDUINO_DIR / "gesture_classifier"

# --- Sensor sampling -------------------------------------------------------
SAMPLE_RATE_HZ = 50        # we read the IMU 50 times per second
WINDOW_SAMPLES = 50        # one gesture = 50 readings = 1.0 second
CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")   # accel in g, gyro in deg/s
N_CHANNELS = len(CHANNELS)
N_FEATURES = WINDOW_SAMPLES * N_CHANNELS          # 300 numbers per gesture

ACCEL_RANGE_G = 4.0        # default full-scale range of the Arduino IMU libraries
GYRO_RANGE_DPS = 2000.0

# A recording starts when the board "feels" motion: either the acceleration
# magnitude deviates from 1 g by more than this, or it rotates faster than this.
TRIGGER_ACCEL_G = 0.35
TRIGGER_GYRO_DPS = 120.0
# The board keeps the last few readings in memory, so the recorded window
# also contains the start of the movement (from just *before* the trigger).
PRETRIGGER_SAMPLES = 10    # = 0.2 s
COOLDOWN_MS = 300          # ignore motion for a moment after each gesture

# --- The task --------------------------------------------------------------
# Class order matters: index 0 is "punch", etc. The Arduino sketch receives the
# same list through the generated model_data.h.
GESTURES = ("punch", "lift", "wave", "circle", "twist")
N_CLASSES = len(GESTURES)

# --- Target hardware: Arduino Nano 33 BLE Sense (Rev1 and Rev2) ------------
MCU_CLOCK_HZ = 64_000_000
MCU_FLASH_BYTES = 1024 * 1024
