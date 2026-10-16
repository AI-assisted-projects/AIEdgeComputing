"""Talk to the Arduino over the USB serial port (needs the ``pyserial`` package).

* ``record_gestures`` -- with the ``gesture_capture`` sketch running: saves
  each recorded gesture to ``data/recorded/<gesture>.csv``.
* ``show_predictions`` -- with the ``gesture_classifier`` sketch running: prints
  what the board recognises.

Close the Arduino IDE's Serial Monitor first: only one program can use the
port at a time.
"""

from __future__ import annotations

import time
from pathlib import Path

from . import config as C


def list_ports() -> list[str]:
    from serial.tools import list_ports as lp
    ports = list(lp.comports())
    for p in ports:
        print(f"{p.device:<20} {p.description}")
    return [p.device for p in ports]


def find_arduino_port() -> str:
    """Guess the serial port of the Nano 33 BLE (USB vendor id 0x2341 = Arduino)."""
    from serial.tools import list_ports as lp
    for p in lp.comports():
        if p.vid == 0x2341 or "Nano 33" in (p.description or "") or "Arduino" in (p.description or ""):
            return p.device
    raise RuntimeError("No Arduino found. Is it plugged in? Try list_ports() and pass port=...")


def _open(port: str | None):
    import serial
    port = port or find_arduino_port()
    ser = serial.Serial(port, 115200, timeout=1)
    time.sleep(0.5)
    ser.reset_input_buffer()
    print(f"connected to {port}")
    return ser


def record_gestures(gesture: str, count: int = 30, port: str | None = None,
                    folder: str | Path = C.RECORDED_DIR) -> Path:
    """Record ``count`` repetitions of ``gesture`` and append them to ``<folder>/<gesture>.csv``."""
    if gesture not in C.GESTURES:
        raise ValueError(f"gesture must be one of {C.GESTURES}")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{gesture}.csv"
    new_file = not path.exists()
    ser = _open(port)
    print(f"Perform '{gesture}' {count} times. Pause ~1 s between repetitions. "
          "(Interrupt the kernel to stop early.)")
    done, rows = 0, []
    try:
        with open(path, "a") as fh:
            if new_file:
                fh.write(",".join(C.CHANNELS) + "\n")
            while done < count:
                line = ser.readline().decode(errors="ignore").strip()
                if not line or line.startswith("#") or line.startswith("ax"):
                    continue
                values = line.split(",")
                if len(values) != C.N_CHANNELS:
                    continue
                rows.append(line)
                if len(rows) == C.WINDOW_SAMPLES:
                    fh.write("\n".join(rows) + "\n\n")
                    fh.flush()
                    rows = []
                    done += 1
                    print(f"  {gesture}: {done}/{count}")
    finally:
        ser.close()
    print(f"saved to {path}")
    return path


def show_predictions(seconds: float = 60, port: str | None = None) -> None:
    """Print everything the classifier sketch sends, for ``seconds`` seconds."""
    ser = _open(port)
    end = time.time() + seconds
    try:
        while time.time() < end:
            line = ser.readline().decode(errors="ignore").rstrip()
            if line:
                print(line)
    finally:
        ser.close()
