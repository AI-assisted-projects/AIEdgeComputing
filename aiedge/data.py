"""Gesture data: a physics-based synthetic generator and a loader for your own
recordings, both producing the same format.

One *sample* is one gesture: a window of ``WINDOW_SAMPLES`` consecutive IMU
readings (50 Hz x 1 s), each reading being 6 numbers
``[ax, ay, az, gx, gy, gz]`` -- acceleration in g and rotation speed in deg/s.
So one sample is a ``(50, 6)`` array, flattened to 300 numbers for the network.

Board frame used by the synthetic generator
-------------------------------------------
Hold the board flat in your palm, components facing up, USB connector pointing
towards your wrist.  We call:
    +x  "forward"  (away from you, the direction you punch)
    +y  "left"
    +z  "up"       (a board lying still on a table reads az = +1 g)
The real IMU axes on your board may be oriented differently (Rev1 and Rev2 even
differ from each other). That is one reason why recording your own data
(``load_recorded``) gives a better model on the real board.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import config as C

G = 9.80665  # m/s^2 per g


# ---------------------------------------------------------------------------
# Small math helpers
# ---------------------------------------------------------------------------

def _min_jerk(tau: np.ndarray) -> np.ndarray:
    """Smooth 0 -> 1 profile (the 'minimum-jerk' curve human arms follow)."""
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def _out_and_back(t: np.ndarray, t_out: float, t_hold: float, t_back: float) -> np.ndarray:
    """0 -> 1 in ``t_out`` seconds, stay for ``t_hold``, then 1 -> 0 in ``t_back``."""
    up = _min_jerk(t / t_out)
    down = _min_jerk((t - t_out - t_hold) / t_back)
    return up - down


def _rot(axis: str, angle: np.ndarray | float) -> np.ndarray:
    """Rotation matrix/matrices about a body axis. Works for scalar or 1-D angles."""
    a = np.asarray(angle, dtype=np.float64)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    if axis == "x":
        m = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == "y":
        m = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        m = [[c, -s, z], [s, c, z], [z, z, o]]
    m = np.array(m, dtype=np.float64)          # (3, 3) or (3, 3, T)
    return m if m.ndim == 2 else np.moveaxis(m, -1, 0)  # -> (T, 3, 3)


def _random_rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    """A random small rotation (each of roll/pitch/yaw uniform in +-max_deg)."""
    r, p, y = np.radians(rng.uniform(-max_deg, max_deg, 3))
    return _rot("z", y) @ _rot("y", p) @ _rot("x", r)


# ---------------------------------------------------------------------------
# The five gestures. Each returns, on a fine time grid ``t`` (seconds):
#   pos   (T, 3)  hand position in metres, in the "hand frame"
#   axis  str     body axis the board rotates about ('x', 'y', 'z')
#   angle (T,)    rotation angle about that axis, in radians
# ---------------------------------------------------------------------------

def _punch(t, rng):
    amp = rng.uniform(0.15, 0.35)                     # 15-35 cm forward
    t_out, t_hold, t_back = rng.uniform(0.12, 0.25), rng.uniform(0.0, 0.08), rng.uniform(0.25, 0.45)
    pos = np.zeros((t.size, 3))
    pos[:, 0] = amp * _out_and_back(t, t_out, t_hold, t_back)
    pos[:, 2] = rng.uniform(-0.08, 0.12) * _out_and_back(t, t_out, t_hold, t_back)
    angle = np.radians(rng.uniform(-10, 10)) * _out_and_back(t, t_out, t_hold, t_back)
    return pos, "y", angle, t_out + t_hold + t_back


def _lift(t, rng):
    amp = rng.uniform(0.12, 0.35)                     # raise 12-35 cm and lower again
    t_out, t_hold, t_back = rng.uniform(0.2, 0.4), rng.uniform(0.0, 0.1), rng.uniform(0.2, 0.4)
    pos = np.zeros((t.size, 3))
    pos[:, 2] = amp * _out_and_back(t, t_out, t_hold, t_back)
    pos[:, 0] = rng.uniform(-0.08, 0.15) * _out_and_back(t, t_out, t_hold, t_back)
    angle = np.radians(rng.uniform(-20, 5)) * _out_and_back(t, t_out, t_hold, t_back)
    return pos, "y", angle, t_out + t_hold + t_back


def _wave(t, rng):
    dur = rng.uniform(0.5, 0.9)
    cycles = rng.uniform(1.0, 3.0)                    # side-to-side 1-3 times
    amp = rng.uniform(0.04, 0.12)
    env = np.where(t < dur, np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2, 0.0)
    phase = 2 * np.pi * cycles * t / dur
    pos = np.zeros((t.size, 3))
    pos[:, 1] = amp * env * np.sin(phase)
    angle = np.radians(rng.uniform(10, 35)) * env * np.sin(phase)  # wrist yaws along
    return pos, "z", angle, dur


def _circle(t, rng):
    dur = rng.uniform(0.55, 0.95)
    radius = rng.uniform(0.05, 0.18)
    phi = 2 * np.pi * _min_jerk(t / dur)              # one smooth revolution
    pos = np.zeros((t.size, 3))
    pos[:, 0] = radius * np.sin(phi)                  # vertical circle in front of you
    pos[:, 2] = radius * (1 - np.cos(phi))
    angle = np.radians(rng.uniform(-10, 10)) * np.sin(phi)
    return pos, "y", angle, dur


def _twist(t, rng):
    turn = np.radians(rng.uniform(35, 110))           # turn the wrist 35-110 degrees
    t_out, t_hold, t_back = rng.uniform(0.15, 0.35), rng.uniform(0.0, 0.1), rng.uniform(0.15, 0.35)
    prof = _out_and_back(t, t_out, t_hold, t_back)
    pos = np.zeros((t.size, 3))
    pos[:, 1] = rng.uniform(-0.03, 0.03) * prof
    return pos, "x", turn * prof, t_out + t_hold + t_back


_GESTURE_FUNCS = {"punch": _punch, "lift": _lift, "wave": _wave, "circle": _circle, "twist": _twist}


def synthesize_gesture(name: str, rng: np.random.Generator) -> np.ndarray:
    """Simulate what the IMU measures during one gesture -> ``(WINDOW_SAMPLES, 6)``.

    Physics in three lines:
      * the accelerometer measures (motion acceleration + 1 g "upwards"),
        expressed in the board's own (rotating, tilted) frame;
      * the gyroscope measures how fast the board rotates, in deg/s;
      * both add a little noise and a small constant offset (bias).
    """
    fs_fine = 1000.0                                   # simulate at 1 kHz ...
    n_fine = int(C.WINDOW_SAMPLES / C.SAMPLE_RATE_HZ * fs_fine)
    start = rng.uniform(-0.05, 0.25)                   # trigger fires slightly early/late
    t = np.arange(n_fine) / fs_fine - start

    pos, axis, angle, _ = _GESTURE_FUNCS[name](np.clip(t, 0, None), rng)
    # Everybody performs a gesture a bit differently: random scale + direction.
    pos = pos * rng.uniform(0.6, 1.5)
    pos = pos @ _random_rotation(rng, 45.0).T
    # Nobody holds their arm perfectly still: add a slow random sway
    # (a few gentle 0.5-2 Hz movements of up to ~2 cm in every direction).
    tt = np.arange(n_fine)[:, None] / fs_fine
    for _ in range(3):
        freq, phase = rng.uniform(0.5, 2.0), rng.uniform(0, 2 * np.pi, 3)
        pos = pos + rng.uniform(0, 0.02, 3) * np.sin(2 * np.pi * freq * tt + phase)

    acc_world = np.gradient(np.gradient(pos, 1 / fs_fine, axis=0), 1 / fs_fine, axis=0) / G
    acc_world[:, 2] += 1.0                             # the 1 g reaction to gravity

    # Orientation of the board: random grip tilt, then the gesture's own rotation.
    r_grip = _random_rotation(rng, 50.0)
    r_t = r_grip[None] @ _rot(axis, angle)             # (T, 3, 3), body -> world
    acc_body = np.einsum("tji,tj->ti", r_t, acc_world)  # world -> body (R^T a)

    omega = np.gradient(angle, 1 / fs_fine)            # rad/s about `axis`
    gyro_body = np.zeros((n_fine, 3))
    gyro_body[:, "xyz".index(axis)] = np.degrees(omega)

    # Sample at the IMU rate, then add sensor imperfections.
    step = int(fs_fine / C.SAMPLE_RATE_HZ)
    acc = acc_body[::step][: C.WINDOW_SAMPLES]
    gyr = gyro_body[::step][: C.WINDOW_SAMPLES]
    acc = acc + rng.normal(0, 0.02, acc.shape) + rng.normal(0, 0.02, 3)   # noise + bias
    gyr = gyr + rng.normal(0, 1.5, gyr.shape) + rng.normal(0, 3.0, 3)
    # Hand tremor: a slow random wobble in all accel axes.
    tremor = np.cumsum(rng.normal(0, 0.01, acc.shape), axis=0)
    acc = acc + tremor - tremor.mean(axis=0)

    acc = np.clip(acc, -C.ACCEL_RANGE_G, C.ACCEL_RANGE_G)   # the sensor saturates
    gyr = np.clip(gyr, -C.GYRO_RANGE_DPS, C.GYRO_RANGE_DPS)
    return np.concatenate([acc, gyr], axis=1).astype(np.float32)


def generate_synthetic(n_per_class: int = 800, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate a balanced synthetic dataset. Returns ``X (N, 50, 6)``, ``y (N,)``."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label, name in enumerate(C.GESTURES):
        for _ in range(n_per_class):
            X.append(synthesize_gesture(name, rng))
            y.append(label)
    return np.stack(X), np.array(y, dtype=np.int64)


# ---------------------------------------------------------------------------
# Your own recordings (see arduino/gesture_capture/)
# ---------------------------------------------------------------------------

def load_recorded(folder: str | Path = C.RECORDED_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Load gestures recorded with the ``gesture_capture`` sketch.

    Expects one CSV file per gesture, named after it (``punch.csv``,
    ``lift.csv``, ...). Each file has a header line ``ax,ay,az,gx,gy,gz`` and
    then blocks of ``WINDOW_SAMPLES`` rows; blocks are separated by empty
    lines and lines starting with ``#`` are ignored. Gestures without a file
    are simply missing from the dataset.
    """
    folder = Path(folder)
    X, y = [], []
    for label, name in enumerate(C.GESTURES):
        path = folder / f"{name}.csv"
        if not path.exists():
            print(f"  (no recordings for '{name}' -- expected {path})")
            continue
        rows: list[list[float]] = []
        with open(path, newline="") as fh:
            for row in csv.reader(fh):
                if not row or not row[0].strip():
                    rows = []                     # blank line: start of a new gesture
                    continue
                if row[0].strip() == C.CHANNELS[0] or row[0].startswith("#"):
                    continue                      # header or comment line
                rows.append([float(v) for v in row[: C.N_CHANNELS]])
                if len(rows) == C.WINDOW_SAMPLES:
                    X.append(np.array(rows, dtype=np.float32))
                    y.append(label)
                    rows = []
    if not X:
        raise FileNotFoundError(f"No recordings found in {folder}")
    return np.stack(X), np.array(y, dtype=np.int64)


# ---------------------------------------------------------------------------
# Train / validation / test split + normalization
# ---------------------------------------------------------------------------

@dataclass
class GestureData:
    """The three data splits as ready-to-train torch tensors.

    ``X_*`` are already *normalized* and *flattened* to ``(N, 300)``. The raw
    (un-normalized) windows are kept in ``raw_*`` for plotting.
    """

    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    channel_mean: np.ndarray            # (6,) -- also baked into the Arduino code
    channel_std: np.ndarray             # (6,)
    raw_train: np.ndarray = field(repr=False)
    raw_val: np.ndarray = field(repr=False)
    raw_test: np.ndarray = field(repr=False)
    source: str = "synthetic"

    def normalize(self, raw: np.ndarray) -> torch.Tensor:
        """Raw windows ``(N, 50, 6)`` -> normalized, flattened ``(N, 300)`` tensor."""
        z = (raw - self.channel_mean) / self.channel_std
        return torch.from_numpy(z.reshape(len(raw), -1).astype(np.float32))

    def summary(self) -> str:
        counts = np.bincount(self.y_train.numpy(), minlength=C.N_CLASSES)
        per_class = ", ".join(f"{g}={n}" for g, n in zip(C.GESTURES, counts))
        return (f"{self.source} data: {len(self.y_train)} train / {len(self.y_val)} val / "
                f"{len(self.y_test)} test samples, {self.X_train.shape[1]} features each\n"
                f"train samples per class: {per_class}")


def make_splits(X: np.ndarray, y: np.ndarray, seed: int = 0, val_frac: float = 0.15,
                test_frac: float = 0.15, source: str = "synthetic") -> GestureData:
    """Shuffle, split per class (stratified), and normalize with training statistics only."""
    rng = np.random.default_rng(seed)
    idx_train, idx_val, idx_test = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        n_val, n_test = int(round(len(idx) * val_frac)), int(round(len(idx) * test_frac))
        idx_val += list(idx[:n_val])
        idx_test += list(idx[n_val:n_val + n_test])
        idx_train += list(idx[n_val + n_test:])
    idx_train, idx_val, idx_test = (rng.permutation(np.array(i)) for i in (idx_train, idx_val, idx_test))

    # One mean / std per sensor channel, computed on the TRAINING set only.
    mean = X[idx_train].reshape(-1, C.N_CHANNELS).mean(axis=0)
    std = X[idx_train].reshape(-1, C.N_CHANNELS).std(axis=0) + 1e-6

    def prep(i):
        z = (X[i] - mean) / std
        return torch.from_numpy(z.reshape(len(i), -1).astype(np.float32)), torch.from_numpy(y[i])

    (Xtr, ytr), (Xv, yv), (Xte, yte) = prep(idx_train), prep(idx_val), prep(idx_test)
    return GestureData(Xtr, ytr, Xv, yv, Xte, yte, mean.astype(np.float32), std.astype(np.float32),
                       raw_train=X[idx_train], raw_val=X[idx_val], raw_test=X[idx_test], source=source)


def limit_training_data(data: GestureData, n_per_class: int, seed: int = 0) -> GestureData:
    """Keep only ``n_per_class`` training examples per gesture (validation and test
    sets stay the same). Simulates having recorded only a few examples yourself."""
    rng = np.random.default_rng(seed)
    y = data.y_train.numpy()
    keep = np.sort(np.concatenate([rng.permutation(np.flatnonzero(y == c))[:n_per_class]
                                   for c in np.unique(y)]))
    return GestureData(data.X_train[keep], data.y_train[keep], data.X_val, data.y_val,
                       data.X_test, data.y_test, data.channel_mean, data.channel_std,
                       raw_train=data.raw_train[keep], raw_val=data.raw_val,
                       raw_test=data.raw_test, source=data.source)


def load_gesture_data(source: str = "synthetic", n_per_class: int = 800, seed: int = 0) -> GestureData:
    """One-stop loader used by every notebook.

    ``source="synthetic"``  -- generate data with the physics simulator (no board needed)
    ``source="recorded"``   -- load your own recordings from ``data/recorded/``
    """
    if source == "synthetic":
        X, y = generate_synthetic(n_per_class=n_per_class, seed=seed)
    elif source == "recorded":
        X, y = load_recorded()
    else:
        raise ValueError("source must be 'synthetic' or 'recorded'")
    return make_splits(X, y, seed=seed, source=source)
