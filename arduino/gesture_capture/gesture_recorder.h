// gesture_recorder.h -- samples the IMU at a fixed rate and cuts out one
// gesture-sized window every time the board starts to move.
//
// This file is IDENTICAL in arduino/gesture_capture/ and
// arduino/gesture_classifier/ (an Arduino sketch can only include files from
// its own folder). If you change one, copy it to the other: the data you
// record and the data the classifier sees must be produced the same way.
//
// How it works
//   * Every 1/SAMPLE_RATE seconds we read the accelerometer (g) and the
//     gyroscope (deg/s): 6 numbers per reading.
//   * We always remember the last PRETRIGGER_SAMPLES readings.
//   * When the board moves (|acceleration| differs from 1 g, or it rotates
//     fast) we start a window with those remembered readings and keep
//     recording until WINDOW_SAMPLES readings are collected.
//   * After a window we ignore motion for COOLDOWN_MS, so the end of one
//     gesture does not immediately trigger the next window.
//
// Requires the global `IMU` object, so include the IMU library first.

#pragma once
#include <math.h>

// The classifier gets these values from model_data.h. The capture sketch uses
// the defaults below, which must match aiedge/config.py.
#ifndef MODEL_SAMPLE_RATE_HZ
#define MODEL_SAMPLE_RATE_HZ 50
#endif
#ifndef MODEL_WINDOW_SAMPLES
#define MODEL_WINDOW_SAMPLES 50
#endif
#ifndef MODEL_N_CHANNELS
#define MODEL_N_CHANNELS 6
#endif
#ifndef MODEL_PRETRIGGER_SAMPLES
#define MODEL_PRETRIGGER_SAMPLES 10
#endif
#ifndef MODEL_COOLDOWN_MS
#define MODEL_COOLDOWN_MS 300
#endif
#ifndef MODEL_TRIGGER_ACCEL_G
#define MODEL_TRIGGER_ACCEL_G 0.35f
#endif
#ifndef MODEL_TRIGGER_GYRO_DPS
#define MODEL_TRIGGER_GYRO_DPS 120.0f
#endif

class GestureRecorder {
 public:
  // Call this as often as possible from loop(). Returns true exactly once per
  // gesture, when a complete window is available in window().
  bool update() {
    const unsigned long now = micros();
    if (!started_) {
      next_us_ = now;
      started_ = true;
    }
    if ((long)(now - next_us_) < 0) return false;  // not time for a reading yet
    next_us_ += kPeriodUs;
    if ((long)(now - next_us_) > (long)kPeriodUs) next_us_ = now + kPeriodUs;  // fell behind: resync

    float s[MODEL_N_CHANNELS];
    IMU.readAcceleration(s[0], s[1], s[2]);
    IMU.readGyroscope(s[3], s[4], s[5]);

    if (state_ == kRecording) {
      store(count_++, s);
      if (count_ == MODEL_WINDOW_SAMPLES) {
        state_ = kCooldown;
        cooldown_until_ms_ = millis() + MODEL_COOLDOWN_MS;
        return true;
      }
      return false;
    }

    // Idle or cooling down: remember this reading in the pre-trigger history.
    for (int c = 0; c < MODEL_N_CHANNELS; ++c) history_[head_][c] = s[c];
    head_ = (head_ + 1) % MODEL_PRETRIGGER_SAMPLES;
    if (filled_ < MODEL_PRETRIGGER_SAMPLES) ++filled_;

    if (state_ == kCooldown) {
      if ((long)(millis() - cooldown_until_ms_) < 0) return false;
      state_ = kIdle;
    }
    if (filled_ == MODEL_PRETRIGGER_SAMPLES && isMoving(s)) {
      // Copy the history, oldest first. The newest entry is this reading.
      for (int k = 0; k < MODEL_PRETRIGGER_SAMPLES; ++k) {
        store(k, history_[(head_ + k) % MODEL_PRETRIGGER_SAMPLES]);
      }
      count_ = MODEL_PRETRIGGER_SAMPLES;
      state_ = kRecording;
    }
    return false;
  }

  bool isRecording() const { return state_ == kRecording; }

  // MODEL_WINDOW_SAMPLES rows of MODEL_N_CHANNELS values: ax, ay, az, gx, gy, gz.
  const float* window() const { return window_; }

 private:
  static constexpr unsigned long kPeriodUs = 1000000UL / MODEL_SAMPLE_RATE_HZ;
  enum State { kIdle, kRecording, kCooldown };

  static bool isMoving(const float* s) {
    const float a = sqrtf(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    const float g = sqrtf(s[3] * s[3] + s[4] * s[4] + s[5] * s[5]);
    return fabsf(a - 1.0f) > MODEL_TRIGGER_ACCEL_G || g > MODEL_TRIGGER_GYRO_DPS;
  }

  void store(int row, const float* s) {
    for (int c = 0; c < MODEL_N_CHANNELS; ++c) window_[row * MODEL_N_CHANNELS + c] = s[c];
  }

  float window_[MODEL_WINDOW_SAMPLES * MODEL_N_CHANNELS];
  float history_[MODEL_PRETRIGGER_SAMPLES][MODEL_N_CHANNELS];
  int head_ = 0, filled_ = 0, count_ = 0;
  State state_ = kIdle;
  bool started_ = false;
  unsigned long next_us_ = 0, cooldown_until_ms_ = 0;
};
