// edge_nn.h -- the neural network, in plain C++ (no libraries).
//
// It computes exactly what the Python code computes:
//   * normalize()     -- aiedge.export.device_features
//   * forward_int8()  -- aiedge.quantization.int8_forward   (bit-identical)
//   * forward_float() -- the PyTorch MLP (equal up to float rounding)
// All numbers (weights, scales, normalization) come from model_data.h, which
// notebook 05 generates.
//
// Memory: every buffer is static (no malloc), so the RAM use is fixed and is
// reported by the Arduino IDE when you compile.

#pragma once
#include <math.h>
#include <stdint.h>

#include "model_data.h"

namespace edge_nn {

// raw: MODEL_WINDOW_SAMPLES rows of MODEL_N_CHANNELS sensor values.
// features: MODEL_N_FEATURES normalized values, in the same (row-major) order.
inline void normalize(const float* raw, float* features) {
  for (int t = 0; t < MODEL_WINDOW_SAMPLES; ++t) {
    for (int c = 0; c < MODEL_N_CHANNELS; ++c) {
      const int k = t * MODEL_N_CHANNELS + c;
      features[k] = (raw[k] - kChannelMean[c]) / kChannelStd[c];
    }
  }
}

// ---------------------------------------------------------------------------
// int8 inference: the heavy loop only uses integer multiply-adds.
// ---------------------------------------------------------------------------
inline void forward_int8(const float* features, float* logits) {
  static int8_t buf_a[MODEL_N_FEATURES > MODEL_MAX_WIDTH ? MODEL_N_FEATURES : MODEL_MAX_WIDTH];
  static int8_t buf_b[MODEL_MAX_WIDTH];
  static int16_t centered[MODEL_N_FEATURES > MODEL_MAX_WIDTH ? MODEL_N_FEATURES : MODEL_MAX_WIDTH];

  // 1. Quantize the input: real value -> int8.
  for (int i = 0; i < MODEL_N_FEATURES; ++i) {
    int32_t q = (int32_t)roundf(features[i] / kInputScale) + kInputZeroPoint;
    buf_a[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
  }

  // 2. The layers. `in` and `out` swap roles after every layer ("ping-pong").
  int8_t* in = buf_a;
  int8_t* out = buf_b;
  for (int l = 0; l < MODEL_N_LAYERS; ++l) {
    const int n_in = kLayerIn[l];
    const int n_out = kLayerOut[l];
    const int8_t* W = kWeightsQ[l];
    const int32_t* B = kBiasesQ[l];
    const float* M = kMultipliers[l];
    const int32_t z_in = kInZeroPoint[l];
    const int32_t z_out = kOutZeroPoint[l];
    const bool last = (l == MODEL_N_LAYERS - 1);

    for (int i = 0; i < n_in; ++i) centered[i] = (int16_t)(in[i] - z_in);

    for (int j = 0; j < n_out; ++j) {
      const int8_t* row = W + j * n_in;
      int32_t acc = B[j];
      for (int i = 0; i < n_in; ++i) acc += (int32_t)row[i] * centered[i];   // the MACs

      if (last) {
        logits[j] = (float)acc * M[j];                 // back to real numbers
      } else {
        int32_t q = (int32_t)roundf((float)acc * M[j]) + z_out;   // re-quantize
        if (q < z_out) q = z_out;                     // ReLU (z_out is real 0)
        if (q > 127) q = 127;
        out[j] = (int8_t)q;
      }
    }
    int8_t* tmp = in;
    in = out;
    out = tmp;
  }
}

// ---------------------------------------------------------------------------
// float32 inference (the same network before quantization), for comparison.
// ---------------------------------------------------------------------------
inline void forward_float(const float* features, float* logits) {
  static float buf_a[MODEL_MAX_WIDTH];
  static float buf_b[MODEL_MAX_WIDTH];
  const float* in = features;
  float* out = buf_a;
  for (int l = 0; l < MODEL_N_LAYERS; ++l) {
    const int n_in = kLayerIn[l];
    const int n_out = kLayerOut[l];
    const bool last = (l == MODEL_N_LAYERS - 1);
    if (last) out = logits;
    for (int j = 0; j < n_out; ++j) {
      const float* row = kWeightsF[l] + j * n_in;
      float acc = kBiasesF[l][j];
      for (int i = 0; i < n_in; ++i) acc += row[i] * in[i];
      out[j] = (last || acc > 0.0f) ? acc : 0.0f;     // ReLU on hidden layers
    }
    in = out;
    out = (out == buf_a) ? buf_b : buf_a;
  }
}

// Logits -> probabilities that sum to 1.
inline void softmax(const float* logits, float* probs, int n) {
  float mx = logits[0];
  for (int j = 1; j < n; ++j) mx = logits[j] > mx ? logits[j] : mx;
  float sum = 0.0f;
  for (int j = 0; j < n; ++j) {
    probs[j] = expf(logits[j] - mx);
    sum += probs[j];
  }
  for (int j = 0; j < n; ++j) probs[j] /= sum;
}

inline int argmax(const float* v, int n) {
  int best = 0;
  for (int j = 1; j < n; ++j) if (v[j] > v[best]) best = j;
  return best;
}

}  // namespace edge_nn
