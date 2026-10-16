// Host-side test driver for arduino/gesture_classifier/edge_nn.h.
// Reads raw gesture windows (float32) from argv[1], writes int8 logits and
// float logits (float32) to argv[2]. Used by tests/test_edge_nn.py.
#include <cstdio>
#include <vector>

#include "edge_nn.h"

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  FILE* fin = std::fopen(argv[1], "rb");
  FILE* fout = std::fopen(argv[2], "wb");
  if (!fin || !fout) return 3;
  std::vector<float> raw(MODEL_N_FEATURES), feats(MODEL_N_FEATURES);
  float logits[MODEL_N_CLASSES];
  while (std::fread(raw.data(), sizeof(float), MODEL_N_FEATURES, fin) == MODEL_N_FEATURES) {
    edge_nn::normalize(raw.data(), feats.data());
    edge_nn::forward_int8(feats.data(), logits);
    std::fwrite(logits, sizeof(float), MODEL_N_CLASSES, fout);
    edge_nn::forward_float(feats.data(), logits);
    std::fwrite(logits, sizeof(float), MODEL_N_CLASSES, fout);
  }
  std::fclose(fin);
  std::fclose(fout);
  return 0;
}
