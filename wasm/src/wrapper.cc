// wrapper.cc
//
// Thin C ABI exposing TFLite-Micro's audio frontend (mel + PCAN + log) to
// JavaScript via Emscripten. Configuration matches the canonical
// microWakeWord preprocessor (rhasspy/pymicro-features), which is the
// same DSP ESPHome's `micro_wake_word` runs natively on the ESP32.
//
// Functions are intentionally minimal C-ABI — no embind, no Python — so
// the resulting WASM is small and the JS glue is straightforward.

#include <cstdint>
#include <cstdlib>
#include <emscripten/emscripten.h>

extern "C" {
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
}

namespace {

constexpr int kFeatureSize        = 40;   // mel channels
constexpr int kWindowMs           = 30;
constexpr int kStepMs             = 10;
constexpr float kFeatureScale     = 1.0f / 25.6f;  // matches pymicro-features
constexpr float kLowerHz          = 125.0f;
constexpr float kUpperHz          = 7500.0f;

void InitDefaultConfig(FrontendConfig* cfg) {
  cfg->window.size_ms                       = kWindowMs;
  cfg->window.step_size_ms                  = kStepMs;
  cfg->filterbank.num_channels              = kFeatureSize;
  cfg->filterbank.lower_band_limit          = kLowerHz;
  cfg->filterbank.upper_band_limit          = kUpperHz;
  cfg->noise_reduction.smoothing_bits       = 10;
  cfg->noise_reduction.even_smoothing       = 0.025f;
  cfg->noise_reduction.odd_smoothing        = 0.06f;
  cfg->noise_reduction.min_signal_remaining = 0.05f;
  cfg->pcan_gain_control.enable_pcan        = 1;
  cfg->pcan_gain_control.strength           = 0.95f;
  cfg->pcan_gain_control.offset             = 80.0f;
  cfg->pcan_gain_control.gain_bits          = 21;
  cfg->log_scale.enable_log                 = 1;
  cfg->log_scale.scale_shift                = 6;
}

struct UwwFrontend {
  FrontendConfig cfg;
  FrontendState  state;
  int            sample_rate;
};

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
UwwFrontend* uww_frontend_create(int sample_rate) {
  auto* h = static_cast<UwwFrontend*>(std::malloc(sizeof(UwwFrontend)));
  if (!h) return nullptr;
  InitDefaultConfig(&h->cfg);
  h->sample_rate = sample_rate;
  if (!FrontendPopulateState(&h->cfg, &h->state, sample_rate)) {
    std::free(h);
    return nullptr;
  }
  return h;
}

EMSCRIPTEN_KEEPALIVE
void uww_frontend_destroy(UwwFrontend* h) {
  if (!h) return;
  FrontendFreeStateContents(&h->state);
  std::free(h);
}

EMSCRIPTEN_KEEPALIVE
void uww_frontend_reset(UwwFrontend* h) {
  if (!h) return;
  FrontendFreeStateContents(&h->state);
  FrontendPopulateState(&h->cfg, &h->state, h->sample_rate);
}

// Process an int16 audio chunk. Writes up to `kFeatureSize` float32 values
// into `out_features` when a window completes; otherwise writes 0 values.
//
// Returns:
//   >0  = number of features written (typically kFeatureSize when ready)
//    0  = window not yet complete; call again with more samples
//   -1  = error
EMSCRIPTEN_KEEPALIVE
int uww_frontend_process(
    UwwFrontend* h,
    const int16_t* samples,
    int num_samples,
    float* out_features) {
  if (!h || !samples || !out_features || num_samples < 0) return -1;
  size_t samples_read = 0;
  FrontendOutput out = FrontendProcessSamples(
      &h->state, samples, static_cast<size_t>(num_samples), &samples_read);
  if (out.size == 0) return 0;
  for (size_t i = 0; i < out.size; ++i) {
    out_features[i] = static_cast<float>(out.values[i]) * kFeatureScale;
  }
  return static_cast<int>(out.size);
}

// Static getters so JS doesn't have to hard-code the same constants.
EMSCRIPTEN_KEEPALIVE int uww_frontend_feature_size(void)  { return kFeatureSize; }
EMSCRIPTEN_KEEPALIVE int uww_frontend_step_ms(void)       { return kStepMs; }
EMSCRIPTEN_KEEPALIVE int uww_frontend_window_ms(void)     { return kWindowMs; }

}  // extern "C"
