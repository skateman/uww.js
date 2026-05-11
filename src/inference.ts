import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import * as tflite from '@tensorflow/tfjs-tflite';
import { MicroFrontend } from './wrapper.js';

// tfjs-tflite logs `WARNING: converting 'int32' to 'int8'` on every
// `predict()` call when the model is INT8-quantized — which is every
// microWakeWord model. That floods the console at ~100 warnings/sec.
// Install a one-time filter that drops exactly this one message.
const NOISY_WARNING = "WARNING: converting 'int32' to 'int8'";
let warnPatched = false;
function patchTfliteWarn(): void {
  if (warnPatched) return;
  warnPatched = true;
  const orig = console.warn.bind(console);
  console.warn = (...args: unknown[]) => {
    if (typeof args[0] === 'string' && args[0].startsWith(NOISY_WARNING)) return;
    orig(...args);
  };
}

export interface InferenceConfig {
  wakeWordModel: string | ArrayBuffer;
  wasmPath?: string;
  sampleRate?: number;
}

export interface InferenceInfo {
  /** Raw audio samples expected per `process()` call (== one window step). */
  frameSize: number;
  /** Number of streaming-state tensors carried between wake-word calls. */
  stateTensorCount: number;
  /** Number of feature rows the wake-word model takes per inference (the time
   *  dimension of input 0). microWakeWord exports range from 1 to ~10. */
  featureWindow: number;
}

/** Diagnostic snapshot for debugging the pipeline. */
export interface InferenceDebug {
  /** Frames seen by the audio frontend so far. */
  audioFrames: number;
  /** Feature rows emitted by the audio frontend so far. */
  featuresProduced: number;
  /** Wake-word inferences run so far. */
  inferences: number;
  /** Stats for the most recent feature row. */
  lastFeatureRow: { mean: number; min: number; max: number; nonZero: number } | null;
  /** Wake-word input shape `[1, T, F]`. */
  featureShape: number[];
  /** Feature input quantization (scale 1, zero 0 → un-quantized). */
  featureQuant: { scale: number; zero: number; dtype: string };
  /** Output dequantization. */
  outputQuant: { scale: number; zero: number; dtype: string };
  /** Last raw probability returned. */
  lastProbability: number;
}

/**
 * Streaming wake-word inference pipeline:
 *
 *   raw int16 audio → WASM MicroFrontend → 40-bin float features
 *   features stacked over a sliding window → wake-word `.tflite` →
 *     probability + new state
 *
 * The audio frontend is the same DSP that ESPHome's `micro_wake_word`
 * runs natively on the ESP32, compiled to WASM from the TFLite-Micro
 * sources via Emscripten — so features are bit-identical to what the
 * model was trained against.
 *
 * The wake-word model's input shape is `[1, T, F]` where T is the
 * number of feature frames consumed per call (a small sliding window,
 * typically 1–10 frames) and F is the feature size (40). We carry a
 * ring buffer of the most recent T feature frames and step the model
 * once per new frame the frontend emits.
 *
 * INT8 / UINT8 quantized inputs are quantized on the fly using the
 * scale and zero-point published by the model's input metadata.
 *
 * State tensors are owned by this class and reused across calls.
 * Disposed via `dispose()`.
 */
export class InferencePipeline {
  private readonly frontend = new MicroFrontend();
  private wakeWord: tflite.TFLiteModel | null = null;
  private state: tf.Tensor[] = [];

  /** Concrete shape `[1, T, F]` of the wake-word feature input. */
  private featureShape: number[] = [];
  /** Number of frames the wake-word model wants per call (T = stride). */
  private featureWindow = 1;
  /** Number of values per frame (F). */
  private featureSize = 0;
  /** Buffer of `featureWindow` feature rows (flat, length T*F). */
  private featureBuffer: Float32Array = new Float32Array(0);
  /**
   * Cyclic slot index in [0, featureWindow). microWakeWord models are
   * **streaming** — they hold internal `Stream/Variable` state that
   * persists across `predict()` calls and assumes each call delivers
   * `featureWindow` BRAND NEW frames (non-overlapping). We write each
   * new frame into the buffer at this slot, increment, and only invoke
   * when the slot wraps. Feeding overlapping windows (sliding by 1)
   * silently corrupts the streaming state and the model returns 0
   * forever. This matches ESPHome's `current_stride_step_` logic.
   */
  private strideStep = 0;
  private lastProbability = 0;

  private featureDtype: string = 'float32';
  private featureScale = 1;
  private featureZero = 0;
  private outputDtype: string = 'float32';
  private outputScale = 1;
  private outputZero = 0;

  // Diagnostics
  private audioFrames = 0;
  private featuresProduced = 0;
  private inferences = 0;
  private lastFeatureRow: Float32Array | null = null;

  async load(config: InferenceConfig): Promise<InferenceInfo> {
    // tfjs-tflite defaults its wasmPath to "" which resolves the WASM/JS
    // glue files to the page origin and 404s. Default to the official CDN
    // unless the caller provides an explicit path. Note that the package
    // also runs a hard-coded preload at module-import time (with empty
    // path) which 404s harmlessly; the loader cache is keyed on path so
    // the call below creates a fresh, working loader.
    tflite.setWasmPath(
      config.wasmPath ??
        'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm/'
    );
    patchTfliteWarn();

    await this.frontend.load(config.sampleRate ?? 16000);
    await tf.setBackend('cpu');
    await tf.ready();
    this.wakeWord = await tflite.loadTFLiteModel(config.wakeWordModel as string);

    const stateInputCount = this.wakeWord.inputs.length - 1;
    const stateOutputCount = this.wakeWord.outputs.length - 1;
    if (stateInputCount < 0) {
      throw new Error('uww: wake-word model must have at least one input');
    }
    if (stateInputCount !== stateOutputCount) {
      throw new Error(
        `uww: wake-word model state mismatch — ${stateInputCount} state inputs vs ` +
          `${stateOutputCount} state outputs. This pipeline expects a streaming ` +
          'microWakeWord export where each state input has a matching output.'
      );
    }

    // Parse the feature input shape [1, T, F] (any null/-1 dims become 1).
    const featInfo = this.wakeWord.inputs[0];
    if (!featInfo?.shape || featInfo.shape.length < 2) {
      throw new Error('uww: wake-word model input 0 has no usable shape');
    }
    this.featureShape = featInfo.shape.map((d) =>
      d == null || d < 0 ? 1 : d
    );
    // Last dim is the feature size; the dim before it is the time window.
    // For the typical [1, T, 40] shape that's [1, T, F]. Some exports use
    // [1, T, F, 1] (extra channel dim) — handled by collapsing to T*F flat.
    this.featureSize = this.featureShape[this.featureShape.length - 1] ?? 40;
    this.featureWindow =
      this.featureShape.length >= 3
        ? this.featureShape[this.featureShape.length - 2] ?? 1
        : 1;
    if (this.featureSize !== this.frontend.featureSize) {
      throw new Error(
        `uww: wake-word model expects features of size ${this.featureSize} but ` +
          `the audio frontend produces ${this.frontend.featureSize}`
      );
    }
    this.featureBuffer = new Float32Array(this.featureWindow * this.featureSize);
    this.strideStep = 0;

    this.featureDtype = (featInfo.dtype as string | undefined) ?? 'float32';
    // The quantization params reported by tfjs-tflite's public API are
    // not the actual model quantization (it strips that info). microWakeWord
    // int8 inputs use a fixed mapping: int8 = round(float * 256/26) - 128,
    // i.e., scale=26/256 ≈ 0.1016 and zero_point = -128. Recorded here so
    // the diagnostic snapshot matches what quantizeFeatures() actually does.
    this.featureScale = 26 / 255;
    this.featureZero = -128;

    const outInfo = this.wakeWord.outputs[0];
    this.outputDtype = (outInfo?.dtype as string | undefined) ?? 'float32';
    // microWakeWord v2 uint8 output: scale = 1/256, zero_point = 0
    this.outputScale = 1 / 256;
    this.outputZero = 0;

    this.resetState();

    const sampleRate = config.sampleRate ?? 16000;
    const frameSize = (this.frontend.stepMs * sampleRate) / 1000;

    return {
      frameSize,
      stateTensorCount: stateInputCount,
      featureWindow: this.featureWindow,
    };
  }

  resetState(): void {
    if (!this.wakeWord) return;
    this.disposeState();
    this.state = this.wakeWord.inputs.slice(1).map((info) => {
      const shape = (info.shape ?? [1]).map((d) => (d == null || d < 0 ? 1 : d));
      const dtype = (info.dtype as 'float32' | 'int32' | 'bool') ?? 'float32';
      return tf.zeros(shape, dtype);
    });
    this.frontend.reset();
    this.featureBuffer.fill(0);
    this.strideStep = 0;
    this.lastProbability = 0;
  }

  /**
   * Run one raw audio frame through the full pipeline.
   * Returns the wake-word probability for the most recent inference, or
   * the previous probability if this frame did not yet complete a window.
   */
  process(frame: Float32Array): number {
    if (!this.wakeWord) {
      throw new Error('uww: pipeline not loaded');
    }
    this.audioFrames++;
    const rows = this.frontend.processFloat32(frame);
    for (const row of rows) {
      this.featuresProduced++;
      this.lastFeatureRow = row;
      // Write the new feature row into the current slot of the buffer.
      this.featureBuffer.set(row, this.strideStep * this.featureSize);
      this.strideStep++;
      // Only invoke when we've filled all T slots with NEW frames. The
      // model's internal streaming state assumes non-overlapping inputs.
      if (this.strideStep >= this.featureWindow) {
        this.strideStep = 0;
        this.inferences++;
        this.lastProbability = this.runWakeWord();
      }
    }
    return this.lastProbability;
  }

  getDebug(): InferenceDebug {
    let stats: { mean: number; min: number; max: number; nonZero: number } | null = null;
    if (this.lastFeatureRow) {
      let sum = 0;
      let min = Infinity;
      let max = -Infinity;
      let nz = 0;
      for (const v of this.lastFeatureRow) {
        sum += v;
        if (v < min) min = v;
        if (v > max) max = v;
        if (v !== 0) nz++;
      }
      stats = {
        mean: sum / this.lastFeatureRow.length,
        min,
        max,
        nonZero: nz,
      };
    }
    return {
      audioFrames: this.audioFrames,
      featuresProduced: this.featuresProduced,
      inferences: this.inferences,
      lastFeatureRow: stats,
      featureShape: this.featureShape.slice(),
      featureQuant: {
        scale: this.featureScale,
        zero: this.featureZero,
        dtype: this.featureDtype,
      },
      outputQuant: {
        scale: this.outputScale,
        zero: this.outputZero,
        dtype: this.outputDtype,
      },
      lastProbability: this.lastProbability,
    };
  }

  dispose(): void {
    this.disposeState();
    this.frontend.dispose();
    this.wakeWord = null;
  }

  private runWakeWord(): number {
    const wakeWord = this.wakeWord;
    if (!wakeWord) throw new Error('uww: pipeline not loaded');

    const featTensor = this.makeFeatureTensor();
    const wwOut = wakeWord.predict([featTensor, ...this.state]);
    featTensor.dispose();

    const outputs = unwrapMany(wwOut);
    const probTensor = outputs[0];
    if (!probTensor) {
      throw new Error('uww: wake-word model produced no outputs');
    }
    const prob = this.readProbability(probTensor);

    this.disposeState();
    this.state = outputs.slice(1);
    probTensor.dispose();
    return prob;
  }

  /**
   * Build the input tensor in the dtype the model expects, quantizing
   * float features on the fly when the model is INT8/UINT8.
   *
   * tfjs-tflite reports all integer dtypes as `int32` in its public
   * metadata, but the underlying model may be int8/uint8. We detect
   * that via the presence of non-trivial quantization params and feed
   * pre-quantized integer values as int32 — the runtime narrows them
   * to the actual int8/uint8 tensor underneath.
   */
  private makeFeatureTensor(): tf.Tensor {
    const isQuantized = this.featureScale !== 1 || this.featureZero !== 0;
    if (this.featureDtype === 'float32' && !isQuantized) {
      return tf.tensor(this.featureBuffer, this.featureShape, 'float32');
    }
    if (this.featureDtype === 'int32' || isQuantized) {
      const buf = this.quantizeFeatures();
      return tf.tensor(buf, this.featureShape, 'int32');
    }
    if (this.featureDtype === 'float32') {
      return tf.tensor(this.featureBuffer, this.featureShape, 'float32');
    }
    throw new Error(
      `uww: unsupported feature input dtype "${this.featureDtype}"`
    );
  }

  private quantizeFeatures(): Int32Array {
    const out = new Int32Array(this.featureBuffer.length);
    // microWakeWord v2 models share fixed input quantization (same across
    // every model in esphome/micro-wake-word-models): scale = 0.10196079
    // (= 26/255), zero_point = -128. Equivalent rearranged:
    //   int8 = round(float * 255 / 26) - 128
    // tfjs-tflite's public API does not expose quantization params, so
    // we hardcode them here. Verified against the .tflite flatbuffer for
    // hey_jarvis, alexa, hey_mycroft, okay_nabu.
    const SCALE = 255 / 26;   // ≈ 9.8077
    for (let i = 0; i < this.featureBuffer.length; i++) {
      const q = Math.round((this.featureBuffer[i] ?? 0) * SCALE) - 128;
      out[i] = q < -128 ? -128 : q > 127 ? 127 : q;
    }
    return out;
  }

  private readProbability(probTensor: tf.Tensor): number {
    const data = probTensor.dataSync();
    const raw = data[0] ?? 0;
    // microWakeWord v2 output: uint8 with scale = 1/256, zero_point = 0.
    return raw / 256;
  }

  private disposeState(): void {
    for (const t of this.state) t.dispose();
    this.state = [];
  }
}

function unwrapMany(out: tf.Tensor | tf.Tensor[] | tf.NamedTensorMap): tf.Tensor[] {
  if (out instanceof tf.Tensor) return [out];
  if (Array.isArray(out)) return out;
  return Object.values(out);
}
