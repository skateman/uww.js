import createFrontendModule, {
  type UwwFrontendModule,
} from './_wasm/uww-frontend.js';

/**
 * Thin TS wrapper around the Emscripten-compiled TFLite-Micro audio
 * frontend. Configuration mirrors microWakeWord training (and ESPHome's
 * `micro_wake_word`) bit-for-bit:
 *
 *   - 16 kHz mono input, int16 audio
 *   - 30 ms window, 10 ms step
 *   - 40 mel-filterbank channels (125–7500 Hz)
 *   - PCAN gain control + log compression
 *
 * Streaming: feed the frontend a continuous 16 kHz audio stream in
 * arbitrary-size int16 chunks; it emits one 40-element feature row each
 * time enough samples have accumulated for a new window.
 */
export class MicroFrontend {
  private module: UwwFrontendModule | null = null;
  private handle = 0;
  private samplesPtr = 0;
  private samplesCapacity = 0;
  private featuresPtr = 0;
  private _featureSize = 0;
  private _stepMs = 0;
  private _windowMs = 0;

  /** Number of mel channels per feature row. Always 40 for microWakeWord. */
  get featureSize(): number {
    return this._featureSize;
  }
  /** Window step in milliseconds (10). */
  get stepMs(): number {
    return this._stepMs;
  }
  /** Window length in milliseconds (30). */
  get windowMs(): number {
    return this._windowMs;
  }

  async load(sampleRate = 16000): Promise<void> {
    if (this.module) return;
    this.module = await createFrontendModule();
    this._featureSize = this.module._uww_frontend_feature_size();
    this._stepMs = this.module._uww_frontend_step_ms();
    this._windowMs = this.module._uww_frontend_window_ms();
    this.handle = this.module._uww_frontend_create(sampleRate);
    if (!this.handle) {
      throw new Error('uww: MicroFrontend failed to initialize WASM frontend');
    }
    this.featuresPtr = this.module._malloc(this._featureSize * 4);
    if (!this.featuresPtr) {
      this.dispose();
      throw new Error('uww: MicroFrontend failed to allocate feature buffer');
    }
  }

  /**
   * Process a chunk of 16-bit PCM audio. Each call consumes the chunk and
   * may emit zero or more feature rows depending on how much new data
   * crosses a window boundary. Most callers feed 10 ms (= 160 samples)
   * per call, in which case at most one row is produced.
   *
   * Returns: an array of Float32Array rows, each of length `featureSize`.
   * The returned arrays are copies; the underlying WASM buffer is reused.
   */
  process(samples: Int16Array): Float32Array[] {
    const m = this.assertReady();
    this.ensureSamplesCapacity(samples.length);
    m.HEAP16.set(samples, this.samplesPtr / 2);

    const rows: Float32Array[] = [];
    // The frontend consumes one window's worth per call. To keep the JS
    // glue simple and bound the number of rows per call, feed the chunk
    // in step-sized slices (160 samples by default).
    const stepSamples = (this._stepMs * 16000) / 1000; // 160 at 16 kHz
    for (let off = 0; off < samples.length; off += stepSamples) {
      const remaining = samples.length - off;
      const n = m._uww_frontend_process(
        this.handle,
        this.samplesPtr + off * 2,
        Math.min(stepSamples, remaining),
        this.featuresPtr
      );
      if (n > 0) {
        const start = this.featuresPtr / 4;
        rows.push(m.HEAPF32.slice(start, start + n));
      } else if (n < 0) {
        throw new Error('uww: MicroFrontend.process returned an error');
      }
    }
    return rows;
  }

  /** Convert a Float32 PCM frame ([-1, 1]) to int16 and process it. */
  processFloat32(samples: Float32Array): Float32Array[] {
    const int16 = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const v = Math.max(-1, Math.min(1, samples[i] ?? 0));
      int16[i] = v < 0 ? Math.round(v * 32768) : Math.round(v * 32767);
    }
    return this.process(int16);
  }

  reset(): void {
    if (this.module && this.handle) {
      this.module._uww_frontend_reset(this.handle);
    }
  }

  dispose(): void {
    const m = this.module;
    if (!m) return;
    if (this.samplesPtr) m._free(this.samplesPtr);
    if (this.featuresPtr) m._free(this.featuresPtr);
    if (this.handle) m._uww_frontend_destroy(this.handle);
    this.samplesPtr = 0;
    this.featuresPtr = 0;
    this.handle = 0;
    this.samplesCapacity = 0;
    this.module = null;
  }

  private assertReady(): UwwFrontendModule {
    if (!this.module || !this.handle) {
      throw new Error('uww: MicroFrontend.load() must be called first');
    }
    return this.module;
  }

  private ensureSamplesCapacity(samples: number): void {
    const m = this.assertReady();
    if (samples <= this.samplesCapacity) return;
    if (this.samplesPtr) m._free(this.samplesPtr);
    const bytes = samples * 2;
    this.samplesPtr = m._malloc(bytes);
    if (!this.samplesPtr) {
      this.samplesCapacity = 0;
      throw new Error('uww: MicroFrontend failed to allocate samples buffer');
    }
    this.samplesCapacity = samples;
  }
}
