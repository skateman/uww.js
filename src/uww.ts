import { AudioCapture } from './audio-capture.js';
import { Detector } from './detector.js';
import { InferencePipeline } from './inference.js';
import { fetchManifest } from './manifest.js';
import type {
  ErrorEventDetail,
  ProbabilityEventDetail,
  UWWOptions,
  UWWStatus,
  WakeEventDetail,
  WakeWordSource,
} from './types.js';

const DEFAULTS = {
  threshold: 0.7,
  slidingWindowSize: 5,
  refractoryMs: 2000,
  sampleRate: 16000,
};

/**
 * In-browser microWakeWord detector.
 *
 * ```ts
 * const uww = new UWW({
 *   wakeWord: { manifestUrl: 'https://.../hey_jarvis.json' },
 * });
 * await uww.load();
 * uww.addEventListener('wake', (e) => console.log('wake!', e.detail));
 * await uww.start();
 * ```
 */
export class UWW extends EventTarget {
  private readonly opts: UWWOptions;
  private readonly pipeline = new InferencePipeline();
  private detector: Detector | null = null;
  private capture: AudioCapture | null = null;
  private _status: UWWStatus = 'idle';
  private _lastProbability = 0;
  private _wakeWordName: string | null = null;
  private frameSize = 0;
  private processingFrame = false;

  constructor(options: UWWOptions) {
    super();
    validateSource(options.wakeWord);
    this.opts = options;
  }

  get status(): UWWStatus {
    return this._status;
  }

  get lastProbability(): number {
    return this._lastProbability;
  }

  /** Human-readable wake-word name from the manifest, if available. */
  get wakeWordName(): string | null {
    return this._wakeWordName;
  }

  /** Diagnostic snapshot — sample rate, frame counts, feature stats, quant params. */
  getDebug(): { sampleRate: number | null } & ReturnType<InferencePipeline['getDebug']> {
    return {
      sampleRate: this.capture?.actualSampleRate ?? null,
      ...this.pipeline.getDebug(),
    };
  }

  /** Resolve the source, load the model, and prepare detection state. Idempotent. */
  async load(): Promise<void> {
    if (this.frameSize > 0) return;
    this.setStatus('loading');
    try {
      const resolved = await this.resolveSource(this.opts.wakeWord);
      this._wakeWordName = resolved.wakeWordName;

      const sampleRate = this.opts.sampleRate ?? DEFAULTS.sampleRate;
      const info = await this.pipeline.load({
        wakeWordModel: resolved.modelData,
        wasmPath: this.opts.wasmPath,
        sampleRate,
      });
      this.frameSize = info.frameSize;

      // Manifest values are defaults; explicit options always win.
      const threshold = this.opts.threshold ?? resolved.threshold ?? DEFAULTS.threshold;
      const slidingWindowSize =
        this.opts.slidingWindowSize ??
        resolved.slidingWindowSize ??
        DEFAULTS.slidingWindowSize;
      const refractoryMs = this.opts.refractoryMs ?? DEFAULTS.refractoryMs;

      this.detector = new Detector({
        threshold,
        windowSize: slidingWindowSize,
        refractoryMs,
      });
      this.setStatus('idle');
    } catch (error) {
      this.fail(error);
      throw error;
    }
  }

  /** Begin listening on the microphone. Calls `load()` if needed. */
  async start(): Promise<void> {
    if (this._status === 'listening') return;
    if (this.frameSize === 0) await this.load();
    if (!this.detector) throw new Error('uww: detector not initialized');

    try {
      this.capture = new AudioCapture({
        sampleRate: this.opts.sampleRate ?? DEFAULTS.sampleRate,
        frameSize: this.frameSize,
        mediaStream: this.opts.mediaStream,
        onFrame: (frame) => this.handleFrame(frame),
      });
      await this.capture.start();
      this.setStatus('listening');
    } catch (error) {
      this.fail(error);
      throw error;
    }
  }

  /** Stop listening. Models stay loaded; call `dispose()` to free them. */
  async stop(): Promise<void> {
    if (this.capture) {
      const c = this.capture;
      this.capture = null;
      await c.stop();
    }
    this.detector?.reset();
    if (this._status !== 'error') this.setStatus('idle');
  }

  /** Stop listening and free model + state tensors. */
  async dispose(): Promise<void> {
    await this.stop();
    this.pipeline.dispose();
    this.detector = null;
    this.frameSize = 0;
  }

  private async resolveSource(source: WakeWordSource): Promise<{
    modelData: string | ArrayBuffer;
    threshold?: number;
    slidingWindowSize?: number;
    wakeWordName: string | null;
  }> {
    if ('manifestUrl' in source) {
      const { manifest, modelUrl } = await fetchManifest(source.manifestUrl);
      return {
        modelData: modelUrl,
        threshold: manifest.micro.probability_cutoff,
        slidingWindowSize: manifest.micro.sliding_window_size,
        wakeWordName: manifest.wake_word,
      };
    }
    if ('manifest' in source) {
      return {
        modelData: source.modelData,
        threshold: source.manifest.micro.probability_cutoff,
        slidingWindowSize: source.manifest.micro.sliding_window_size,
        wakeWordName: source.manifest.wake_word,
      };
    }
    return {
      modelData: source.wakeWordModel,
      wakeWordName: null,
    };
  }

  private handleFrame(frame: Float32Array): void {
    if (!this.detector || this.processingFrame) {
      // Drop frames if we're still busy with the previous one. This keeps
      // the audio thread from queuing unbounded backlog if inference stalls.
      return;
    }
    this.processingFrame = true;
    try {
      const prob = this.pipeline.process(frame);
      this._lastProbability = prob;
      this.dispatchEvent(
        new CustomEvent<ProbabilityEventDetail>('probability', {
          detail: { probability: prob },
        })
      );
      const { fired, mean } = this.detector.push(prob);
      if (fired) {
        this.dispatchEvent(
          new CustomEvent<WakeEventDetail>('wake', {
            detail: { probability: mean, timestamp: performance.now() },
          })
        );
      }
    } catch (error) {
      this.fail(error);
    } finally {
      this.processingFrame = false;
    }
  }

  private setStatus(status: UWWStatus): void {
    if (this._status === status) return;
    this._status = status;
    this.dispatchEvent(
      new CustomEvent<{ status: UWWStatus }>('statuschange', { detail: { status } })
    );
  }

  private fail(error: unknown): void {
    const err = error instanceof Error ? error : new Error(String(error));
    this.setStatus('error');
    this.dispatchEvent(
      new CustomEvent<ErrorEventDetail>('error', { detail: { error: err } })
    );
  }
}

function validateSource(source: WakeWordSource | undefined): void {
  if (!source || typeof source !== 'object') {
    throw new Error(
      'uww: options.wakeWord must be { manifestUrl }, { manifest, modelData }, or { wakeWordModel }'
    );
  }
  const variants = [
    'manifestUrl' in source && source.manifestUrl,
    'manifest' in source && source.manifest,
    'wakeWordModel' in source && source.wakeWordModel,
  ].filter(Boolean).length;
  if (variants !== 1) {
    throw new Error(
      'uww: options.wakeWord must specify exactly one of manifestUrl, manifest+modelData, or wakeWordModel'
    );
  }
}
