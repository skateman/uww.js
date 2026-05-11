import type { WakeWordManifest } from './manifest.js';

export type WakeWordSource =
  | {
      /**
       * URL of a microWakeWord JSON manifest. The library fetches the
       * manifest, then fetches the `.tflite` whose path is resolved
       * against this URL — the exact same flow ESPHome uses on the ESP32.
       *
       * `threshold` and `slidingWindowSize` default to the manifest's
       * `probability_cutoff` and `sliding_window_size` unless overridden.
       */
      manifestUrl: string;
    }
  | {
      /**
       * A parsed manifest object plus the model bytes (or its URL).
       * Use this when you already have the files locally — e.g., from a
       * file picker that selected both the .json and the .tflite.
       */
      manifest: WakeWordManifest;
      modelData: ArrayBuffer | string;
    }
  | {
      /**
       * Raw `.tflite` bytes or URL with no manifest. The library uses
       * defaults for `threshold` and `slidingWindowSize`; pass them
       * explicitly if you know better.
       */
      wakeWordModel: string | ArrayBuffer;
    };

export interface UWWOptions {
  /**
   * Wake-word source. Pass exactly one of:
   *   - `{ manifestUrl }` — fetch a microWakeWord manifest from a URL.
   *   - `{ manifest, modelData }` — provide both directly.
   *   - `{ wakeWordModel }` — raw `.tflite` bytes/URL, no manifest.
   */
  wakeWord: WakeWordSource;
  /** Probability threshold (0..1). Defaults to manifest's probability_cutoff or 0.7. */
  threshold?: number;
  /** Frames averaged. Defaults to manifest's sliding_window_size or 5. */
  slidingWindowSize?: number;
  /** Cooldown after a detection, in ms. Default: 2000 */
  refractoryMs?: number;
  /** Audio sample rate in Hz. Default: 16000. */
  sampleRate?: number;
  /**
   * Override the path/URL where `@tensorflow/tfjs-tflite` looks for its
   * WASM files. Defaults to the package CDN.
   */
  wasmPath?: string;
  /** Optional MediaStream to use instead of `getUserMedia`. */
  mediaStream?: MediaStream;
}

export type UWWStatus = 'idle' | 'loading' | 'listening' | 'error';

export interface WakeEventDetail {
  /** Mean probability across the sliding window at the moment of firing. */
  probability: number;
  /** High-resolution timestamp (ms since navigation start). */
  timestamp: number;
}

export interface ProbabilityEventDetail {
  /** Latest raw probability emitted by the wake-word model. */
  probability: number;
}

export interface ErrorEventDetail {
  error: Error;
}

export type UWWEventMap = {
  wake: CustomEvent<WakeEventDetail>;
  probability: CustomEvent<ProbabilityEventDetail>;
  statuschange: CustomEvent<{ status: UWWStatus }>;
  error: CustomEvent<ErrorEventDetail>;
};
