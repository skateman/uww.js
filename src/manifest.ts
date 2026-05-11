/**
 * The microWakeWord JSON manifest schema as published in
 * https://github.com/esphome/micro-wake-word-models (v2). This is the
 * same schema ESPHome's `micro_wake_word` component consumes when you
 * point it at a model URL.
 */
export interface WakeWordManifest {
  /** Always "micro" for microWakeWord models. */
  type: 'micro';
  /** Human-readable wake phrase (e.g., "Hey Jarvis"). */
  wake_word: string;
  /** Model author. */
  author?: string;
  /** Author website / source. */
  website?: string;
  /** Filename or URL of the `.tflite` model, resolved relative to this JSON. */
  model: string;
  /** ISO 639-1 language codes the model was trained against. */
  trained_languages?: string[];
  /** Manifest schema version (currently 2). */
  version: number;
  /** Detection parameters. */
  micro: {
    /** Probability threshold the wake word must exceed (the sliding-window mean). */
    probability_cutoff: number;
    /** Number of recent frames averaged before checking the threshold. */
    sliding_window_size: number;
    /** Audio frontend hop size in milliseconds (always 10 for v2). */
    feature_step_size: number;
    /** Tensor arena size — ESP32-only, ignored in the browser. */
    tensor_arena_size?: number;
    /** Minimum ESPHome version — informational. */
    minimum_esphome_version?: string;
  };
}

export interface ResolvedManifest {
  manifest: WakeWordManifest;
  /** Absolute URL of the .tflite model. */
  modelUrl: string;
}

/**
 * Fetch and parse a microWakeWord manifest from a URL, returning the
 * parsed manifest plus the absolute URL of its `.tflite` model.
 */
export async function fetchManifest(manifestUrl: string): Promise<ResolvedManifest> {
  const res = await fetch(manifestUrl);
  if (!res.ok) {
    throw new Error(
      `uww: failed to fetch manifest at ${manifestUrl} (HTTP ${res.status})`
    );
  }
  const json = (await res.json()) as unknown;
  const manifest = validateManifest(json);
  const modelUrl = new URL(manifest.model, manifestUrl).href;
  return { manifest, modelUrl };
}

/**
 * Validate that an object matches the microWakeWord manifest schema.
 * Throws with a descriptive message on mismatch.
 */
export function validateManifest(value: unknown): WakeWordManifest {
  if (!value || typeof value !== 'object') {
    throw new Error('uww: manifest is not an object');
  }
  const m = value as Record<string, unknown>;
  if (m['type'] !== 'micro') {
    throw new Error(`uww: manifest type must be "micro", got ${JSON.stringify(m['type'])}`);
  }
  if (typeof m['wake_word'] !== 'string') {
    throw new Error('uww: manifest.wake_word must be a string');
  }
  if (typeof m['model'] !== 'string') {
    throw new Error('uww: manifest.model must be a string (filename or URL)');
  }
  const micro = m['micro'];
  if (!micro || typeof micro !== 'object') {
    throw new Error('uww: manifest.micro must be an object');
  }
  const mm = micro as Record<string, unknown>;
  if (typeof mm['probability_cutoff'] !== 'number') {
    throw new Error('uww: manifest.micro.probability_cutoff must be a number');
  }
  if (typeof mm['sliding_window_size'] !== 'number') {
    throw new Error('uww: manifest.micro.sliding_window_size must be a number');
  }
  if (typeof mm['feature_step_size'] !== 'number') {
    throw new Error('uww: manifest.micro.feature_step_size must be a number');
  }
  return value as WakeWordManifest;
}
