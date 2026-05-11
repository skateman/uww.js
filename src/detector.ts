export interface DetectorOptions {
  threshold: number;
  windowSize: number;
  refractoryMs: number;
  /** Time source. Defaults to `performance.now`. Override in tests. */
  now?: () => number;
}

/**
 * Tracks a sliding window of wake-word probabilities and decides when to
 * fire. Implements a refractory period to suppress repeated triggers.
 */
export class Detector {
  private readonly window: number[] = [];
  private lastFireTime = 0;
  private readonly now: () => number;

  constructor(private readonly opts: DetectorOptions) {
    this.now = opts.now ?? (() => performance.now());
  }

  /**
   * Push a new probability. Returns the sliding mean, plus whether a wake
   * event should fire now.
   */
  push(probability: number): { mean: number; fired: boolean } {
    this.window.push(probability);
    if (this.window.length > this.opts.windowSize) {
      this.window.shift();
    }
    if (this.window.length < this.opts.windowSize) {
      return { mean: probability, fired: false };
    }

    let sum = 0;
    for (const p of this.window) sum += p;
    const mean = sum / this.window.length;

    if (mean < this.opts.threshold) return { mean, fired: false };

    const now = this.now();
    if (now - this.lastFireTime < this.opts.refractoryMs) {
      return { mean, fired: false };
    }
    this.lastFireTime = now;
    this.window.length = 0;
    return { mean, fired: true };
  }

  reset(): void {
    this.window.length = 0;
    this.lastFireTime = 0;
  }
}
