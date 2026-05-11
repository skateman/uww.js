import { WORKLET_SOURCE } from './worklet-source.js';

export interface AudioCaptureOptions {
  sampleRate: number;
  frameSize: number;
  onFrame: (frame: Float32Array) => void;
  /** Pre-existing MediaStream; if omitted, getUserMedia is called. */
  mediaStream?: MediaStream;
}

/**
 * Wraps mic acquisition and the AudioWorklet that produces fixed-size
 * Float32 frames at the requested sample rate.
 */
export class AudioCapture {
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private ownsStream = false;

  constructor(private readonly opts: AudioCaptureOptions) {}

  get actualSampleRate(): number | null {
    return this.context?.sampleRate ?? null;
  }

  async start(): Promise<void> {
    if (this.context) return;

    if (this.opts.mediaStream) {
      this.stream = this.opts.mediaStream;
      this.ownsStream = false;
    } else {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('uww: getUserMedia is not available in this context');
      }
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: this.opts.sampleRate,
          // Disable browser noise suppression and echo cancellation —
          // wake-word models are trained on raw mic audio with the
          // PCAN gain control inside the audio frontend doing the
          // dynamic-range adjustment. AGC is left ON because typical
          // browser microphones produce too quiet a signal otherwise.
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: true,
        },
      });
      this.ownsStream = true;
    }

    this.context = new AudioContext({ sampleRate: this.opts.sampleRate });
    if (this.context.sampleRate !== this.opts.sampleRate) {
      // Browser ignored our request (Safari often does). Wake-word accuracy
      // depends on matching the training sample rate; warn loudly.
      // eslint-disable-next-line no-console
      console.warn(
        `uww: requested sampleRate=${this.opts.sampleRate} but AudioContext is ${this.context.sampleRate}; ` +
          'detection accuracy may degrade. Consider providing a pre-resampled MediaStream.'
      );
    }

    const blob = new Blob([WORKLET_SOURCE], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    try {
      await this.context.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }

    this.node = new AudioWorkletNode(this.context, 'uww-capture', {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      processorOptions: { frameSize: this.opts.frameSize },
    });
    this.node.port.onmessage = (ev) => {
      this.opts.onFrame(ev.data as Float32Array);
    };

    this.source = this.context.createMediaStreamSource(this.stream);
    this.source.connect(this.node);

    if (this.context.state === 'suspended') {
      await this.context.resume();
    }
  }

  async stop(): Promise<void> {
    try {
      this.node?.port.close();
    } catch {
      /* ignore */
    }
    this.node?.disconnect();
    this.source?.disconnect();
    if (this.ownsStream) {
      this.stream?.getTracks().forEach((t) => t.stop());
    }
    if (this.context && this.context.state !== 'closed') {
      await this.context.close();
    }
    this.context = null;
    this.stream = null;
    this.node = null;
    this.source = null;
  }
}
