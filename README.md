# uww.js

In-browser **microWakeWord** detection. Loads your `.tflite` wake-word
model, captures the microphone with an `AudioWorklet`, computes mel-PCAN
features with a tiny **WASM build of the TFLite-Micro audio frontend**
(the exact same DSP ESPHome's `micro_wake_word` runs on the ESP32), runs
streaming inference via `@tensorflow/tfjs-tflite`, and emits `wake`
events.

> Status: experimental. Built and tested against the v2 microWakeWord
> models from
> [`esphome/micro-wake-word-models`](https://github.com/esphome/micro-wake-word-models)
> (Hey Jarvis, Alexa, Hey Mycroft, Okay Nabu, …) — the same models
> ESPHome's `micro_wake_word` component runs on the ESP32.

## Install

```bash
npm install uww.js \
  @tensorflow/tfjs-core \
  @tensorflow/tfjs-backend-cpu \
  @tensorflow/tfjs-tflite
```

The three `@tensorflow/*` packages are peer dependencies so you
control their versions. `tfjs-core` must be exactly `4.9.0` (matching
the `tfjs-tflite` peer requirement). The audio frontend is bundled
inside `uww.js` itself (~40 KB of WASM, embedded as base64) — no separate
file or CDN.

## Quick start

```ts
import { UWW } from 'uww.js';

const uww = new UWW({
  // Same manifest format ESPHome's micro_wake_word reads.
  // The .tflite path is resolved relative to the manifest URL.
  wakeWord: {
    manifestUrl:
      'https://cdn.jsdelivr.net/gh/esphome/micro-wake-word-models@main/models/v2/hey_jarvis.json',
  },
  // threshold + slidingWindowSize default to the manifest's
  // probability_cutoff and sliding_window_size — override only if needed.
  refractoryMs: 2000,
});

uww.addEventListener('wake', (e) => {
  console.log('wake!', (e as CustomEvent).detail);
  // e.detail = { probability, timestamp }
});

await uww.start();           // implicitly calls load()
console.log(uww.wakeWordName); // → "Hey Jarvis"
```

### Other ways to specify a wake word

```ts
// Already have a parsed manifest + the model bytes? Pass them directly.
new UWW({ wakeWord: { manifest, modelData: arrayBuffer } });

// Just a raw .tflite (no manifest)? Defaults are used for threshold/window.
new UWW({ wakeWord: { wakeWordModel: '/models/my_wake_word.tflite' } });
new UWW({ wakeWord: { wakeWordModel: arrayBuffer } });
```

## Why no preprocessor file?

Standard microWakeWord training and ESPHome's `micro_wake_word` both run
the [TFLite-Micro audio frontend](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/experimental/microfrontend)
(window → real FFT → 40-bin mel filterbank → noise reduction → PCAN
gain → log compression). The exported `.tflite` preprocessor uses
TFLite-Micro's "signal" custom ops, which `tfjs-tflite` does **not**
ship — so even though you can fetch the model, it can't execute in the
browser via that runtime.

uww.js solves this by compiling the same C audio frontend (vendored from
[`rhasspy/pymicro-features`](https://github.com/rhasspy/pymicro-features))
to WASM via Emscripten. Features are bit-identical to what your model
was trained against. See `wasm/` for the C wrapper and the build script.

## Options

| Option              | Type                       | Default | Notes                                                                 |
| ------------------- | -------------------------- | ------- | --------------------------------------------------------------------- |
| `wakeWord`          | see [Quick start](#quick-start) | —  | Required. One of `{ manifestUrl }`, `{ manifest, modelData }`, or `{ wakeWordModel }`. |
| `threshold`         | `number`                   | manifest's `probability_cutoff`, else `0.7` | Mean probability over the sliding window required to fire. |
| `slidingWindowSize` | `number`                   | manifest's `sliding_window_size`, else `5`  | Frames averaged before threshold check. |
| `refractoryMs`      | `number`                   | `2000`  | Suppress further detections for this many ms after firing.            |
| `sampleRate`        | `number`                   | `16000` | microWakeWord trains on 16 kHz mono.                                  |
| `wasmPath`          | `string`                   | jsDelivr CDN of `tfjs-tflite/wasm/` | Where `tfjs-tflite` looks for its `.wasm` files. |
| `mediaStream`       | `MediaStream`              | —       | Skip `getUserMedia` and use a stream you already have.                |

## Events

| Event          | `detail`                                   |
| -------------- | ------------------------------------------ |
| `wake`         | `{ probability: number, timestamp: number }` |
| `probability`  | `{ probability: number }` (one per frame)  |
| `statuschange` | `{ status: 'idle' \| 'loading' \| 'listening' \| 'error' }` |
| `error`        | `{ error: Error }`                         |

## Triggering Home Assistant Assist

After a `wake` event, open a WebSocket to HA and run the Assist pipeline
starting at `stt`:

```ts
ws.send(JSON.stringify({
  id: ++msgId,
  type: 'assist_pipeline/run',
  start_stage: 'stt',
  end_stage: 'tts',
  input: { sample_rate: 16000 },
}));
// then stream raw 16-bit PCM chunks prefixed with the handler id from the
// stt-start event
```

## Bundler notes

- **`tfjs-tflite` ESM is broken.** Its `module` entry points at
  `dist/index.js` which imports a file that doesn't exist in the
  package. Alias the FESM bundle in your bundler:
  ```ts
  // vite.config.ts
  resolve: {
    alias: {
      '@tensorflow/tfjs-tflite':
        'node_modules/@tensorflow/tfjs-tflite/dist/tf-tflite.fesm.js',
    },
  }
  ```
  See `examples/vite.config.ts` for a worked example.

- **Import-time WASM probe.** `tfjs-tflite` runs an unconditional
  `loader.load(true)` at module-import time using a hard-coded empty
  `wasmPath`. This 404s once on the page origin before any of your code
  runs. The error is harmless — the loader cache is keyed on path, so
  the call inside `UWW.load()` constructs a fresh, working loader using
  your `wasmPath`. The demo includes a Vite middleware that 302-redirects
  these probes to the CDN to keep the dev console clean; for production
  you can add a similar redirect or simply ignore the noise.

## Known limits

- **Sample rate.** Some browsers (notably Safari) ignore the requested
  `sampleRate` on `AudioContext` and `getUserMedia`. The library warns
  but does not resample. Provide a pre-resampled `mediaStream` if needed.
- **Background tabs.** `AudioWorklet` is throttled when the tab is
  hidden. For wall-panel use, combine with a Wake Lock and keep the tab
  visible.
- **Browser audio processing.** `getUserMedia` is requested with
  `noiseSuppression: false, echoCancellation: false, autoGainControl: true`.
  The wake-word models were trained against the `MicroFrontend`'s
  built-in PCAN gain control + their own noise reduction; browser
  noise suppression silently shifts the spectrum and kills detection.

## Demo

```bash
npm install
npm run demo
```

Open the page, drop in your wake-word `.tflite`, click **Start
listening**.

## Building

```bash
npm install
npm run build:wasm    # rebuild the WASM frontend (requires Docker)
npm run build         # tsup → dist/
npm run typecheck

# Pin to a specific upstream version instead of "latest":
PMF_VERSION=2.0.2 npm run build:wasm
```

The WASM frontend (`src/_wasm/uww-frontend.js`) is built from a small
C wrapper in `wasm/src/` plus upstream sources fetched on demand —
**no third-party C is committed to this repo**. The build script
([`scripts/build-wasm.mjs`](scripts/build-wasm.mjs)) hits the GitHub
release API to resolve the **latest**
[`rhasspy/pymicro-features`](https://github.com/rhasspy/pymicro-features)
release (which bundles TFLite-Micro's audio frontend + kissfft),
downloads its official sdist asset, extracts only the files we need
into `wasm/.cache/`, then runs Emscripten in Docker to produce the
WASM. The script is plain Node (no shell, no npm deps) so it runs the
same on macOS, Linux, and Windows + WSL.

The exact upstream version used for any given build is recorded as
`src/_wasm/upstream-version.txt` inside the WASM artifact directory
and shipped with the package.

The artifact is rebuilt:

- on every push and pull request by [`.github/workflows/ci.yml`](.github/workflows/ci.yml),
- automatically before `npm publish` via the `prepublishOnly` script,
- on demand locally with `npm run build:wasm`.

The published npm package contains the pre-built WASM (inlined into
`dist/`), so end users never need Docker or Emscripten.

The exact upstream version any given build was compiled against is
shipped as `src/_wasm/upstream-version.txt` inside the package, and
also logged by the CI release job.

## Releasing

Releases are driven by the `version` field in `package.json`.
Upstream pymicro-features tracking is **manual** — when upstream
publishes a new release that you want to ship, bump our `version` and
merge. The CI logs and `src/_wasm/upstream-version.txt` inside the
published package record exactly which upstream version was used.

1. Bump `version` (e.g. `0.1.0` → `0.1.1`) on a PR.
2. (Optional) Note in the PR / release description which upstream
   pymicro-features version is being shipped.
3. Merge to `main`.
4. CI's `release` job notices the new version isn't on npm yet, builds
   the WASM against latest `pymicro-features`, and publishes with
   provenance.

If `version` is unchanged, the `release` job runs but skips the
publish step. No git tags are required — npm's signed provenance
attestation links the published tarball to the exact commit + workflow
run that produced it.

## License

MIT for the JS/TS code (see `LICENSE`).

The WASM frontend incorporates upstream C sources fetched at build
time from the official `pymicro-features` release sdist:

- TensorFlow Lite Micro audio frontend — Apache 2.0
- Kiss FFT — BSD-3-Clause

Full attribution and license texts are in `NOTICE`. The npm package
ships the compiled WASM (no source); both license texts are reproduced
in `NOTICE`, which is included in the published package.
