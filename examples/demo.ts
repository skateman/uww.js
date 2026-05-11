import { UWW, validateManifest, type WakeWordSource } from 'uww';

const $ = <T extends HTMLElement = HTMLElement>(id: string) =>
  document.getElementById(id) as T;

const srcUrlRadio = $<HTMLInputElement>('src-url');
const srcFilesRadio = $<HTMLInputElement>('src-files');
const manifestUrlInput = $<HTMLInputElement>('manifest-url');
const filesInput = $<HTMLInputElement>('files');
const thresholdInput = $<HTMLInputElement>('threshold');
const windowInput = $<HTMLInputElement>('window');
const refractoryInput = $<HTMLInputElement>('refractory');
const startBtn = $<HTMLButtonElement>('start');
const stopBtn = $<HTMLButtonElement>('stop');
const statusEl = $('status');
const wwNameEl = $('ww-name');
const meter = $('meter');
const bar = meter.querySelector('.bar') as HTMLElement;
const thresholdMarker = meter.querySelector('.threshold') as HTMLElement;
const probValue = $('probValue');
const wakeCountEl = $('wakeCount');
const log = $('log');

let uww: UWW | null = null;
let wakeCount = 0;

function appendLog(line: string) {
  const time = new Date().toLocaleTimeString();
  log.textContent = `[${time}] ${line}\n` + log.textContent;
}

function updateInputDisable() {
  manifestUrlInput.disabled = !srcUrlRadio.checked;
  filesInput.disabled = !srcFilesRadio.checked;
}
srcUrlRadio.addEventListener('change', updateInputDisable);
srcFilesRadio.addEventListener('change', updateInputDisable);
manifestUrlInput.addEventListener('focus', () => {
  srcUrlRadio.checked = true;
  updateInputDisable();
});
filesInput.addEventListener('focus', () => {
  srcFilesRadio.checked = true;
  updateInputDisable();
});

thresholdInput.addEventListener('input', () => {
  const t = parseFloat(thresholdInput.value);
  if (Number.isFinite(t)) {
    thresholdMarker.style.left = `${Math.max(0, Math.min(1, t)) * 100}%`;
  }
});

async function buildSource(): Promise<WakeWordSource> {
  if (srcUrlRadio.checked) {
    const url = manifestUrlInput.value.trim();
    if (!url) throw new Error('Enter a manifest URL');
    return { manifestUrl: url };
  }

  // Local files: zero, one, or two of them.
  const files = Array.from(filesInput.files ?? []);
  if (files.length === 0) throw new Error('Pick at least a .tflite file');
  const json = files.find((f) => f.name.toLowerCase().endsWith('.json'));
  const tflite = files.find((f) => f.name.toLowerCase().endsWith('.tflite'));
  if (!tflite) throw new Error('No .tflite file in the selection');

  if (json) {
    const manifest = validateManifest(JSON.parse(await json.text()));
    return { manifest, modelData: await tflite.arrayBuffer() };
  }
  return { wakeWordModel: await tflite.arrayBuffer() };
}

function readNumber(input: HTMLInputElement): number | undefined {
  const v = parseFloat(input.value);
  return Number.isFinite(v) ? v : undefined;
}

startBtn.addEventListener('click', async () => {
  startBtn.disabled = true;
  try {
    const wakeWord = await buildSource();
    uww = new UWW({
      wakeWord,
      threshold: readNumber(thresholdInput),
      slidingWindowSize: readNumber(windowInput),
      refractoryMs: readNumber(refractoryInput) ?? 2000,
    });
    uww.addEventListener('statuschange', (e) => {
      statusEl.textContent = (e as CustomEvent).detail.status;
    });
    uww.addEventListener('probability', (e) => {
      const p = (e as CustomEvent).detail.probability as number;
      bar.style.width = `${Math.max(0, Math.min(1, p)) * 100}%`;
      probValue.textContent = p.toFixed(3);
    });
    uww.addEventListener('wake', (e) => {
      const detail = (e as CustomEvent).detail as {
        probability: number;
        timestamp: number;
      };
      wakeCount += 1;
      wakeCountEl.textContent = String(wakeCount);
      meter.classList.remove('pulse');
      // restart animation
      void meter.offsetWidth;
      meter.classList.add('pulse');
      appendLog(`wake! mean=${detail.probability.toFixed(3)}`);
    });
    uww.addEventListener('error', (e) => {
      const err = (e as CustomEvent).detail.error as Error;
      appendLog(`error: ${err.message}`);
    });
    appendLog('loading…');
    await uww.load();
    if (uww.wakeWordName) {
      wwNameEl.textContent = `· "${uww.wakeWordName}"`;
      appendLog(`loaded "${uww.wakeWordName}"`);
    } else {
      appendLog('loaded (no manifest)');
    }
    appendLog('starting mic…');
    await uww.start();
    stopBtn.disabled = false;
  } catch (err) {
    appendLog(`failed: ${(err as Error).message}`);
    startBtn.disabled = false;
  }
});

stopBtn.addEventListener('click', async () => {
  stopBtn.disabled = true;
  if (uww) {
    await uww.dispose();
    uww = null;
  }
  wwNameEl.textContent = '';
  startBtn.disabled = false;
  appendLog('stopped');
});

// Diagnostic poller — updates the live status panel with real numbers
// flowing through the pipeline so we can see WHERE things go wrong.
setInterval(() => {
  if (!uww || uww.status !== 'listening') return;
  const d = uww.getDebug();
  const r = d.lastFeatureRow;
  const featStats = r
    ? `mean=${r.mean.toFixed(2)} max=${r.max.toFixed(2)} min=${r.min.toFixed(2)} nz=${r.nonZero}/40`
    : 'none yet';
  const q = d.featureQuant;
  const dbg = `rate=${d.sampleRate}Hz · audio=${d.audioFrames} feats=${d.featuresProduced} ` +
    `infer=${d.inferences} · shape=[${d.featureShape.join(',')}] ` +
    `quant=${q.dtype}(scale=${q.scale.toExponential(2)},zero=${q.zero}) · last=${featStats}`;
  $('debug').textContent = dbg;
}, 500);

updateInputDisable();
