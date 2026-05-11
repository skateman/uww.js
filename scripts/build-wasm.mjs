#!/usr/bin/env node
/**
 * Build the uww-frontend WASM module.
 *
 * Resolves the latest release of rhasspy/pymicro-features (which vendors
 * TFLite-Micro's audio_microfrontend lib + kissfft), downloads the
 * official sdist asset, verifies it, extracts it, and compiles it with
 * our wrapper via Emscripten in Docker.
 *
 * Two modes:
 *   PMF_VERSION="latest" (default): hits the GitHub API for the latest tag.
 *   PMF_VERSION="2.0.2":            pin to a specific release.
 *
 * Output:
 *   src/_wasm/uww-frontend.{js,d.ts}
 *   src/_wasm/upstream-version.txt
 *
 * Requires: Node 20+, Docker, system `tar`. No npm dependencies.
 */
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const outDir = resolve(root, 'src/_wasm');
const cacheDir = resolve(root, 'wasm/.cache');
const wrapperRel = 'wasm/src/wrapper.cc';

const PMF_VERSION_REQ = process.env.PMF_VERSION ?? 'latest';
const DOCKER_IMAGE = process.env.EMSCRIPTEN_IMAGE ?? 'emscripten/emsdk:latest';

mkdirSync(outDir, { recursive: true });
mkdirSync(cacheDir, { recursive: true });

// ---------------------------------------------------------------------------
// 1. Resolve upstream version (latest by default).
// ---------------------------------------------------------------------------
async function resolveVersion(req) {
  if (req !== 'latest') return req;
  console.log('==> Resolving latest pymicro-features release');
  const res = await fetch(
    'https://api.github.com/repos/rhasspy/pymicro-features/releases/latest',
    {
      headers: {
        Accept: 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
      },
    }
  );
  if (!res.ok) {
    throw new Error(`GitHub API returned ${res.status}`);
  }
  const json = await res.json();
  const tag = json.tag_name;
  if (typeof tag !== 'string') {
    throw new Error('No tag_name in release JSON');
  }
  return tag.replace(/^v/, '');
}

const version = await resolveVersion(PMF_VERSION_REQ);
console.log(`==> Using pymicro-features v${version}`);

const asset = `pymicro_features-${version}.tar.gz`;
const url = `https://github.com/rhasspy/pymicro-features/releases/download/v${version}/${asset}`;
const tarball = resolve(cacheDir, asset);
const srcDir = resolve(cacheDir, `pymicro_features-${version}`);

// ---------------------------------------------------------------------------
// 2. Download (idempotent).
// ---------------------------------------------------------------------------
if (!existsSync(tarball)) {
  console.log(`==> Downloading ${url}`);
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) {
    throw new Error(`Download failed: ${res.status} ${res.statusText}`);
  }
  const buf = Buffer.from(await res.arrayBuffer());
  writeFileSync(tarball, buf);
}

const sha = createHash('sha256').update(readFileSync(tarball)).digest('hex');
console.log(`==> SHA-256: ${sha}`);

// ---------------------------------------------------------------------------
// 3. Extract.
// ---------------------------------------------------------------------------
if (!existsSync(srcDir)) {
  console.log('==> Extracting');
  mkdirSync(srcDir, { recursive: true });
  const r = spawnSync(
    'tar',
    ['-xzf', tarball, '-C', srcDir, '--strip-components=1'],
    { stdio: 'inherit' }
  );
  if (r.status !== 0) {
    throw new Error(`tar extract failed (status ${r.status})`);
  }
}

const relSrc = `wasm/.cache/pymicro_features-${version}`;
const relFrontendLib = `${relSrc}/tensorflow/lite/experimental/microfrontend/lib`;
const relKissfft = `${relSrc}/kissfft`;

for (const p of [relFrontendLib, relKissfft]) {
  if (!existsSync(resolve(root, p))) {
    throw new Error(`Missing expected path after extract: ${p}`);
  }
}

// ---------------------------------------------------------------------------
// 4. Build with Emscripten in Docker.
// ---------------------------------------------------------------------------
const frontendSources = [
  'fft.cc', 'fft_util.cc', 'filterbank.cc', 'filterbank_util.cc',
  'frontend.cc', 'frontend_util.cc', 'kiss_fft_int16.cc', 'log_lut.cc',
  'log_scale.cc', 'log_scale_util.cc', 'noise_reduction.cc',
  'noise_reduction_util.cc', 'pcan_gain_control.cc',
  'pcan_gain_control_util.cc', 'window.cc', 'window_util.cc',
].map((f) => `${relFrontendLib}/${f}`);

const sources = [
  ...frontendSources,
  `${relKissfft}/kiss_fft.cc`,
  `${relKissfft}/tools/kiss_fftr.cc`,
  wrapperRel,
];

const exportedFunctions = [
  '_uww_frontend_create',
  '_uww_frontend_destroy',
  '_uww_frontend_reset',
  '_uww_frontend_process',
  '_uww_frontend_feature_size',
  '_uww_frontend_step_ms',
  '_uww_frontend_window_ms',
  '_malloc', '_free',
];
const exportedRuntimeMethods = ['HEAP16', 'HEAPF32', 'HEAPU8'];

// --no-entry: no main(); pure library.
// SINGLE_FILE: embed the .wasm as base64 inside the JS so the library
//   ships as a single file with no runtime fetches and no path config.
// MODULARIZE + EXPORT_ES6: factory pattern, ES module form.
// ENVIRONMENT=web,worker: skip Node-only branches.
const dockerArgs = [
  'run', '--rm', '-v', `${root}:/src`, '-w', '/src', DOCKER_IMAGE,
  'emcc',
    '-O3',
    '-DFIXED_POINT=16',
    '-I', relSrc,
    '-I', relKissfft,
    ...sources,
    '--no-entry',
    '-sMODULARIZE=1',
    '-sEXPORT_ES6=1',
    '-sENVIRONMENT=web,worker',
    '-sSINGLE_FILE=1',
    '-sINITIAL_MEMORY=512KB',
    '-sSTACK_SIZE=64KB',
    '-sALLOW_MEMORY_GROWTH=0',
    '-sFILESYSTEM=0',
    '-sASSERTIONS=0',
    `-sEXPORTED_FUNCTIONS=${JSON.stringify(exportedFunctions)}`,
    `-sEXPORTED_RUNTIME_METHODS=${JSON.stringify(exportedRuntimeMethods)}`,
    '-o', 'src/_wasm/uww-frontend.js',
];

console.log('==> docker run emcc ...');
const r = spawnSync('docker', dockerArgs, { stdio: 'inherit' });
if (r.status !== 0) {
  throw new Error(`emcc build failed (status ${r.status})`);
}

// Hand-written .d.ts — the Emscripten output is plain JS.
writeFileSync(
  resolve(outDir, 'uww-frontend.d.ts'),
  `// Generated wrapper around the Emscripten module. The actual JS is
// produced by scripts/build-wasm.mjs and contains the embedded WASM
// as base64.

export interface UwwFrontendModule {
  HEAP16: Int16Array;
  HEAPF32: Float32Array;
  HEAPU8: Uint8Array;
  _malloc(size: number): number;
  _free(ptr: number): void;
  _uww_frontend_create(sampleRate: number): number;
  _uww_frontend_destroy(handle: number): void;
  _uww_frontend_reset(handle: number): void;
  _uww_frontend_process(
    handle: number,
    samplesPtr: number,
    numSamples: number,
    outFeaturesPtr: number
  ): number;
  _uww_frontend_feature_size(): number;
  _uww_frontend_step_ms(): number;
  _uww_frontend_window_ms(): number;
}

export type UwwFrontendFactory = (options?: object) => Promise<UwwFrontendModule>;

declare const factory: UwwFrontendFactory;
export default factory;
`
);

writeFileSync(resolve(outDir, 'upstream-version.txt'), version);

console.log(`\nBuilt from rhasspy/pymicro-features v${version}.`);
