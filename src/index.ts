export { UWW } from './uww.js';
export { Detector } from './detector.js';
export { AudioCapture } from './audio-capture.js';
export { InferencePipeline } from './inference.js';
export { MicroFrontend } from './wrapper.js';
export {
  fetchManifest,
  validateManifest,
  type WakeWordManifest,
  type ResolvedManifest,
} from './manifest.js';
export type {
  UWWOptions,
  UWWStatus,
  UWWEventMap,
  WakeWordSource,
  WakeEventDetail,
  ProbabilityEventDetail,
  ErrorEventDetail,
} from './types.js';
