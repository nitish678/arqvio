// jsdom gaps the workstation relies on. Environment shims only -- none of them produces a value the UI renders.
import { configure } from "@testing-library/react";

configure({ asyncUtilTimeout: 5000 });

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
if (!("ResizeObserver" in globalThis)) {
  (globalThis as unknown as Record<string, unknown>).ResizeObserver = ResizeObserverStub;
}
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = function scrollIntoView() {};
}
// jsdom has no canvas: getContext answers null and the drawings simply skip (their numbers are in the DOM too).
HTMLCanvasElement.prototype.getContext = (() => null) as unknown as HTMLCanvasElement["getContext"];

// No test reaches a real engine: a test that does not stub fetch sees an engine that is offline.
globalThis.fetch = (() => Promise.reject(new TypeError("network disabled in tests"))) as typeof fetch;
