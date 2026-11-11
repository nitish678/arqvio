// Source fences: the app talks to its own engine on loopback only, and the old name is gone from the UI.
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const ROOT = join(__dirname, "..");

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    if (statSync(p).isDirectory()) return n === "__tests__" ? [] : files(p);
    return /\.(tsx?|css)$/.test(n) ? [p] : [];
  });
}

describe("fences", () => {
  const src = files(ROOT).map((p) => ({ p, text: readFileSync(p, "utf8") }));

  it("reaches no host but the local engine", () => {
    for (const { p, text } of src) {
      const hosts = [...text.matchAll(/https?:\/\/([^/"'`\s:$]+)/g)].map((m) => m[1]);
      expect(hosts.filter((h) => h !== "127.0.0.1"), p).toEqual([]);
    }
  });

  it("never says ASTRA", () => {
    for (const { p, text } of src) expect(/astra/i.test(text.replace(/fastrand/gi, "")), p).toBe(false);
  });

  it("has no order path: the UI never calls a trading route", () => {
    for (const { p, text } of src) expect(/order_send|\/orders?\b|\/trade\b/i.test(text), p).toBe(false);
  });
});
