import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: false,
    setupFiles: ["./src/__tests__/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    // a whole-App walk over eleven screens outlives the 5 s default on a loaded machine (see setup.ts)
    testTimeout: 20000,
  },
});
