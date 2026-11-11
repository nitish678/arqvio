// ARQVIO shared config — single source of truth for engine connectivity.
//
// The engine runs on 8741 (engine/config/settings.py: api_port default 8741). The GUI spec once said 8777; that was
// a stale note. Every client (fetch, sidecar spawn) reads from this file; loopback only (L6).

const ENGINE_HOST = "127.0.0.1";
// A dev server may point at a second engine (VITE_ARQVIO_ENGINE_PORT); the packaged app always uses 8741.
const ENGINE_HTTP_PORT = Number(import.meta.env.VITE_ARQVIO_ENGINE_PORT ?? 8741);
export const ENGINE_HTTP_ORIGIN = `http://${ENGINE_HOST}:${ENGINE_HTTP_PORT}`;

// A read waits up to 10 s: the environment route runs the MT5 probe (5 s ceiling) and the installed runtime answers
// its first requests while it is still warming up. Longer actions pass their own timeout.
export const ENGINE_TIMEOUT_MS = 10_000;
