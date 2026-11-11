// A fake engine for the workstation tests: answers the routes the UI calls, records every request.
import { vi } from "vitest";

export interface Call {
  method: string;
  path: string;
  body: unknown;
}

type Handler = (body: unknown, url: URL) => unknown | Promise<unknown>;

export const TERMINAL = {
  id: "t1", path: "C:\\mt5\\2\\terminal64.exe", install_dir: "C:\\mt5\\2", data_dir: "C:\\data", portable: false,
  running: true, remembered_server: "Broker-Demo",
  account: { state: "OK", login: 1, server: "Broker-Demo", company: "Broker Ltd", currency: "USD", balance: 1000,
    equity: 1000, leverage: 100, mode: "DEMO", connected: true, algo_trading: true, max_bars: 100000 },
};

export const SYMBOLS = {
  terminal: "t1",
  symbols: ["EURUSDm", "GBPUSDm", "XAUUSDm", "USDJPYm", "BTCUSDm"].map((n) => ({ name: n, description: n, path: `Forex\\${n}`, visible: true })),
  main_pairs: [
    { pair: "EURUSD", symbol: "EURUSDm" }, { pair: "GBPUSD", symbol: "GBPUSDm" }, { pair: "USDJPY", symbol: "USDJPYm" },
    { pair: "XAUUSD", symbol: "XAUUSDm" }, { pair: "US30", symbol: null },
  ],
};

export const LAUNCH_IDLE = {
  launch: null, busy: false, paused: false,
  depths: { QUICK: { pop: 50, gen: 5, budget: 400, final_n: 20 }, STANDARD: { pop: 200, gen: 25, budget: 8000, final_n: 20 },
    DEEP: { pop: 200, gen: 50, budget: 16000, final_n: 20 } },
  timeframes: ["M5", "M15", "M30", "H1", "H4", "D1"], home: "C:\\Users\\me\\ARQVIO", logical_cores: 8,
  power: { eco: 1, medium: 4, high: 7, extreme: 8 },
};

export const ROW = {
  id: "abcdef1234567890@eurusdm_h1", family: "RSI_REVERSION", cell: "eurusdm_h1", symbol: "EURUSDm", timeframe: "H1",
  tier: "CANDIDATE", score: 45.4, score_max: 80, grade: "D", sharpe: 0.98, n_trades: 34, max_dd: 0.014,
  total_return: 0.04, dsr_final: 0.287, sham_margin: 1.48, cross_sharpe: 0.02, alive_on_both: true,
  campaign_id: "ga-x", added_utc: "2026-10-23T05:32:54+00:00",
};

export const DETAIL = {
  ...ROW, dna: { signal: "RSI_REVERSION", rsi_period: 7, rsi_oversold: 16.6, rsi_overbought: 80, risk_pct: 0.015,
    sl_atr: 1.61, tp_atr: 3.18 }, dna_hash: "abcdef1234567890", cross_cell: "gbpusdm_h1",
  metrics: { total_return: 0.04, sharpe: 0.98, n_trades: 34, max_dd: 0.014, dsr_final: 0.287, cross_sharpe: 0.02, sham_margin: 1.48 },
  gates: [{ gate: "Sharpe", value: 0.98, need: "> 0", passed: true }, { gate: "trades", value: 34, need: ">= 100", passed: false }],
  score_parts: {}, source: { campaign_id: "ga-x" }, robustness_state: "NONE",
};

export const INSPECT = {
  id: ROW.id, bars: 18669, first_bar: "2023-10-23T00:00:00+00:00", last_bar: "2026-10-23T00:00:00+00:00",
  engine: "fast kernel (trade log)", matches_vault: true,
  stats: { trades: 34, win_rate: 0.5294, profit_factor: 1.8157, avg_win: 49.62, avg_loss: -30.75, avg_trade: 11.8,
    largest_win: 94.2, largest_loss: -47.9, max_win_streak: 5, max_loss_streak: 3, long_trades: 8, short_trades: 26,
    avg_hold_hours: 8.29, total_return: 0.0401, max_drawdown: 0.0142, net_profit: 401.28, initial_equity: 10000,
    curve: [[1698019200, 10000], [1761177600, 10401.28]] },
};

export function fakeEngine(over: Record<string, Handler> = {}) {
  const calls: Call[] = [];
  const routes: Record<string, Handler> = {
    "GET /api/v1/ping": () => ({ pong: true }),
    "GET /api/v2/app/terminals": () => ({ terminals: [TERMINAL], chosen: "t1", read_utc: "2026-10-23T06:00:00Z" }),
    "GET /api/v2/app/terminals/t1/symbols": () => SYMBOLS,
    "POST /api/v2/app/terminals/choose": () => ({ chosen: "t1" }),
    "GET /api/v2/app/launch": () => LAUNCH_IDLE,
    "POST /api/v2/app/launch": () => ({ launch: { id: "x" } }),
    "GET /api/v2/app/log": () => ({ lines: [], last: 0 }),
    "GET /api/v2/app/library": () => ({ strategies: [], counts: {}, robustness_running: null }),
    "GET /api/v2/app/deploy": () => ({ bots: [], stop_all: false, incubation_stages: [0.25, 0.5, 1], guardrail_defaults: { lots: 0.01, daily_loss_pct: 3, max_account_positions: 5, friday_close_hour: 20, news_block_minutes: 30, profit_target_pct: 0 } }),
    "GET /api/v1/gene-pool": () => ({ families: { MA_CROSS: { enabled: true, weight: 0.2 }, RSI_REVERSION: { enabled: true, weight: 0.2 } }, genes: {}, hash: "h1" }),
    ...over,
  };
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input));
    const method = (init?.method ?? "GET").toUpperCase();
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ method, path: url.pathname, body });
    const h = routes[`${method} ${url.pathname}`];
    if (!h) return new Response(JSON.stringify({ detail: `no route ${method} ${url.pathname}` }), { status: 404 });
    try {
      return new Response(JSON.stringify(await h(body, url)), { status: 200 });
    } catch (e) {
      return new Response(JSON.stringify({ detail: String((e as Error).message) }), { status: 409 });
    }
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, fetchMock, routes };
}
