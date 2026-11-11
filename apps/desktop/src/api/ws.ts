// The workstation's engine client: loopback only (127.0.0.1:8741). Every value on screen is one of these answers;
// a failed call is an ApiError carrying the engine's own words, never a made-up value.

import { ENGINE_HTTP_ORIGIN, ENGINE_TIMEOUT_MS } from "../config/engine";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(detail);
  }
}

function detailOf(body: unknown, status: number): string {
  if (body && typeof body === "object") {
    const b = body as Record<string, unknown>;
    if (typeof b.detail === "string") return b.detail.replace(/^REFUSED: /, "");
    if (typeof b.error === "string") return b.error;
    if (Array.isArray(b.problems)) return b.problems.join("; ");
  }
  return `HTTP ${status}`;
}

async function call<T>(path: string, init?: RequestInit, timeoutMs = ENGINE_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${ENGINE_HTTP_ORIGIN}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { Accept: "application/json", "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
    let body: unknown = null;
    try {
      body = await res.json();
    } catch (e) {
      if (res.ok) throw e instanceof Error ? e : new Error(String(e));
    }
    if (!res.ok) throw new ApiError(res.status, detailOf(body, res.status));
    return body as T;
  } finally {
    clearTimeout(timer);
  }
}

const get = <T>(path: string, timeoutMs?: number) => call<T>(path, undefined, timeoutMs);
const send = <T>(method: string, path: string, body?: unknown, timeoutMs?: number) =>
  call<T>(path, { method, body: body === undefined ? undefined : JSON.stringify(body) }, timeoutMs);

// ---- MetaTrader 5 terminals -------------------------------------------------------------------------------------------

export interface Account {
  state: "OK" | "CLOSED" | "UNREACHABLE" | "LOGGED_OUT";
  detail?: string;
  login?: number;
  server?: string;
  company?: string;
  name?: string;
  currency?: string;
  balance?: number;
  equity?: number;
  leverage?: number;
  mode?: string;
  connected?: boolean;
  algo_trading?: boolean;
  max_bars?: number;
}

export interface Terminal {
  id: string;
  path: string;
  install_dir: string;
  data_dir: string | null;
  portable: boolean;
  running: boolean;
  remembered_server: string | null;
  account: Account;
}

export interface TerminalsOverview {
  terminals: Terminal[];
  chosen: string | null;
  read_utc: string;
}

export interface SymbolRow {
  name: string;
  description: string;
  path: string;
  visible: boolean;
}

export interface SymbolsOverview {
  terminal: string;
  symbols: SymbolRow[];
  main_pairs: { pair: string; symbol: string | null }[];
}

// ---- launch -------------------------------------------------------------------------------------------------------------

export type Depth = "QUICK" | "STANDARD" | "DEEP";
export type Power = "eco" | "medium" | "high" | "extreme";

export interface Progress {
  generation: number;
  generations: number;
  trials: number;
  budget: number;
  best_sharpe: number | null;  // null on an engine V1 run: it ranks by its research fitness instead
  alive: number;
  utc: string;
  engine?: "v1";
  top_fitness?: number;
}

export interface LaunchTask {
  kind: "data" | "breed";
  role?: "market" | "cross";
  symbol: string;
  timeframe: string;
  cell: string;
  state: string;
  detail: string | null;
  bars?: number;
  progress?: Progress;
  started_utc?: string;
  added?: number;
  run?: string;
  engine?: "v1";
}

export interface LaunchState {
  id: string;
  utc: string;
  state: "RUNNING" | "PAUSED" | "DONE" | "STOPPED" | "FAILED" | "INTERRUPTED" | "BLOCKED_DATA" | "PARTIAL_DATA" | "COMPLETED_ZERO_SURVIVORS";
  request: { terminal: string; symbols: string[]; timeframes: string[]; years: number; depth: Depth; power: Power;
    terminal_path: string; server?: string; company?: string };
  args: { pop: number; gen: number; budget: number; final_n: number };
  tasks: LaunchTask[];
  vaulted: number;
  detail?: string;
  finished_utc?: string;
}

/** The CAMPAIGN tab's generation fan and trade tape (GET /api/v2/app/launch/live). Tape rows are backtest trades. */
export interface LiveBot { id: string; tribe: string; ret: number; dd: number; fit: number; curve: number[] }
export interface LiveTrade { bot: string; dir: number; bar: number; pct: number; reason: string }
export interface LiveView {
  task: { symbol: string; timeframe: string; state: string; engine: "v1" | "v2"; campaign_id: string } | null;
  live: { generation: number; generations: number; phase: string; survivors: number; bots: LiveBot[]; tape: LiveTrade[];
    utc: string } | null;
}

// ---- AUTOMATION (research: named automations, schedule, steps, activity) + named portfolios --------------------------

export type StepKind = "campaign" | "portfolio" | "parity" | "purge" | "replace" | "reassess";
export type AutoStep = { kind: StepKind } & Record<string, unknown>;

export interface AutoRun { ts: string; ok: boolean; paused?: boolean; detail: string }

export interface AutomationRec {
  id: string;
  name: string;
  enabled: boolean;
  every_hours: number;
  every_text: string;
  steps: AutoStep[];
  next_steps?: AutoStep[];
  terminal: string;
  history: AutoRun[];
  next_at: string | null;
  run_now: boolean;
  last_run_utc: string | null;
  alongside: boolean;
}

export interface ActivityRow { ts: string; id: string; name: string; status: "ok" | "failed" | "paused"; detail: string }

/** SETTINGS the engine applies (engine/core/app_settings.py); null in v1 = the depth preset's value. */
export interface AppSettings {
  campaign: { train_split: number; held_out_min_trades: number };
  v1: { proving_grounds: number | null; generations: number | null; tribe_size: number | null; stagnation_after: number | null;
    drawdown_limit: number | null; risk_pct: number | null };
  purge: { criteria: Record<string, number> };
  auto_purge: { enabled: boolean; every_hours: number; older_than_days: number; cap: number };
}

export interface ConfigEcho { line: string; train_split: number; held_out: number; held_out_min_trades: number;
  v1?: Record<string, number | string>; v2?: Record<string, number>; error?: string }

export interface PortfolioCard { id: string; name: string; created_utc: string; members: number; in_vault: number; live: number;
  avg_return: number | null; avg_dd: number | null; brokers: string[]; grade: string | null }

export interface NamedPortfolio { id: string; name: string; created_utc: string; members: { id: string; added_utc: string; by: string }[] }

export interface AutomationList {
  automations: AutomationRec[];
  log: ActivityRow[];
  groups: { id: string; label: string; count: number | null }[];
  timeframes: string[];
  kinds: StepKind[];
  built: StepKind[];
  planned_why: Record<string, string>;
  defaults: Record<StepKind, AutoStep>;
  launch_defaults: { years: number; depth: string; power: string; search: string };
  terminals: { id: string; path: string; server: string | null; running: boolean }[];
  portfolios: NamedPortfolio[];
}

export interface AutomationState {
  running?: { automation_id: string; name: string; step_idx: number; steps_total: number; phase: string;
    progress: { done: number; total: number } | null; resume_in?: number };
  pending?: { id: string; name: string; why: string };
  due_next?: { id: string; name: string; ts: string };
  last_result?: AutoRun;
}

export interface AutomationDraft { id?: string; name: string; enabled: boolean; every_hours: number; steps: AutoStep[]; terminal: string }

export interface LaunchOverview {
  launch: LaunchState | null;
  busy: boolean;
  paused: boolean;
  depths: Record<Depth, { pop: number; gen: number; budget: number; final_n: number }>;
  timeframes: string[];
  home: string;
  logical_cores: number;
  power: Record<Power, number>;
}

// "v1": engine V1, the research engine (UNCERTIFIED strategies); "standard": engine V2, ARQVIO's (can CERTIFY).
export type SearchEngine = "standard" | "v1";

export interface LaunchRequest {
  terminal: string;
  symbols: string[];
  timeframes: string[];
  horizon?: "research" | "intraday" | "swing" | "scalping";
  years: number;
  depth: Depth;
  power: Power;
  search?: SearchEngine;
  after?: AfterStep[];  // AUTOMATION chain steps run when the launch finishes
}

export type AfterStep = "purge_copies" | "purge_held_out_failed" | "purge_grade_f" | "repeat";

export interface Diagnostics {
  utc: string;
  version: string | null;
  os: string;
  python: string;
  cores: number | null;
  memory: { total_gb: number | null; available_gb: number | null; load_pct?: number };
  disk: { path: string; free_gb: number | null; total_gb: number | null };
  vault: Record<string, number | string>;
  recent_problems: { utc: string; src: string; msg: string }[];
}

export interface LogLine {
  seq: number;
  utc: string;
  level: "INFO" | "OK" | "WARN" | "ERROR";
  src: string;
  msg: string;
}

// ---- vault ----------------------------------------------------------------------------------------------------------------

export interface StrategyRow {
  id: string;
  family: string;
  cell: string;
  symbol: string;
  timeframe: string;
  tier: "CERTIFIED" | "CANDIDATE" | "UNCERTIFIED";  // UNCERTIFIED: bred by engine V1 (research engine)
  engine?: "v1" | "v2";
  score: number;
  score_max: number;
  grade: string;
  sharpe: number | null;
  n_trades: number;
  max_dd: number | null;
  total_return: number | null;
  dsr_final: number | null;
  sham_margin: number | null;
  cross_sharpe: number | null;
  alive_on_both: boolean;
  campaign_id: string;
  added_utc: string;
  /** the held-out test: the last 30% of the window, never seen by breeding (v4) */
  held_out?: { state: string | null; return: number | null; trades: number | null; sharpe: number | null };
  /** strategies that made the same trades (same cell, trades, return, Sharpe); 1 = unique */
  copies?: number;
  /** the best-ranked copy this row repeats; absent or null for the lead */
  copy_of?: string | null;
  // research vault columns (routes_app_library._extra)
  broker?: string | null;
  window_years?: number | null;
  first_bar?: string | null;
  cagr?: number | null;
  parity_grade?: string | null;
  win_rate?: number | null;
  robust?: string | null;
  portfolios?: string[];
  robustness?: string | null;
}

export interface VaultOverview {
  strategies: StrategyRow[];
  counts: Record<string, number>;
  robustness_running: string | null;
}

export interface Gate {
  gate: string;
  value: unknown;
  need: string;
  passed: boolean | null;
}

export interface StrategyDetail {
  id: string;
  dna: Record<string, unknown>;
  dna_hash: string;
  family: string;
  cell: string;
  cross_cell: string;
  symbol: string;
  timeframe: string;
  metrics: Record<string, unknown>;
  tier: string;
  score: number;
  score_max: number;
  grade: string;
  gates: Gate[];
  score_parts: Record<string, number>;
  source: { campaign_id: string };
  robustness_state: string;
  engine?: "v1" | "v2";
  address?: { campaign_dir: string | null; campaign_id: string | null; file: string; dataset_id: string | null;
    as_of: string | null; cost_model: string };
  certify?: { state: string; reason?: string; kind?: string; dsr_n_trials?: number } | null;
  parity?: { state: string; grade?: string; score?: number; reason?: string; utc?: string;
    levels?: { level: string; passed: boolean; summary: string }[] } | null;
}

export interface InspectStats {
  trades: number;
  win_rate: number | null;
  profit_factor: number | null;
  avg_win: number | null;
  avg_loss: number | null;
  avg_trade: number | null;
  largest_win: number | null;
  largest_loss: number | null;
  max_win_streak: number;
  max_loss_streak: number;
  long_trades: number;
  short_trades: number;
  avg_hold_hours: number | null;
  total_return: number;
  max_drawdown: number;
  net_profit: number;
  initial_equity: number;
  curve: [number, number][];
}

export interface Inspect {
  id: string;
  bars: number;
  first_bar: string;
  last_bar: string;
  engine: string;
  stats: InspectStats;
  matches_vault: boolean;
  monthly?: InspectMonth[];
  spread_gate?: { mode: "regime" | "fixed" | "off"; detail: string };
}

/** One UTC month of the inspected run: closed trades and their P&L, and bars the spread gate would refuse. */
export interface InspectMonth {
  month: string;
  trades: number;
  pnl: number;
  bars: number;
  gate_bars: number;
}

// ---- deploy -----------------------------------------------------------------------------------------------------------------

export interface Heartbeat {
  state: string;
  age_s?: number;
  standby?: string;
  status?: string; // the EA's own on-chart status (template arqvio-live, spec 11.2), with its reason
  reason?: string;
  position?: { dir: number; volume: number; profit: number } | null;
  day_pnl_pct?: number;
  lots?: number;
}

export interface Bot {
  bot: string;
  magic: number;
  symbol: string;
  timeframe: string;
  stage: number;
  status: string;
  lot_scale: number;
  guardrails: Record<string, unknown>;
  source: { vault_id: string; grade?: string };
  heartbeat: Heartbeat | null;
  live: string | null;
  terminal?: string | null;
}

export interface DeployOverview {
  bots: Bot[];
  stop_all: boolean | null;
  incubation_stages: number[];
  guardrail_defaults: Record<string, number | boolean>;
}

export interface GoRequest {
  vault_id: string;
  terminal: string;
  guardrails: Record<string, number>;
  attach: boolean;
}

// ---- gene pool ----------------------------------------------------------------------------------------------------------------

export interface GenePoolDoc {
  families: Record<string, { enabled: boolean; weight: number }>;
  genes: Record<string, unknown>;
  gates?: Record<string, boolean>;
  /** single switches turned off inside an open gate, registry names ("bias.sma") */
  off?: string[];
  hash: string;
}

/** One gene of the registry (GET /api/v1/gene-pool/registry): what the engine can breed, and why not when it cannot. */
export interface RegistryGene {
  name: string;
  group: string;
  kind: string;
  origin: string;
  status: "READY" | "FIXED" | "EA_ONLY" | "PLANNED";
  params: string[];
  family?: string;
  dna_field?: string;
  backing?: string;
  note?: string;
  research_hidden?: boolean;
}

export interface GeneCapabilities {
  schema: "arqvio.gene_capabilities";
  version: number;
  registry_hash: string;
  scope: string;
  genes: { name: string; legacy_status: string; breedable_declared: boolean;
    support: Record<"oracle" | "kernel" | "live_ea" | "tester_ea", {
      declared: boolean; evidence_status: "UNVERIFIED"; proof_runs: string[];
    }>; note: string }[];
}

export interface GeneRegistry {
  version: number;
  hash: string;
  counts: { total: number; by_status: Record<string, number>; by_group: Record<string, number>;
    engines: Record<string, { genes: number; ready: number; families: string[] }> };
  genes: RegistryGene[];
}

// ---- boot -------------------------------------------------------------------------------------------------------------

export interface BootCheck {
  group?: "engine" | "environment" | "library" | "session";
  check: string;
  ok: boolean;
  detail: string;
  ms: number;
}

export interface LivePerf {
  bots: { bot: string; magic: number; trades: number | null; net?: number; win_rate?: number | null;
    band?: { p5: number; p50: number; p95: number } | null; verdict: string; last_trade?: string | null;
    incubation?: { stage: number; lot_scale: number; next_lot_scale: number | null; ready: boolean; note: string;
      checks: { check: string; value: unknown; need: string; passed: boolean }[] } }[];
  orphans: { magic: number; deals: number; open_positions: number; symbol: string }[];
  notes: string[];
  min_trades: number;
}

export interface QueueItem {
  id: string;
  utc: string;
  request: LaunchRequest;
}

export interface TrueSim {
  state: "DONE" | "EMPTY";
  members: number;
  skipped?: { id: string; reason: string }[];
  e0?: number;
  final_equity?: number;
  return?: number;
  max_dd?: number;
  refused_trades?: number;
  conflicts?: number;
  curve?: { t: number; equity: number }[];
  note?: string;
}

export interface PortfolioView {
  source: string;
  members: { id: string; symbol?: string; timeframe?: string; family?: string; grade?: string; days?: number }[];
  overall: string | null;
  note?: string;
  corr?: (number | null)[][];
  worst_pair?: { ids: [string, string]; corr: number } | null;
  axes?: Record<string, { counts: Record<string, number>; largest: string | null; share: number | null; grade: string | null }>;
  grades?: Record<string, string | null>;
  shared_days?: number;
  combined?: [number, number][];
  tips?: string[];
  bars?: { correlation: [number, string][]; share: [number, string][]; min_shared_days: number };
}

export interface EngineSettings {
  costs: Record<string, string | number>;
  certified: Record<string, number>;
  grades: { min: number; grade: string }[];
  v1?: { presets: Record<string, Record<string, number>>; score_weights: Record<string, number>; good_at: number };
}

export const api = {
  ping: () => get<unknown>("/api/v1/ping", 3000),
  health: () => get<{ version: string; status: string }>("/healthz", 5000),
  engineSettings: () => get<EngineSettings>("/api/v2/app/engine-settings"),
  diagnostics: () => get<Diagnostics>("/api/v2/app/diagnostics", 30_000),
  supportReport: () => get<{ text: string }>("/api/v2/app/support-report", 30_000),
  boot: () => get<{ version: string; checks: BootCheck[]; ok: boolean }>("/api/v2/app/boot", 30_000),
  terminals: () => get<TerminalsOverview>("/api/v2/app/terminals", 30_000),
  chooseTerminal: (terminal: string) => send<{ chosen: string }>("POST", "/api/v2/app/terminals/choose", { terminal }),
  symbols: (terminal: string) => get<SymbolsOverview>(`/api/v2/app/terminals/${encodeURIComponent(terminal)}/symbols`,
    30_000),
  launchState: () => get<LaunchOverview>("/api/v2/app/launch"),
  launchLive: () => get<LiveView>("/api/v2/app/launch/live"),
  launch: (req: LaunchRequest) => send<{ launch: LaunchState }>("POST", "/api/v2/app/launch", req, 60_000),
  pause: () => send<unknown>("POST", "/api/v2/app/launch/pause", {}),
  resume: () => send<unknown>("POST", "/api/v2/app/launch/resume", {}),
  stop: () => send<unknown>("POST", "/api/v2/app/launch/stop", {}),
  skip: () => send<unknown>("POST", "/api/v2/app/launch/skip", {}),
  queue: () => get<{ items: QueueItem[]; busy: boolean }>("/api/v2/app/launch/queue"),
  enqueue: (req: LaunchRequest) => send<{ queued: string; items: QueueItem[] }>("POST", "/api/v2/app/launch/queue", req),
  dequeue: (id: string) => send<{ items: QueueItem[] }>("DELETE", `/api/v2/app/launch/queue/${encodeURIComponent(id)}`),
  automations: () => get<AutomationList>("/api/v2/app/automation", 30_000),
  automationState: () => get<AutomationState>("/api/v2/app/automation/state"),
  automationSave: (a: AutomationDraft) => send<{ automation: AutomationRec; note: string }>("POST", "/api/v2/app/automation/save", a),
  automationToggle: (id: string, enabled: boolean) => send<{ automation: AutomationRec }>("POST", "/api/v2/app/automation/toggle", { id, enabled }),
  automationRun: (id: string) => send<{ automation: AutomationRec; toast: string }>("POST", "/api/v2/app/automation/run", { id }),
  automationCancel: (id: string) => send<{ stopped: boolean }>("POST", "/api/v2/app/automation/cancel", { id }),
  automationDelete: (id: string) => send<{ deleted: string }>("POST", "/api/v2/app/automation/delete", { id }),
  automationPreview: (a: AutomationDraft) => send<{ lines: string[]; note: string }>("POST", "/api/v2/app/automation/preview", a, 60_000),
  vaultBulk: (action: "delete" | "reassess" | "parity", ids: string[]) =>
    send<{ action: string; done?: string[]; skipped?: string[]; queued?: string[]; refused?: Record<string, string>; waiting?: number }>("POST", "/api/v2/app/library/bulk", { action, ids }),
  vaultJobs: () => get<{ running: string | null; waiting: { kind: string; id: string }[] }>("/api/v2/app/library/jobs"),
  purgeCriteria: (criteria: Record<string, number | null>, apply: boolean) =>
    send<{ ids: string[]; count: number; applied: boolean }>("POST", "/api/v2/app/library/purge-criteria", { criteria, apply }),
  dedupe: (apply: boolean) => send<{ ids: string[]; count: number; applied: boolean }>("POST", "/api/v2/app/library/dedupe", { apply }),
  restoreAll: () => send<{ restored: string[] }>("POST", "/api/v2/app/library/recycle/restore-all", {}),
  deleteCampaign: (cid: string) => send<{ removed: string[] }>("POST", `/api/v2/app/library/campaign/${encodeURIComponent(cid)}/delete`, {}),
  passport: (id: string) => get<Record<string, unknown>>(`/api/v2/app/library/strategies/${encodeURIComponent(id)}/passport`),
  portfolios: () => get<{ portfolios: NamedPortfolio[] }>("/api/v2/app/portfolios"),
  settings: () => get<{ settings: AppSettings; defaults: AppSettings; splits: number[] }>("/api/v2/app/settings"),
  patchSettings: (patch: Record<string, unknown>) => send<{ settings: AppSettings }>("PATCH", "/api/v2/app/settings", patch),
  configEcho: (depth: string, engine: string) => get<ConfigEcho>(`/api/v2/app/settings/echo?depth=${encodeURIComponent(depth)}&engine=${encodeURIComponent(engine)}`),
  bugReport: (what: string, contact: string) => send<{ path: string; text: string }>("POST", "/api/v2/app/settings/bug-report", { what, contact }),
  clearQueue: () => send<{ removed: number }>("DELETE", "/api/v2/app/launch/queue"),
  portfolioCards: () => get<{ cards: PortfolioCard[] }>("/api/v2/app/portfolios/cards", 30_000),
  portfolioRename: (id: string, name: string) => send<{ portfolio: NamedPortfolio }>("POST", `/api/v2/app/portfolios/${encodeURIComponent(id)}/rename`, { name }),
  portfolioCreate: (name: string) => send<{ portfolio: NamedPortfolio }>("POST", "/api/v2/app/portfolios", { name }),
  portfolioAdd: (id: string, ids: string[]) => send<{ portfolio: NamedPortfolio }>("POST", `/api/v2/app/portfolios/${encodeURIComponent(id)}/add`, { ids }),
  portfolioRemove: (id: string, ids: string[]) => send<{ portfolio: NamedPortfolio }>("POST", `/api/v2/app/portfolios/${encodeURIComponent(id)}/remove`, { ids }),
  portfolioDelete: (id: string) => send<{ deleted: string }>("DELETE", `/api/v2/app/portfolios/${encodeURIComponent(id)}`),
  queueStart: () => send<{ started: string | null; detail?: string }>("POST", "/api/v2/app/launch/queue/start", {}),
  log: (after: number) => get<{ lines: LogLine[]; last: number }>(`/api/v2/app/log?after=${after}&limit=1000`),
  vault: () => get<VaultOverview>("/api/v2/app/library"),
  vaultExport: () => get<{ schema: string; count: number; exported_utc: string; strategies: unknown[] }>("/api/v2/app/library/export", 60_000),
  recycle: () => get<{ days: number; strategies: (StrategyRow & { deleted_utc: string; deleted_reason: string | null })[] }>("/api/v2/app/library/recycle"),
  restore: (id: string) => send<{ restored: string }>("POST", `/api/v2/app/library/recycle/${encodeURIComponent(id)}/restore`, {}),
  purge: (rule: string, apply: boolean) => send<{ rule: string; what: string; ids: string[]; count: number; applied: boolean }>("POST", "/api/v2/app/library/purge", { rule, apply }),
  truesim: (source: string) => get<TrueSim>(`/api/v2/app/library/portfolio/truesim?source=${source}&n=8`, 900_000),
  portfolio: (source: string) => get<PortfolioView>(`/api/v2/app/library/portfolio?source=${source}&n=8`, 180_000),
  strategy: (id: string) => get<StrategyDetail>(`/api/v2/app/library/strategies/${encodeURIComponent(id)}`),
  inspect: (id: string) => get<Inspect>(`/api/v2/app/library/strategies/${encodeURIComponent(id)}/inspect`, 120_000),
  removeStrategy: (id: string) => send<unknown>("DELETE", `/api/v2/app/library/strategies/${encodeURIComponent(id)}`),
  robustness: (id: string) =>
    send<unknown>("POST", `/api/v2/app/library/strategies/${encodeURIComponent(id)}/robustness`, {}),
  parity: (id: string) =>
    send<unknown>("POST", `/api/v2/app/library/strategies/${encodeURIComponent(id)}/parity`, {}),
  certify: (id: string) =>
    send<unknown>("POST", `/api/v2/app/library/strategies/${encodeURIComponent(id)}/certify`, {}),
  deploy: () => get<DeployOverview>("/api/v2/app/deploy"),
  livePerf: () => get<LivePerf>("/api/v2/app/deploy/live", 120_000),
  go: (req: GoRequest) => send<{ bot: string; symbol: string; terminal: string }>("POST", "/api/v2/app/deploy/go", req,
    180_000),
  setStage: (bot: string, stage: number) =>
    send<unknown>("POST", `/api/v2/app/deploy/bots/${encodeURIComponent(bot)}/stage`, { stage }),
  removeBot: (bot: string) => send<unknown>("DELETE", `/api/v2/app/deploy/bots/${encodeURIComponent(bot)}`),
  stopAll: (stop: boolean) => send<{ stop_all: boolean }>("POST", "/api/v2/app/deploy/stop", { stop }),
  genePool: () => get<GenePoolDoc>("/api/v1/gene-pool"),
  geneRegistry: () => get<GeneRegistry>("/api/v1/gene-pool/registry"),
  geneCapabilities: () => get<GeneCapabilities>("/api/v1/gene-pool/capabilities"),
  smartPriors: () => get<{ weights: Record<string, number>; basis: { survivors: number; vault: number; control_share: number };
    patch: Record<string, unknown> }>("/api/v1/gene-pool/smart-priors"),
  patchGenePool: (body: Record<string, unknown>) => send<GenePoolDoc>("PATCH", "/api/v1/gene-pool", body),
};
