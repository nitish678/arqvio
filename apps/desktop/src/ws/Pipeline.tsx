// The pipeline topology in the top bar: MT5 -> DATA -> CHECK -> BREED -> VAULT -> DEPLOY, each node lit from the
// engine's own state (the chosen terminal, the launch queue, the vault, the bots' heartbeats). Nothing is animated
// unless the engine says that step is running.

import type { Bot, LaunchState, Terminal } from "../api/ws";

type NodeState = "idle" | "ok" | "active" | "bad";

interface Node {
  key: string;
  name: string;
  state: NodeState;
  value: string;
  title: string;
}

export function pipelineNodes(term: Terminal | null, launch: LaunchState | null, busy: boolean, vaultCount: number | null,
  bots: Bot[] | null): Node[] {
  const acc = term?.account;
  const mt5: Node = !term
    ? { key: "mt5", name: "MT5", state: "bad", value: "not found", title: "No MetaTrader 5 terminal on this machine" }
    : acc?.state === "OK"
      ? { key: "mt5", name: "MT5", state: "ok", value: `${acc.server ?? ""}`, title: `${term.path}\n${acc.company ?? ""} · ${acc.mode ?? ""}` }
      : { key: "mt5", name: "MT5", state: "bad", value: acc?.state === "CLOSED" ? "closed" : (acc?.state ?? "—").toLowerCase(),
        title: acc?.detail ?? term.path };

  const tasks = busy || launch ? launch?.tasks ?? [] : [];
  const data = tasks.filter((t) => t.kind === "data");
  const breed = tasks.filter((t) => t.kind === "breed");
  const running = (ts: typeof tasks, ...states: string[]) => ts.find((t) => states.includes(t.state));
  const live = busy && launch && (launch.state === "RUNNING" || launch.state === "PAUSED");

  const d = running(data, "RUNNING");
  const dataNode: Node = d && live
    ? { key: "data", name: "DATA", state: "active", value: `${d.symbol} ${d.timeframe}`, title: "Reading bars, specs and spread from MetaTrader 5" }
    : data.length && data.some((t) => t.state === "DONE")
      ? { key: "data", name: "DATA", state: "ok", value: `${data.filter((t) => t.state === "DONE").length} market(s)`,
        title: data.map((t) => `${t.symbol} ${t.timeframe}: ${t.detail ?? t.state}`).join("\n") }
      : data.length && data.every((t) => t.state === "FAILED")
        ? { key: "data", name: "DATA", state: "bad", value: "failed", title: data.map((t) => t.detail ?? "").join("\n") }
        : { key: "data", name: "DATA", state: "idle", value: "—", title: "Market data comes from your MetaTrader 5" };

  const c = running(breed, "CHECKING");
  const checked = breed.filter((t) => ["RUNNING", "PAUSED", "DONE", "STOPPED"].includes(t.state)).length;
  const check: Node = c && live
    ? { key: "check", name: "CHECK", state: "active", value: `${c.symbol} ${c.timeframe}`, title: "A short trial run on a copy of the data" }
    : checked
      ? { key: "check", name: "CHECK", state: "ok", value: `${checked} passed`, title: "Each market's safety check passed" }
      : breed.some((t) => t.state === "FAILED" && (t.detail ?? "").startsWith("safety"))
        ? { key: "check", name: "CHECK", state: "bad", value: "failed", title: breed.map((t) => t.detail ?? "").join("\n") }
        : { key: "check", name: "CHECK", state: "idle", value: "—", title: "A short trial run before each campaign" };

  const b = running(breed, "RUNNING", "PAUSED");
  const breedNode: Node = b && live
    ? { key: "breed", name: "BREED", state: "active",
      value: b.state === "PAUSED" ? `${b.symbol} paused` : b.progress
        ? `${b.symbol} ${b.timeframe} · ${b.progress.generation + 1}/${b.progress.generations}` : `${b.symbol} ${b.timeframe}`,
      title: "The genetic search on this market" }
    : breed.some((t) => t.state === "DONE")
      ? { key: "breed", name: "BREED", state: "ok", value: `${breed.filter((t) => t.state === "DONE").length} done`, title: "Campaigns completed" }
      : { key: "breed", name: "BREED", state: "idle", value: "—", title: "Strategies are bred and tested here" };

  const vault: Node = vaultCount
    ? { key: "vault", name: "VAULT", state: "ok", value: `${vaultCount} strateg${vaultCount === 1 ? "y" : "ies"}`, title: "Survivors kept in the vault" }
    : { key: "vault", name: "VAULT", state: "idle", value: "empty", title: "Survivors land in the vault" };

  const liveBots = (bots ?? []).filter((x) => x.live === "LIVE").length;
  const deploy: Node = liveBots
    ? { key: "deploy", name: "DEPLOY", state: "ok", value: `${liveBots} live`, title: "EAs whose heartbeat is fresh" }
    : { key: "deploy", name: "DEPLOY", state: "idle", value: bots?.length ? `${bots.length} installed` : "none", title: "EAs running in MetaTrader 5" };

  return [mt5, dataNode, check, breedNode, vault, deploy];
}

export function Pipeline({ nodes }: { nodes: Node[] }) {
  return (
    <nav className="pipe" aria-label="Pipeline">
      {nodes.map((n, i) => (
        <span key={n.key} style={{ display: "contents" }}>
          {i > 0 && <span className={`pipe-link${nodes[i - 1].state === "ok" && n.state !== "idle" ? " lit" : ""}`} aria-hidden="true" />}
          <span className={`pipe-node ${n.state}`} title={n.title} data-node={n.key} data-state={n.state}>
            <span className="pipe-dot" aria-hidden="true" />
            <span className="pipe-txt">
              <span className="pipe-name">{n.name}</span>
              <span className="pipe-val">{n.value}</span>
            </span>
          </span>
        </span>
      ))}
    </nav>
  );
}
