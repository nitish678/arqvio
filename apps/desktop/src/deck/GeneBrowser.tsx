// BREEDING: every gene the engine knows (the registry), by class, with what the breeder may draw. A READY switch has
// its own on/off inside its class gate; a gene that is not READY is shown greyed with the engine's own reason, so
// nothing pretends to breed. Families (the signal engines) and Smart priors stay here too.
// idea: research PART 4.1 (genome classes: signals, bias, filters, management, risk, exec) + BREEDING settings tab.

import { useMemo, useState } from "react";
import { api, type GenePoolDoc, type GeneRegistry, type RegistryGene } from "../api/ws";
import { errText, FAMILY_NAMES, usePoll } from "../ws/util";

const GROUPS: { id: string; title: string; gate?: string; cls: string; hint: string }[] = [
  { id: "signal", title: "Signals", gate: "signal_switches", cls: "sig", hint: "when to enter: the entry signals" },
  { id: "bias", title: "Bias", gate: "bias_switches", cls: "bias", hint: "which side: long only above, short only below" },
  { id: "filter", title: "Filters", gate: "filter_switches", cls: "filt", hint: "when not to trade" },
  { id: "management", title: "Trade management", gate: "mgmt_switches", cls: "mgmt", hint: "how an open trade is run" },
  { id: "risk", title: "Risk guards", cls: "risk", hint: "account guardrails (set at deploy)" },
  { id: "exec", title: "Execution", cls: "exec", hint: "how orders fill" },
  { id: "core", title: "Core", cls: "core", hint: "stops, targets, hours" },
  { id: "limits", title: "Limits", cls: "core", hint: "broker limits" },
  { id: "meta", title: "Bookkeeping", cls: "core", hint: "what the engine records about a genome" },
];
const STATUS_TEXT: Record<string, string> = { READY: "breeds", FIXED: "fixed", EA_ONLY: "EA only", PLANNED: "planned" };
const ENTRY_GATES: [string, string][] = [["session", "Session window (Asia / London / NY / London+NY)"],
  ["trend_ema", "Trend EMA (50 / 100 / 200)"]];

const short = (g: RegistryGene) => g.name.split(".").slice(1).join(".");

export function GeneBrowser({ pool, busy, onPatch, error }: {
  pool: GenePoolDoc | null;
  busy: boolean;
  onPatch: (body: Record<string, unknown>) => Promise<void>;
  error: string | null;
}) {
  const reg = usePoll(() => api.geneRegistry(), 120_000);
  const capabilities = usePoll(() => api.geneCapabilities(), 120_000);
  const support = new Map((capabilities.data?.registry_hash === reg.data?.hash ? capabilities.data?.genes ?? [] : [])
    .map((g) => [g.name, g]));
  const [q, setQ] = useState("");
  const [onlyReady, setOnlyReady] = useState(false);
  const data: GeneRegistry | null | undefined = reg.data;
  const off = useMemo(() => new Set(pool?.off ?? []), [pool?.off]);
  const gates = pool?.gates ?? {};
  const toggleOff = (name: string) => {
    const next = new Set(off);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    void onPatch({ off: [...next].sort() });
  };
  const shown = (g: RegistryGene) => (!onlyReady || g.status === "READY")
    && (!q || g.name.toLowerCase().includes(q.toLowerCase()) || (g.note ?? "").toLowerCase().includes(q.toLowerCase()));

  return (
    <div className="gb">
      <section className="dk-panel">
        <div className="dk-ph"><span className="dk-pt">Genome</span>
          <span className="dk-ps">{data ? `${data.counts.total} genes · ${data.counts.by_status.READY ?? 0} breed · V1 ${data.counts.engines.v1?.ready ?? "—"} · V2 ${data.counts.engines.v2?.ready ?? "—"}` : "reading the registry…"}</span>
          <span className="sp" />
          <input className="gb-q" placeholder="search genes" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search genes" />
          <button type="button" className="chip" aria-pressed={onlyReady} onClick={() => setOnlyReady(!onlyReady)}>BREEDABLE ONLY</button>
        </div>
        <div className="dk-pb">
          {error && <div className="dk-note bad">{error}</div>}
          {reg.load.state === "error" && <div className="dk-note bad">{errText(reg.load.error)}</div>}
          {busy && <div className="dk-note">A campaign is running: the gene pool is locked until it ends.</div>}
          <div className="dk-note">READY is a legacy implementation declaration, not verified parity. Support below is unverified until matching run evidence is available.</div>
          <div className="gb-fams">
            <span className="k">FAMILIES</span>
            {pool && Object.entries(pool.families).map(([f, v]) => (
              <button key={f} type="button" className="chip" aria-pressed={v.enabled} disabled={busy}
                title={`${FAMILY_NAMES[f] ?? f} · weight ${(v.weight * 100).toFixed(0)}%`}
                onClick={() => void onPatch({ families: { [f]: { enabled: !v.enabled } } })}>{FAMILY_NAMES[f] ?? f}</button>
            ))}
            <button type="button" className="chip" disabled={busy}
              title="Weigh the families by what survives in your vault; half of every draw stays uniform as the control"
              onClick={() => void api.smartPriors().then((s) => onPatch(s.patch))}>SMART PRIORS</button>
          </div>
          <div className="gb-fams">
            <span className="k">ENTRY GATES</span>
            {ENTRY_GATES.map(([g, label]) => (
              <button key={g} type="button" className="chip" aria-pressed={!!gates[g]} disabled={busy}
                onClick={() => void onPatch({ gates: { [g]: !gates[g] } })}>{label}</button>
            ))}
          </div>
          {pool && <div className="dk-note">Pool <span className="mono">{pool.hash}</span> · saved at once · {off.size} switch{off.size === 1 ? "" : "es"} turned off</div>}
        </div>
      </section>

      <div className="gb-groups">
        {GROUPS.map((grp) => {
          const genes = (data?.genes ?? []).filter((g) => g.group === grp.id && shown(g));
          if (!genes.length) return null;
          const gateOn = grp.gate ? !!gates[grp.gate] : null;
          const ready = genes.filter((g) => g.status === "READY").length;
          return (
            <section key={grp.id} className={`dk-panel gb-g ${grp.cls}`} aria-label={grp.title}>
              <div className="dk-ph"><span className="dk-pt">{grp.title}</span><span className="dk-ps">{ready} / {genes.length} breed</span>
                <span className="sp" />
                {grp.gate && (
                  <button type="button" className="chip" aria-pressed={!!gateOn} disabled={busy}
                    title={`${grp.hint}: the breeder ${gateOn ? "draws" : "does not draw"} these switches`}
                    onClick={() => void onPatch({ gates: { [grp.gate as string]: !gateOn } })}>{gateOn ? "ON" : "OFF"}</button>
                )}
              </div>
              <div className="dk-pb">
                <div className="gb-hint">{grp.hint}</div>
                <ul className="gb-list">
                  {genes.map((g) => {
                    const switchable = g.status === "READY" && !!g.dna_field && !!grp.gate;
                    const on = switchable ? !!gateOn && !off.has(g.name) : g.status === "READY";
                    return (
                      <li key={g.name} className={`gb-row ${g.status.toLowerCase()}`} title={g.note ?? ""}>
                        <span className={`gb-dot ${on ? "on" : ""}`} />
                        <b>{short(g)}</b>
                        <span className={`gb-st ${g.status.toLowerCase()}`}>{STATUS_TEXT[g.status]}</span>
                        <span className="gb-note">{g.note ?? (g.family ? `family ${FAMILY_NAMES[g.family] ?? g.family}` : g.params.join(", "))}
                          <small className="dim" aria-label={`Support for ${g.name}`}>
                            {support.has(g.name)
                              ? ` · Declared: ${Object.entries(support.get(g.name)!.support).filter(([, v]) => v.declared).map(([k]) => k.replaceAll("_", " ")).join(", ") || "none"} · Evidence: UNVERIFIED`
                              : " · Support evidence unavailable"}
                          </small>
                        </span>
                        {switchable && (
                          <button type="button" className="gb-tog" aria-pressed={on} disabled={busy || !gateOn}
                            aria-label={`${on ? "Turn off" : "Turn on"} ${g.name}`} onClick={() => toggleOff(g.name)}>{on ? "ON" : "OFF"}</button>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}
