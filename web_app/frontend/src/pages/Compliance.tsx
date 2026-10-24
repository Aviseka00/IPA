import { useEffect, useState } from "react";
import { api } from "../api";

type Status = Awaited<ReturnType<typeof api.complianceStatus>>;

export default function Compliance() {
  const [status, setStatus] = useState<Status | null>(null);
  const [audit, setAudit] = useState<Awaited<ReturnType<typeof api.systemAudit>>>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .complianceStatus()
      .then(setStatus)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"));
    api.systemAudit(50).then(setAudit).catch(() => setAudit([]));
  }, []);

  if (error) {
    return <p className="badge-warn rounded-lg px-4 py-3">{error}</p>;
  }
  if (!status) {
    return (
      <div className="flex flex-col items-center py-20">
        <div className="loading-ring" />
        <p className="mt-4 text-sm text-ink-muted font-mono">Loading compliance status…</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <header>
        <p className="section-label mb-2">Regulated software controls</p>
        <h1 className="page-title">21 CFR Part 11 · GAMP 5</h1>
        <p className="page-subtitle max-w-3xl">
          Technical controls supporting electronic records, electronic signatures, and computer
          system validation for AI-driven cell analysis.
        </p>
      </header>

      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="stat-card">
          <p className="section-label">Part 11 mode</p>
          <p className="text-xl font-serif font-semibold text-status-ok mt-1">
            {status.part11_mode ? "Enabled" : "Off"}
          </p>
        </div>
        <div className="stat-card">
          <p className="section-label">GAMP category</p>
          <p className="text-sm font-sans text-ink mt-2 leading-snug">{status.gamp_category}</p>
        </div>
        <div className="stat-card">
          <p className="section-label">Session timeout</p>
          <p className="text-xl font-serif font-semibold text-cyan-400 mt-1">
            {status.session_timeout_minutes} min
          </p>
        </div>
        <div className="stat-card">
          <p className="section-label">E-signature on publish</p>
          <p className="text-xl font-serif font-semibold text-ink mt-1">
            {status.esignature_required_on_publish ? "Required" : "Optional"}
          </p>
        </div>
      </div>

      <div className="glass-panel rounded-xl p-5 border border-amber-500/30 bg-amber-500/5">
        <p className="section-label text-amber-300/90 mb-2">Compliance posture</p>
        <p className="font-mono text-sm text-ink">{status.compliance_posture}</p>
        <p className="text-sm text-ink-muted mt-3 font-sans leading-relaxed">{status.disclaimer}</p>
        <p className="text-xs text-ink-faint mt-2 font-mono">
          Validation package: {status.validation_package_path} · v{status.app_version}
        </p>
      </div>

      <section className="glass-panel rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-ink-line bg-lab-850/60">
          <p className="section-label">Control matrix</p>
        </div>
        <div className="overflow-x-auto">
          <table className="results-table w-full text-sm">
            <thead>
              <tr>
                <th className="text-left">Clause</th>
                <th className="text-left">Control</th>
                <th>Status</th>
                <th className="text-left">Evidence</th>
              </tr>
            </thead>
            <tbody>
              {status.controls.map((c) => (
                <tr key={c.id}>
                  <td className="text-left font-mono text-cyan-400">{c.id}</td>
                  <td className="text-left text-ink">{c.title}</td>
                  <td>
                    <span className={c.status === "implemented" ? "badge-live" : "badge-warn"}>
                      {c.status}
                    </span>
                  </td>
                  <td className="text-left text-xs text-ink-muted max-w-md">{c.evidence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="glass-panel rounded-xl p-5">
        <p className="section-label mb-3">System audit trail (recent)</p>
        {audit.length === 0 ? (
          <p className="text-sm text-ink-muted">No system audit entries yet.</p>
        ) : (
          <ul className="space-y-2 max-h-80 overflow-y-auto text-xs font-mono">
            {audit.map((a) => (
              <li key={a.log_id} className="border-b border-ink-line pb-2">
                <span className="text-cyan-400">{a.action}</span>
                <span className="text-ink-faint ml-2">{a.employee_id}</span>
                <span className="text-ink-faint ml-2">{new Date(a.created_at).toLocaleString()}</span>
                {a.entry_hash && (
                  <span className="block text-[10px] text-ink-faint mt-0.5 truncate">
                    hash {a.entry_hash.slice(0, 16)}…
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
