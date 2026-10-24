import type { AnalysisItem, Batch } from "../api";

export type ResultRow = {
  serial: number;
  filename: string;
  aiCount: number;
  humanVerified: number | null;
  fn: number;
  fp: number;
  total: number;
  status: string;
  reviewStatus: string;
  itemId: string;
};

export function itemToResultRow(item: AnalysisItem, serial: number): ResultRow {
  const ai = item.ai_infected_count ?? item.infected_count ?? 0;
  const fn = item.false_negative_count ?? item.review?.false_negative_count ?? 0;
  const fp = item.false_positive_count ?? item.review?.false_positive_count ?? 0;
  const humanDraft = item.review?.human_infected_count ?? Math.max(0, ai + fn - fp);
  const humanVerified =
    item.review_status === "published" && item.human_infected_count != null
      ? item.human_infected_count
      : null;
  const total = humanVerified ?? humanDraft;

  return {
    serial,
    filename: item.filename,
    aiCount: ai,
    humanVerified,
    fn,
    fp,
    total,
    status: item.status,
    reviewStatus: item.review_status ?? "draft",
    itemId: item.id,
  };
}

export function batchToResultRows(batch: Batch): ResultRow[] {
  return batch.items.map((item, idx) => itemToResultRow(item, idx + 1));
}

function rowLabel(row: ResultRow): { text: string; className: string } {
  if (row.status === "queued") return { text: "Queued", className: "badge-warn text-[10px]" };
  if (row.status === "processing") return { text: "Running", className: "badge-live text-[10px]" };
  if (row.status === "failed") return { text: "Failed", className: "badge-warn text-[10px]" };
  if (row.reviewStatus === "published") return { text: "Published", className: "badge-live text-[10px]" };
  return { text: "Ready", className: "badge-live text-[10px]" };
}

type Props = {
  rows: ResultRow[];
  title?: string;
  onRowClick?: (itemId: string) => void;
  selectedItemId?: string | null;
  showTotals?: boolean;
};

export default function ResultsTable({
  rows,
  title = "Analysis results summary",
  onRowClick,
  selectedItemId,
  showTotals = true,
}: Props) {
  if (rows.length === 0) return null;

  const finished = rows.filter((r) => r.status === "completed");
  const running = rows.filter((r) => r.status === "queued" || r.status === "processing").length;
  const totals = finished.reduce(
    (acc, r) => ({
      ai: acc.ai + r.aiCount,
      fn: acc.fn + r.fn,
      fp: acc.fp + r.fp,
      total: acc.total + r.total,
      published: acc.published + (r.reviewStatus === "published" ? 1 : 0),
    }),
    { ai: 0, fn: 0, fp: 0, total: 0, published: 0 }
  );

  const allPublished = finished.length > 0 && totals.published === finished.length;

  return (
    <section className="glass-panel rounded-xl overflow-hidden border border-ink-line">
      <div className="px-5 py-4 border-b border-ink-line bg-lab-850/60 flex flex-wrap justify-between items-center gap-2">
        <div>
          <p className="section-label">{title}</p>
          <p className="text-xs text-ink-muted mt-1 font-sans">
            {finished.length}/{rows.length} finished
            {running > 0 ? ` · ${running} still running` : ""}
            {finished.length > 0
              ? allPublished
                ? " · all published"
                : ` · ${totals.published}/${finished.length} published`
              : ""}
          </p>
        </div>
        {showTotals && (
          <div className="text-right">
            <p className="section-label">Batch total (human)</p>
            <p className="text-2xl font-serif font-semibold text-cyan-400">{totals.total}</p>
          </div>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="results-table w-full text-sm">
          <thead>
            <tr>
              <th className="w-12">S.No</th>
              <th className="text-left min-w-[140px]">Image name</th>
              <th>AI count</th>
              <th>Human verified</th>
              <th className="text-emerald-400">FN</th>
              <th className="text-red-400">FP</th>
              <th>Total</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.itemId}
                className={
                  onRowClick
                    ? `cursor-pointer transition-colors ${
                        selectedItemId === row.itemId ? "bg-status-ok-bg/40" : "hover:bg-lab-800/50"
                      }`
                    : undefined
                }
                onClick={onRowClick ? () => onRowClick(row.itemId) : undefined}
              >
                <td className="font-mono text-ink-faint">{row.serial}</td>
                <td className="text-left font-mono text-xs text-ink truncate max-w-[220px]" title={row.filename}>
                  {row.filename}
                </td>
                <td className="font-semibold text-ink-muted">
                  {row.status === "completed" ? row.aiCount : "—"}
                </td>
                <td className="font-semibold text-cyan-400">
                  {row.status !== "completed" ? (
                    "—"
                  ) : row.humanVerified != null ? (
                    row.humanVerified
                  ) : (
                    <span className="text-ink-faint font-normal" title="Draft — not yet published">
                      {row.total}*
                    </span>
                  )}
                </td>
                <td className="font-mono text-emerald-400">{row.status === "completed" ? row.fn : "—"}</td>
                <td className="font-mono text-red-400">{row.status === "completed" ? row.fp : "—"}</td>
                <td className="font-serif font-semibold text-ink">
                  {row.status === "completed" ? row.total : "—"}
                </td>
                <td>
                  <span className={rowLabel(row).className}>{rowLabel(row).text}</span>
                </td>
              </tr>
            ))}
          </tbody>
          {showTotals && (
            <tfoot>
              <tr className="border-t-2 border-ink-line bg-lab-850/80 font-semibold">
                <td colSpan={2} className="text-right section-label uppercase tracking-wider">
                  Total
                </td>
                <td className="text-ink-muted">{totals.ai}</td>
                <td className="text-cyan-400">{allPublished ? totals.total : "—"}</td>
                <td className="text-emerald-400">{totals.fn}</td>
                <td className="text-red-400">{totals.fp}</td>
                <td className="text-lg font-serif text-cyan-400">{totals.total}</td>
                <td />
              </tr>
            </tfoot>
          )}
        </table>
      </div>

      {!allPublished && (
        <p className="px-5 py-3 text-[11px] text-ink-faint border-t border-ink-line">
          * Draft human count — publish each image to finalize verified totals.
        </p>
      )}
    </section>
  );
}
