import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { Link } from "react-router-dom";
import { api, type AnalysisItem, type Batch, type BatchSummary, type ReviewState } from "../api";
import { IconAnalyze, IconCell } from "../components/Icons";
import MicroscopyShowcase from "../components/MicroscopyShowcase";
import ReviewPanel from "../components/ReviewPanel";
import { batchToResultRows } from "../components/ResultsTable";

export default function Dashboard() {
  const [groupBy, setGroupBy] = useState<"batch" | "date">("batch");
  const [batches, setBatches] = useState<BatchSummary[]>([]);
  const [byDate, setByDate] = useState<Record<string, BatchSummary[]> | null>(null);
  const [loading, setLoading] = useState(true);
  const [detail, setDetail] = useState<Batch | null>(null);
  const [selectedItem, setSelectedItem] = useState<AnalysisItem | null>(null);
  const [editorOpen, setEditorOpen] = useState(false);
  const [openError, setOpenError] = useState("");

  useEffect(() => {
    setLoading(true);
    api
      .history(groupBy)
      .then((h) => {
        setBatches(h.batches);
        setByDate(h.by_date);
      })
      .finally(() => setLoading(false));
  }, [groupBy]);

  async function openBatch(id: string, refresh = false) {
    setOpenError("");
    try {
      const b = await api.getBatch(id);
      setDetail(b);
      if (!refresh) setEditorOpen(true);
      setSelectedItem((prev) => {
        const keep = prev && b.items.some((i) => i.id === prev.id) ? b.items.find((i) => i.id === prev.id) : null;
        return keep ?? b.items.find((i) => i.status === "completed") ?? b.items[0] ?? null;
      });
    } catch (err) {
      setOpenError(err instanceof Error ? err.message : "Could not open this batch");
    }
  }

  useEffect(() => {
    if (!detail || detail.status !== "processing") return;
    const timer = window.setInterval(() => {
      void openBatch(detail.id, true);
    }, 2000);
    return () => window.clearInterval(timer);
  }, [detail?.id, detail?.status]);

  function updateItemReview(itemId: string, review: ReviewState) {
    if (!detail) return;
    const items = detail.items.map((it) =>
      it.id === itemId
        ? {
            ...it,
            review_status: review.review_status,
            human_infected_count: review.human_infected_count,
            false_negative_count: review.false_negative_count,
            false_positive_count: review.false_positive_count,
            infected_count:
              review.review_status === "published"
                ? review.human_infected_count
                : it.ai_infected_count ?? it.infected_count,
            review,
          }
        : it
    );
    const published = items.filter((i) => i.status === "completed" && i.review_status === "published");
    const completed = items.filter((i) => i.status === "completed");
    const humanTotal = items.reduce(
      (s, it) =>
        s +
        (it.review_status === "published" && it.human_infected_count != null
          ? it.human_infected_count
          : it.ai_infected_count ?? it.infected_count),
      0
    );
    const reviewStatus: Batch["review_status"] =
      completed.length > 0 && published.length === completed.length
        ? "published"
        : published.length > 0
          ? "partial"
          : "draft";
    const next: Batch = { ...detail, items, human_total_infected: humanTotal, review_status: reviewStatus };
    setDetail(next);
    setSelectedItem(items.find((i) => i.id === itemId) ?? null);
    const patch = (b: BatchSummary): BatchSummary =>
      b.id === next.id
        ? { ...b, human_total_infected: humanTotal, review_status: reviewStatus, total_infected: humanTotal }
        : b;
    setBatches((list) => list.map(patch));
    setByDate((groups) => {
      if (!groups) return groups;
      return Object.fromEntries(Object.entries(groups).map(([day, rows]) => [day, rows.map(patch)]));
    });
  }

  const active = selectedItem ?? detail?.items.find((i) => i.status === "completed") ?? null;

  useEffect(() => {
    if (!editorOpen) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setEditorOpen(false);
    }
    window.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [editorOpen]);

  const totalRuns = batches.length;
  const totalCells = batches.reduce((s, b) => s + b.total_infected, 0);

  return (
    <div className="space-y-8">
      <MicroscopyShowcase variant="banner" />

      <header className="flex flex-wrap items-start justify-between gap-6">
        <div>
          <p className="section-label mb-2">Analysis archive</p>
          <h1 className="page-title">Run history</h1>
          <p className="page-subtitle">Click a draft to open it in a full window and edit the review.</p>
        </div>
        <Link to="/analyze" className="btn-primary flex items-center gap-2 shrink-0">
          <IconAnalyze className="w-4 h-4" />
          New analysis
        </Link>
      </header>

      {!loading && totalRuns > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          <div className="stat-card">
            <p className="section-label">Total batches</p>
            <p className="text-3xl font-serif font-semibold text-ink mt-1">{totalRuns}</p>
          </div>
          <div className="stat-card">
            <p className="section-label">Cells detected</p>
            <p className="text-3xl font-serif font-semibold text-status-ok mt-1">{totalCells}</p>
          </div>
          <div className="stat-card col-span-2 sm:col-span-1">
            <p className="section-label">View mode</p>
            <p className="text-lg font-sans font-medium text-ink mt-1 capitalize">{groupBy}</p>
          </div>
        </div>
      )}

      <div className="flex gap-2 p-1 glass-panel rounded-lg w-fit">
        <button
          type="button"
          className={groupBy === "batch" ? "btn-ghost-active text-sm py-2" : "btn-ghost text-sm py-2 border-transparent"}
          onClick={() => setGroupBy("batch")}
        >
          By batch
        </button>
        <button
          type="button"
          className={groupBy === "date" ? "btn-ghost-active text-sm py-2" : "btn-ghost text-sm py-2 border-transparent"}
          onClick={() => setGroupBy("date")}
        >
          By date
        </button>
      </div>

      <div className="glass-panel rounded-xl p-0 max-h-[72vh] overflow-y-auto divide-y divide-ink-line">
          {loading && (
            <div className="flex flex-col items-center py-16">
              <div className="loading-ring scale-75" />
              <p className="text-sm text-ink-muted mt-4 font-mono">Loading archive…</p>
            </div>
          )}

          {!loading && groupBy === "batch" && batches.length === 0 && (
            <div className="text-center py-16 px-6">
              <IconCell className="w-10 h-10 text-ink-faint mx-auto mb-3" />
              <p className="text-ink-muted font-sans">No analyses recorded yet.</p>
              <Link to="/analyze" className="text-cyan-400 text-sm mt-2 inline-block hover:underline font-medium">
                Run your first analysis →
              </Link>
            </div>
          )}

          {!loading && groupBy === "batch" &&
            batches.map((b) => (
              <button
                key={b.id}
                type="button"
                onClick={() => openBatch(b.id)}
                className={`w-full text-left p-4 transition-colors row-divider ${
                  detail?.id === b.id ? "bg-status-ok-bg" : "hover:bg-lab-800/60"
                }`}
              >
                <div className="flex justify-between items-start gap-2">
                  <span className="font-mono text-xs text-ink truncate">{b.id}</span>
                  <span className="text-[10px] text-ink-faint whitespace-nowrap font-mono">
                    {new Date(b.created_at).toLocaleString(undefined, {
                      month: "short",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </span>
                </div>
                <p className="text-sm font-sans font-medium text-ink mt-2">{b.model_name}</p>
                <div className="flex items-center gap-3 mt-2 text-xs font-sans">
                  <span className="text-ink-muted">AI {b.ai_total_infected ?? b.total_infected}</span>
                  {b.review_status === "published" && b.human_total_infected != null && (
                    <>
                      <span className="text-ink-faint">→</span>
                      <span className="text-cyan-400 font-semibold">{b.human_total_infected} verified</span>
                    </>
                  )}
                  <span className="text-ink-faint">·</span>
                  <span className="text-ink-muted">{b.image_count} images</span>
                  <span className={`ml-auto badge ${b.review_status === "published" ? "badge-live" : "badge-warn"}`}>
                    {(b.review_status ?? "draft").toUpperCase()}
                  </span>
                </div>
              </button>
            ))}

          {!loading && groupBy === "date" && byDate &&
            Object.keys(byDate)
              .sort()
              .reverse()
              .map((date) => (
                <div key={date} className="px-2 py-2">
                  <h3 className="section-label mb-2 sticky top-0 bg-lab-900 py-2 px-2">{date}</h3>
                  {byDate[date].map((b) => (
                    <button
                      key={b.id}
                      type="button"
                      onClick={() => openBatch(b.id)}
                      className={`w-full text-left p-3 rounded-lg mb-1 text-sm font-sans transition-colors ${
                        detail?.id === b.id ? "bg-status-ok-bg border border-status-ok-border" : "hover:bg-lab-800/60"
                      }`}
                    >
                      <span className="text-ink">{b.model_name}</span>
                      <span className="text-status-ok font-medium ml-2">{b.total_infected}</span>
                      <span className="text-ink-faint ml-1">· {b.image_count} img</span>
                    </button>
                  ))}
                </div>
              ))}
        </div>

      {openError && !editorOpen && <p className="text-sm text-status-warn font-sans">{openError}</p>}

      {editorOpen &&
        detail &&
        createPortal(
          <div className="fixed inset-0 z-[80] bg-lab-950 flex flex-col">
            <header className="shrink-0 border-b border-ink-line bg-lab-900 px-4 sm:px-6 py-3 flex flex-wrap items-center gap-3">
              <button type="button" className="btn-ghost text-sm py-2" onClick={() => setEditorOpen(false)}>
                ← Back to history
              </button>
              <div className="min-w-0">
                <p className="font-serif text-lg text-ink leading-tight truncate">{detail.model_name}</p>
                <p className="font-mono text-[11px] text-ink-faint truncate">{detail.id}</p>
              </div>
              <span className={`badge ${detail.review_status === "published" ? "badge-live" : "badge-warn"}`}>
                {(detail.review_status ?? "draft").toUpperCase()}
              </span>
              <div className="ml-auto text-right">
                <p className="section-label">Batch total</p>
                <p className="text-xl font-serif font-semibold text-cyan-400">
                  {detail.review_status === "published" && detail.human_total_infected != null
                    ? detail.human_total_infected
                    : batchToResultRows(detail).reduce((s, r) => s + (r.status === "completed" ? r.total : 0), 0)}
                </p>
              </div>
            </header>

            <div className="flex-1 min-h-0 grid lg:grid-cols-[280px_minmax(0,1fr)]">
              <aside className="border-r border-ink-line overflow-y-auto bg-lab-900/80">
                <p className="section-label px-4 pt-4 pb-2">Images in this draft</p>
                {detail.items.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setSelectedItem(item)}
                    className={`w-full text-left px-4 py-3 border-b border-ink-line text-xs font-mono ${
                      active?.id === item.id ? "bg-status-ok-bg text-status-ok" : "text-ink-muted hover:bg-lab-800"
                    }`}
                  >
                    <span className="block truncate">{item.filename}</span>
                    <span className="block mt-1 text-[10px] uppercase tracking-wide">
                      {item.status === "completed"
                        ? item.review_status === "published"
                          ? "Published"
                          : "Draft"
                        : item.status}
                    </span>
                  </button>
                ))}
              </aside>

              <div className="overflow-y-auto">
                {openError && <p className="px-6 pt-4 text-sm text-status-warn font-sans">{openError}</p>}
                {active && active.status === "completed" && (
                  <div>
                    <div className="px-6 py-4 border-b border-ink-line flex flex-wrap justify-between gap-3">
                      <div>
                        <p className="section-label">Edit review</p>
                        <span className="font-mono text-sm text-ink">{active.filename}</span>
                      </div>
                      <span className="flex items-center gap-2 badge-live">
                        <IconCell className="w-3.5 h-3.5" />
                        AI {active.ai_infected_count ?? active.infected_count}
                      </span>
                    </div>
                    <ReviewPanel
                      itemId={active.id}
                      batchId={detail.id}
                      imageUrl={active.output_url || active.collage_url}
                      initialReview={active.review ?? null}
                      onDraftSaved={(review) => updateItemReview(active.id, review)}
                      onPublished={(review) => updateItemReview(active.id, review)}
                    />
                  </div>
                )}
                {active && active.status !== "completed" && (
                  <p className="p-6 text-sm text-ink-muted font-sans">
                    {active.filename} is {active.status}. Choose a finished image on the left to edit its review.
                  </p>
                )}
              </div>
            </div>
          </div>,
          document.body
        )}
    </div>
  );
}
