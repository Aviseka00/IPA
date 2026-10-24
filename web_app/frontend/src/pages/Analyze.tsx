import { useCallback, useEffect, useState } from "react";
import { api, type AnalysisItem, type Batch, type BatchSummary, type ModelInfo, type ReviewState } from "../api";
import { IconCell, IconMicroscope, IconSparkle, IconUpload } from "../components/Icons";
import MicroscopyShowcase from "../components/MicroscopyShowcase";
import ReviewPanel from "../components/ReviewPanel";
import ResultsTable, { batchToResultRows } from "../components/ResultsTable";

function modelBadge(id: string): { label: string; className: string } | null {
  if (id === "ipa8") return { label: "Recommended", className: "badge-live" };
  if (id === "ipa8_a" || id === "ipa8_b") return { label: "IPA8 test", className: "badge-ai" };
  if (id === "ipa6") return { label: "Dev", className: "badge-ai" };
  if (id === "ipa5") return { label: "Fallback", className: "badge-ai" };
  if (id === "ipa4_fusion") return { label: "Fusion", className: "badge-ai" };
  if (id.includes("phase2")) return { label: "Production", className: "badge-ai" };
  return null;
}

export default function Analyze() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelId, setModelId] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [live, setLive] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<Batch | null>(null);
  const [selectedItem, setSelectedItem] = useState<AnalysisItem | null>(null);
  const [savedBatches, setSavedBatches] = useState<BatchSummary[]>([]);
  const [savedBatchId, setSavedBatchId] = useState("");
  const [restitchMosaic, setRestitchMosaic] = useState(true);

  useEffect(() => {
    api.models().then((m) => {
      setModels(m);
      const preferred = m.find((x) => x.id === "ipa8" && x.available) ?? m.find((x) => x.id === "ipa5" && x.available) ?? m.find((x) => x.available);
      if (preferred) setModelId(preferred.id);
    });
    api.history("batch").then((h) => {
      setSavedBatches(h.batches);
      if (h.batches[0]) setSavedBatchId(h.batches[0].id);
    }).catch(() => setSavedBatches([]));
  }, []);

  async function openSavedBatch(id: string) {
    if (!id) return;
    setError("");
    setBusy(true);
    setLive(false);
    try {
      const batch = await api.getBatch(id);
      setResult(batch);
      setSelectedItem(batch.items.find((i) => i.status === "completed") ?? batch.items[0] ?? null);
      if (batch.status === "processing") setLive(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not open this batch");
    } finally {
      setBusy(false);
    }
  }

  const onFiles = useCallback((list: FileList | null) => {
    if (!list) return;
    setFiles(Array.from(list));
    setResult(null);
    setSelectedItem(null);
    setLive(false);
  }, []);

  async function analyze() {
    if (!modelId || files.length === 0) return;
    setBusy(true);
    setLive(false);
    setError("");
    setResult(null);
    setSelectedItem(null);
    try {
      const batch = await api.runAnalysis(modelId, files, { restitchMosaic });
      setResult(batch);
      setSelectedItem(batch.items[0] ?? null);
      setBusy(false);
      if (batch.status !== "processing") {
        if (batch.status === "failed") setError("All images failed.");
        return;
      }
      setLive(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (!live || !result || result.status !== "processing") return;
    const batchId = result.id;
    let stop = false;
    const timer = window.setInterval(async () => {
      try {
        const current = await api.getBatch(batchId);
        if (stop) return;
        setResult(current);
        setSelectedItem((prev) => {
          const done = current.items.find((i) => i.status === "completed");
          if (!prev) return done ?? current.items[0] ?? null;
          if (prev.status === "completed") {
            return current.items.find((i) => i.id === prev.id) ?? prev;
          }
          const updated = current.items.find((i) => i.id === prev.id);
          if (updated?.status === "completed") return updated;
          return done ?? updated ?? prev;
        });
        if (current.status !== "processing") {
          setLive(false);
          if (current.status === "failed") {
            const err0 = current.items.find((i) => i.error_message)?.error_message;
            setError(err0 || "All images failed.");
          }
        }
      } catch {
        /* keep polling until the batch leaves processing */
      }
    }, 2000);
    return () => {
      stop = true;
      window.clearInterval(timer);
    };
  }, [live, result?.id, result?.status]);

  const active = selectedItem ?? result?.items[0] ?? null;
  const selectedModel = models.find((m) => m.id === modelId);

  function updateItemReview(itemId: string, review: ReviewState) {
    setResult((prev) => {
      if (!prev) return prev;
      const items = prev.items.map((it) =>
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
      const humanTotal = items.reduce(
        (s, it) =>
          s +
          (it.review_status === "published" && it.human_infected_count != null
            ? it.human_infected_count
            : it.ai_infected_count ?? it.infected_count),
        0
      );
      return {
        ...prev,
        items,
        human_total_infected: humanTotal,
        review_status: items.every((i) => i.review_status === "published")
          ? "published"
          : items.some((i) => i.review_status === "published")
            ? "partial"
            : "draft",
      };
    });
    setSelectedItem((prev) =>
      prev?.id === itemId
        ? {
            ...prev,
            review_status: review.review_status,
            human_infected_count: review.human_infected_count,
            review,
            infected_count:
              review.review_status === "published"
                ? review.human_infected_count
                : prev.ai_infected_count ?? prev.infected_count,
          }
        : prev
    );
  }

  return (
    <div className="space-y-8">
      <MicroscopyShowcase variant="compact" />

      <header>
        <p className="section-label mb-2">Inference workbench</p>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="page-title">Microscopy analysis</h1>
            <p className="page-subtitle">
              Upload slide images, select an AI model, and receive side-by-side detection
              overlays with infected cell counts.
            </p>
          </div>
          <div className="stat-card min-w-[120px] text-center py-4">
            <p className="text-2xl font-serif font-semibold text-ink">{models.filter((m) => m.available).length}</p>
            <p className="section-label mt-1">Models ready</p>
          </div>
        </div>
      </header>

      <div className="grid lg:grid-cols-12 gap-6 lg:gap-8">
        <aside className="lg:col-span-4 xl:col-span-3 space-y-5">
          <section className="glass-panel rounded-xl p-6 space-y-5 card-hover">
            <h2 className="text-sm font-sans font-semibold text-ink flex items-center gap-2">
              <IconSparkle className="w-4 h-4 text-status-ok" />
              AI model
            </h2>

            <div className="space-y-0 divide-y divide-ink-line max-h-[280px] overflow-y-auto">
              {models.map((m) => {
                const badge = modelBadge(m.id);
                const selected = modelId === m.id;
                return (
                  <button
                    key={m.id}
                    type="button"
                    disabled={!m.available}
                    onClick={() => setModelId(m.id)}
                    className={`${selected ? "model-card-selected" : "model-card"} !rounded-none !border-0 !border-b !border-ink-line last:!border-b-0 disabled:opacity-40 disabled:cursor-not-allowed`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <span className="font-mono text-sm text-ink">{m.name}</span>
                      {badge && <span className={badge.className}>{badge.label}</span>}
                      {!m.available && <span className="badge-warn">Offline</span>}
                    </div>
                    <p className="text-xs text-ink-muted mt-1.5 leading-relaxed font-sans">{m.description}</p>
                  </button>
                );
              })}
            </div>
          </section>

          <section className="glass-panel rounded-xl p-6 space-y-3 card-hover">
            <h2 className="text-sm font-sans font-semibold text-ink">Open a saved batch</h2>
            <p className="text-xs text-ink-muted font-sans">
              Continue review on any earlier run — mark missed cells, correct over-counts, and publish.
            </p>
            {savedBatches.length === 0 ? (
              <p className="text-xs text-ink-faint font-sans">No saved batches yet.</p>
            ) : (
              <>
                <select
                  className="input-field w-full text-sm"
                  value={savedBatchId}
                  onChange={(e) => setSavedBatchId(e.target.value)}
                >
                  {savedBatches.map((b) => (
                    <option key={b.id} value={b.id}>
                      {new Date(b.created_at).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                      {" · "}
                      {b.model_name}
                      {" · "}
                      {b.image_count} img
                    </option>
                  ))}
                </select>
                <button type="button" className="btn-ghost w-full" disabled={busy || !savedBatchId} onClick={() => openSavedBatch(savedBatchId)}>
                  Open and review
                </button>
              </>
            )}
          </section>

          <section className="glass-panel rounded-xl p-6 space-y-4 card-hover">
            <h2 className="text-sm font-sans font-semibold text-ink flex items-center gap-2">
              <IconUpload className="w-4 h-4 text-status-ok" />
              Image upload
            </h2>

            <label className="upload-zone group">
              <input
                type="file"
                accept="image/*,.tif,.tiff"
                multiple
                className="hidden"
                onChange={(e) => onFiles(e.target.files)}
              />
              <div className="relative z-10 flex flex-col items-center text-center">
                <div className="w-14 h-14 rounded-xl bg-status-ok-bg border border-status-ok-border flex items-center justify-center mb-4">
                  <IconUpload className="w-7 h-7 text-status-ok" />
                </div>
                <span className="text-ink font-medium font-sans">Drop or browse files</span>
                <span className="text-xs text-ink-muted mt-2">JPG · PNG · TIF · Up to 5000 images per batch</span>
                {files.length > 0 && (
                  <span className="mt-3 badge-live">{files.length} file{files.length !== 1 ? "s" : ""} selected</span>
                )}
              </div>
            </label>

            {files.length > 0 && (
              <ul className="space-y-0 divide-y divide-ink-line text-xs font-mono text-ink-muted">
                {files.slice(0, 5).map((f) => (
                  <li key={f.name} className="truncate py-2">{f.name}</li>
                ))}
                {files.length > 5 && <li className="py-2 text-ink-faint">+{files.length - 5} more</li>}
              </ul>
            )}

            <label className="flex items-start gap-3 cursor-pointer select-none rounded-lg border border-ink-line bg-lab-850/40 px-3 py-3">
              <input
                type="checkbox"
                className="mt-1 accent-cyan-400"
                checked={restitchMosaic}
                onChange={(e) => setRestitchMosaic(e.target.checked)}
              />
              <span>
                <span className="block text-sm font-sans font-medium text-ink">EVOS mosaic restitch</span>
                <span className="block text-xs text-ink-muted font-sans mt-1 leading-relaxed">
                  Detect instrument tiled previews, measure ~50% field overlap, restitch, then run the selected model.
                </span>
              </span>
            </label>

            {error && (
              <p className="text-sm badge-warn rounded-lg px-4 py-3 w-full justify-start">{error}</p>
            )}

            <button type="button" className="btn-primary w-full" disabled={busy || live || !files.length || !modelId} onClick={analyze}>
              {busy || live ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  {busy ? "Uploading…" : "Publishing results…"}
                </span>
              ) : (
                "Run analysis"
              )}
            </button>

            {selectedModel && (
              <p className="text-[10px] text-center font-mono text-ink-faint uppercase tracking-wider">
                Pipeline: {selectedModel.id}
              </p>
            )}
          </section>
        </aside>

        <div className="lg:col-span-8 xl:col-span-9 space-y-5">
          {result && (
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              {[
                {
                  label: "AI counted",
                  value: result.ai_total_infected ?? result.total_infected,
                  accent: "text-ink-muted",
                  icon: IconCell,
                },
                {
                  label: "Human verified",
                  value:
                    result.review_status === "published"
                      ? result.human_total_infected ?? result.total_infected
                      : "—",
                  accent: "text-cyan-400",
                  icon: IconCell,
                },
                { label: "Images", value: result.image_count, accent: "text-ink", icon: IconMicroscope },
                { label: "Model", value: result.model_name, accent: "text-ink text-sm font-sans", icon: IconSparkle, small: true },
                {
                  label: "Review",
                  value: result.review_status ?? "draft",
                  accent: "text-status-ok text-sm capitalize",
                  icon: null,
                  small: true,
                },
              ].map((s) => (
                <div key={s.label} className="stat-card">
                  {s.icon && <s.icon className="w-4 h-4 text-ink-faint mb-2" />}
                  <p className="section-label">{s.label}</p>
                  <p className={`font-serif font-semibold mt-1 ${s.small ? "text-sm truncate" : "text-3xl"} ${s.accent}`}>
                    {s.value}
                  </p>
                </div>
              ))}
            </div>
          )}

          {live && result && (
            <div className="glass-panel rounded-xl px-5 py-4 flex items-center gap-4">
              <div className="loading-ring scale-50" />
              <div>
                <p className="font-serif text-lg text-ink">Results are publishing as each image finishes</p>
                <p className="text-sm text-ink-muted font-sans">
                  {result.items.filter((i) => i.status === "completed").length} of {result.image_count} complete
                  {" · "}
                  AI total so far {result.ai_total_infected ?? result.total_infected}
                </p>
              </div>
            </div>
          )}

          {result && result.items.some((i) => (i.warnings && i.warnings.length > 0) || i.restitch_status) && (
            <div className="glass-panel rounded-xl px-5 py-4 space-y-2 border border-status-warn-border/40">
              <p className="section-label">Preprocessing / restitch</p>
              {result.items
                .filter((i) => i.status === "completed")
                .slice(0, 8)
                .map((i) => (
                  <p key={i.id} className="text-xs font-sans text-ink-muted">
                    <span className="font-mono text-ink">{i.filename}</span>
                    {" · "}
                    {i.restitch_used ? "restitched" : i.restitch_status || "passthrough"}
                    {i.warnings && i.warnings.length > 0 ? ` — ${i.warnings[0]}` : ""}
                  </p>
                ))}
            </div>
          )}

          {result && batchToResultRows(result).length > 0 && (
            <ResultsTable
              rows={batchToResultRows(result)}
              title="Results summary table"
              selectedItemId={active?.id}
              onRowClick={(itemId) => {
                const item = result.items.find((i) => i.id === itemId);
                if (item) setSelectedItem(item);
              }}
            />
          )}

          {result && result.items.length > 1 && (
            <div className="flex gap-2 flex-wrap">
              {result.items.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setSelectedItem(item)}
                  className={`px-3 py-2 rounded-lg text-xs font-mono border transition-all ${
                    active?.id === item.id
                      ? "border-status-ok-border bg-status-ok-bg text-status-ok"
                      : "border-ink-line text-ink-muted hover:bg-lab-800 hover:text-ink"
                  }`}
                >
                  {item.filename}{" "}
                  <span className="text-status-ok ml-1">
                    (AI {item.ai_infected_count ?? item.infected_count}
                    {item.review_status === "published" ? ` → ${item.human_infected_count}` : ""})
                  </span>
                </button>
              ))}
            </div>
          )}

          {active && active.status === "completed" && (
            <div className="glass-panel rounded-xl overflow-hidden border border-ink-line">
              <div className="px-5 py-4 border-b border-ink-line flex flex-wrap justify-between items-center gap-3 bg-lab-850/60">
                <div>
                  <p className="section-label">Result — model testing & review</p>
                  <span className="font-mono text-sm text-ink truncate block max-w-md">{active.filename}</span>
                </div>
                <div className="flex items-center gap-4">
                  <div className="text-right">
                    <p className="section-label">AI count</p>
                    <p className="text-xl font-serif font-semibold text-ink-muted">
                      {active.ai_infected_count ?? active.infected_count}
                    </p>
                  </div>
                  <div className="text-right pl-4 border-l border-ink-line">
                    <p className="section-label">Human count</p>
                    <p className="text-2xl font-serif font-semibold text-cyan-400">
                      {active.review_status === "published" && active.human_infected_count != null
                        ? active.human_infected_count
                        : active.review
                          ? active.review.human_infected_count
                          : "—"}
                    </p>
                  </div>
                </div>
              </div>

              <ReviewPanel
                itemId={active.id}
                batchId={result!.id}
                inputUrl={active.input_url}
                overlayUrl={active.output_url}
                collageUrl={active.collage_url}
                initialReview={active.review ?? null}
                onDraftSaved={(review) => updateItemReview(active.id, review)}
                onPublished={(review) => updateItemReview(active.id, review)}
                onAniImported={async () => {
                  try {
                    const batch = await api.getBatch(result!.id);
                    setResult(batch);
                    setSelectedItem(
                      batch.items.find((i) => i.id === active.id) ??
                        batch.items.find((i) => i.status === "completed") ??
                        batch.items[0] ??
                        null
                    );
                  } catch {
                    /* keep current UI */
                  }
                }}
              />
            </div>
          )}

          {active && active.status === "failed" && (
            <div className="glass-panel rounded-xl p-6 border border-status-warn-border space-y-2 bg-status-warn-bg/30">
              <p className="font-semibold text-status-warn flex items-center gap-2 font-sans">
                <span className="badge-warn">Failed</span>
                Analysis failed
              </p>
              {active.error_message && (
                <p className="text-sm font-mono text-status-warn break-words bg-lab-950/60 rounded-lg p-3 border border-status-warn-border">
                  {active.error_message}
                </p>
              )}
            </div>
          )}

          {!result && !busy && (
            <div className="glass-panel rounded-xl p-12 text-center relative overflow-hidden min-h-[280px] flex flex-col items-center justify-center">
              <div className="absolute inset-0 opacity-15">
                <img src="/images/fluorescence-ring.png" alt="" className="w-full h-full object-cover ken-burns-slow" />
                <div className="absolute inset-0 bg-lab-900/80" />
              </div>
              <div className="relative z-10">
                <IconMicroscope className="w-12 h-12 text-status-ok mx-auto mb-4" />
                <h3 className="font-serif text-xl font-semibold text-ink">Awaiting specimens</h3>
                <p className="text-ink-muted mt-2 max-w-md text-sm leading-relaxed font-sans">
                  Upload microscopy images and select a model to generate side-by-side collages.
                </p>
              </div>
            </div>
          )}

          {busy && !result && (
            <div className="glass-panel rounded-xl p-16 text-center">
              <div className="loading-ring mx-auto" />
              <p className="mt-8 font-serif text-lg font-semibold text-ink">Uploading images</p>
              <p className="mt-3 text-sm text-ink-muted max-w-sm mx-auto font-sans italic">
                Counts will appear in the table as soon as each image finishes. The batch does not wait for every file.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
