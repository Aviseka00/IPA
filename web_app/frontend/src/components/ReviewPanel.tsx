import { useCallback, useEffect, useRef, useState } from "react";
import { api, mediaUrl, type ReviewMarker, type ReviewState } from "../api";

type MarkerMode = "false_negative" | "false_positive";

type ImageView = "overlay" | "original" | "collage";

type Props = {
  itemId: string;
  batchId: string;
  inputUrl?: string;
  overlayUrl?: string;
  collageUrl?: string;
  /** @deprecated use overlayUrl */
  imageUrl?: string;
  initialReview?: ReviewState | null;
  onPublished: (review: ReviewState) => void;
  onDraftSaved: (review: ReviewState) => void;
  /** Called after Ani ground-truth import so parent can refresh the batch */
  onAniImported?: () => void;
};

function newId() {
  return Math.random().toString(36).slice(2, 14);
}

export default function ReviewPanel({
  itemId,
  batchId,
  inputUrl,
  overlayUrl,
  collageUrl,
  imageUrl,
  initialReview,
  onPublished,
  onDraftSaved,
  onAniImported,
}: Props) {
  const [markers, setMarkers] = useState<ReviewMarker[]>(initialReview?.markers ?? []);
  const [mode, setMode] = useState<MarkerMode>("false_negative");
  const [reviewStatus, setReviewStatus] = useState<"draft" | "published">(
    initialReview?.review_status ?? "draft"
  );
  const [audit, setAudit] = useState<Awaited<ReturnType<typeof api.getItemAudit>>>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [aniMsg, setAniMsg] = useState("");
  const [aniStatus, setAniStatus] = useState<Awaited<ReturnType<typeof api.aniStatus>> | null>(null);
  const [showAudit, setShowAudit] = useState(false);
  const [view, setView] = useState<ImageView>("overlay");
  const [showEsign, setShowEsign] = useState(false);
  const [esignPassword, setEsignPassword] = useState("");
  const [esignName, setEsignName] = useState("");
  const [esignMeaning, setEsignMeaning] = useState(
    "I have reviewed this AI analysis result and approve the human-verified count as accurate."
  );
  const imgRef = useRef<HTMLImageElement>(null);

  const aiCount = initialReview?.ai_infected_count ?? 0;
  const markerFn = markers.filter((m) => m.type === "false_negative").length;
  const markerFp = markers.filter((m) => m.type === "false_positive").length;
  // Prefer server/Ani ground-truth counts when present
  const fnCount = initialReview?.false_negative_count ?? markerFn;
  const fpCount = initialReview?.false_positive_count ?? markerFp;
  const humanCount =
    initialReview?.human_infected_count != null && markers.length === 0
      ? initialReview.human_infected_count
      : Math.max(0, aiCount + markerFn - markerFp);
  const isPublished = reviewStatus === "published";

  useEffect(() => {
    setMarkers(initialReview?.markers ?? []);
    setReviewStatus(initialReview?.review_status ?? "draft");
  }, [initialReview, itemId]);

  useEffect(() => {
    setView(overlayUrl || imageUrl ? "overlay" : inputUrl ? "original" : "collage");
  }, [itemId, overlayUrl, imageUrl, inputUrl, collageUrl]);

  const displayUrl =
    view === "original"
      ? inputUrl
      : view === "collage"
        ? collageUrl
        : overlayUrl || imageUrl || collageUrl || inputUrl || "";

  const loadAudit = useCallback(async () => {
    try {
      const entries = await api.getItemAudit(itemId);
      setAudit(entries);
    } catch {
      setAudit([]);
    }
  }, [itemId]);

  useEffect(() => {
    if (showAudit) loadAudit();
  }, [showAudit, loadAudit]);

  const refreshAniStatus = useCallback(async () => {
    try {
      setAniStatus(await api.aniStatus(batchId));
    } catch {
      setAniStatus(null);
    }
  }, [batchId]);

  useEffect(() => {
    refreshAniStatus();
  }, [refreshAniStatus, itemId]);

  async function openInAni() {
    setBusy(true);
    setError("");
    setAniMsg("");
    try {
      const res = await api.aniOpen(batchId);
      setAniMsg(res.message || `Ani opened with ${res.image_count} image(s).`);
      await refreshAniStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not open Ani");
    } finally {
      setBusy(false);
    }
  }

  async function importAniLabels() {
    setBusy(true);
    setError("");
    setAniMsg("");
    try {
      const res = await api.aniImport(batchId);
      setAniMsg(res.message || `Imported ${res.imported} ground-truth mask(s).`);
      await refreshAniStatus();
      onAniImported?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ani import failed");
    } finally {
      setBusy(false);
    }
  }

  function onImageClick(e: React.MouseEvent<HTMLDivElement>) {
    if (isPublished || busy) return;
    const img = imgRef.current;
    if (!img) return;
    const rect = img.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    setMarkers((prev) => [...prev, { id: newId(), x, y, type: mode }]);
  }

  function removeMarker(id: string) {
    if (isPublished) return;
    setMarkers((prev) => prev.filter((m) => m.id !== id));
  }

  async function saveDraft() {
    setBusy(true);
    setError("");
    try {
      const review = await api.saveReview(itemId, markers);
      setReviewStatus(review.review_status);
      onDraftSaved(review);
      await loadAudit();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setBusy(false);
    }
  }

  async function publishWithEsign() {
    if (!esignName.trim() || !esignPassword) {
      setError("Electronic signature requires printed name and password (21 CFR Part 11)");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await api.saveReview(itemId, markers);
      const review = await api.publishReview(itemId, {
        password: esignPassword,
        printed_name: esignName.trim(),
        meaning: esignMeaning.trim(),
      });
      setReviewStatus("published");
      setShowEsign(false);
      setEsignPassword("");
      onPublished(review);
      await loadAudit();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Publish failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3 px-5 py-3 border-b border-ink-line bg-lab-850/40">
        <span className="section-label">Human review</span>
        {isPublished ? (
          <span className="badge-live">Published</span>
        ) : (
          <span className="badge-warn">Draft — review required before publish</span>
        )}
      </div>

      {!isPublished && (
        <div className="mx-5 mt-4 glass-panel rounded-xl p-4 border border-cyan-500/25 space-y-3">
          <div>
            <p className="section-label mb-1">Ani v0.4.0 — pixel-accurate labeling</p>
            <p className="text-xs text-ink-muted font-sans leading-relaxed">
              Model detections are provisional labels. Open Ani to keep confirmed infected cells,
              add false negatives, and remove false positives. The full corrected mask becomes
              ground truth for the retrain dataset.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 items-center">
            <button type="button" className="btn-primary text-sm" disabled={busy} onClick={openInAni}>
              Open in Ani
            </button>
            <button type="button" className="btn-ghost text-sm" disabled={busy} onClick={importAniLabels}>
              Import Ani labels
            </button>
            {aniStatus && (
              <span className="text-[11px] font-mono text-ink-faint">
                {aniStatus.labeled_count}/{aniStatus.image_count || "?"} labeled
                {aniStatus.last_import_at ? ` · last import ${new Date(aniStatus.last_import_at).toLocaleString()}` : ""}
              </span>
            )}
          </div>
          {aniMsg && <p className="text-xs text-cyan-300 font-sans">{aniMsg}</p>}
          {aniStatus && !aniStatus.ani_python_ok && (
            <p className="text-xs text-status-warn">
              Ani Python not found — run Ani\setup_windows.bat once (Ani is bundled in this IPA repo).
            </p>
          )}
        </div>
      )}

      <div className="px-5 grid sm:grid-cols-3 gap-3">
        <div className="stat-card py-3">
          <p className="section-label">AI counted</p>
          <p className="text-2xl font-serif font-semibold text-ink-muted">{aiCount}</p>
        </div>
        <div className="stat-card py-3">
          <p className="section-label">Human adjusted</p>
          <p className="text-2xl font-serif font-semibold text-cyan-400">{humanCount}</p>
          <p className="text-[10px] text-ink-faint mt-1">
            +{fnCount} FN · −{fpCount} FP
          </p>
        </div>
        <div className="stat-card py-3">
          <p className="section-label">Intervention</p>
          <p className="text-2xl font-serif font-semibold text-status-ok">
            {fnCount + fpCount}
          </p>
          <p className="text-[10px] text-ink-faint mt-1">markers placed</p>
        </div>
      </div>

      <div className="px-5 flex flex-wrap gap-2 items-center">
        <span className="section-label mr-1">View:</span>
        {overlayUrl || imageUrl ? (
          <button
            type="button"
            className={view === "overlay" ? "btn-primary text-xs py-1.5 px-3" : "btn-ghost text-xs py-1.5 px-3"}
            onClick={() => setView("overlay")}
          >
            Detection overlay
          </button>
        ) : null}
        {inputUrl ? (
          <button
            type="button"
            className={view === "original" ? "btn-primary text-xs py-1.5 px-3" : "btn-ghost text-xs py-1.5 px-3"}
            onClick={() => setView("original")}
          >
            Original
          </button>
        ) : null}
        {collageUrl ? (
          <button
            type="button"
            className={view === "collage" ? "btn-primary text-xs py-1.5 px-3" : "btn-ghost text-xs py-1.5 px-3"}
            onClick={() => setView("collage")}
          >
            Side-by-side
          </button>
        ) : null}
      </div>

      {!isPublished && (
        <div className="px-5 flex flex-wrap gap-2 items-center">
          <span className="section-label mr-1">Mark mode:</span>
          <button
            type="button"
            className={mode === "false_negative" ? "btn-primary text-xs py-1.5 px-3" : "btn-ghost text-xs py-1.5 px-3"}
            onClick={() => setMode("false_negative")}
          >
            🟢 False negative (missed)
          </button>
          <button
            type="button"
            className={mode === "false_positive" ? "btn-primary text-xs py-1.5 px-3 !bg-red-600" : "btn-ghost text-xs py-1.5 px-3"}
            onClick={() => setMode("false_positive")}
          >
            🔴 False positive (over-count)
          </button>
          <span className="text-xs text-ink-muted ml-2">Click image to place marker · click marker to remove</span>
        </div>
      )}

      <div
        className={`relative mx-5 mb-2 ${!isPublished ? "cursor-crosshair" : ""}`}
        onClick={onImageClick}
      >
        {displayUrl ? (
          <img
            ref={imgRef}
            src={mediaUrl(displayUrl)}
            alt={view === "overlay" ? "Infected cell detection overlay" : "Analysis image"}
            className="w-full h-auto rounded-lg border border-ink-line"
            draggable={false}
          />
        ) : (
          <p className="text-sm text-ink-muted py-8 text-center">No preview available for this image.</p>
        )}
        {markers.map((m) => (
          <button
            key={m.id}
            type="button"
            title={m.type === "false_negative" ? "False negative — click to remove" : "False positive — click to remove"}
            className="absolute w-5 h-5 -ml-2.5 -mt-2.5 rounded-full border-2 border-white shadow-lg transform hover:scale-110 transition-transform"
            style={{
              left: `${m.x * 100}%`,
              top: `${m.y * 100}%`,
              backgroundColor: m.type === "false_negative" ? "#22c55e" : "#ef4444",
            }}
            onClick={(e) => {
              e.stopPropagation();
              removeMarker(m.id);
            }}
          />
        ))}
        {(initialReview?.ai_detections ?? [])
          .filter((d) => d.class_id === 0)
          .map((d) => (
            <span
              key={d.id}
              className="absolute w-3 h-3 -ml-1.5 -mt-1.5 rounded-sm border border-amber-400/80 pointer-events-none"
              style={{ left: `${d.x * 100}%`, top: `${d.y * 100}%` }}
              title={`AI detection ${(d.confidence * 100).toFixed(0)}%`}
            />
          ))}
      </div>

      <div className="px-5 pb-5 flex flex-wrap gap-3 items-center">
        {!isPublished && (
          <>
            <button type="button" className="btn-ghost text-sm" disabled={busy} onClick={saveDraft}>
              Save draft
            </button>
            <button
              type="button"
              className="btn-primary text-sm"
              disabled={busy}
              onClick={() => setShowEsign(true)}
            >
              Publish with e-signature
            </button>
          </>
        )}
        <button type="button" className="btn-ghost text-sm" onClick={() => setShowAudit((v) => !v)}>
          {showAudit ? "Hide audit log" : "View audit trail"}
        </button>
        <button
          type="button"
          className="btn-ghost text-sm"
          onClick={async () => {
            const report = await api.getAuditReport(batchId);
            console.info("Audit report", report);
            alert(
              `Batch ${batchId.slice(0, 8)}…\nAI total: ${report.ai_total_infected}\nHuman total: ${report.human_total_infected ?? "—"}\nAudit entries: ${report.audit_log.length}`
            );
          }}
        >
          Batch report
        </button>
      </div>

      {showEsign && !isPublished && (
        <div className="mx-5 mb-4 glass-panel rounded-xl p-5 border border-cyan-500/30 space-y-4">
          <div>
            <p className="section-label mb-1">21 CFR Part 11 electronic signature</p>
            <p className="text-xs text-ink-muted font-sans">
              Re-enter your password to legally bind this human-verified count. Signature is linked
              to the record (§11.50 / §11.70 / §11.300).
            </p>
          </div>
          <div>
            <label className="block section-label mb-1">Printed name</label>
            <input
              className="input-field"
              value={esignName}
              onChange={(e) => setEsignName(e.target.value)}
              placeholder="Full legal name"
              autoComplete="name"
            />
          </div>
          <div>
            <label className="block section-label mb-1">Password (re-authenticate)</label>
            <input
              type="password"
              className="input-field"
              value={esignPassword}
              onChange={(e) => setEsignPassword(e.target.value)}
              autoComplete="current-password"
            />
          </div>
          <div>
            <label className="block section-label mb-1">Meaning of signature</label>
            <textarea
              className="input-field min-h-[72px]"
              value={esignMeaning}
              onChange={(e) => setEsignMeaning(e.target.value)}
            />
          </div>
          <div className="flex gap-2">
            <button type="button" className="btn-primary text-sm" disabled={busy} onClick={publishWithEsign}>
              {busy ? "Signing…" : "Sign & publish"}
            </button>
            <button type="button" className="btn-ghost text-sm" disabled={busy} onClick={() => setShowEsign(false)}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {error && (
        <p className="mx-5 mb-4 text-sm badge-warn rounded-lg px-4 py-3">{error}</p>
      )}

      {showAudit && (
        <div className="mx-5 mb-5 glass-panel rounded-lg p-4 max-h-48 overflow-y-auto">
          <p className="section-label mb-2">Audit trail</p>
          {audit.length === 0 ? (
            <p className="text-xs text-ink-muted">No activity logged yet.</p>
          ) : (
            <ul className="space-y-2 text-xs font-mono">
              {audit.map((a) => (
                <li key={a.log_id} className="border-b border-ink-line pb-2">
                  <span className="text-cyan-400">{a.action}</span>
                  <span className="text-ink-faint ml-2">
                    {new Date(a.created_at).toLocaleString()}
                  </span>
                  {Object.keys(a.detail).length > 0 && (
                    <span className="text-ink-muted ml-2">{JSON.stringify(a.detail)}</span>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
