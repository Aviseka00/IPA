const $ = (id) => document.getElementById(id);

const drop = $("dropzone");
const input = $("files");
const btn = $("btnAnalyze");
const statusEl = $("status");
const out = $("out");
const cards = $("cards");
const jobMeta = $("jobMeta");
const resultsTables = $("resultsTables");
const raw = $("raw");

let selectedFiles = [];

window.addEventListener("DOMContentLoaded", () => {
  if (location.protocol === "file:") {
    const w = document.createElement("p");
    w.className = "warn-banner";
    w.textContent =
      "You opened this page as a local file — uploads will not work. Start the API (uvicorn) and open http://127.0.0.1:8000 instead.";
    document.querySelector(".wrap")?.prepend(w);
  }
});

function setFiles(fileList) {
  selectedFiles = Array.from(fileList || []);
  btn.disabled = selectedFiles.length === 0;
  statusEl.textContent = selectedFiles.length
    ? `${selectedFiles.length} file(s) selected`
    : "";
}

input.addEventListener("change", () => setFiles(input.files));

drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.style.borderColor = "#58a6ff";
});
drop.addEventListener("dragleave", () => {
  drop.style.borderColor = "";
});
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.style.borderColor = "";
  setFiles(e.dataTransfer.files);
});

function readOptionalIds() {
  const ids = [
    "minArea",
    "borderMargin",
    "borderTouchKeep",
    "claheClip",
    "blendAlpha",
    "saveObservation",
    "wsDist",
    "opening",
    "useWs",
  ];
  const missing = ids.filter((id) => !$(id));
  if (missing.length) {
    console.warn("Missing DOM nodes:", missing.join(", "));
  }
}

function formatApiError(data, res, rawText) {
  if (rawText && !data) return rawText.slice(0, 400) || res.statusText;
  if (!data) return res.statusText;
  const d = data.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d))
    return d.map((x) => (typeof x === "object" ? x.msg || JSON.stringify(x) : String(x))).join("; ");
  if (d != null && typeof d === "object") return JSON.stringify(d);
  return res.statusText;
}

btn.addEventListener("click", async () => {
  if (!selectedFiles.length) return;

  readOptionalIds();

  const fd = new FormData();
  selectedFiles.forEach((f) => fd.append("files", f));

  const minArea = $("minArea").value || "100";
  const thr = $("threshold").value;

  const qs = new URLSearchParams({
    min_area: minArea,
    border_margin: $("borderMargin").value || "10",
    border_touch_keep_min_area: $("borderTouchKeep").value || "500",
    clahe_clip: $("claheClip").value || "2",
    blend_alpha: $("blendAlpha").value || "0.52",
    write_observation_panel: $("saveObservation").checked ? "true" : "false",
    watershed_peak_distance: $("wsDist").value || "18",
    opening_ksize: $("opening").value || "5",
    use_watershed: $("useWs").checked ? "true" : "false",
  });
  if (thr) qs.set("threshold", thr);

  statusEl.textContent = "Running model…";
  btn.disabled = true;
  out.classList.add("hidden");

  try {
    const res = await fetch(`/api/analyze?${qs}`, {
      method: "POST",
      body: fd,
    });
    const ct = (res.headers.get("content-type") || "").toLowerCase();
    let data = null;
    let rawText = "";
    if (ct.includes("application/json")) {
      data = await res.json();
    } else {
      rawText = await res.text();
    }
    if (!res.ok) throw new Error(formatApiError(data, res, rawText));

    document.getElementById("persistWarn")?.remove();
    jobMeta.textContent = `Job ${data.job_id} — ${data.job_type} — ${data.status}`;
    if (resultsTables) resultsTables.innerHTML = renderResultsTablesHtml(data);
    cards.innerHTML = "";
    if (data.persistence_warning) {
      console.warn(data.persistence_warning);
      const note = document.createElement("p");
      note.id = "persistWarn";
      note.className = "warn-banner";
      note.textContent =
        "MongoDB is not running — your analysis completed and images are below, but this job was not saved to the database.";
      cards.before(note);
    }

    (data.results || []).forEach((r) => {
      const div = document.createElement("div");
      div.className = "card" + (r.observation_url ? " card-with-obs" : "");

      const meta = document.createElement("div");
      meta.className = "meta";
      const settingsHtml = formatAnalysisSettings(r.analysis_settings);
      meta.innerHTML = `
        <div class="meta-head"><strong class="count-val">${r.count}</strong> <span class="meta-label">true positives</span></div>
        <div class="filename">${escapeHtml(r.original_filename)}</div>
        ${settingsHtml}
        <div class="dl-row">
          <a href="${r.processed_url}" download>Download overlay</a>
          ${
            r.observation_url
              ? ` · <a href="${r.observation_url}" download>Download observation</a>`
              : ""
          }
        </div>`;

      const figOverlay = document.createElement("figure");
      figOverlay.className = "fig";
      const img = document.createElement("img");
      img.src = r.processed_url;
      img.alt = "Overlay with QC rings if applicable";
      figOverlay.appendChild(img);
      const cap1 = document.createElement("figcaption");
      cap1.textContent = "Brightfield overlay — count card + instance IDs; red rings = excluded artifacts (QC)";
      figOverlay.appendChild(cap1);

      div.appendChild(meta);
      div.appendChild(figOverlay);

      if (r.observation_url) {
        const figObs = document.createElement("figure");
        figObs.className = "fig fig-obs";
        const obs = document.createElement("img");
        obs.src = r.observation_url;
        obs.alt = "Observation triptych";
        obs.className = "img-observation";
        figObs.appendChild(obs);
        const cap2 = document.createElement("figcaption");
        cap2.textContent =
          "Observation panel — probability map · instance masks · overlay (no QC rings)";
        figObs.appendChild(cap2);
        div.appendChild(figObs);
      }

      cards.appendChild(div);
    });

    if ((data.errors || []).length) {
      const err = document.createElement("p");
      err.style.color = "#f85149";
      err.textContent = "Errors: " + JSON.stringify(data.errors);
      cards.appendChild(err);
    }

    raw.textContent = JSON.stringify(data, null, 2);
    raw.classList.remove("hidden");
    out.classList.remove("hidden");
    statusEl.textContent = "Done.";
    input.value = "";
    selectedFiles = [];
  } catch (e) {
    statusEl.textContent = "Error: " + e.message;
  } finally {
    btn.disabled = selectedFiles.length === 0;
  }
});

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function renderResultsTablesHtml(data) {
  const jobRows = [
    ["Job ID", escapeHtml(String(data.job_id ?? ""))],
    ["Type", escapeHtml(String(data.job_type ?? ""))],
    ["Status", escapeHtml(String(data.status ?? ""))],
    ["Files processed", String((data.results || []).length)],
    ["Errors", (data.errors || []).length ? escapeHtml(JSON.stringify(data.errors)) : "none"],
  ];
  if (data.persistence_warning) {
    jobRows.push(["Persistence", escapeHtml(String(data.persistence_warning))]);
  }

  let html = '<div class="table-wrap"><h3 class="table-title">Job summary</h3><table class="data-table">';
  jobRows.forEach(([k, v]) => {
    html += `<tr><th scope="row">${k}</th><td>${v}</td></tr>`;
  });
  html += "</table></div>";

  const rows = data.results || [];
  if (!rows.length) {
    html += '<p class="table-empty">No result rows.</p>';
    return html;
  }

  html +=
    '<div class="table-wrap table-wrap-scroll"><h3 class="table-title">Per-file analysis</h3><table class="data-table data-table-wide"><thead><tr>';
  const headers = [
    "Original file",
    "True positives",
    "Threshold",
    "Min area (px)",
    "Border margin",
    "Edge keep (px)",
    "CLAHE",
    "Blend α",
    "Watershed",
    "Peak dist.",
    "Opening",
    "Artifacts marked",
    "Obs. saved",
    "Stored file",
    "Overlay",
    "Observation",
  ];
  headers.forEach((h) => {
    html += `<th>${escapeHtml(h)}</th>`;
  });
  html += "</tr></thead><tbody>";

  rows.forEach((r) => {
    const s = r.analysis_settings || {};
    const ws = s.use_watershed ? "yes" : "no";
    html += "<tr>";
    html += `<td class="td-file">${escapeHtml(r.original_filename || "")}</td>`;
    html += `<td class="td-num">${escapeHtml(String(r.count ?? ""))}</td>`;
    html += `<td>${escapeHtml(String(s.threshold ?? ""))}</td>`;
    html += `<td>${escapeHtml(String(s.min_area_px ?? ""))}</td>`;
    html += `<td>${escapeHtml(String(s.border_margin_px ?? ""))}</td>`;
    html += `<td>${escapeHtml(String(s.border_touch_keep_min_area_px ?? ""))}</td>`;
    html += `<td>${escapeHtml(String(s.clahe_clip ?? ""))}</td>`;
    html += `<td>${escapeHtml(String(s.blend_alpha ?? ""))}</td>`;
    html += `<td>${escapeHtml(ws)}</td>`;
    html += `<td>${escapeHtml(String(s.watershed_peak_distance ?? ""))}</td>`;
    html += `<td>${escapeHtml(String(s.opening_ksize ?? ""))}</td>`;
    html += `<td>${escapeHtml(String(s.artifacts_marked ?? ""))}</td>`;
    html += `<td>${s.observation_saved ? "yes" : "no"}</td>`;
    html += `<td class="td-mono">${escapeHtml(r.stored_filename || "")}</td>`;
    html += `<td><a href="${escapeHtml(r.processed_url)}">view</a></td>`;
    html += `<td>${
      r.observation_url
        ? `<a href="${escapeHtml(r.observation_url)}">view</a>`
        : "—"
    }</td>`;
    html += "</tr>";
  });

  html += "</tbody></table></div>";
  return html;
}

function formatAnalysisSettings(s) {
  if (!s || typeof s !== "object") return "";
  const order = [
    ["threshold", "Threshold"],
    ["min_area_px", "Min area (px)"],
    ["border_margin_px", "Border margin (px)"],
    ["border_touch_keep_min_area_px", "Edge keep ≥ (px)"],
    ["clahe_clip", "CLAHE clip"],
    ["blend_alpha", "Blend α"],
    ["use_watershed", "Watershed"],
    ["watershed_peak_distance", "Watershed peak px"],
    ["opening_ksize", "Opening k"],
    ["artifacts_marked", "Artifacts marked"],
    ["observation_saved", "Observation saved"],
  ];
  const rows = [];
  for (const [key, label] of order) {
    if (s[key] === undefined || s[key] === null) continue;
    let v = s[key];
    if (typeof v === "boolean") v = v ? "yes" : "no";
    rows.push(
      `<div class="kv"><span class="kv-k">${escapeHtml(label)}</span><span class="kv-v">${escapeHtml(String(v))}</span></div>`
    );
  }
  return `<div class="settings-grid">${rows.join("")}</div>`;
}
