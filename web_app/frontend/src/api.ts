/** Use Vite proxy in dev; direct URL if proxy fails or frontend opened without Vite */
const API = import.meta.env.VITE_API_URL?.replace(/\/$/, "") || "/api";

export type ModelInfo = {
  id: string;
  name: string;
  description: string;
  available: boolean;
};

export type ReviewMarker = {
  id: string;
  x: number;
  y: number;
  type: "false_negative" | "false_positive";
  note?: string;
};

export type ReviewState = {
  item_id: string;
  review_status: "draft" | "published";
  ai_infected_count: number;
  ai_artifact_count: number;
  human_infected_count: number;
  false_negative_count: number;
  false_positive_count: number;
  markers: ReviewMarker[];
  ai_detections: { id: string; x: number; y: number; class_id: number; confidence: number; box?: number[] }[];
  published_at?: string | null;
  published_by?: string | null;
};

export type AuditEntry = {
  log_id: string;
  item_id: string;
  batch_id: string;
  employee_id: string;
  action: string;
  detail: Record<string, unknown>;
  created_at: string;
};

export type AnalysisItem = {
  id: string;
  filename: string;
  infected_count: number;
  artifact_count: number;
  processing_ms: number;
  input_url: string;
  output_url: string;
  collage_url: string;
  status: string;
  error_message?: string | null;
  review_status?: "draft" | "published";
  ai_infected_count?: number;
  human_infected_count?: number | null;
  false_negative_count?: number;
  false_positive_count?: number;
  review?: ReviewState | null;
  warnings?: string[];
  restitch_status?: string | null;
  restitch_used?: boolean;
};

export type Batch = {
  id: string;
  employee_id: string;
  model_id: string;
  model_name: string;
  created_at: string;
  status: string;
  image_count: number;
  total_infected: number;
  ai_total_infected?: number;
  human_total_infected?: number | null;
  review_status?: "draft" | "partial" | "published";
  items: AnalysisItem[];
};

export type BatchSummary = {
  id: string;
  model_id: string;
  model_name: string;
  created_at: string;
  status: string;
  image_count: number;
  total_infected: number;
  ai_total_infected?: number;
  human_total_infected?: number | null;
  review_status?: "draft" | "partial" | "published";
};

export type HistoryResponse = {
  group_by: "batch" | "date";
  batches: BatchSummary[];
  by_date: Record<string, BatchSummary[]> | null;
};

function authHeaders(): HeadersInit {
  const token = localStorage.getItem("ipa_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function clearAuth(): void {
  localStorage.removeItem("ipa_token");
  localStorage.removeItem("ipa_employee_id");
}

function parseErrorDetail(detail: unknown): string {
  if (Array.isArray(detail)) {
    return detail
      .map((d: { msg?: string }) => d.msg || JSON.stringify(d))
      .join("; ");
  }
  return typeof detail === "string" ? detail : "Request failed";
}

async function handleAuthResponse(res: Response): Promise<void> {
  if (res.status !== 401) return;
  const err = await res.json().catch(() => ({ detail: "Session expired" }));
  clearAuth();
  const msg = parseErrorDetail(err.detail);
  window.location.href = `/login?reason=${encodeURIComponent(msg)}`;
  throw new Error(msg);
}

function apiErrorMessage(res: Response, detail: unknown): string {
  if (res.status === 404) {
    return (
      "API not found — start the backend with web_app\\backend\\run_server.bat (port 8010)"
    );
  }
  if (res.status === 500 || res.status === 502 || res.status === 503) {
    const msg = parseErrorDetail(detail);
    if (msg === "Internal Server Error" || msg === "Bad Gateway" || !msg) {
      return (
        "Backend unavailable — open a terminal, run web_app\\backend\\run_server.bat, " +
        "then refresh this page (API must be on http://127.0.0.1:8010)"
      );
    }
    return msg;
  }
  return parseErrorDetail(detail);
}

async function readErrorDetail(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text) return res.statusText;
  try {
    return JSON.parse(text).detail;
  } catch {
    return text;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API}${path}`, {
      ...init,
      headers: {
        ...(init?.headers || {}),
        ...authHeaders(),
      },
    });
  } catch {
    throw new Error(
      "Cannot reach API — start web_app\\backend\\run_server.bat (port 8010) and ensure npm run dev is running"
    );
  }
  await handleAuthResponse(res);
  if (!res.ok) {
    const detail = await readErrorDetail(res);
    throw new Error(apiErrorMessage(res, detail));
  }
  return res.json();
}

async function authRequest<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers || {}),
      },
    });
  } catch {
    throw new Error(
      "Cannot reach API — start web_app\\backend\\run_server.bat (port 8010) and ensure npm run dev is running"
    );
  }
  if (!res.ok) {
    const detail = await readErrorDetail(res);
    throw new Error(apiErrorMessage(res, detail));
  }
  return res.json();
}

export const api = {
  /** Start Ani companion if not already running (no auth — called when IPA UI opens). */
  ensureAniRunning: async () => {
    try {
      const res = await fetch(`${API}/ani/ensure-running`, { method: "POST" });
      if (!res.ok) return null;
      return (await res.json()) as {
        ok: boolean;
        skipped?: boolean;
        message: string;
        already_running?: boolean;
        project_dir?: string;
        pid?: number | null;
      };
    } catch {
      return null;
    }
  },

  register: (employee_id: string, password: string) =>
    authRequest<{ employee_id: string; created_at?: string }>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ employee_id, password }),
    }),

  login: (employee_id: string, password: string) =>
    authRequest<{ access_token: string; employee_id: string }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ employee_id, password }),
    }),

  resetPassword: (employee_id: string, reset_code: string, new_password: string) =>
    authRequest<{ message: string }>("/auth/reset-password", {
      method: "POST",
      body: JSON.stringify({ employee_id, reset_code, new_password }),
    }),

  me: () => request<{ employee_id: string; created_at: string }>("/auth/me"),

  models: () => request<ModelInfo[]>("/models"),

  runAnalysis: async (modelId: string, files: File[], opts?: { restitchMosaic?: boolean }) => {
    const fd = new FormData();
    fd.append("model_id", modelId);
    fd.append("restitch_mosaic", opts?.restitchMosaic === false ? "false" : "true");
    files.forEach((f) => fd.append("files", f));
    let res: Response;
    try {
      res = await fetch(`${API}/analysis/run`, {
        method: "POST",
        headers: authHeaders(),
        body: fd,
      });
    } catch {
      throw new Error(
        "Cannot reach API — start web_app\\backend\\run_server.bat (port 8010)"
      );
    }
    await handleAuthResponse(res);
    if (!res.ok) {
      const detail = await readErrorDetail(res);
      throw new Error(apiErrorMessage(res, detail));
    }
    return res.json() as Promise<Batch>;
  },

  history: (groupBy: "batch" | "date" = "batch") =>
    request<HistoryResponse>(`/analysis/history?group_by=${groupBy}`),

  getBatch: (id: string) => request<Batch>(`/analysis/batches/${id}`),

  getReview: (itemId: string) => request<ReviewState>(`/analysis/items/${itemId}/review`),

  saveReview: (itemId: string, markers: ReviewMarker[]) =>
    request<ReviewState>(`/analysis/items/${itemId}/review`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ markers }),
    }),

  publishReview: (
    itemId: string,
    signature?: { password: string; printed_name: string; meaning: string }
  ) =>
    request<ReviewState>(`/analysis/items/${itemId}/publish`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(signature ? { signature } : {}),
    }),

  complianceStatus: () =>
    request<{
      app_name: string;
      app_version: string;
      part11_mode: boolean;
      gamp_category: string;
      compliance_posture: string;
      disclaimer: string;
      controls: { id: string; title: string; status: string; evidence: string }[];
      password_policy: Record<string, unknown>;
      session_timeout_minutes: number;
      esignature_required_on_publish: boolean;
      validation_package_path: string;
    }>("/compliance/status"),

  systemAudit: (limit = 100) =>
    request<
      {
        log_id: string;
        employee_id: string;
        action: string;
        detail: Record<string, unknown>;
        created_at: string;
        entry_hash?: string;
      }[]
    >(`/compliance/audit?limit=${limit}`),

  getItemAudit: (itemId: string) => request<AuditEntry[]>(`/analysis/items/${itemId}/audit`),

  getAuditReport: (batchId: string) =>
    request<{
      batch_id: string;
      model_name: string;
      review_status: string;
      ai_total_infected: number;
      human_total_infected: number | null;
      items: Record<string, unknown>[];
      audit_log: AuditEntry[];
    }>(`/analysis/batches/${batchId}/audit-report`),

  aniStatus: (batchId: string) =>
    request<{
      project_exists: boolean;
      project_dir: string;
      image_count: number;
      labeled_count: number;
      labeled_stems: string[];
      ani_root: string;
      ani_python_ok: boolean;
      retrain_dataset_dir: string;
      last_import_at?: string | null;
    }>(`/analysis/batches/${batchId}/ani/status`),

  aniOpen: (batchId: string) =>
    request<{
      project_dir: string;
      image_count: number;
      pid?: number | null;
      message: string;
    }>(`/analysis/batches/${batchId}/ani/open`, { method: "POST" }),

  aniImport: (batchId: string) =>
    request<{
      imported: number;
      items: {
        item_id: string;
        stem: string;
        human_infected_count: number;
        false_negative_count: number;
        false_positive_count: number;
      }[];
      retrain_dataset_dir: string;
      imported_at: string;
      message: string;
    }>(`/analysis/batches/${batchId}/ani/import`, { method: "POST" }),
};

export function mediaUrl(apiPath: string): string {
  const token = localStorage.getItem("ipa_token");
  const base = apiPath.startsWith("/") ? apiPath : `/${apiPath}`;
  if (!token) return base;
  const sep = base.includes("?") ? "&" : "?";
  return `${base}${sep}access_token=${encodeURIComponent(token)}`;
}
