import { adminSession, clearSession } from "./lib/auth.js";

// Mọi lời gọi backend đi qua file này.
// Local: để trống biến môi trường → đi qua proxy của Vite (/detection, /rag).
// Deploy: đặt VITE_DETECTION_URL, VITE_RAG_URL trong .env.production rồi `npm run build`.
export const DETECTION_URL = (import.meta.env.VITE_DETECTION_URL || "/detection").replace(/\/$/, "");
export const RAG_URL = (import.meta.env.VITE_RAG_URL || "/rag").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(status, message, service) {
    super(message);
    this.status = status;
    this.service = service;
  }
}

const SERVICE_NAME = { detection: "detection-server (cổng 8005)", rag: "RAG server (cổng 8010)" };

function errorMessage(status, data, service) {
  const detail = data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
  // Proxy của Vite trả 500/502/504 không kèm JSON khi backend chưa chạy.
  if ([500, 502, 503, 504].includes(status)) return `Không kết nối được ${SERVICE_NAME[service]}. Backend đã chạy chưa?`;
  return `Lỗi ${status}`;
}

// `admin`: API /admin/* của detection, cần token đăng nhập; 401 là phiên hết hạn → về trang đăng nhập.
async function request(service, path, { method = "GET", body, form, signal, admin = false } = {}) {
  const base = service === "rag" ? RAG_URL : DETECTION_URL;
  const init = { method, signal, headers: {}, cache: "no-store" };
  const session = admin ? adminSession() : null;
  if (session) init.headers.Authorization = `Bearer ${session.token}`;
  if (form) {
    init.body = form;
  } else if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers["Content-Type"] = "application/json";
  }
  let response;
  try {
    response = await fetch(base + path, init);
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiError(0, `Không kết nối được ${SERVICE_NAME[service]}. Kiểm tra mạng hoặc backend.`, service);
  }
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (admin && response.status === 401) clearSession("expired");
  if (!response.ok) throw new ApiError(response.status, errorMessage(response.status, data, service), service);
  return data;
}

const query = (params) => {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") search.set(key, String(value));
  }
  const text = search.toString();
  return text ? `?${text}` : "";
};

// `apple/apple_scab.txt` giữ dấu "/" làm đường dẫn; mã hóa từng phần.
const sourcePath = (source) => source.split("/").map(encodeURIComponent).join("/");

// --- detection-server ------------------------------------------------------------

export const detection = {
  health: () => request("detection", "/health"),
  models: () => request("detection", "/api/models"),
  chat: ({ sessionId, message, image, imageName, llmProvider, webSearch, signal }) => {
    const form = new FormData();
    form.append("session_id", sessionId);
    form.append("message", message || "");
    if (webSearch) form.append("web_search", "true");
    else if (llmProvider) form.append("llm_provider", llmProvider);
    if (image) form.append("image", image, imageName || "leaf.jpg");
    return request("detection", "/api/chat", { method: "POST", form, signal });
  },
  conversations: () => request("detection", "/api/conversations"),
  conversation: (id) => request("detection", `/api/conversations/${encodeURIComponent(id)}`),
  deleteConversation: (id) => request("detection", `/api/conversations/${encodeURIComponent(id)}`, { method: "DELETE" }),
  feedback: (feedbackId, rating, reasons = []) =>
    request("detection", "/feedback", { method: "PUT", body: { feedback_id: feedbackId, rating, reasons } }),
  login: (username, password) => request("detection", "/admin/login", { method: "POST", body: { username, password } }),
  adminStatus: () => request("detection", "/admin/status", { admin: true }),
  reviews: ({ all = false, from, to } = {}) =>
    request("detection", `/admin/reviews${query({ all: all || undefined, from, to })}`, { admin: true }),
  deleteReview: (id) => request("detection", `/admin/review/${encodeURIComponent(id)}`, { method: "DELETE", admin: true }),
  train: (items) => request("detection", "/admin/training", { method: "POST", body: { items }, admin: true }),
};

// --- RAG server ------------------------------------------------------------------

export const rag = {
  health: () => request("rag", "/health"),
  status: () => request("rag", "/v1/status"),
  llm: (probe = false) => request("rag", `/v1/llm${query({ probe: probe || undefined })}`),
  taxonomy: () => request("rag", "/v1/taxonomy"),
  retrieve: (body) => request("rag", "/v1/retrieve", { method: "POST", body }),
  answer: (body) => request("rag", "/v1/answer", { method: "POST", body }),
  overview: () => request("rag", "/v1/admin/overview"),
  documents: (index) => request("rag", `/v1/admin/documents${query({ index })}`),
  document: (source, { index, includeText } = {}) =>
    request("rag", `/v1/admin/documents/${sourcePath(source)}${query({ index, include_text: includeText || undefined })}`),
  searchChunks: (q, { index, limit } = {}) => request("rag", `/v1/admin/chunks${query({ q, index, limit })}`),
  chunk: (id, index) => request("rag", `/v1/admin/chunks/${encodeURIComponent(id)}${query({ index })}`),
};
