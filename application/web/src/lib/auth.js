import { readJSON, writeJSON } from "./storage.js";

// Phiên đăng nhập trang admin: token của POST /admin/login (detection), gửi lại trong
// `Authorization: Bearer`. Giữ cả trong bộ nhớ vì localStorage có thể bị chặn.
const KEY = "plant.web.admin";
export const LOGOUT_EVENT = "plant:admin-logout";

let memory = null;

export function adminSession() {
  const session = memory || readJSON(KEY, null);
  if (!session?.token || !session.expires_at) return null;
  return new Date(session.expires_at).getTime() > Date.now() ? session : null;
}

export function saveSession(session) {
  memory = session;
  writeJSON(KEY, session);
}

// reason: "logout" (bấm Đăng xuất) hoặc "expired" (server trả 401).
export function clearSession(reason = "logout") {
  memory = null;
  writeJSON(KEY, null);
  window.dispatchEvent(new CustomEvent(LOGOUT_EVENT, { detail: reason }));
}
