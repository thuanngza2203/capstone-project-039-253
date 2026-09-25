// localStorage có thể bị chặn (chế độ riêng tư, trình duyệt nhúng): mọi thao tác đều bọc try/catch.

export function readJSON(key, fallback) {
  try {
    const text = window.localStorage.getItem(key);
    return text ? JSON.parse(text) : fallback;
  } catch {
    return fallback;
  }
}

export function writeJSON(key, value) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Không lưu được thì thôi: web vẫn chạy, chỉ mất danh sách hội thoại khi tải lại trang.
  }
}

// crypto.randomUUID() chỉ có trong secure context (HTTPS hoặc localhost); mở web qua
// http://<IP máy host> trên điện thoại thì không có, nên tự tạo UUID v4 bằng getRandomValues.
export function newId() {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

// --- Danh sách hội thoại của trình duyệt này -------------------------------------

const HISTORY_KEY = "plant.web.conversations";
const CURRENT_KEY = "plant.web.current";

export const history = {
  list: () => readJSON(HISTORY_KEY, []),
  save: (items) => writeJSON(HISTORY_KEY, items.slice(0, 100)),
  // Đưa hội thoại lên đầu; giữ tiêu đề cũ (câu hỏi đầu tiên) nếu đã có.
  upsert(entry) {
    const all = history.list();
    const previous = all.find((item) => item.id === entry.id);
    const items = [{ ...previous, ...entry, title: previous?.title || entry.title },
      ...all.filter((item) => item.id !== entry.id)];
    history.save(items);
    return items;
  },
  remove(id) {
    const items = history.list().filter((item) => item.id !== id);
    history.save(items);
    return items;
  },
  current: () => readJSON(CURRENT_KEY, null),
  setCurrent: (id) => writeJSON(CURRENT_KEY, id),
};
