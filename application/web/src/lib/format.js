const dateTime = new Intl.DateTimeFormat("vi-VN", { dateStyle: "short", timeStyle: "short" });
const dayOnly = new Intl.DateTimeFormat("vi-VN", { day: "2-digit", month: "2-digit" });
const number = new Intl.NumberFormat("vi-VN");

// MongoDB trả thời gian không kèm múi giờ (UTC): thêm "Z" để trình duyệt không hiểu nhầm là giờ địa phương.
export function toDate(value) {
  if (!value) return null;
  if (value instanceof Date) return value;
  const text = String(value);
  const date = new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(text) ? text : `${text}Z`);
  return Number.isNaN(date.getTime()) ? null : date;
}

export const formatDateTime = (value) => {
  const date = toDate(value);
  return date ? dateTime.format(date) : "—";
};

export const formatDay = (value) => {
  const date = toDate(value);
  return date ? dayOnly.format(date) : "—";
};

export function formatRelative(value) {
  const date = toDate(value);
  if (!date) return "";
  const seconds = Math.round((Date.now() - date.getTime()) / 1000);
  if (seconds < 60) return "vừa xong";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} phút trước`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} giờ trước`;
  if (seconds < 7 * 86400) return `${Math.floor(seconds / 86400)} ngày trước`;
  return formatDateTime(date);
}

export const formatNumber = (value) => (value === null || value === undefined ? "—" : number.format(value));

export function formatPercent(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits).replace(".", ",")}%`;
}

export function formatMs(value) {
  if (value === null || value === undefined) return "—";
  return value >= 1000 ? `${(value / 1000).toFixed(1).replace(".", ",")} s` : `${Math.round(value)} ms`;
}

// Bỏ nhãn [Nguồn n] trong câu trả lời; tài liệu và link được liệt kê riêng dưới câu trả lời.
// Detection đã bỏ ở câu trả lời mới (strip_citations trong rag_http.py); ở đây cho các lượt lưu trước đó.
const CITATIONS = /[ \t]*\[\s*Nguồn\s+\d[^\]]*\](?:[ \t]*(?:,|;|và|and)?[ \t]*\[\s*Nguồn\s+\d[^\]]*\])*/giu;
const LEFTOVER = new Set(["", "nguồn", "nguồn tham khảo", "tài liệu tham khảo"]);

export function stripCitations(text) {
  if (!text) return text;
  const lines = [];
  for (const line of text.normalize("NFC").split("\n")) {
    const cleaned = line.replace(CITATIONS, "");
    if (cleaned !== line && LEFTOVER.has(cleaned.replace(/^[\s\-*•_:.,;]+|[\s\-*•_:.,;]+$/g, "").toLowerCase())) continue;
    lines.push(cleaned.trimEnd());
  }
  return lines.join("\n").replace(/\n{3,}/g, "\n\n").trim() || text.trim();
}

// Ngày theo giờ máy người xem, dạng YYYY-MM-DD, để gom lượt dùng theo ngày.
export function dayKey(value) {
  const date = toDate(value);
  if (!date) return null;
  const pad = (n) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}
