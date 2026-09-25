import { useEffect, useState } from "react";
import { rag } from "../api.js";
import { sourceFallback } from "./labels.js";

// Tiêu đề tài liệu viết hoa toàn bộ ("BỆNH THỐI ĐEN TRÊN CÂY TÁO") → viết hoa chữ đầu.
export function niceTitle(title) {
  if (!title) return title;
  const letters = title.replace(/[^\p{L}]/gu, "");
  if (letters && letters === letters.toLocaleUpperCase("vi")) {
    const lower = title.toLocaleLowerCase("vi");
    return lower.charAt(0).toLocaleUpperCase("vi") + lower.slice(1);
  }
  return title;
}

let cached = null;
let pending = null;

// Tên tài liệu theo `source` (lấy một lần từ RAG, dùng chung cho mọi trang).
export function useDocTitles() {
  const [titles, setTitles] = useState(cached || {});
  useEffect(() => {
    if (cached) return undefined;
    let alive = true;
    if (!pending) {
      pending = rag
        .documents()
        .then((rows) => {
          cached = Object.fromEntries(rows.map((row) => [row.source, niceTitle(row.title)]));
          return cached;
        })
        .catch(() => {
          pending = null; // RAG chưa chạy: lần sau thử lại, tạm hiện tên file.
          return {};
        });
    }
    pending.then((value) => alive && setTitles(value));
    return () => {
      alive = false;
    };
  }, []);
  return (source) => titles[source] || sourceFallback(source);
}
