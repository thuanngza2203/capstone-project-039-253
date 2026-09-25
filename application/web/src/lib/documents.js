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
