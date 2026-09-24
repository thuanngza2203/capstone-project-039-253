const HEADER_LABELS = ["Tài liệu:", "Tên tra cứu:", "Mục:"];

// Tách header định danh khỏi thân chunk để xem trước cho gọn (giống server/knowledge.py).
export function split(content = "") {
  const index = content.indexOf("\n\n");
  if (index > 0) {
    const head = content.slice(0, index);
    if (head.split("\n").every((line) => HEADER_LABELS.some((label) => line.startsWith(label)))) {
      return { header: head, body: content.slice(index + 2) };
    }
  }
  return { header: "", body: content };
}
