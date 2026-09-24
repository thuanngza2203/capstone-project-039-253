# Design — PlantGPT (detection-server-module web UI)

A locked design system for the two pages in `app/static/`: the chat (`index.html`)
and the feedback review (`admin_review.html`). Read this before changing either page;
amend it here instead of restyling a page locally.

Source: the 2026-09-17 Hallmark redesign of `plant_disease_chatbot` ("field notebook"),
ported unchanged for the chat page on 2026-09-23. The admin page was rebuilt as a
table in the same system.

## Genre
Editorial, applied to an app. Tone: field notebook.

## Macrostructure family
- Chat: Workbench app shell — N3 side rail (brand, "Cuộc chat mới", conversation list,
  "Kết quả gần đây" specimen card), drawer below 52rem; the composer closes the page.
- Admin: same brand block in a top bar, page title, one hairline stats strip, filter
  pills + search, **a table** (select · đánh giá · câu hỏi · bot trả lời + tài liệu RAG ·
  lý do · câu trả lời đúng · thời gian · xóa), sticky bottom action bar. On narrow
  screens the table scrolls horizontally inside its frame; the page itself never does.

## Theme — Garden
Tokens: `app/static/tokens.css` (shared by `styles.css` and `admin.css`).
Warm herbarium paper `oklch(97.5% 0.010 105)`, ink `oklch(24% 0.028 142)`, one leaf-green
accent `oklch(47% 0.110 146)` for state, focus and small marks. Error `oklch(51% 0.150 35)`,
warn `oklch(49% 0.090 75)`.

## Typography
Fraunces 700 (display, opsz), Be Vietnam Pro 400/600 (body), IBM Plex Mono 400/500
(data: dates, counts, file paths, labels). Small uppercase labels at 0.08em tracking
for section/column labels only. No italic headings.

## CTA voice
- Primary: ink-filled, `--radius-md`, one per view (Send; "Đưa vào Feedback RAG").
- Secondary: bordered sheet button (Cuộc chat mới) or ghost (top-bar actions).
- Filters: pills, pressed = ink fill.

## Icons
One hand-drawn stroke sprite: 24px grid, 1.75 stroke, round joins. No emoji, no icon library.

## Motion
`--ease-out` / `--ease-in` / `--ease-in-out`; Garden runs calm (140 / 260 / 480ms).
Reduced motion collapses to ≤150ms.

## Exports
### tokens.css
Canonical file: `app/static/tokens.css`.
