const $ = (id) => document.getElementById(id);

const tbody = $("reviewTableBody");
const searchInput = $("searchInput");
const selectAll = $("selectAll");
const trainBtn = $("trainBtn");
const trainLabel = $("trainLabel");
const barStatus = $("barStatus");

let reviews = [];
let currentFilter = "all";
const selectedIds = new Set();
// Correct answers being typed survive re-renders (filter, search, refresh).
const drafts = new Map();

/* =========================================================
   LOAD
   ========================================================= */

async function loadReviews() {
  tbody.innerHTML = '<tr><td colspan="8" class="empty">Đang tải phản hồi…</td></tr>';

  try {
    const response = await fetch("/admin/reviews", { cache: "no-store" });
    if (!response.ok) throw new Error(await response.text());
    reviews = await response.json();

    // Keep selections only if the record still exists.
    const existing = new Set(reviews.map((item) => item._id));
    for (const id of [...selectedIds]) if (!existing.has(id)) selectedIds.delete(id);

    updateStats();
    renderTable();
    updateSelectedInfo();
  } catch (error) {
    console.error(error);
    tbody.innerHTML = '<tr><td colspan="8" class="empty is-error">Không tải được /admin/reviews. Kiểm tra MongoDB rồi bấm Tải lại.</td></tr>';
  }
}

async function loadStatus() {
  try {
    const response = await fetch("/admin/status", { cache: "no-store" });
    if (!response.ok) return;
    const status = await response.json();
    $("backendNote").hidden = status.feedback_examples_used !== false;
  } catch {}
}

function updateStats() {
  $("totalCount").textContent = reviews.length;
  $("likeCount").textContent = reviews.filter((x) => x.user_feedback?.rating === "like").length;
  $("unlikeCount").textContent = reviews.filter((x) => x.user_feedback?.rating === "unlike").length;
  $("trainingCount").textContent = reviews.filter((x) => x.training?.selected === true).length;
}

/* =========================================================
   FILTER / SEARCH / SELECT
   ========================================================= */

document.querySelectorAll(".filter-btn").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".filter-btn").forEach((x) => x.setAttribute("aria-pressed", "false"));
    button.setAttribute("aria-pressed", "true");
    currentFilter = button.dataset.filter;
    renderTable();
  });
});

searchInput.addEventListener("input", renderTable);

selectAll.addEventListener("change", () => {
  for (const item of getFilteredReviews()) {
    if (selectAll.checked) selectedIds.add(item._id);
    else selectedIds.delete(item._id);
  }
  renderTable();
  updateSelectedInfo();
});

function getFilteredReviews() {
  const search = searchInput.value.trim().toLowerCase();

  return reviews.filter((item) => {
    const rating = item.user_feedback?.rating;
    if (currentFilter === "like" && rating !== "like") return false;
    if (currentFilter === "unlike" && rating !== "unlike") return false;
    if (currentFilter === "training" && item.training?.selected !== true) return false;

    if (search) {
      const text = `${item.question || ""} ${item.answer || ""} ${(item.user_feedback?.reasons || []).join(" ")}`.toLowerCase();
      if (!text.includes(search)) return false;
    }
    return true;
  });
}

/* =========================================================
   TABLE
   ========================================================= */

function renderTable() {
  const data = getFilteredReviews();

  if (!data.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="empty">${
      reviews.length ? "Không có phản hồi khớp bộ lọc." : "Chưa có phản hồi nào. Khi người dùng bấm hữu ích / chưa tốt, câu trả lời sẽ hiện ở đây."
    }</td></tr>`;
    selectAll.checked = false;
    return;
  }

  tbody.innerHTML = data.map(createRow).join("");

  tbody.querySelectorAll(".row-check").forEach((box) => {
    box.addEventListener("change", () => toggleSelection(box.dataset.id, box.checked));
  });
  tbody.querySelectorAll(".correct-answer").forEach((textarea) => {
    textarea.addEventListener("input", () => {
      drafts.set(textarea.dataset.id, textarea.value);
      textarea.removeAttribute("aria-invalid");
    });
  });
  tbody.querySelectorAll(".delete-btn").forEach((button) => {
    button.addEventListener("click", () => deleteQA(button.dataset.id));
  });

  selectAll.checked = data.every((item) => selectedIds.has(item._id));
}

function createRow(item) {
  const id = escapeHtml(item._id);
  const rating = item.user_feedback?.rating || "";
  const reasons = item.user_feedback?.reasons || [];
  const meta = item.metadata || {};
  const isTraining = item.training?.selected === true;
  const correct = drafts.has(item._id) ? drafts.get(item._id) : (item.admin_feedback?.correct_answer || "");

  const feedbackHtml = rating === "like"
    ? '<span class="badge badge-like"><svg class="icon" aria-hidden="true"><use href="#i-thumb-up"/></svg>Hữu ích</span>'
    : '<span class="badge badge-unlike"><svg class="icon" aria-hidden="true"><use href="#i-thumb-down"/></svg>Chưa tốt</span>';

  const reasonHtml = reasons.length
    ? `<div class="reason-list">${reasons.map((r) => `<span class="reason-chip">${escapeHtml(r)}</span>`).join("")}</div>`
    : '<span class="muted">—</span>';

  const subject = [meta.plant, meta.disease].filter(Boolean).map(formatLabel).join(" · ");

  // Documents the RAG answer was grounded on (saved by the pipeline in metadata).
  const sources = meta.sources || [];
  let sourcesHtml = "";
  if (sources.length) {
    sourcesHtml = `<div class="sources">${sources.map((s) =>
      `<span><svg class="icon" aria-hidden="true"><use href="#i-doc"/></svg>${escapeHtml(s)}</span>`).join("")}</div>`;
  } else if (meta.grounded === false) {
    sourcesHtml = '<div class="sources"><span class="is-warn"><svg class="icon" aria-hidden="true"><use href="#i-alert"/></svg>Kho chưa có tài liệu cho câu này</span></div>';
  }

  return `
    <tr class="${selectedIds.has(item._id) ? "is-selected" : ""}">
      <td class="c-select">
        <input class="row-check" type="checkbox" data-id="${id}" aria-label="Chọn QA này" ${selectedIds.has(item._id) ? "checked" : ""}>
      </td>
      <td>
        ${feedbackHtml}
        ${isTraining ? '<span class="rag-badge"><svg class="icon" aria-hidden="true"><use href="#i-stack"/></svg>trong RAG</span>' : ""}
      </td>
      <td>
        <div class="question">${escapeHtml(item.question || "—")}</div>
        ${subject ? `<div class="subject">${escapeHtml(subject)}</div>` : ""}
      </td>
      <td>
        <div class="answer">${escapeHtml(item.answer || "—")}</div>
        ${sourcesHtml}
      </td>
      <td>${reasonHtml}</td>
      <td>
        <label class="visually-hidden" for="correct-${id}">Câu trả lời đúng</label>
        <textarea class="correct-answer" id="correct-${id}" data-id="${id}" placeholder="${
          rating === "unlike" ? "Bắt buộc khi câu trả lời bị chê…" : "Để trống nếu câu trả lời của bot đã tốt…"
        }">${escapeHtml(correct)}</textarea>
      </td>
      <td class="date">${formatDate(item.created_at)}</td>
      <td class="c-action">
        <button type="button" class="delete-btn" data-id="${id}" aria-label="Xóa QA này" title="Xóa QA">
          <svg class="icon" aria-hidden="true"><use href="#i-trash"/></svg>
        </button>
      </td>
    </tr>
  `;
}

function toggleSelection(id, checked) {
  if (checked) selectedIds.add(id);
  else selectedIds.delete(id);

  const box = tbody.querySelector(`.row-check[data-id="${CSS.escape(id)}"]`);
  box?.closest("tr")?.classList.toggle("is-selected", checked);

  const visible = getFilteredReviews();
  selectAll.checked = visible.length > 0 && visible.every((item) => selectedIds.has(item._id));
  updateSelectedInfo();
}

function updateSelectedInfo() {
  $("selectedCount").textContent = selectedIds.size;
  trainBtn.disabled = selectedIds.size === 0 || trainBtn.dataset.state === "loading";
}

/* =========================================================
   ACTIONS
   ========================================================= */

async function deleteQA(id) {
  const item = reviews.find((x) => x._id === id);
  const shortQuestion = item?.question ? `\n\n${item.question.slice(0, 100)}` : "";
  if (!confirm(`Xóa QA này khỏi feedback và Feedback RAG?${shortQuestion}`)) return;

  try {
    const response = await fetch(`/admin/review/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (!response.ok) throw new Error(await errorText(response));
    selectedIds.delete(id);
    drafts.delete(id);
    reviews = reviews.filter((x) => x._id !== id);
    updateStats();
    renderTable();
    updateSelectedInfo();
    say("Đã xóa QA.", "ok");
  } catch (error) {
    console.error(error);
    say(`Xóa QA thất bại: ${error.message}`, "error");
  }
}

trainBtn.addEventListener("click", async () => {
  if (!selectedIds.size) return;

  const items = [];
  for (const id of selectedIds) {
    const review = reviews.find((x) => x._id === id);
    if (!review) continue;

    const saved = review.admin_feedback?.correct_answer || "";
    const correctAnswer = (drafts.has(id) ? drafts.get(id) : saved).trim();

    if (review.user_feedback?.rating === "unlike" && !correctAnswer) {
      // The row may be hidden by the current filter: show everything, then point at it.
      if (!tbody.querySelector(`#correct-${CSS.escape(id)}`)) {
        currentFilter = "all";
        searchInput.value = "";
        document.querySelectorAll(".filter-btn").forEach((x) => x.setAttribute("aria-pressed", String(x.dataset.filter === "all")));
        renderTable();
      }
      const textarea = tbody.querySelector(`#correct-${CSS.escape(id)}`);
      textarea?.setAttribute("aria-invalid", "true");
      textarea?.focus();
      say("QA bị chê phải có câu trả lời đúng trước khi đưa vào Feedback RAG.", "error");
      return;
    }

    items.push({ id, correct_answer: correctAnswer });
  }
  if (!items.length) return;

  trainBtn.dataset.state = "loading";
  trainBtn.disabled = true;
  trainLabel.textContent = "Đang tạo embedding…";

  try {
    const response = await fetch("/admin/training", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items }),
    });
    if (!response.ok) throw new Error(await errorText(response));
    const data = await response.json();

    items.forEach(({ id }) => drafts.delete(id));
    selectedIds.clear();
    say(`Đã đưa ${data.indexed_count || data.trained_count || items.length} QA vào Feedback RAG.`, "ok");
    await loadReviews();
  } catch (error) {
    console.error(error);
    say(error.message, "error");
  } finally {
    delete trainBtn.dataset.state;
    trainLabel.textContent = "Đưa vào Feedback RAG";
    updateSelectedInfo();
  }
});

$("refreshBtn").addEventListener("click", loadReviews);

/* =========================================================
   UTIL
   ========================================================= */

let sayTimer;
function say(message, kind = "") {
  barStatus.textContent = message;
  barStatus.className = `bar-status${kind ? ` is-${kind}` : ""}`;
  clearTimeout(sayTimer);
  sayTimer = setTimeout(() => { barStatus.textContent = ""; }, 5000);
}

async function errorText(response) {
  const raw = await response.text();
  try { return JSON.parse(raw).detail || raw; } catch { return raw || `HTTP ${response.status}`; }
}

function formatLabel(value) {
  // Same as the chat page: "Tomato___Late_blight" -> "Tomato Late blight".
  return String(value).replace(/_+/g, " ").trim();
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString("vi-VN", {
    day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

updateSelectedInfo();
loadStatus();
loadReviews();
