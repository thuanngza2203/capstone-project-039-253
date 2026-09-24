const $ = (id) => document.getElementById(id);

const appShell = $("appShell");
const sidebar = $("sidebar");
const main = $("main");
const menuBtn = $("menuBtn");
const drawerBackdrop = $("drawerBackdrop");
const topbarTitle = $("topbarTitle");
const chatArea = $("chatArea");
const chatForm = $("chatForm");
const messageInput = $("messageInput");
const imageInput = $("imageInput");
const attachBtn = $("attachBtn");
const sendBtn = $("sendBtn");
const attachmentPreview = $("attachmentPreview");
const previewImage = $("previewImage");
const previewName = $("previewName");
const removeImageBtn = $("removeImageBtn");
const newChatBtn = $("newChatBtn");
const newChatMobileBtn = $("newChatMobileBtn");
const conversationList = $("conversationList");
const memoryCard = $("memoryCard");
const memoryPlant = $("memoryPlant");
const memoryDisease = $("memoryDisease");
const memoryConfidence = $("memoryConfidence");
const memoryMeter = $("memoryMeter");
const welcomeTemplate = $("welcomeTemplate");

const NEW_CONVERSATION_TITLE = "Cuộc trò chuyện mới";

const FEEDBACK_REASONS = [
  "Chẩn đoán sai",
  "Không liên quan",
  "Thiếu thông tin",
  "Thông tin chưa chính xác",
  "Khó hiểu",
  "Khác",
];

let sessionId = localStorage.getItem("plant_session_id") || crypto.randomUUID();
let selectedFile = null;
let isLoading = false;
setSessionId(sessionId);

function setSessionId(value) {
  sessionId = value;
  localStorage.setItem("plant_session_id", sessionId);
}

function icon(name, className = "icon") {
  return `<svg class="${className}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
}

/* =========================================================
   MOBILE DRAWER
   ========================================================= */

const desktopQuery = window.matchMedia("(min-width: 52rem)");

function isDrawerOpen() {
  return appShell.dataset.drawer === "open";
}

function syncDrawerAccessibility() {
  const mobile = !desktopQuery.matches;
  // Off-canvas sidebar must not be reachable by Tab; the page behind an open drawer must not be either.
  sidebar.inert = mobile && !isDrawerOpen();
  main.inert = mobile && isDrawerOpen();
}

function openDrawer() {
  appShell.dataset.drawer = "open";
  menuBtn.setAttribute("aria-expanded", "true");
  syncDrawerAccessibility();
  newChatBtn.focus({ preventScroll: true });
}

function closeDrawer({ returnFocus = true } = {}) {
  if (!isDrawerOpen()) return;
  delete appShell.dataset.drawer;
  menuBtn.setAttribute("aria-expanded", "false");
  syncDrawerAccessibility();
  if (returnFocus && !desktopQuery.matches) menuBtn.focus({ preventScroll: true });
}

menuBtn.addEventListener("click", openDrawer);
drawerBackdrop.addEventListener("click", () => closeDrawer());
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeDrawer();
});
desktopQuery.addEventListener("change", () => {
  closeDrawer({ returnFocus: false });
  syncDrawerAccessibility();
});
syncDrawerAccessibility();

/* =========================================================
   WELCOME / EXAMPLES
   ========================================================= */

function bindExamples(scope) {
  scope.querySelectorAll("[data-example]").forEach((btn) => {
    btn.addEventListener("click", () => {
      messageInput.value = btn.dataset.example;
      autoResize();
      updateSendState();
      if (btn.hasAttribute("data-pick-image")) {
        imageInput.click();
      } else {
        messageInput.focus();
      }
    });
  });
}

function showWelcome() {
  chatArea.innerHTML = "";
  const fragment = welcomeTemplate.content.cloneNode(true);
  bindExamples(fragment);
  chatArea.appendChild(fragment);
  setTopbarTitle(NEW_CONVERSATION_TITLE);
}

function hideWelcome() {
  document.querySelector(".welcome")?.remove();
}

function setTopbarTitle(title) {
  topbarTitle.textContent = title || NEW_CONVERSATION_TITLE;
}

/* =========================================================
   CONVERSATION HISTORY
   ========================================================= */

async function loadConversationList() {
  if (!conversationList.children.length || conversationList.querySelector(".conversation-empty")) {
    conversationList.innerHTML = '<div class="conversation-loading">Đang tải…</div>';
  }

  try {
    const response = await fetch("/api/conversations", { cache: "no-store" });
    if (!response.ok) throw new Error(await response.text());

    const conversations = await response.json();
    renderConversationList(conversations);
  } catch (error) {
    console.error("Load conversations error:", error);
    conversationList.innerHTML = '<div class="conversation-empty">Không tải được lịch sử.</div>';
  }
}

function renderConversationList(conversations) {
  if (!conversations.length) {
    conversationList.innerHTML = '<div class="conversation-empty">Chưa có cuộc trò chuyện.</div>';
    return;
  }

  conversationList.innerHTML = "";

  conversations.forEach((conversation) => {
    const isActive = conversation.session_id === sessionId;
    const row = document.createElement("div");
    row.className = `conversation-item ${isActive ? "active" : ""}`;
    row.dataset.sessionId = conversation.session_id;

    const mainBtn = document.createElement("button");
    mainBtn.type = "button";
    mainBtn.className = "conversation-item-main";
    if (isActive) mainBtn.setAttribute("aria-current", "true");

    const title = document.createElement("span");
    title.className = "conversation-title";
    title.textContent = conversation.title || NEW_CONVERSATION_TITLE;

    const preview = document.createElement("span");
    preview.className = "conversation-preview";
    preview.textContent = conversation.last_message || "";

    mainBtn.append(title, preview);

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "conversation-delete";
    deleteBtn.setAttribute("aria-label", `Xóa cuộc trò chuyện: ${title.textContent}`);
    deleteBtn.innerHTML = icon("close");

    mainBtn.addEventListener("click", () => {
      closeDrawer({ returnFocus: false });
      openConversation(conversation.session_id);
    });
    deleteBtn.addEventListener("click", async (event) => {
      event.stopPropagation();
      await deleteConversation(conversation.session_id);
    });

    row.append(mainBtn, deleteBtn);
    conversationList.appendChild(row);

    if (isActive) setTopbarTitle(title.textContent);
  });
}

async function openConversation(id) {
  setSessionId(id);
  setLoading(true);
  chatArea.innerHTML = '<div class="conversation-loading chat-loading">Đang tải cuộc trò chuyện…</div>';

  try {
    const response = await fetch(`/api/conversations/${encodeURIComponent(id)}`, { cache: "no-store" });
    if (!response.ok) throw new Error(await response.text());

    const conversation = await response.json();
    chatArea.innerHTML = "";

    const messages = conversation.messages || [];
    if (!messages.length) {
      showWelcome();
    } else {
      messages.forEach((message) => {
        addMessage(message.role, message.content, {
          imageUploaded: message.image_uploaded,
          imageFilename: message.image_filename,
          feedbackId: message.feedback_id,
          feedbackRating: message.feedback_rating,
          feedbackReasons: message.feedback_reasons || [],
          action: message.action,
          scroll: false,
        });
      });
      scrollBottom();
    }

    updateMemory(conversation.last_detection);
    await loadConversationList();
  } catch (error) {
    console.error("Open conversation error:", error);
    showWelcome();
    updateMemory(null);
  } finally {
    setLoading(false);
  }
}

async function deleteConversation(id) {
  if (!confirm("Xóa cuộc trò chuyện này?")) return;

  try {
    const response = await fetch(`/api/conversations/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    if (!response.ok) throw new Error(await response.text());

    if (id === sessionId) {
      setSessionId(crypto.randomUUID());
      showWelcome();
      updateMemory(null);
    }
    await loadConversationList();
  } catch (error) {
    console.error("Delete conversation error:", error);
    alert("Không xóa được cuộc trò chuyện.");
  }
}

async function startNewChat() {
  // Important: do NOT delete the old conversation. Start a new MongoDB session.
  closeDrawer({ returnFocus: false });
  setSessionId(crypto.randomUUID());
  clearAttachment();
  updateMemory(null);
  showWelcome();
  await loadConversationList();
  messageInput.focus();
}

newChatBtn.addEventListener("click", startNewChat);
newChatMobileBtn.addEventListener("click", startNewChat);

/* =========================================================
   IMAGE ATTACHMENT
   ========================================================= */

attachBtn.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", () => {
  selectedFile = imageInput.files[0] || null;
  renderAttachment();
  if (selectedFile) messageInput.focus();
});

removeImageBtn.addEventListener("click", () => {
  clearAttachment();
  messageInput.focus();
});

function renderAttachment() {
  if (previewImage.src.startsWith("blob:")) URL.revokeObjectURL(previewImage.src);

  if (!selectedFile) {
    attachmentPreview.classList.add("hidden");
    previewImage.removeAttribute("src");
    previewName.textContent = "";
    updateSendState();
    return;
  }

  previewImage.src = URL.createObjectURL(selectedFile);
  previewName.textContent = selectedFile.name;
  attachmentPreview.classList.remove("hidden");
  updateSendState();
}

function clearAttachment() {
  selectedFile = null;
  imageInput.value = "";
  renderAttachment();
}

/* =========================================================
   INPUT / SEND
   ========================================================= */

messageInput.addEventListener("input", () => {
  autoResize();
  updateSendState();
});
messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

function autoResize() {
  messageInput.style.height = "auto";
  messageInput.style.height = Math.min(messageInput.scrollHeight, 152) + "px";
}

function updateSendState() {
  sendBtn.disabled = isLoading || (!messageInput.value.trim() && !selectedFile);
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (isLoading) return;

  const message = messageInput.value.trim();
  if (!message && !selectedFile) return;

  hideWelcome();

  const effectiveQuestion = message || "Ảnh này đang bị bệnh gì?";
  const fileToSend = selectedFile;
  const imageUrl = fileToSend ? URL.createObjectURL(fileToSend) : null;

  addMessage("user", effectiveQuestion, { imageUrl, animate: true });

  messageInput.value = "";
  autoResize();
  clearAttachment();

  const typingId = addTyping(Boolean(fileToSend));
  setLoading(true);

  try {
    const form = new FormData();
    form.append("session_id", sessionId);
    form.append("message", message);
    if (fileToSend) form.append("image", fileToSend);

    const response = await fetch("/api/chat", {
      method: "POST",
      body: form,
    });

    const raw = await response.text();
    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      throw new Error(`Backend ${response.status}: ${raw}`);
    }

    if (!response.ok) {
      const error = new Error(data.detail || `HTTP ${response.status}`);
      // 400 (ví dụ ảnh không phải lá) và 503 (dịch vụ tra cứu tạm lỗi) mang thông báo
      // tiếng Việt viết cho người dùng: hiện nguyên văn. 500 vẫn dùng câu chung.
      if (data.detail && (response.status < 500 || response.status === 503)) {
        error.userMessage = data.detail;
      }
      throw error;
    }

    // Keep browser session aligned with the session persisted by the backend.
    if (data.session_id && data.session_id !== sessionId) {
      setSessionId(data.session_id);
    }

    removeTyping(typingId);
    addMessage("assistant", data.answer, {
      action: data.action,
      feedbackId: data.feedback_id,
      animate: true,
      live: true,
    });

    updateMemory(data.memory);
    // Update title/history after every new turn.
    await loadConversationList();
  } catch (error) {
    removeTyping(typingId);
    console.error("Chat request error:", error);
    addMessage("assistant", error.userMessage || "Mình chưa thể xử lý yêu cầu lúc này. Bạn thử lại sau nhé.", {
      action: "ERROR",
      animate: true,
    });
  } finally {
    setLoading(false);
    messageInput.focus({ preventScroll: true });
  }
});

/* =========================================================
   MESSAGE RENDERING
   ========================================================= */

function addMessage(role, content, options = {}) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;
  if (options.animate) row.classList.add("is-new");
  if (options.action === "ERROR") row.classList.add("is-error");

  const inner = document.createElement("div");
  inner.className = "message-inner";

  if (role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.innerHTML = icon("leaf");
    inner.appendChild(avatar);
  }

  const body = document.createElement("div");
  body.className = "message-content";

  if (options.imageUrl) {
    const img = document.createElement("img");
    img.className = "user-image";
    img.src = options.imageUrl;
    img.alt = "Ảnh lá cây đã gửi";
    body.appendChild(img);
  } else if (role === "user" && options.imageUploaded) {
    const note = document.createElement("div");
    note.className = "image-history-note";
    note.innerHTML = icon("image");
    const name = document.createElement("span");
    name.textContent = options.imageFilename || "Ảnh đã upload";
    note.appendChild(name);
    body.appendChild(note);
  }

  const textWrap = document.createElement("div");
  textWrap.className = "message-text";
  textWrap.innerHTML = options.action === "ERROR"
    ? `${icon("alert")}<p>${escapeHtml(content)}</p>`
    : renderText(content);
  body.appendChild(textWrap);

  // When the assistant asks for a photo, offer the picker right under the question.
  if (role === "assistant" && options.live && options.action === "REQUEST_IMAGE") {
    const pick = document.createElement("button");
    pick.type = "button";
    pick.className = "inline-action";
    pick.innerHTML = `${icon("camera")}<span>Gửi ảnh lá</span>`;
    pick.addEventListener("click", () => imageInput.click());
    body.appendChild(pick);
  }

  // Only real assistant responses with a valid MongoDB feedback id can be rated.
  if (role === "assistant" && options.feedbackId) {
    body.appendChild(
      createFeedbackBox(
        options.feedbackId,
        options.feedbackRating || null,
        options.feedbackReasons || [],
      ),
    );
  }

  inner.appendChild(body);
  row.appendChild(inner);
  chatArea.appendChild(row);

  if (options.scroll !== false) scrollBottom();
  return row;
}

/* =========================================================
   FEEDBACK
   ========================================================= */

function createFeedbackBox(feedbackId, existingRating = null, existingReasons = []) {
  const feedback = document.createElement("div");
  feedback.className = "feedback-box";

  const top = document.createElement("div");
  top.className = "feedback-top";

  const label = document.createElement("span");
  label.className = "feedback-label";
  label.textContent = "Câu trả lời có hữu ích không?";

  const likeBtn = document.createElement("button");
  likeBtn.type = "button";
  likeBtn.className = "feedback-btn like-btn";
  likeBtn.innerHTML = icon("thumb-up");
  likeBtn.setAttribute("aria-label", "Hữu ích");
  likeBtn.setAttribute("aria-pressed", "false");

  const unlikeBtn = document.createElement("button");
  unlikeBtn.type = "button";
  unlikeBtn.className = "feedback-btn unlike-btn";
  unlikeBtn.innerHTML = icon("thumb-down");
  unlikeBtn.setAttribute("aria-label", "Không hữu ích");
  unlikeBtn.setAttribute("aria-pressed", "false");

  top.append(label, likeBtn, unlikeBtn);

  const reasonsPanel = document.createElement("div");
  reasonsPanel.className = "feedback-reasons hidden";

  const reasonsTitle = document.createElement("div");
  reasonsTitle.className = "feedback-reasons-title";
  reasonsTitle.textContent = "Vì sao câu trả lời chưa tốt?";

  const reasonList = document.createElement("div");
  reasonList.className = "feedback-reason-list";

  FEEDBACK_REASONS.forEach((reason) => {
    const item = document.createElement("label");
    item.className = "feedback-reason-item";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = reason;
    checkbox.checked = existingReasons.includes(reason);

    const text = document.createElement("span");
    text.textContent = reason;

    item.append(checkbox, text);
    reasonList.appendChild(item);
  });

  const submitBtn = document.createElement("button");
  submitBtn.type = "button";
  submitBtn.className = "feedback-submit";
  submitBtn.textContent = "Gửi phản hồi";

  const status = document.createElement("div");
  status.className = "feedback-status";
  status.setAttribute("role", "status");

  reasonsPanel.append(reasonsTitle, reasonList, submitBtn);
  feedback.append(top, reasonsPanel, status);

  function markSelected(rating) {
    likeBtn.classList.toggle("selected", rating === "like");
    unlikeBtn.classList.toggle("selected", rating === "unlike");
    likeBtn.setAttribute("aria-pressed", String(rating === "like"));
    unlikeBtn.setAttribute("aria-pressed", String(rating === "unlike"));
  }

  if (existingRating === "like") {
    markSelected("like");
    setFeedbackStatus(status, "Đã lưu: hữu ích", "success");
  } else if (existingRating === "unlike") {
    markSelected("unlike");
    setFeedbackStatus(status, "Đã lưu: chưa hữu ích", "success");
  }

  likeBtn.addEventListener("click", async () => {
    setFeedbackBusy(true);
    setFeedbackStatus(status, "Đang lưu…", "");

    const result = await sendFeedback(feedbackId, "like", []);
    setFeedbackBusy(false);

    if (!result.ok) {
      setFeedbackStatus(status, result.message, "error");
      return;
    }

    markSelected("like");
    reasonsPanel.classList.add("hidden");
    clearReasonChecks(reasonsPanel);
    setFeedbackStatus(status, "Đã lưu: hữu ích", "success");
  });

  unlikeBtn.addEventListener("click", () => {
    // Dislike is not saved until a reason is selected and submitted.
    markSelected("unlike");
    reasonsPanel.classList.remove("hidden");
    setFeedbackStatus(status, "Chọn ít nhất một lý do rồi bấm Gửi phản hồi.", "warning");
    reasonsPanel.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });

  submitBtn.addEventListener("click", async () => {
    const reasons = getSelectedReasons(reasonsPanel);
    if (!reasons.length) {
      setFeedbackStatus(status, "Bạn cần chọn ít nhất một lý do.", "warning");
      return;
    }

    setFeedbackBusy(true);
    setFeedbackStatus(status, "Đang lưu…", "");

    const result = await sendFeedback(feedbackId, "unlike", reasons);
    setFeedbackBusy(false);

    if (!result.ok) {
      setFeedbackStatus(status, result.message, "error");
      return;
    }

    markSelected("unlike");
    reasonsPanel.classList.add("hidden");
    setFeedbackStatus(status, "Đã lưu: chưa hữu ích", "success");
  });

  function setFeedbackBusy(value) {
    likeBtn.disabled = value;
    unlikeBtn.disabled = value;
    submitBtn.disabled = value;
  }

  return feedback;
}

function getSelectedReasons(panel) {
  return Array.from(panel.querySelectorAll('input[type="checkbox"]:checked')).map(
    (input) => input.value,
  );
}

function clearReasonChecks(panel) {
  panel.querySelectorAll('input[type="checkbox"]').forEach((input) => {
    input.checked = false;
  });
}

function setFeedbackStatus(element, message, type = "") {
  element.textContent = message;
  element.className = "feedback-status";
  if (type) element.classList.add(type);
}

async function sendFeedback(feedbackId, rating, reasons = []) {
  if (!feedbackId) {
    return { ok: false, message: "Thiếu feedback_id. Hãy reload trang và thử lại." };
  }

  try {
    const response = await fetch("/feedback", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        feedback_id: feedbackId,
        rating,
        reasons,
      }),
    });

    const raw = await response.text();
    let data = {};
    try {
      data = raw ? JSON.parse(raw) : {};
    } catch {}

    if (!response.ok) {
      console.error("Feedback error:", response.status, raw);
      return {
        ok: false,
        message: data.detail || `Không lưu được phản hồi (${response.status}).`,
      };
    }

    return { ok: true, data };
  } catch (error) {
    console.error("Feedback request error:", error);
    return { ok: false, message: "Không kết nối được backend feedback." };
  }
}

/* =========================================================
   TEXT — a small, escape-first Markdown subset
   (headings, bold/italic/code, bullet and numbered lists, tables, rules)
   ========================================================= */

const RE_TABLE_ROW = /^\|.*\|$/;
const RE_TABLE_SEPARATOR = /^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$/;
const RE_HEADING = /^#{1,6}\s+(.*)$/;
const RE_RULE = /^(-{3,}|\*{3,}|_{3,})$/;
const RE_BULLET = /^[-*•+]\s+(.*)$/;
const RE_NUMBERED = /^(\d+)[.)]\s+(.*)$/;

function renderText(text) {
  const lines = String(text || "").replace(/\r\n?/g, "\n").split("\n");
  const html = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i].trim();

    if (!line) {
      i += 1;
      continue;
    }

    if (RE_TABLE_ROW.test(line) && RE_TABLE_SEPARATOR.test((lines[i + 1] || "").trim())) {
      const header = splitTableRow(line);
      const rows = [];
      i += 2;
      while (i < lines.length && RE_TABLE_ROW.test(lines[i].trim())) {
        rows.push(splitTableRow(lines[i].trim()));
        i += 1;
      }
      html.push(renderTable(header, rows));
      continue;
    }

    const heading = line.match(RE_HEADING);
    if (heading) {
      html.push(`<h3>${renderInline(heading[1])}</h3>`);
      i += 1;
      continue;
    }

    if (RE_RULE.test(line)) {
      html.push("<hr>");
      i += 1;
      continue;
    }

    if (RE_BULLET.test(line)) {
      const items = [];
      while (i < lines.length && RE_BULLET.test(lines[i].trim())) {
        items.push(`<li>${renderInline(lines[i].trim().match(RE_BULLET)[1])}</li>`);
        i += 1;
      }
      html.push(`<ul>${items.join("")}</ul>`);
      continue;
    }

    const numbered = line.match(RE_NUMBERED);
    if (numbered) {
      const start = Number(numbered[1]);
      const items = [];
      while (i < lines.length && RE_NUMBERED.test(lines[i].trim())) {
        items.push(`<li>${renderInline(lines[i].trim().match(RE_NUMBERED)[2])}</li>`);
        i += 1;
      }
      html.push(`<ol${start !== 1 ? ` start="${start}"` : ""}>${items.join("")}</ol>`);
      continue;
    }

    html.push(`<p>${renderInline(line)}</p>`);
    i += 1;
  }

  return html.join("");
}

function renderInline(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*(?!\s)([^*]+?)\*(?!\*)/g, "$1<em>$2</em>");
}

function splitTableRow(row) {
  return row.replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
}

function renderTable(header, rows) {
  const head = header.map((cell) => `<th scope="col">${renderInline(cell)}</th>`).join("");
  const body = rows
    .map((row) => `<tr>${header.map((_, index) => `<td>${renderInline(row[index] || "")}</td>`).join("")}</tr>`)
    .join("");
  return `<div class="table-scroll"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

/* =========================================================
   TYPING / LOADING / MEMORY
   ========================================================= */

function addTyping(withImage = false) {
  const id = `typing-${Date.now()}`;
  const label = withImage ? "Đang phân tích ảnh lá…" : "Đang soạn câu trả lời…";
  const row = document.createElement("div");
  row.className = "message-row assistant is-new";
  row.id = id;
  row.innerHTML = `
    <div class="message-inner">
      <div class="avatar">${icon("leaf")}</div>
      <div class="message-content">
        <div class="typing" role="status">
          <span class="typing-dots" aria-hidden="true"><span></span><span></span><span></span></span>
          <span>${label}</span>
        </div>
      </div>
    </div>
  `;
  chatArea.appendChild(row);
  scrollBottom();
  return id;
}

function removeTyping(id) {
  document.getElementById(id)?.remove();
}

function setLoading(value) {
  isLoading = value;
  attachBtn.disabled = value;
  updateSendState();
}

function formatLabel(value) {
  // Detector class names arrive as e.g. "Early_blight" or "Tomato___Late_blight".
  return String(value).replace(/_+/g, " ").trim();
}

function updateMemory(memory) {
  if (!memory) {
    memoryCard.classList.add("is-empty");
    memoryPlant.textContent = "Chưa có";
    memoryDisease.textContent = "Chưa có";
    memoryConfidence.textContent = "—";
    memoryMeter.style.transform = "scaleX(0)";
    return;
  }

  memoryCard.classList.remove("is-empty");
  memoryPlant.textContent = memory.plant ? formatLabel(memory.plant) : "—";
  memoryDisease.textContent = memory.disease ? formatLabel(memory.disease) : "—";

  if (memory.confidence == null) {
    memoryConfidence.textContent = "—";
    memoryMeter.style.transform = "scaleX(0)";
  } else {
    const ratio = Math.min(Math.max(Number(memory.confidence), 0), 1);
    memoryConfidence.textContent = `${Math.round(ratio * 100)}%`;
    memoryMeter.style.transform = `scaleX(${ratio})`;
  }
}

function scrollBottom() {
  requestAnimationFrame(() => {
    chatArea.scrollTop = chatArea.scrollHeight;
  });
}

/* =========================================================
   STARTUP
   ========================================================= */

(async function init() {
  await loadConversationList();

  try {
    const response = await fetch(`/api/conversations/${encodeURIComponent(sessionId)}`);
    if (response.ok) {
      await openConversation(sessionId);
      return;
    }
  } catch (error) {
    console.error("Restore conversation error:", error);
  }

  showWelcome();
  updateMemory(null);
})();
