const $ = (id) => document.getElementById(id);

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
const conversationList = $("conversationList");
const memoryPlant = $("memoryPlant");
const memoryDisease = $("memoryDisease");
const memoryConfidence = $("memoryConfidence");

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
setSessionId(sessionId);

function setSessionId(value) {
  sessionId = value;
  localStorage.setItem("plant_session_id", sessionId);
}

/* =========================================================
   WELCOME / EXAMPLES
   ========================================================= */

function welcomeTemplate() {
  const wrap = document.createElement("div");
  wrap.className = "welcome";
  wrap.id = "welcome";
  wrap.innerHTML = `
    <div class="welcome-icon">🌱</div>
    <h1>Hỏi mình về bệnh cây</h1>
    <div class="suggestions">
      <button data-example="bệnh này là bệnh gì?">
        <span>🔍</span>
        <div><strong>Nhận diện bệnh trên lá cây</strong><small>Gửi ảnh lá cây để bắt đầu</small></div>
      </button>
      <button data-example="bệnh đốm lá trên cây cà chua chữa sao?">
        <span>💬</span>
        <div><strong>Cách chữa bệnh trên cà chua</strong><small>Hỏi hướng xử lý cụ thể</small></div>
      </button>
      <button data-example="cà chua nhà tui bị đóm nâu chữa s?">
        <span>🍅</span>
        <div><strong>Lá cây có đốm nâu</strong><small>Mô tả triệu chứng bạn nhìn thấy</small></div>
      </button>
      <button data-example="bệnh này chữa s?">
        <span>🧠</span>
        <div><strong>Cách phòng bệnh này</strong><small>Tiếp tục cuộc trò chuyện</small></div>
      </button>
    </div>
  `;
  bindExamples(wrap);
  return wrap;
}

function bindExamples(scope = document) {
  scope.querySelectorAll("[data-example]").forEach((btn) => {
    btn.addEventListener("click", () => {
      messageInput.value = btn.dataset.example;
      autoResize();
      messageInput.focus();
    });
  });
}

bindExamples();

function showWelcome() {
  chatArea.innerHTML = "";
  chatArea.appendChild(welcomeTemplate());
}

function hideWelcome() {
  document.querySelector(".welcome")?.remove();
}

/* =========================================================
   CONVERSATION HISTORY
   ========================================================= */

async function loadConversationList() {
  conversationList.innerHTML = '<div class="conversation-loading">Đang tải...</div>';

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
    const row = document.createElement("div");
    row.className = `conversation-item ${conversation.session_id === sessionId ? "active" : ""}`;
    row.dataset.sessionId = conversation.session_id;

    const main = document.createElement("div");
    main.className = "conversation-item-main";

    const title = document.createElement("span");
    title.className = "conversation-title";
    title.textContent = conversation.title || "Cuộc trò chuyện mới";

    const preview = document.createElement("span");
    preview.className = "conversation-preview";
    preview.textContent = conversation.last_message || "";

    main.appendChild(title);
    main.appendChild(preview);

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "conversation-delete";
    deleteBtn.title = "Xóa cuộc trò chuyện";
    deleteBtn.textContent = "×";

    main.addEventListener("click", () => openConversation(conversation.session_id));
    deleteBtn.addEventListener("click", async (event) => {
      event.stopPropagation();
      await deleteConversation(conversation.session_id);
    });

    row.appendChild(main);
    row.appendChild(deleteBtn);
    conversationList.appendChild(row);
  });
}

async function openConversation(id) {
  setSessionId(id);
  setLoading(true);
  chatArea.innerHTML = '<div class="conversation-loading" style="text-align:center;padding:40px">Đang tải cuộc trò chuyện...</div>';

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

newChatBtn.addEventListener("click", async () => {
  // Important: do NOT delete the old conversation. Start a new MongoDB session.
  setSessionId(crypto.randomUUID());
  selectedFile = null;
  imageInput.value = "";
  renderAttachment();
  updateMemory(null);
  showWelcome();
  await loadConversationList();
  messageInput.focus();
});

/* =========================================================
   IMAGE ATTACHMENT
   ========================================================= */

attachBtn.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", () => {
  selectedFile = imageInput.files[0] || null;
  renderAttachment();
});

removeImageBtn.addEventListener("click", clearAttachment);

function renderAttachment() {
  if (!selectedFile) {
    attachmentPreview.classList.add("hidden");
    previewImage.src = "";
    previewName.textContent = "";
    return;
  }

  previewImage.src = URL.createObjectURL(selectedFile);
  previewName.textContent = selectedFile.name;
  attachmentPreview.classList.remove("hidden");
}

function clearAttachment() {
  selectedFile = null;
  imageInput.value = "";
  renderAttachment();
}

/* =========================================================
   INPUT / SEND
   ========================================================= */

messageInput.addEventListener("input", autoResize);
messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

function autoResize() {
  messageInput.style.height = "auto";
  messageInput.style.height = Math.min(messageInput.scrollHeight, 150) + "px";
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = messageInput.value.trim();
  if (!message && !selectedFile) return;

  hideWelcome();

  const effectiveQuestion = message || "Ảnh này đang bị bệnh gì?";
  const imageUrl = selectedFile ? URL.createObjectURL(selectedFile) : null;

  addMessage("user", effectiveQuestion, { imageUrl });

  const fileToSend = selectedFile;
  messageInput.value = "";
  autoResize();
  clearAttachment();

  const typingId = addTyping();
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

    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);

    // Keep browser session aligned with the session persisted by the backend.
    if (data.session_id && data.session_id !== sessionId) {
      setSessionId(data.session_id);
    }

    removeTyping(typingId);
    addMessage("assistant", data.answer, {
      action: data.action,
      feedbackId: data.feedback_id,
    });

    updateMemory(data.memory);
    // Update title/history after every new turn.
    await loadConversationList();
  } catch (error) {
    removeTyping(typingId);
    console.error("Chat request error:", error);
    addMessage("assistant", "Mình chưa thể xử lý yêu cầu lúc này. Bạn thử lại sau nhé.", {
      action: "ERROR",
    });
  } finally {
    setLoading(false);
  }
});

/* =========================================================
   MESSAGE RENDERING
   ========================================================= */

function addMessage(role, content, options = {}) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const inner = document.createElement("div");
  inner.className = "message-inner";

  if (role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "🌿";
    inner.appendChild(avatar);
  }

  const body = document.createElement("div");
  body.className = "message-content";

  if (options.imageUrl) {
    const img = document.createElement("img");
    img.className = "user-image";
    img.src = options.imageUrl;
    img.alt = "Ảnh lá cây";
    body.appendChild(img);
  } else if (role === "user" && options.imageUploaded) {
    const note = document.createElement("div");
    note.className = "image-history-note";
    note.textContent = `🖼️ ${options.imageFilename || "Ảnh đã upload"}`;
    body.appendChild(note);
  }

  const textWrap = document.createElement("div");
  textWrap.innerHTML = renderText(content);
  body.appendChild(textWrap);

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
  likeBtn.textContent = "👍";
  likeBtn.title = "Hữu ích";

  const unlikeBtn = document.createElement("button");
  unlikeBtn.type = "button";
  unlikeBtn.className = "feedback-btn unlike-btn";
  unlikeBtn.textContent = "👎";
  unlikeBtn.title = "Không hữu ích";

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

  reasonsPanel.append(reasonsTitle, reasonList, submitBtn);
  feedback.append(top, reasonsPanel, status);

  if (existingRating === "like") {
    likeBtn.classList.add("selected");
    setFeedbackStatus(status, "Đã lưu phản hồi 👍", "success");
  } else if (existingRating === "unlike") {
    unlikeBtn.classList.add("selected");
    setFeedbackStatus(status, "Đã lưu phản hồi 👎", "success");
  }

  likeBtn.addEventListener("click", async () => {
    setFeedbackBusy(true);
    setFeedbackStatus(status, "Đang lưu...", "");

    const result = await sendFeedback(feedbackId, "like", []);
    setFeedbackBusy(false);

    if (!result.ok) {
      setFeedbackStatus(status, result.message, "error");
      return;
    }

    likeBtn.classList.add("selected");
    unlikeBtn.classList.remove("selected");
    reasonsPanel.classList.add("hidden");
    clearReasonChecks(reasonsPanel);
    setFeedbackStatus(status, "Đã lưu phản hồi 👍", "success");
  });

  unlikeBtn.addEventListener("click", () => {
    // Dislike is not saved until a reason is selected and submitted.
    unlikeBtn.classList.add("selected");
    likeBtn.classList.remove("selected");
    reasonsPanel.classList.remove("hidden");
    setFeedbackStatus(status, "Chọn ít nhất một lý do rồi bấm Gửi phản hồi.", "warning");
    scrollBottom();
  });

  submitBtn.addEventListener("click", async () => {
    const reasons = getSelectedReasons(reasonsPanel);
    if (!reasons.length) {
      setFeedbackStatus(status, "Bạn cần chọn ít nhất một lý do.", "warning");
      return;
    }

    setFeedbackBusy(true);
    setFeedbackStatus(status, "Đang lưu...", "");

    const result = await sendFeedback(feedbackId, "unlike", reasons);
    setFeedbackBusy(false);

    if (!result.ok) {
      setFeedbackStatus(status, result.message, "error");
      return;
    }

    unlikeBtn.classList.add("selected");
    likeBtn.classList.remove("selected");
    reasonsPanel.classList.add("hidden");
    setFeedbackStatus(status, "Đã lưu phản hồi 👎", "success");
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
   TEXT / TYPING / MEMORY
   ========================================================= */

function renderText(text) {
  const escaped = escapeHtml(text || "");
  const bold = escaped.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  const lines = bold.split(/\n+/).filter(Boolean);
  return lines.map((line) => `<p>${line}</p>`).join("");
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function addTyping() {
  const id = `typing-${Date.now()}`;
  const row = document.createElement("div");
  row.className = "message-row assistant";
  row.id = id;
  row.innerHTML = `
    <div class="message-inner">
      <div class="avatar">🌿</div>
      <div class="message-content">
        <div class="typing"><span></span><span></span><span></span></div>
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
  sendBtn.disabled = value;
  attachBtn.disabled = value;
}

function updateMemory(memory) {
  if (!memory) {
    memoryPlant.textContent = "Chưa có";
    memoryDisease.textContent = "Chưa có";
    memoryConfidence.textContent = "—";
    return;
  }

  memoryPlant.textContent = memory.plant || "—";
  memoryDisease.textContent = memory.disease || "—";
  memoryConfidence.textContent =
    memory.confidence == null ? "—" : `${Math.round(memory.confidence * 100)}%`;
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
