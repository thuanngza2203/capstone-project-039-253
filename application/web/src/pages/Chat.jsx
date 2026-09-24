import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import Icon from "../components/Icon.jsx";
import Composer from "../components/chat/Composer.jsx";
import Message from "../components/chat/Message.jsx";
import { detection } from "../api.js";
import { history, newId } from "../lib/storage.js";
import { formatRelative } from "../lib/format.js";
import { diseaseName, plantName } from "../lib/labels.js";

const SUGGESTIONS = [
  "Lá cà chua bị mốc sương thì xịt thuốc gì?",
  "Bệnh ghẻ táo chữa như thế nào?",
  "khoai tay bi chay som phong sao",
  "Gỉ sắt trên lá ngô lây lan thế nào?",
];

function fromStored(message, index) {
  const debug = message.debug || null;
  return {
    key: `stored-${index}`,
    role: message.role,
    content: message.content,
    imageUploaded: message.image_uploaded,
    imageName: message.image_filename,
    feedbackId: message.feedback_id,
    rating: message.feedback_rating,
    reasons: message.feedback_reasons || [],
    action: message.action,
    debug,
    sources: debug?.rag_sources || [],
    detection: debug?.detection || null,
  };
}

function fromResponse(key, response) {
  return {
    key,
    role: "assistant",
    content: response.answer,
    action: response.action,
    feedbackId: response.feedback_id,
    sources: response.sources || [],
    detection: response.debug?.detection || null,
    debug: response.debug,
    rating: null,
    reasons: [],
  };
}

function titleFrom(text, image) {
  const value = (text || "").trim();
  if (value) return value.length > 60 ? `${value.slice(0, 57)}…` : value;
  return image ? "Ảnh lá cây" : "Cuộc trò chuyện mới";
}

function Welcome({ onPick }) {
  return (
    <div className="welcome">
      <div className="welcome-mark"><Icon name="leaf" size={28} /></div>
      <h1>Bác sĩ cây trồng</h1>
      <p>
        Chụp ảnh lá cây bị bệnh hoặc mô tả triệu chứng. Hệ thống nhận diện cây và bệnh, rồi trả lời dựa trên tài
        liệu, có ghi nguồn.
      </p>
      <div className="suggestions">
        {SUGGESTIONS.map((text) => (
          <button key={text} type="button" className="suggestion" onClick={() => onPick(text)}>
            {text}
          </button>
        ))}
      </div>
      <p className="small muted">
        Hỗ trợ: táo, anh đào, ngô, nho, đào, ớt chuông, khoai tây, dâu tây, cà chua (bí chỉ hỏi bằng chữ).
      </p>
    </div>
  );
}

export default function Chat() {
  const [params, setParams] = useSearchParams();
  // `/?session=<id>` mở một cuộc trò chuyện có sẵn (ví dụ từ trang admin), kể cả khi trình duyệt này chưa có nó.
  const linked = useRef(params.get("session"));
  const [sessionId, setSessionId] = useState(() => linked.current || history.current() || newId());
  const [conversations, setConversations] = useState(() => history.list());
  const [messages, setMessages] = useState([]);
  const [memory, setMemory] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState(null);
  const [sending, setSending] = useState(false);
  const [drawer, setDrawer] = useState(false);
  const composer = useRef(null);
  const scroller = useRef(null);

  useEffect(() => {
    if (params.get("session")) setParams({}, { replace: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Mở lại hội thoại đã có trên backend; hội thoại mới thì để trống.
  useEffect(() => {
    history.setCurrent(sessionId);
    const fromLink = linked.current === sessionId;
    if (!fromLink && !history.list().some((item) => item.id === sessionId)) {
      setMessages([]);
      setMemory(null);
      return undefined;
    }
    let alive = true;
    setLoading(true);
    setLoadError(null);
    detection
      .conversation(sessionId)
      .then((conversation) => {
        if (!alive) return;
        setMessages((conversation.messages || []).map(fromStored));
        setMemory(conversation.last_detection || null);
        if (fromLink && conversation.messages?.length) {
          setConversations(history.upsert({ id: sessionId, title: conversation.title, updatedAt: new Date().toISOString() }));
        }
      })
      .catch((error) => {
        if (!alive) return;
        setMessages([]);
        setMemory(null);
        if (error.status === 404) setConversations(history.remove(sessionId));
        else setLoadError(error);
      })
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [sessionId]);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const send = useCallback(async ({ text, image }) => {
    const pendingKey = newId();
    setMessages((current) => [
      ...current,
      { key: newId(), role: "user", content: text, imageUrl: image?.previewUrl, imageName: image?.name },
      { key: pendingKey, role: "assistant", pending: true, startedAt: Date.now() },
    ]);
    setSending(true);
    try {
      const response = await detection.chat({ sessionId, message: text, image: image?.blob, imageName: image?.name });
      setMessages((current) => current.map((item) => (item.key === pendingKey ? fromResponse(pendingKey, response) : item)));
      setMemory(response.memory || null);
      setConversations(history.upsert({ id: sessionId, title: titleFrom(text, image), updatedAt: new Date().toISOString() }));
    } catch (error) {
      setMessages((current) => current.map((item) => (
        item.key === pendingKey ? { key: pendingKey, role: "assistant", error, retry: { text, image } } : item
      )));
    } finally {
      setSending(false);
    }
  }, [sessionId]);

  // Backend không lưu lượt bị lỗi, nên gửi lại là an toàn: bỏ tin nhắn lỗi và câu hỏi của nó rồi gửi lại.
  function retry(failed) {
    setMessages((current) => {
      const index = current.findIndex((item) => item.key === failed.key);
      return index > 0 ? current.slice(0, index - 1) : current;
    });
    send(failed.retry);
  }

  function newChat() {
    setSessionId(newId());
    setDrawer(false);
    setLoadError(null);
  }

  function open(id) {
    if (id !== sessionId) setSessionId(id);
    setDrawer(false);
  }

  async function remove(id) {
    if (!window.confirm("Xóa cuộc trò chuyện này?")) return;
    try {
      await detection.deleteConversation(id);
    } catch {
      // Vẫn bỏ khỏi danh sách của trình duyệt này.
    }
    setConversations(history.remove(id));
    if (id === sessionId) newChat();
  }

  const title = conversations.find((item) => item.id === sessionId)?.title || "Cuộc trò chuyện mới";

  return (
    <div className="chat-app">
      <div className={`scrim ${drawer ? "is-open" : ""}`} onClick={() => setDrawer(false)} aria-hidden="true" />
      <aside className={`chat-sidebar ${drawer ? "is-open" : ""}`} aria-label="Lịch sử trò chuyện">
        <div className="sidebar-brand">
          <Icon name="leaf" size={22} />
          <span>Bác sĩ cây trồng</span>
          <button type="button" className="icon-btn only-mobile" onClick={() => setDrawer(false)} aria-label="Đóng">
            <Icon name="close" />
          </button>
        </div>
        <button type="button" className="btn btn-accent btn-block" onClick={newChat}>
          <Icon name="plus" size={18} /> Cuộc trò chuyện mới
        </button>
        <div className="history">
          <div className="history-title">Gần đây trên thiết bị này</div>
          {conversations.length === 0 ? <p className="muted small">Chưa có cuộc trò chuyện nào.</p> : null}
          <ul>
            {conversations.map((item) => (
              <li key={item.id} className={item.id === sessionId ? "is-active" : ""}>
                <button type="button" className="history-item" onClick={() => open(item.id)}>
                  <span className="history-name">{item.title}</span>
                  <span className="history-time">{formatRelative(item.updatedAt)}</span>
                </button>
                <button type="button" className="icon-btn history-delete" onClick={() => remove(item.id)} aria-label={`Xóa ${item.title}`}>
                  <Icon name="trash" size={16} />
                </button>
              </li>
            ))}
          </ul>
        </div>
        <nav className="sidebar-footer">
          <Link to="/admin"><Icon name="chart" size={18} /> Trang quản trị</Link>
        </nav>
      </aside>

      <main className="chat-main">
        <header className="chat-topbar">
          <button type="button" className="icon-btn only-mobile" onClick={() => setDrawer(true)} aria-label="Mở lịch sử">
            <Icon name="menu" />
          </button>
          <div className="chat-title">{title}</div>
          {memory ? (
            <div className="memory-pill" title="Kết quả nhận diện gần nhất trong cuộc trò chuyện; câu hỏi tiếp theo như 'bệnh này chữa sao?' sẽ dùng kết quả này.">
              <Icon name="leaf" size={16} />
              <span>{plantName(memory.plant)}{memory.disease ? ` · ${diseaseName(memory.disease)}` : ""}</span>
            </div>
          ) : null}
          <button type="button" className="icon-btn only-mobile" onClick={newChat} aria-label="Cuộc trò chuyện mới">
            <Icon name="plus" />
          </button>
        </header>

        <div className="chat-scroll" ref={scroller}>
          <div className="chat-column">
            {loading ? <div className="spinner-center muted">Đang tải cuộc trò chuyện…</div> : null}
            {loadError ? (
              <div className="bubble bubble-error" role="alert">
                <Icon name="alert" />
                <div>Không tải được cuộc trò chuyện: {loadError.message}</div>
              </div>
            ) : null}
            {!loading && !loadError && messages.length === 0 ? (
              <Welcome onPick={(text) => send({ text, image: null })} />
            ) : null}
            {messages.map((message) => (
              <Message
                key={message.key}
                message={message}
                onRetry={sending ? null : retry}
                onPickImage={() => composer.current?.pickImage()}
              />
            ))}
          </div>
        </div>

        <div className="chat-bottom">
          <div className="chat-column">
            <Composer ref={composer} disabled={sending} onSend={send} />
            <p className="disclaimer">Thông tin chỉ để tham khảo. Hỏi cán bộ kỹ thuật nông nghiệp trước khi dùng thuốc.</p>
          </div>
        </div>
      </main>
    </div>
  );
}
