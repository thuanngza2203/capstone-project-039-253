import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { detection } from "../../api.js";
import DebugPanel from "../../components/DebugPanel.jsx";
import Icon from "../../components/Icon.jsx";
import Markdown from "../../components/Markdown.jsx";
import { Badge, ErrorNote, PageHeader, RefreshButton, Spinner, useLoad } from "../../components/ui.jsx";
import { formatDateTime, formatPercent } from "../../lib/format.js";
import { ACTION_LABEL, SCOPE_LABEL, diseaseName, plantName } from "../../lib/labels.js";

function BotTurn({ message }) {
  const [open, setOpen] = useState(false);
  const debug = message.debug;
  return (
    <div className="card conversation-turn">
      <div className="toolbar">
        <Badge tone="neutral">Bot</Badge>
        {message.action ? <Badge tone={message.action === "ACCEPT_QUERY" ? "ok" : "warn"}>{ACTION_LABEL[message.action] || message.action}</Badge> : null}
        {debug?.rag_scope_status ? <Badge tone="neutral">{SCOPE_LABEL[debug.rag_scope_status] || debug.rag_scope_status}</Badge> : null}
        {message.feedback_rating === "like" ? <Badge tone="ok">Được thích</Badge> : null}
        {message.feedback_rating === "unlike" ? <Badge tone="error">Không thích: {(message.feedback_reasons || []).join(", ")}</Badge> : null}
        <span className="small muted" style={{ marginLeft: "auto" }}>{formatDateTime(message.created_at)}</span>
      </div>
      <Markdown>{message.content}</Markdown>
      {debug?.rag_sources?.length ? (
        <div className="small">
          <span className="muted">Nguồn: </span>
          <span className="inline-list">
            {debug.rag_sources.map((source) => <Link key={source} to={`/admin/kb/doc/${source}`}>{source}</Link>)}
          </span>
        </div>
      ) : null}
      {debug ? (
        <div>
          <button type="button" className="link-btn" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
            <Icon name={open ? "chevronDown" : "chevronRight"} size={16} /> Chi tiết pipeline
          </button>
          {open ? <DebugPanel debug={debug} /> : null}
        </div>
      ) : null}
    </div>
  );
}

function UserTurn({ message }) {
  return (
    <div className="card conversation-turn" style={{ background: "var(--color-paper-2)" }}>
      <div className="toolbar">
        <Badge tone="neutral">Người dùng</Badge>
        {message.image_uploaded ? <Badge tone="ok"><Icon name="image" size={14} /> {message.image_filename || "có ảnh"}</Badge> : null}
        <span className="small muted" style={{ marginLeft: "auto" }}>{formatDateTime(message.created_at)}</span>
      </div>
      <div style={{ whiteSpace: "pre-wrap" }}>{message.content || <span className="muted">(chỉ gửi ảnh)</span>}</div>
    </div>
  );
}

export default function ConversationDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { data, error, loading, reload } = useLoad(() => detection.conversation(id), [id]);
  const [deleteError, setDeleteError] = useState(null);

  async function remove() {
    if (!window.confirm("Xóa cuộc trò chuyện này khỏi MongoDB?")) return;
    try {
      await detection.deleteConversation(id);
      navigate("/admin/conversations");
    } catch (failure) {
      setDeleteError(failure);
    }
  }

  const detected = data?.last_detection;

  return (
    <>
      <div className="breadcrumb"><Link to="/admin/conversations">Hội thoại</Link> <Icon name="chevronRight" size={14} /> <span>{id.slice(0, 8)}…</span></div>
      <PageHeader
        title={data?.title || "Cuộc trò chuyện"}
        description={detected
          ? `Nhận diện gần nhất: ${plantName(detected.plant)}${detected.disease ? ` · ${diseaseName(detected.disease)}` : ""} (${formatPercent(detected.confidence)})`
          : "Chưa có ảnh nào trong cuộc trò chuyện này."}
        actions={(
          <>
            <RefreshButton onClick={reload} loading={loading} />
            <Link className="btn btn-small" to={`/?session=${encodeURIComponent(id)}`}><Icon name="chat" size={16} /> Mở trong trang chat</Link>
            <button type="button" className="btn btn-danger btn-small" onClick={remove}><Icon name="trash" size={16} /> Xóa</button>
          </>
        )}
      />
      <ErrorNote error={error} onRetry={reload} />
      <ErrorNote error={deleteError} />
      {loading && !data ? <Spinner /> : null}
      <div className="grid" style={{ gap: "0.75rem" }}>
        {(data?.messages || []).map((message, index) => (
          message.role === "user"
            ? <UserTurn key={index} message={message} />
            : <BotTurn key={index} message={message} />
        ))}
      </div>
    </>
  );
}
