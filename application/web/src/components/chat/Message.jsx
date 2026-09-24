import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import Icon from "../Icon.jsx";
import Markdown from "../Markdown.jsx";
import DebugPanel from "../DebugPanel.jsx";
import FeedbackBar from "./FeedbackBar.jsx";
import { diseaseName, plantName } from "../../lib/labels.js";
import { formatPercent } from "../../lib/format.js";
import { useDocTitles } from "../../lib/documents.js";

function Waiting({ startedAt }) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setSeconds(Math.floor((Date.now() - startedAt) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [startedAt]);
  return (
    <div className="bubble bubble-bot is-waiting" role="status">
      <span className="typing" aria-hidden="true"><i /><i /><i /></span>
      <div>
        <div>Đang nhận diện và tra cứu… {seconds > 0 ? `${seconds} giây` : ""}</div>
        <div className="muted small">Có thể mất 10–30 giây.</div>
      </div>
    </div>
  );
}

function DetectionCard({ detection }) {
  return (
    <div className="detection-card">
      <Icon name="leaf" size={18} />
      <div>
        <div className="small muted">Nhận diện từ ảnh</div>
        <div>
          <strong>{plantName(detection.plant)}</strong>
          {detection.disease ? <> · {diseaseName(detection.disease)}</> : <span className="muted"> · chưa có model bệnh cho cây này</span>}
        </div>
      </div>
      <div className="confidence" title="Độ tin cậy của model">{formatPercent(detection.confidence)}</div>
    </div>
  );
}

function Sources({ sources }) {
  const titleOf = useDocTitles();
  if (!sources?.length) return null;
  return (
    <div className="sources">
      <div className="small muted">Tài liệu tham khảo</div>
      <ul>
        {sources.map((source, index) => (
          <li key={source}>
            <Link to={`/admin/kb/doc/${source}`} title={source}>
              <span className="source-number">{index + 1}</span>
              {titleOf(source)}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function Message({ message, onRetry, onPickImage }) {
  const [showTrace, setShowTrace] = useState(false);

  if (message.role === "user") {
    return (
      <div className="message message-user">
        <div className="bubble bubble-user">
          {message.imageUrl ? <img className="bubble-image" src={message.imageUrl} alt="Ảnh đã gửi" /> : null}
          {!message.imageUrl && message.imageUploaded ? (
            <div className="image-placeholder"><Icon name="image" size={18} /> {message.imageName || "Ảnh lá cây"}</div>
          ) : null}
          {message.content ? <div className="bubble-text">{message.content}</div> : null}
        </div>
      </div>
    );
  }

  if (message.pending) {
    return (
      <div className="message message-bot">
        <Waiting startedAt={message.startedAt} />
      </div>
    );
  }

  if (message.error) {
    return (
      <div className="message message-bot">
        <div className="bubble bubble-error" role="alert">
          <Icon name="alert" />
          <div>
            <div>{message.error.message}</div>
            {onRetry ? (
              <button type="button" className="btn btn-small" onClick={() => onRetry(message)}>
                <Icon name="refresh" size={16} /> Gửi lại
              </button>
            ) : null}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="message message-bot">
      <div className="bubble bubble-bot">
        {message.detection ? <DetectionCard detection={message.detection} /> : null}
        <Markdown>{message.content}</Markdown>
        {message.action === "REQUEST_IMAGE" && onPickImage ? (
          <button type="button" className="btn btn-accent" onClick={onPickImage}>
            <Icon name="camera" size={18} /> Tải ảnh lá
          </button>
        ) : null}
        <Sources sources={message.sources} />
        <FeedbackBar feedbackId={message.feedbackId} rating={message.rating} reasons={message.reasons} />
        {message.debug ? (
          <div className="trace-toggle">
            <button type="button" className="link-btn" onClick={() => setShowTrace((value) => !value)} aria-expanded={showTrace}>
              <Icon name={showTrace ? "chevronDown" : "chevronRight"} size={16} /> Cách hệ thống xử lý câu này
            </button>
            {showTrace ? <DebugPanel debug={message.debug} /> : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}
