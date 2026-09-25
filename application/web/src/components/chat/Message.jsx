import { useEffect, useState } from "react";
import Icon from "../Icon.jsx";
import Markdown from "../Markdown.jsx";
import DebugPanel from "../DebugPanel.jsx";
import FeedbackBar from "./FeedbackBar.jsx";
import { diseaseName, plantName, providerName, sourceFallback } from "../../lib/labels.js";
import { formatPercent, stripCitations } from "../../lib/format.js";
import { niceTitle } from "../../lib/documents.js";

function Waiting({ startedAt, webSearch }) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setSeconds(Math.floor((Date.now() - startedAt) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [startedAt]);
  return (
    <div className="bubble bubble-bot is-waiting" role="status">
      <span className="typing" aria-hidden="true"><i /><i /><i /></span>
      <div>
        <div>{webSearch ? "Đang tìm trên web…" : "Đang nhận diện và tra cứu…"} {seconds > 0 ? `${seconds} giây` : ""}</div>
        <div className="muted small">{webSearch ? "Thường mất 5–15 giây." : "Có thể mất 10–30 giây."}</div>
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

const fileName = (source) => String(source).split("/").pop();

function hostName(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

// Mỗi tài liệu: file dữ liệu trong kho của hệ thống, rồi link nguồn gốc ghi trong file đó.
function Sources({ documents }) {
  if (!documents?.length) return null;
  return (
    <div className="sources">
      <div className="small muted">Tài liệu tham khảo</div>
      <ul>
        {documents.map((doc) => (
          <li key={doc.source} className="source-item">
            <div className="source-doc">
              <Icon name="doc" size={16} />
              <span>
                {doc.title ? niceTitle(doc.title) : sourceFallback(doc.source)}
                <span className="source-file"> · dữ liệu: {fileName(doc.source)}</span>
              </span>
            </div>
            {doc.links?.length ? (
              <ul className="source-links">
                {doc.links.map((link) => (
                  <li key={link.url}>
                    <a href={link.url} target="_blank" rel="noopener noreferrer" title={link.url}>
                      <Icon name="external" size={14} />
                      <span>{link.label || hostName(link.url)}</span>
                    </a>
                  </li>
                ))}
              </ul>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}

// Trang web Groq đã đọc khi "Tìm trên web". Không qua kho tài liệu nên ghi rõ mức tin cậy.
function WebSources({ links }) {
  return (
    <div className="sources">
      <div className="small muted">Nguồn trên web</div>
      {links?.length ? (
        <ul className="source-links is-web">
          {links.map((link) => (
            <li key={link.url}>
              <a href={link.url} target="_blank" rel="noopener noreferrer" title={link.url}>
                <Icon name="external" size={14} />
                <span>
                  {link.label || hostName(link.url)}
                  {link.label ? <span className="source-file"> · {hostName(link.url)}</span> : null}
                </span>
              </a>
            </li>
          ))}
        </ul>
      ) : <p className="small muted">Model trả lời mà không mở trang web nào.</p>}
      <p className="web-caution">
        Thông tin tổng hợp từ web, chưa được kiểm chứng như kho tài liệu của hệ thống. Hỏi cán bộ kỹ thuật
        trước khi dùng thuốc.
      </p>
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
        <Waiting startedAt={message.startedAt} webSearch={message.webSearch} />
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
        <Markdown>{stripCitations(message.content)}</Markdown>
        {message.action === "REQUEST_IMAGE" && onPickImage ? (
          <button type="button" className="btn btn-accent" onClick={onPickImage}>
            <Icon name="camera" size={18} /> Tải ảnh lá
          </button>
        ) : null}
        {message.webSearch ? <WebSources links={message.webSources} /> : <Sources documents={message.documents} />}
        {message.llmProvider ? (
          <div className="answered-by">
            {message.webSearch ? <Icon name="globe" size={14} /> : null}
            Trả lời bởi {providerName(message.llmProvider)}{message.webSearch ? " · tìm trên web" : ""}
          </div>
        ) : null}
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
