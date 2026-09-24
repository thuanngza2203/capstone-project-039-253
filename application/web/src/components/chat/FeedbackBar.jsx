import { useState } from "react";
import Icon from "../Icon.jsx";
import { detection } from "../../api.js";

export const FEEDBACK_REASONS = [
  "Chẩn đoán sai",
  "Không liên quan",
  "Thiếu thông tin",
  "Thông tin chưa chính xác",
  "Khó hiểu",
  "Khác",
];

// Thích / không thích một câu trả lời; không thích thì phải chọn ít nhất một lý do (backend bắt buộc).
export default function FeedbackBar({ feedbackId, rating: initialRating, reasons: initialReasons = [] }) {
  const [rating, setRating] = useState(initialRating || null);
  const [open, setOpen] = useState(false);
  const [picked, setPicked] = useState(new Set(initialReasons));
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);

  if (!feedbackId) return null;

  async function save(nextRating, reasons = []) {
    setBusy(true);
    setStatus(null);
    try {
      await detection.feedback(feedbackId, nextRating, reasons);
      setRating(nextRating);
      setOpen(false);
      setStatus({ tone: "ok", text: "Cảm ơn bạn đã góp ý." });
    } catch (error) {
      setStatus({ tone: "error", text: error.message });
    } finally {
      setBusy(false);
    }
  }

  function toggle(reason) {
    setPicked((previous) => {
      const next = new Set(previous);
      if (next.has(reason)) next.delete(reason);
      else next.add(reason);
      return next;
    });
  }

  return (
    <div className="feedback">
      <div className="feedback-row">
        <span className="muted">Câu trả lời có hữu ích không?</span>
        <button
          type="button"
          className={`icon-btn feedback-btn ${rating === "like" ? "is-like" : ""}`}
          aria-pressed={rating === "like"}
          aria-label="Hữu ích"
          disabled={busy}
          onClick={() => save("like")}
        >
          <Icon name="thumbUp" size={18} />
        </button>
        <button
          type="button"
          className={`icon-btn feedback-btn ${rating === "unlike" ? "is-unlike" : ""}`}
          aria-pressed={rating === "unlike"}
          aria-label="Chưa tốt"
          disabled={busy}
          onClick={() => setOpen((value) => !value)}
        >
          <Icon name="thumbDown" size={18} />
        </button>
      </div>
      {open ? (
        <div className="feedback-reasons">
          <div className="feedback-title">Vì sao câu trả lời chưa tốt?</div>
          <div className="reason-grid">
            {FEEDBACK_REASONS.map((reason) => (
              <label key={reason} className={`reason ${picked.has(reason) ? "is-picked" : ""}`}>
                <input type="checkbox" checked={picked.has(reason)} onChange={() => toggle(reason)} />
                {reason}
              </label>
            ))}
          </div>
          <button
            type="button"
            className="btn btn-small"
            disabled={busy || picked.size === 0}
            onClick={() => save("unlike", [...picked])}
          >
            Gửi góp ý
          </button>
        </div>
      ) : null}
      {status ? <div className={`feedback-status text-${status.tone}`}>{status.text}</div> : null}
    </div>
  );
}
