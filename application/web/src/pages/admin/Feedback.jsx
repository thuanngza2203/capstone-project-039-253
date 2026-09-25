import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { detection } from "../../api.js";
import Icon from "../../components/Icon.jsx";
import { Badge, ErrorNote, Note, PageHeader, RefreshButton, Segmented, Spinner, Stat, useLoad } from "../../components/ui.jsx";
import { formatDateTime } from "../../lib/format.js";
import { subjectLabel } from "../../lib/labels.js";

const FILTERS = [
  { value: "all", label: "Tất cả" },
  { value: "like", label: "Hữu ích" },
  { value: "unlike", label: "Chưa tốt" },
  { value: "training", label: "Trong RAG" },
];

async function load() {
  const [reviews, status] = await Promise.all([detection.reviews(), detection.adminStatus().catch(() => null)]);
  return { reviews, status };
}

// Chuyển nguyên tính năng của trang admin_review.html cũ sang web mới.
export default function Feedback() {
  const { data, error, loading, reload } = useLoad(load, []);
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState(new Set());
  const [drafts, setDrafts] = useState({});
  const [invalid, setInvalid] = useState(null);
  const [message, setMessage] = useState(null);
  const [busy, setBusy] = useState(false);

  const reviews = useMemo(() => data?.reviews || [], [data]);
  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return reviews.filter((item) => {
      const rating = item.user_feedback?.rating;
      if (filter === "like" && rating !== "like") return false;
      if (filter === "unlike" && rating !== "unlike") return false;
      if (filter === "training" && item.training?.selected !== true) return false;
      if (!needle) return true;
      const text = `${item.question || ""} ${item.answer || ""} ${(item.user_feedback?.reasons || []).join(" ")}`;
      return text.toLowerCase().includes(needle);
    });
  }, [reviews, filter, search]);

  const draftOf = (item) => (item._id in drafts ? drafts[item._id] : item.admin_feedback?.correct_answer || "");

  function toggle(id) {
    setSelected((previous) => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function toggleAll(checked) {
    setSelected((previous) => {
      const next = new Set(previous);
      for (const item of visible) {
        if (checked) next.add(item._id);
        else next.delete(item._id);
      }
      return next;
    });
  }

  async function train() {
    const items = [];
    for (const id of selected) {
      const item = reviews.find((review) => review._id === id);
      if (!item) continue;
      const correct = draftOf(item).trim();
      if (item.user_feedback?.rating === "unlike" && !correct) {
        setFilter("all");
        setSearch("");
        setInvalid(id);
        setMessage({ tone: "error", text: "QA bị chê phải có câu trả lời đúng trước khi đưa vào Feedback RAG." });
        setTimeout(() => document.getElementById(`correct-${id}`)?.focus(), 50);
        return;
      }
      items.push({ id, correct_answer: correct });
    }
    if (!items.length) return;
    setBusy(true);
    setMessage({ tone: "info", text: "Đang tạo embedding…" });
    try {
      const result = await detection.train(items);
      setSelected(new Set());
      setDrafts((previous) => {
        const next = { ...previous };
        items.forEach(({ id }) => delete next[id]);
        return next;
      });
      setMessage({ tone: "ok", text: `Đã đưa ${result.indexed_count || items.length} QA vào Feedback RAG.` });
      reload();
    } catch (failure) {
      setMessage({ tone: "error", text: failure.message });
    } finally {
      setBusy(false);
    }
  }

  async function remove(item) {
    if (!window.confirm(`Xóa QA này khỏi feedback và Feedback RAG?\n\n${(item.question || "").slice(0, 100)}`)) return;
    try {
      await detection.deleteReview(item._id);
      setSelected((previous) => {
        const next = new Set(previous);
        next.delete(item._id);
        return next;
      });
      setMessage({ tone: "ok", text: "Đã xóa QA." });
      reload();
    } catch (failure) {
      setMessage({ tone: "error", text: `Xóa QA thất bại: ${failure.message}` });
    }
  }

  const likes = reviews.filter((item) => item.user_feedback?.rating === "like").length;
  const unlikes = reviews.filter((item) => item.user_feedback?.rating === "unlike").length;
  const inRag = reviews.filter((item) => item.training?.selected === true).length;
  const allVisibleSelected = visible.length > 0 && visible.every((item) => selected.has(item._id));

  return (
    <>
      <PageHeader
        title="Feedback"
        description="Các câu trả lời người dùng đã bấm hữu ích hoặc chưa tốt. Chọn QA, sửa câu trả lời đúng nếu cần, rồi đưa vào Feedback RAG."
        actions={<RefreshButton onClick={reload} loading={loading} />}
      />
      {data?.status && data.status.feedback_examples_used === false ? (
        <Note tone="warn">
          Detection đang dùng <code>ANSWER_BACKEND={data.status.answer_backend}</code>: ví dụ trong Feedback RAG
          <strong> chưa được dùng</strong> khi trả lời (API RAG chưa nhận ví dụ admin). Vẫn thu thập và duyệt được.
        </Note>
      ) : null}
      <ErrorNote error={error} onRetry={reload} />

      <div className="grid grid-stats">
        <Stat label="Tổng" value={reviews.length} />
        <Stat label="Hữu ích" value={likes} />
        <Stat label="Chưa tốt" value={unlikes} />
        <Stat label="Trong Feedback RAG" value={inRag} />
      </div>

      <div className="toolbar">
        <Segmented label="Lọc" options={FILTERS} value={filter} onChange={setFilter} />
        <input className="input" type="search" placeholder="Tìm câu hỏi, câu trả lời, lý do…" value={search}
          onChange={(event) => setSearch(event.target.value)} aria-label="Tìm" />
      </div>

      <div className="toolbar">
        <label className="check">
          <input type="checkbox" checked={allVisibleSelected} onChange={(event) => toggleAll(event.target.checked)} />
          Chọn tất cả đang hiện
        </label>
        <span className="muted small">Đã chọn {selected.size}</span>
        <button type="button" className="btn btn-accent" disabled={busy || selected.size === 0} onClick={train}>
          <Icon name="database" size={18} /> {busy ? "Đang xử lý…" : "Đưa vào Feedback RAG"}
        </button>
      </div>
      {message ? <Note tone={message.tone}>{message.text}</Note> : null}

      {loading && !data ? <Spinner /> : null}
      {data && visible.length === 0 ? (
        <p className="muted">{reviews.length ? "Không có phản hồi khớp bộ lọc." : "Chưa có phản hồi nào. Khi người dùng bấm hữu ích / chưa tốt, câu trả lời sẽ hiện ở đây."}</p>
      ) : null}

      {visible.length ? (
        <table className="table responsive">
          <thead>
            <tr>
              <th aria-label="Chọn" />
              <th>Đánh giá</th>
              <th>Câu hỏi</th>
              <th>Câu trả lời của bot</th>
              <th>Câu trả lời đúng</th>
              <th>Thời gian</th>
              <th aria-label="Xóa" />
            </tr>
          </thead>
          <tbody>
            {visible.map((item) => {
              const rating = item.user_feedback?.rating;
              const meta = item.metadata || {};
              return (
                <tr key={item._id}>
                  <td data-label="">
                    <label className="check">
                      <input type="checkbox" checked={selected.has(item._id)} onChange={() => toggle(item._id)} aria-label="Chọn QA này" />
                    </label>
                  </td>
                  <td data-label="Đánh giá">
                    <div className="toolbar">
                      {rating === "like" ? <Badge tone="ok">Hữu ích</Badge> : <Badge tone="error">Chưa tốt</Badge>}
                      {item.training?.selected ? <Badge tone="neutral">trong RAG</Badge> : null}
                    </div>
                    {(item.user_feedback?.reasons || []).map((reason) => <div key={reason} className="small muted">• {reason}</div>)}
                  </td>
                  <td data-label="Câu hỏi">
                    <div>{item.question || "—"}</div>
                    {meta.plant || meta.disease ? <div className="small muted">{subjectLabel(meta.plant, meta.disease)}</div> : null}
                    <Link className="small" to={`/admin/conversations/${encodeURIComponent(item.session_id)}`}>Xem hội thoại</Link>
                  </td>
                  <td data-label="Bot trả lời">
                    <details>
                      <summary className="clamp-2">{item.answer || "—"}</summary>
                      <div style={{ whiteSpace: "pre-wrap" }}>{item.answer}</div>
                    </details>
                    {meta.sources?.length ? <div className="small muted">Nguồn: {meta.sources.join(", ")}</div> : null}
                    {!meta.sources?.length && meta.grounded === false ? <div className="small text-warn">Kho chưa có tài liệu cho câu này</div> : null}
                  </td>
                  <td data-label="Trả lời đúng">
                    <textarea
                      id={`correct-${item._id}`}
                      className="textarea"
                      value={draftOf(item)}
                      aria-invalid={invalid === item._id ? "true" : undefined}
                      placeholder={rating === "unlike" ? "Bắt buộc khi câu trả lời bị chê…" : "Để trống nếu câu trả lời của bot đã tốt…"}
                      onChange={(event) => {
                        const value = event.target.value;
                        setDrafts((previous) => ({ ...previous, [item._id]: value }));
                        if (invalid === item._id) setInvalid(null);
                      }}
                    />
                  </td>
                  <td data-label="Thời gian" className="small">{formatDateTime(item.created_at)}</td>
                  <td data-label="">
                    <button type="button" className="icon-btn" onClick={() => remove(item)} aria-label="Xóa QA này" title="Xóa QA">
                      <Icon name="trash" size={18} />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : null}
    </>
  );
}
