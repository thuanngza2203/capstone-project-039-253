import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { detection } from "../../api.js";
import Icon from "../../components/Icon.jsx";
import { ErrorNote, PageHeader, RefreshButton, Spinner, useLoad } from "../../components/ui.jsx";
import { formatDateTime, formatRelative } from "../../lib/format.js";

export default function Conversations() {
  const { data, error, loading, reload } = useLoad(() => detection.conversations(), []);
  const [search, setSearch] = useState("");

  const rows = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return (data || []).filter((item) => !needle
      || `${item.title} ${item.last_message}`.toLowerCase().includes(needle));
  }, [data, search]);

  return (
    <>
      <PageHeader
        title="Hội thoại"
        description="100 cuộc trò chuyện mới nhất của mọi người dùng. Mở một cuộc để xem từng lượt và cách pipeline xử lý."
        actions={<RefreshButton onClick={reload} loading={loading} />}
      />
      <div className="toolbar">
        <label className="visually-hidden" htmlFor="conversation-search">Tìm</label>
        <input
          id="conversation-search"
          className="input"
          type="search"
          placeholder="Tìm theo tiêu đề hoặc tin nhắn cuối…"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
      </div>
      <ErrorNote error={error} onRetry={reload} />
      {loading && !data ? <Spinner /> : null}
      {data && rows.length === 0 ? <p className="muted">Không có cuộc trò chuyện nào.</p> : null}
      <ul className="list-plain">
        {rows.map((item) => (
          <li key={item.session_id}>
            <Link className="row-link" to={`/admin/conversations/${encodeURIComponent(item.session_id)}`}>
              <Icon name="message" />
              <div className="row-main">
                <div className="row-title">{item.title}</div>
                <div className="row-sub clamp-2">{item.last_message || "—"}</div>
              </div>
              <div className="small muted" title={formatDateTime(item.updated_at)}>{formatRelative(item.updated_at)}</div>
            </Link>
          </li>
        ))}
      </ul>
    </>
  );
}
