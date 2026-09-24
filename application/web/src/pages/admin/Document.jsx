import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { rag } from "../../api.js";
import Icon from "../../components/Icon.jsx";
import { Badge, ErrorNote, PageHeader, RefreshButton, Spinner, Stat, Tabs, useLoad } from "../../components/ui.jsx";
import { formatNumber } from "../../lib/format.js";
import { niceTitle } from "../../lib/documents.js";
import { INDEX_STATE, useIndexParam, withIndex } from "./KnowledgeBase.jsx";

// Gom chunk theo mục cấp 1 ("TRIỆU CHỨNG") lấy từ heading_path "Tài liệu > Mục > Mục con".
function groupBySection(chunks) {
  const groups = [];
  for (const chunk of chunks) {
    const parts = (chunk.heading_path || "").split(" > ");
    const name = chunk.section || parts[1] || "Không có mục";
    const last = groups[groups.length - 1];
    if (last && last.name === name) last.chunks.push(chunk);
    else groups.push({ name, chunks: [chunk] });
  }
  return groups;
}

export default function DocumentPage() {
  const source = useParams()["*"];
  const [index] = useIndexParam();
  const { data, error, loading, reload } = useLoad(
    () => rag.document(source, { index: index || undefined, includeText: true }), [source, index]);
  const [tab, setTab] = useState("chunks");
  const groups = useMemo(() => groupBySection(data?.chunk_list || []), [data]);
  const state = data ? INDEX_STATE[data.index_state] : null;

  return (
    <>
      <div className="breadcrumb">
        <Link to={withIndex("/admin/kb", index)}>Kho tri thức</Link> <Icon name="chevronRight" size={14} /> <span>{source}</span>
      </div>
      <PageHeader
        title={data ? niceTitle(data.title) : source}
        description={data ? `${source} · index ${data.index}` : null}
        actions={<RefreshButton onClick={reload} loading={loading} />}
      />
      <ErrorNote error={error} onRetry={reload} />
      {loading && !data ? <Spinner /> : null}
      {data ? (
        <>
          <div className="grid grid-stats">
            <Stat label="Chunk" value={formatNumber(data.chunks)} />
            <Stat label="Token (tổng các chunk)" value={formatNumber(data.tokens)} />
            <Stat label="Ký tự" value={formatNumber(data.chars)} />
            <Stat label="Trạng thái" value={<Badge tone={state?.tone}>{state?.label || data.index_state}</Badge>}
              hint={data.taxonomy.length ? `taxonomy: ${data.taxonomy.map((link) => `${link.plant}/${link.disease}`).join(", ")}` : "chưa gắn taxonomy"} />
          </div>
          <Tabs tabs={[{ value: "chunks", label: `Chunk (${data.chunk_list.length})` }, { value: "text", label: "Văn bản gốc" }]} value={tab} onChange={setTab} />
          {tab === "chunks" ? (
            <section className="card">
              {groups.length === 0 ? <p className="muted">Tài liệu này chưa có chunk trong index {data.index}.</p> : null}
              {groups.map((group, position) => (
                <details key={`${group.name}-${position}`} className="section-group" open={groups.length <= 3}>
                  <summary><Icon name="chevronRight" size={16} /> {group.name} <span className="muted small">({group.chunks.length})</span></summary>
                  <ul className="list-plain" style={{ marginBottom: "0.75rem" }}>
                    {group.chunks.map((chunk) => (
                      <li key={chunk.chunk_id}>
                        <Link className="row-link" to={withIndex(`/admin/kb/chunk/${chunk.chunk_id}`, index)}>
                          <span className="source-number">{chunk.position + 1}</span>
                          <div className="row-main">
                            <div className="row-title">{(chunk.heading_path || "").split(" > ").slice(1).join(" › ") || "—"}</div>
                            <div className="row-sub clamp-2">{chunk.preview}</div>
                          </div>
                          <div className="small muted" style={{ textAlign: "right" }}>
                            {chunk.tokens} token
                            {chunk.start_line ? <div>dòng {chunk.start_line}–{chunk.end_line}</div> : null}
                          </div>
                        </Link>
                      </li>
                    ))}
                  </ul>
                </details>
              ))}
            </section>
          ) : (
            <section className="card">
              {data.text ? <div className="doc-text">{data.text}</div> : <p className="muted">Tài liệu không còn trong data/.</p>}
            </section>
          )}
        </>
      ) : null}
    </>
  );
}
