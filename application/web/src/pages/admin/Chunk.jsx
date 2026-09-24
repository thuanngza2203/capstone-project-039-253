import { Link, useParams } from "react-router-dom";
import { rag } from "../../api.js";
import Icon from "../../components/Icon.jsx";
import { ErrorNote, KeyValue, PageHeader, Spinner, Stat, useLoad } from "../../components/ui.jsx";
import { formatNumber } from "../../lib/format.js";
import { useIndexParam, withIndex } from "./KnowledgeBase.jsx";

function formatValue(value) {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value === "object") return <code>{JSON.stringify(value)}</code>;
  return String(value);
}

export default function ChunkPage() {
  const { id } = useParams();
  const [index] = useIndexParam();
  const { data, error, loading, reload } = useLoad(() => rag.chunk(id, index || undefined), [id, index]);

  return (
    <>
      <div className="breadcrumb">
        <Link to={withIndex("/admin/kb", index)}>Kho tri thức</Link>
        <Icon name="chevronRight" size={14} />
        {data ? <Link to={withIndex(`/admin/kb/doc/${data.source}`, index)}>{data.source}</Link> : <span>…</span>}
        <Icon name="chevronRight" size={14} />
        <span>chunk {data ? `${data.position + 1}/${data.total_in_document}` : ""}</span>
      </div>
      <PageHeader
        title={data?.heading_path?.split(" > ").slice(1).join(" › ") || "Chunk"}
        description={data ? `Index ${data.index} · id ${id.slice(0, 12)}…` : null}
        actions={data ? (
          <div className="toolbar">
            <Link className={`btn btn-small ${data.prev_id ? "" : "is-disabled"}`} aria-disabled={!data.prev_id}
              to={data.prev_id ? withIndex(`/admin/kb/chunk/${data.prev_id}`, index) : "#"}
              onClick={(event) => !data.prev_id && event.preventDefault()}>
              <Icon name="chevronLeft" size={16} /> Trước
            </Link>
            <Link className={`btn btn-small ${data.next_id ? "" : "is-disabled"}`} aria-disabled={!data.next_id}
              to={data.next_id ? withIndex(`/admin/kb/chunk/${data.next_id}`, index) : "#"}
              onClick={(event) => !data.next_id && event.preventDefault()}>
              Sau <Icon name="chevronRight" size={16} />
            </Link>
          </div>
        ) : null}
      />
      <ErrorNote error={error} onRetry={reload} />
      {loading && !data ? <Spinner /> : null}
      {data ? (
        <>
          <div className="grid grid-stats">
            <Stat label="Token" value={formatNumber(data.tokens)} hint="gồm cả header" />
            <Stat label="Ký tự" value={formatNumber(data.chars)} />
            <Stat label="Dòng trong tài liệu" value={data.start_line ? `${data.start_line}–${data.end_line}` : "—"} />
            <Stat label="Vị trí" value={`${data.position + 1}/${data.total_in_document}`} />
          </div>
          <section className="card">
            <h2>Header</h2>
            <p className="small muted" style={{ marginTop: 0 }}>Chèn vào đầu chunk khi index để chunk tự nói nó thuộc tài liệu và mục nào.</p>
            {data.header ? <div className="chunk-header">{data.header}</div> : <p className="muted">Chunk này không có header.</p>}
          </section>
          <section className="card">
            <h2>Nội dung</h2>
            <div className="chunk-body">{data.body}</div>
          </section>
          <section className="card">
            <h2>Metadata</h2>
            <KeyValue rows={Object.entries(data.metadata).sort(([a], [b]) => a.localeCompare(b)).map(([key, value]) => [key, formatValue(value)])} />
          </section>
        </>
      ) : null}
    </>
  );
}
