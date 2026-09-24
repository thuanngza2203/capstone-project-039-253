import { useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { rag } from "../../api.js";
import Icon from "../../components/Icon.jsx";
import { Badge, ErrorNote, Note, PageHeader, RefreshButton, Segmented, Spinner, Stat, Tabs, useLoad } from "../../components/ui.jsx";
import { formatDateTime, formatNumber } from "../../lib/format.js";
import { niceTitle } from "../../lib/documents.js";
import { PLANTS_VI, diseaseName } from "../../lib/labels.js";

export const INDEX_STATE = {
  current: { tone: "ok", label: "Đã index" },
  changed: { tone: "warn", label: "Đã sửa sau lần index" },
  not_indexed: { tone: "warn", label: "Chưa index" },
  deleted: { tone: "error", label: "Đã xóa khỏi data/" },
  unknown: { tone: "neutral", label: "Không rõ (index cũ)" },
};

export function useIndexParam() {
  const [params, setParams] = useSearchParams();
  const index = params.get("index") || "";
  const setIndex = (value) => {
    const next = new URLSearchParams(params);
    if (value) next.set("index", value);
    else next.delete("index");
    setParams(next, { replace: true });
  };
  return [index, setIndex];
}

export const withIndex = (path, index) => (index ? `${path}?index=${index}` : path);

function IndexCard({ entry }) {
  return (
    <section className="card">
      <div className="card-header">
        <h2>Index {entry.name}</h2>
        <div className="toolbar">
          {entry.default ? <Badge tone="ok">đang dùng</Badge> : null}
          {entry.matches_data === true ? <Badge tone="ok">Khớp dữ liệu</Badge> : null}
          {entry.matches_data === false ? <Badge tone="warn">Cần index lại</Badge> : null}
        </div>
      </div>
      {entry.chunks === null ? (
        <Note tone="warn">{entry.detail || "Index chưa build."}</Note>
      ) : (
        <>
          <div className="grid grid-stats">
            <Stat label="Tài liệu" value={formatNumber(entry.documents)} />
            <Stat label="Chunk" value={formatNumber(entry.chunks)} />
            <Stat label="Token/chunk" value={entry.tokens ? formatNumber(entry.tokens.mean) : "—"}
              hint={entry.tokens ? `trung vị ${entry.tokens.median} · tối đa ${entry.tokens.max}` : null} />
            <Stat label="Build lúc" value={<span className="small">{formatDateTime(entry.built_at)}</span>} />
          </div>
          {entry.matches_data === false ? <p className="small text-warn">{entry.detail}</p> : null}
          <p className="small muted" style={{ marginBottom: 0 }}>
            Chunk theo cây: {Object.entries(entry.chunks_per_crop || {}).map(([crop, count]) => `${crop} ${count}`).join(" · ")}
          </p>
        </>
      )}
    </section>
  );
}

function Documents({ index }) {
  const { data, error, loading, reload } = useLoad(() => rag.documents(index || undefined), [index]);
  const [crop, setCrop] = useState("");
  const [search, setSearch] = useState("");
  const navigate = useNavigate();

  const crops = useMemo(() => [...new Set((data || []).map((row) => row.crop))].sort(), [data]);
  const rows = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return (data || []).filter((row) => (!crop || row.crop === crop)
      && (!needle || `${row.title} ${row.source}`.toLowerCase().includes(needle)));
  }, [data, crop, search]);
  const unmapped = (data || []).filter((row) => row.index_state !== "deleted" && row.taxonomy.length === 0);

  return (
    <>
      {unmapped.length ? (
        <Note tone="warn">
          {unmapped.length} tài liệu chưa gắn vào taxonomy: RAG không khoanh được phạm vi tới các tài liệu này khi biết cây/bệnh
          ({unmapped.map((row) => row.source).join(", ")}). Sửa trong <code>RAG-module/taxonomy.py</code>.
        </Note>
      ) : null}
      <div className="toolbar">
        <select className="select" style={{ width: "auto" }} value={crop} onChange={(event) => setCrop(event.target.value)} aria-label="Lọc theo cây">
          <option value="">Mọi cây</option>
          {crops.map((value) => <option key={value} value={value}>{value}</option>)}
        </select>
        <input className="input" type="search" placeholder="Tìm theo tên tài liệu…" value={search}
          onChange={(event) => setSearch(event.target.value)} aria-label="Tìm tài liệu" />
        <RefreshButton onClick={reload} loading={loading} />
      </div>
      <ErrorNote error={error} onRetry={reload} />
      {loading && !data ? <Spinner /> : null}
      {rows.length ? (
        <table className="table responsive">
          <thead>
            <tr><th>Tài liệu</th><th>Cây</th><th className="num">Chunk</th><th className="num">Token</th><th className="num">Ký tự</th><th>Trạng thái</th><th>Taxonomy</th></tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.source} className="is-clickable" onClick={() => navigate(withIndex(`/admin/kb/doc/${row.source}`, index))}>
                <td data-label="Tài liệu">
                  <Link to={withIndex(`/admin/kb/doc/${row.source}`, index)} onClick={(event) => event.stopPropagation()}>{niceTitle(row.title)}</Link>
                  <div className="small muted">{row.source}</div>
                </td>
                <td data-label="Cây">{row.crop}</td>
                <td data-label="Chunk" className="num">{formatNumber(row.chunks)}</td>
                <td data-label="Token" className="num">{formatNumber(row.tokens)}</td>
                <td data-label="Ký tự" className="num">{formatNumber(row.chars)}</td>
                <td data-label="Trạng thái"><Badge tone={INDEX_STATE[row.index_state]?.tone}>{INDEX_STATE[row.index_state]?.label || row.index_state}</Badge></td>
                <td data-label="Taxonomy">
                  {row.taxonomy.length
                    ? row.taxonomy.map((link) => <div key={`${link.plant}-${link.disease}`} className="small">{link.plant} · {link.disease}</div>)
                    : <Badge tone="warn">chưa gắn</Badge>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : data ? <p className="muted">Không có tài liệu khớp bộ lọc.</p> : null}
    </>
  );
}

function Taxonomy() {
  const { data, error, loading, reload } = useLoad(() => rag.taxonomy(), []);
  return (
    <>
      <ErrorNote error={error} onRetry={reload} />
      {loading && !data ? <Spinner /> : null}
      {data ? (
        <table className="table responsive">
          <thead><tr><th>Cây</th><th>Bệnh</th><th>Tài liệu</th><th>Tên khác (alias)</th></tr></thead>
          <tbody>
            {data.plants.flatMap((plant) => plant.diseases.map((disease, position) => (
              <tr key={`${plant.plant}-${disease.disease}`}>
                <td data-label="Cây">{position === 0 ? <strong>{PLANTS_VI[plant.plant] || plant.plant} <span className="muted">({plant.plant})</span></strong> : null}</td>
                <td data-label="Bệnh">{diseaseName(disease.disease)} <span className="small muted">{disease.disease}</span></td>
                <td data-label="Tài liệu">
                  {disease.has_document
                    ? <Link to={`/admin/kb/doc/${disease.source}`}>{disease.source}</Link>
                    : <Badge tone="warn">chưa có tài liệu</Badge>}
                </td>
                <td data-label="Alias" className="small muted">{disease.aliases.join(", ") || "—"}</td>
              </tr>
            )))}
          </tbody>
        </table>
      ) : null}
    </>
  );
}

function ChunkSearch({ index }) {
  const [query, setQuery] = useState("");
  const [state, setState] = useState({ rows: null, error: null, loading: false });

  async function submit(event) {
    event.preventDefault();
    if (!query.trim()) return;
    setState({ rows: null, error: null, loading: true });
    try {
      setState({ rows: await rag.searchChunks(query.trim(), { index: index || undefined, limit: 50 }), error: null, loading: false });
    } catch (error) {
      setState({ rows: null, error, loading: false });
    }
  }

  return (
    <section className="card">
      <h2>Tìm chunk theo nội dung</h2>
      <form className="toolbar" onSubmit={submit}>
        <input className="input" type="search" placeholder="Ví dụ: đốm mắt ếch (không phân biệt dấu)" value={query}
          onChange={(event) => setQuery(event.target.value)} aria-label="Nội dung cần tìm" />
        <button type="submit" className="btn" disabled={state.loading}><Icon name="search" size={18} /> Tìm</button>
      </form>
      <ErrorNote error={state.error} />
      {state.rows ? (
        state.rows.length ? (
          <ul className="list-plain" style={{ marginTop: "0.75rem" }}>
            {state.rows.map((chunk) => (
              <li key={chunk.chunk_id}>
                <Link className="row-link" to={withIndex(`/admin/kb/chunk/${chunk.chunk_id}`, index)}>
                  <div className="row-main">
                    <div className="row-title">{chunk.heading_path || chunk.source}</div>
                    <div className="row-sub clamp-2">{chunk.preview}</div>
                  </div>
                  <span className="small muted">{chunk.tokens} token</span>
                </Link>
              </li>
            ))}
          </ul>
        ) : <p className="muted">Không có chunk nào chứa nội dung này.</p>
      ) : null}
    </section>
  );
}

export default function KnowledgeBase() {
  const overview = useLoad(() => rag.overview(), []);
  const [index, setIndex] = useIndexParam();
  const [tab, setTab] = useState("documents");
  const indexes = overview.data?.indexes || [];

  return (
    <>
      <PageHeader
        title="Kho tri thức"
        description={overview.data
          ? `${overview.data.data.documents} tài liệu trong data/ (dấu vân tay ${overview.data.data.fingerprint}). Sửa tài liệu trong RAG-module/data/ rồi chạy python main.py index.`
          : "Tài liệu trong RAG-module/data/ và các chunk đã index."}
        actions={(
          <>
            <Link className="btn" to="/admin/kb/playground"><Icon name="flask" size={18} /> Thử truy vấn</Link>
            <RefreshButton onClick={overview.reload} loading={overview.loading} />
          </>
        )}
      />
      <ErrorNote error={overview.error} onRetry={overview.reload} />
      {overview.loading && !overview.data ? <Spinner label="Đang đọc index (lần đầu mất vài giây)…" /> : null}
      <div className="grid grid-2">
        {indexes.map((entry) => <IndexCard key={entry.name} entry={entry} />)}
      </div>

      <div className="toolbar">
        <span className="small muted">Xem theo index:</span>
        <Segmented
          label="Index"
          options={[{ value: "", label: `Mặc định${overview.data ? ` (${overview.data.default_index})` : ""}` },
            ...indexes.map((entry) => ({ value: entry.name, label: entry.name }))]}
          value={index}
          onChange={setIndex}
        />
      </div>

      <ChunkSearch index={index} />

      <Tabs tabs={[{ value: "documents", label: "Tài liệu" }, { value: "taxonomy", label: "Cây · bệnh" }]} value={tab} onChange={setTab} />
      {tab === "documents" ? <Documents index={index} /> : <Taxonomy />}
    </>
  );
}
