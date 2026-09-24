import { useState } from "react";
import { Link } from "react-router-dom";
import { rag } from "../../api.js";
import Icon from "../../components/Icon.jsx";
import Markdown from "../../components/Markdown.jsx";
import { Badge, ErrorNote, KeyValue, PageHeader, useLoad } from "../../components/ui.jsx";
import { formatMs, formatNumber } from "../../lib/format.js";
import { SCOPE_LABEL } from "../../lib/labels.js";
import { split } from "./playgroundUtils.js";
import { withIndex } from "./KnowledgeBase.jsx";

const EMPTY = { query: "", extra: "", plant: "", disease: "", index: "", mode: "", topK: 4, rerank: false, provider: "" };

function Scope({ scope }) {
  const searchable = scope.searchable;
  return (
    <div className={`note ${searchable ? "note-info" : "note-warn"}`}>
      <Icon name={searchable ? "info" : "alert"} />
      <div className="note-body">
        <div><strong>{SCOPE_LABEL[scope.status] || scope.status}</strong> <code>{scope.status}</code></div>
        <div className="small">{scope.message}</div>
        {scope.sources?.length ? <div className="small muted">Chỉ tìm trong: {scope.sources.join(", ")}</div> : null}
      </div>
    </div>
  );
}

function Chunks({ chunks, index }) {
  if (!chunks?.length) return <p className="muted">Không có chunk nào.</p>;
  return (
    <table className="table responsive">
      <thead>
        <tr><th className="num">Hạng</th><th>Chunk</th><th className="num">Semantic</th><th className="num">BM25</th><th className="num">RRF</th><th className="num">Rerank</th></tr>
      </thead>
      <tbody>
        {chunks.map((chunk) => {
          const { body } = split(chunk.content);
          return (
            <tr key={chunk.chunk_id}>
              <td data-label="Hạng" className="num"><strong>{chunk.rank}</strong></td>
              <td data-label="Chunk">
                <Link to={withIndex(`/admin/kb/chunk/${chunk.chunk_id}`, index)}>{chunk.heading_path || chunk.source}</Link>
                <div className="small muted">{chunk.source}</div>
                <div className="small clamp-2">{body}</div>
              </td>
              <td data-label="Semantic" className="num">{chunk.semantic_rank ?? "—"}</td>
              <td data-label="BM25" className="num">{chunk.bm25_rank ?? "—"}</td>
              <td data-label="RRF" className="num">{chunk.rrf_score?.toFixed(4) ?? "—"}</td>
              <td data-label="Rerank" className="num">{chunk.rerank_score?.toFixed(3) ?? "—"}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function Meta({ meta }) {
  const timing = meta.timing_ms || {};
  return (
    <KeyValue rows={[
      ["Index", `${meta.index} (${meta.chunking_strategy || "?"})`],
      ["Cách tìm", meta.retrieval_mode ? `${meta.retrieval_mode}${meta.reranker_enabled ? " + reranker" : ""}` : "không tìm (phạm vi chặn)"],
      ["Câu đã dùng để tìm", meta.search_queries?.length ? meta.search_queries.join(" | ") : null],
      ["LLM", meta.llm ? `${meta.llm.provider} · ${meta.llm.model}` : undefined],
      ["Token", meta.llm ? `${formatNumber(meta.llm.input_tokens)} vào · ${formatNumber(meta.llm.output_tokens)} ra` : undefined],
      ["Kết thúc", meta.llm ? `${meta.llm.finish_reason || "—"}${meta.llm.truncated ? " (bị cắt ở giới hạn token)" : ""}` : undefined],
      ["Thời gian", `tìm ${formatMs(timing.retrieve)}${timing.generate !== null && timing.generate !== undefined ? ` · sinh ${formatMs(timing.generate)}` : ""} · tổng ${formatMs(timing.total)}`],
    ]} />
  );
}

export default function Playground() {
  const [form, setForm] = useState(EMPTY);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(null);
  const taxonomy = useLoad(() => rag.taxonomy(), []);

  const set = (key) => (event) => {
    const value = event.target.type === "checkbox" ? event.target.checked : event.target.value;
    setForm((previous) => ({ ...previous, [key]: value }));
  };

  function body() {
    return {
      query: form.query.trim(),
      plant_type: form.plant.trim() || undefined,
      disease: form.disease.trim() || undefined,
      top_k: Number(form.topK) || undefined,
      mode: form.mode || undefined,
      rerank: form.rerank || undefined,
      index: form.index || undefined,
      extra_queries: form.extra.trim() ? [form.extra.trim()] : undefined,
      debug: true,
    };
  }

  async function run(kind) {
    if (!form.query.trim()) return;
    setBusy(kind);
    setError(null);
    try {
      const response = kind === "answer"
        ? await rag.answer({ ...body(), llm_provider: form.provider || undefined })
        : await rag.retrieve(body());
      setResult({ kind, response });
    } catch (failure) {
      setError(failure);
    } finally {
      setBusy(null);
    }
  }

  const plants = taxonomy.data?.plants || [];
  const diseases = [...new Set(plants.flatMap((plant) => plant.diseases.map((disease) => disease.disease)))].sort();
  const response = result?.response;

  return (
    <>
      <PageHeader
        title="Thử truy vấn"
        description="Chạy riêng phần của RAG (bước ⑦–⑨), không qua Groq và nhận diện ảnh. Cây/bệnh ở đây giả lập nhãn detection gửi sang."
      />
      <form className="card grid" onSubmit={(event) => { event.preventDefault(); run("retrieve"); }}>
        <label className="field">
          <span>Câu hỏi (câu dùng để tìm)</span>
          <textarea className="textarea" rows={2} value={form.query} onChange={set("query")} placeholder="Ví dụ: Lá táo có đốm xanh ô liu là bệnh gì?" required />
        </label>
        <div className="form-grid form-grid-4">
          <label className="field">
            <span>Cây (plant_type)</span>
            <input className="input" list="plant-options" value={form.plant} onChange={set("plant")} placeholder="apple, Apple…" />
            <datalist id="plant-options">{plants.map((plant) => <option key={plant.plant} value={plant.plant} />)}</datalist>
          </label>
          <label className="field">
            <span>Bệnh (disease)</span>
            <input className="input" list="disease-options" value={form.disease} onChange={set("disease")} placeholder="apple_scab, Apple___Black_rot…" />
            <datalist id="disease-options">{diseases.map((disease) => <option key={disease} value={disease} />)}</datalist>
          </label>
          <label className="field">
            <span>Index</span>
            <select className="select" value={form.index} onChange={set("index")}>
              <option value="">Mặc định</option>
              <option value="structure">structure</option>
              <option value="recursive">recursive</option>
            </select>
          </label>
          <label className="field">
            <span>Cách tìm</span>
            <select className="select" value={form.mode} onChange={set("mode")}>
              <option value="">Mặc định (.env)</option>
              <option value="hybrid">hybrid</option>
              <option value="semantic">semantic</option>
              <option value="bm25">bm25</option>
            </select>
          </label>
          <label className="field">
            <span>Số chunk (top_k)</span>
            <input className="input" type="number" min={1} max={20} value={form.topK} onChange={set("topK")} />
          </label>
          <label className="field">
            <span>LLM (khi sinh câu trả lời)</span>
            <select className="select" value={form.provider} onChange={set("provider")}>
              <option value="">Mặc định (.env)</option>
              <option value="vllm">vllm (Vast)</option>
              <option value="ollama">ollama</option>
              <option value="gemini">gemini</option>
            </select>
          </label>
          <label className="field" style={{ gridColumn: "span 2" }}>
            <span>Câu tìm phụ (extra_queries, tùy chọn)</span>
            <input className="input" value={form.extra} onChange={set("extra")} placeholder="Ví dụ câu gốc không dấu" />
          </label>
        </div>
        <div className="toolbar">
          <label className="check"><input type="checkbox" checked={form.rerank} onChange={set("rerank")} /> Dùng reranker</label>
          <span style={{ flex: 1 }} />
          <button type="button" className="btn btn-quiet" onClick={() => { setForm(EMPTY); setResult(null); setError(null); }}>Xóa</button>
          <button type="submit" className="btn" disabled={!!busy || !form.query.trim()}>
            <Icon name="search" size={18} /> {busy === "retrieve" ? "Đang tìm…" : "Tìm tài liệu"}
          </button>
          <button type="button" className="btn btn-accent" disabled={!!busy || !form.query.trim()} onClick={() => run("answer")}>
            <Icon name="send" size={18} /> {busy === "answer" ? "Đang sinh… (10–30 giây)" : "Sinh câu trả lời"}
          </button>
        </div>
      </form>

      <ErrorNote error={error} />

      {response ? (
        <>
          <Scope scope={response.scope} />
          {result.kind === "answer" ? (
            <section className="card">
              <div className="card-header">
                <h2>Câu trả lời</h2>
                <div className="toolbar">
                  <Badge tone={response.grounded ? "ok" : "warn"}>{response.grounded ? "có tài liệu" : "câu từ chối dựng sẵn"}</Badge>
                  {response.citations?.invalid?.length ? <Badge tone="error">trích nguồn không có: {response.citations.invalid.join(", ")}</Badge> : null}
                </div>
              </div>
              <Markdown>{response.answer}</Markdown>
              {response.sources?.length ? <p className="small muted">Nguồn: {response.sources.join(", ")}</p> : null}
            </section>
          ) : null}
          <section className="card">
            <h2>Cấu hình và thời gian</h2>
            <Meta meta={response.meta} />
          </section>
          <section className="card">
            <h2>Chunk được chọn</h2>
            <Chunks chunks={response.chunks} index={response.meta.index} />
          </section>
          <details className="raw-json card">
            <summary>JSON đầy đủ</summary>
            <pre>{JSON.stringify(response, null, 2)}</pre>
          </details>
        </>
      ) : null}
    </>
  );
}
