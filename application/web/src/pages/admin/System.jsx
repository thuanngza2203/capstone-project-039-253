import { detection, rag, DETECTION_URL, RAG_URL } from "../../api.js";
import { Badge, ErrorNote, KeyValue, PageHeader, RefreshButton, Spinner, useLoad } from "../../components/ui.jsx";
import { formatDateTime, formatNumber } from "../../lib/format.js";

async function settle(promises) {
  const entries = await Promise.allSettled(Object.values(promises));
  return Object.fromEntries(Object.keys(promises).map((key, position) => [key, entries[position]]));
}

const value = (entry) => (entry?.status === "fulfilled" ? entry.value : null);
const failure = (entry) => (entry?.status === "rejected" ? entry.reason : null);

function Health({ entry }) {
  if (!entry) return null;
  return entry.status === "fulfilled" ? <Badge tone="ok">hoạt động</Badge> : <Badge tone="error">không kết nối được</Badge>;
}

export default function System() {
  const { data, loading, reload } = useLoad(() => settle({
    detectionHealth: detection.health(),
    detectionStatus: detection.adminStatus(),
    ragHealth: rag.health(),
    ragStatus: rag.status(),
    llm: rag.llm(true),
  }), []);

  const status = value(data?.ragStatus);
  const llm = value(data?.llm);
  const backend = value(data?.detectionStatus);

  return (
    <>
      <PageHeader
        title="Hệ thống"
        description="Trạng thái từng thành phần. LLM được hỏi trực tiếp (probe) để biết model thật đang chạy trên Vast."
        actions={<RefreshButton onClick={reload} loading={loading} />}
      />
      {loading && !data ? <Spinner /> : null}
      <div className="grid grid-2">
        <section className="card">
          <div className="card-header"><h2>detection-server</h2><Health entry={data?.detectionHealth} /></div>
          <ErrorNote error={failure(data?.detectionHealth)} />
          <KeyValue rows={[
            ["Địa chỉ", <code key="url">{DETECTION_URL}</code>],
            ["Nguồn câu trả lời", backend ? <>
              <code>ANSWER_BACKEND={backend.answer_backend}</code>{" "}
              {backend.answer_backend === "rag" ? <Badge tone="ok">gọi RAG server</Badge> : <Badge tone="warn">rag/ nội bộ + Groq, không qua RAG server</Badge>}
            </> : null],
            ["Feedback RAG", backend ? (backend.feedback_examples_used ? "dùng khi trả lời" : "không dùng khi trả lời") : null],
          ]} />
        </section>

        <section className="card">
          <div className="card-header"><h2>RAG server</h2><Health entry={data?.ragHealth} /></div>
          <ErrorNote error={failure(data?.ragStatus)} />
          {status ? (
            <>
              {!status.ready ? <ErrorNote>{status.detail}</ErrorNote> : null}
              <KeyValue rows={[
                ["Địa chỉ", <code key="url">{RAG_URL}</code>],
                ["Index mặc định", `${status.default_index} · ${formatNumber(status.chunk_count)} chunk`],
                ["Thư mục index", <code key="dir">{status.index_directory}</code>],
                ["Embedding", status.embedding_model],
                ["Cách tìm", `${status.retrieval_mode}${status.reranker_enabled ? " + reranker" : ""}`],
                ["LLM", `${status.llm_provider} · ${status.llm_model || "—"}`],
                ["Endpoint LLM", status.llm_endpoint],
                ["Xác thực API", status.auth_enabled ? "bật (RAG_API_KEY)" : "tắt (public)"],
              ]} />
              <h3 style={{ marginTop: "1rem" }}>Index</h3>
              <table className="table responsive">
                <thead><tr><th>Index</th><th>Chunk</th><th>Build</th><th>Khớp data/</th></tr></thead>
                <tbody>
                  {status.indexes.map((index) => (
                    <tr key={index.name}>
                      <td data-label="Index">{index.name}{index.default ? " (mặc định)" : ""}</td>
                      <td data-label="Chunk">{formatNumber(index.chunk_count)}</td>
                      <td data-label="Build">{formatDateTime(index.built_at)}</td>
                      <td data-label="Khớp data/">
                        {index.matches_data === true ? <Badge tone="ok">có</Badge> : index.matches_data === false ? <Badge tone="warn">không</Badge> : "—"}
                        {index.detail ? <div className="small muted">{index.detail}</div> : null}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          ) : null}
        </section>
      </div>

      <section className="card">
        <div className="card-header">
          <h2>LLM</h2>
          {llm ? (llm.reachable ? <Badge tone="ok">kết nối được</Badge> : llm.reachable === false ? <Badge tone="error">không kết nối được</Badge> : <Badge tone="neutral">không probe</Badge>) : null}
        </div>
        <ErrorNote error={failure(data?.llm)} />
        {llm ? (
          <>
            {llm.detail ? <p className={llm.reachable === false ? "text-error" : "text-warn"}>{llm.detail}</p> : null}
            <KeyValue rows={[
              ["Provider", llm.provider],
              ["Model trong .env", llm.model],
              ["Endpoint", llm.endpoint],
            ]} />
            {llm.served?.length ? (
              <table className="table responsive" style={{ marginTop: "0.75rem" }}>
                <thead><tr><th>Tên gọi (id)</th><th>Model thật (root)</th><th>max_model_len</th></tr></thead>
                <tbody>
                  {llm.served.map((model) => (
                    <tr key={model.id}>
                      <td data-label="Tên gọi">{model.id}</td>
                      <td data-label="Model thật">{model.root || model.details || "—"}</td>
                      <td data-label="max_model_len">{formatNumber(model.max_model_len)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : null}
          </>
        ) : null}
      </section>
    </>
  );
}
