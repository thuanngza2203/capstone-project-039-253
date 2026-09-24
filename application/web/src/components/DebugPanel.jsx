import { Link } from "react-router-dom";
import { ACTION_LABEL, INTENT_LABEL, SCOPE_LABEL, diseaseName, plantName, subjectLabel } from "../lib/labels.js";
import { formatPercent } from "../lib/format.js";
import { KeyValue } from "./ui.jsx";

const CONTEXT_SOURCE = {
  query: "câu hỏi",
  current_image: "ảnh lượt này",
  session_memory: "ảnh lượt trước",
  query_only: "chỉ câu hỏi",
};

function contextSource(value) {
  if (!value) return null;
  return value.split("+").map((part) => CONTEXT_SOURCE[part] || part).join(" + ");
}

function Step({ number, title, children }) {
  return (
    <section className="trace-step">
      <h4>
        <span className="trace-number">{number}</span> {title}
      </h4>
      {children}
    </section>
  );
}

// Dấu vết pipeline của một lượt trả lời (trường `debug` của /api/chat), theo các bước ②–⑧ trong plan.
export default function DebugPanel({ debug }) {
  if (!debug) return <p className="muted">Lượt này không có dữ liệu debug.</p>;
  const request = debug.rag_request || {};
  const detection = debug.detection;
  return (
    <div className="trace">
      <Step number="②" title="Nhận diện ảnh">
        {detection ? (
          <KeyValue rows={[
            ["Cây", `${plantName(detection.plant)} (${detection.plant})`],
            ["Bệnh", detection.disease ? `${diseaseName(detection.disease)} (${detection.disease})` : "không có model bệnh cho cây này"],
            ["Độ tin cậy", formatPercent(detection.confidence, 1)],
          ]} />
        ) : (
          <p className="muted">Lượt này không có ảnh.</p>
        )}
      </Step>
      <Step number="③" title="Chuẩn hóa câu hỏi (Groq)">
        {debug.normalizer_failed ? (
          <p className="text-warn">Groq lỗi: câu gốc được gửi thẳng sang RAG (rewrite_query=true).</p>
        ) : null}
        <KeyValue rows={[
          ["Câu gốc", debug.raw_query],
          ["Câu chuẩn hóa", debug.normalized_query],
          ["Ý định", debug.intent ? `${INTENT_LABEL[debug.intent] || debug.intent} (${debug.intent})` : null],
          ["Cây trong câu", debug.explicit_plant],
          ["Bệnh trong câu", debug.explicit_disease
            ? `${debug.explicit_disease}${debug.disease_named === false ? " — chỉ đoán, không dùng để khoanh phạm vi" : ""}`
            : null],
          ["Triệu chứng", debug.symptoms?.length ? debug.symptoms.join("; ") : null],
          ["Trọng tâm", debug.focus],
          ["Ám chỉ lượt trước", debug.refers_to_previous_context ? "có" : "không"],
        ]} />
      </Step>
      <Step number="④" title="Cây và bệnh đã xác định">
        <KeyValue rows={[
          ["Cây", debug.resolved_plant],
          ["Bệnh", debug.resolved_disease],
          ["Bệnh đoán (không dùng)", debug.suspected_disease || undefined],
          ["Lấy từ", contextSource(debug.context_source)],
          ["Đối tượng", subjectLabel(debug.resolved_plant, debug.resolved_disease) || null],
        ]} />
      </Step>
      <Step number="⑤" title="Điều hướng">
        <p>
          <strong>{ACTION_LABEL[debug.action] || debug.action}</strong> <code>{debug.action}</code>
        </p>
      </Step>
      <Step number="⑥" title="Câu gửi sang RAG">
        <KeyValue rows={[
          ["Câu tìm (retrieval_query)", debug.retrieval_query],
          ["Câu tìm phụ", request.extra_queries?.length ? request.extra_queries.join(" | ") : undefined],
          ["plant_type / disease", request.plant_type || request.disease
            ? `${request.plant_type || "—"} / ${request.disease || "—"}` : request.query ? "không gửi" : null],
          ["rewrite_query", request.rewrite_query ? "true" : undefined],
          ["Lịch sử gửi kèm", request.history ? `${request.history.length} tin nhắn` : null],
        ]} />
      </Step>
      <Step number="⑦–⑧" title="Tìm tài liệu và sinh câu trả lời">
        <KeyValue rows={[
          ["Backend", debug.answer_backend],
          ["Phạm vi tìm", debug.rag_scope_status
            ? `${SCOPE_LABEL[debug.rag_scope_status] || debug.rag_scope_status} (${debug.rag_scope_status})` : null],
          ["Có tài liệu", debug.rag_grounded === null || debug.rag_grounded === undefined ? null : debug.rag_grounded ? "có" : "không"],
          ["Nguồn", debug.rag_sources?.length ? (
            <span className="inline-list">
              {debug.rag_sources.map((source) => (
                <Link key={source} to={`/admin/kb/doc/${source}`}>{source}</Link>
              ))}
            </span>
          ) : null],
        ]} />
      </Step>
      <details className="raw-json">
        <summary>JSON đầy đủ</summary>
        <pre>{JSON.stringify(debug, null, 2)}</pre>
      </details>
    </div>
  );
}
