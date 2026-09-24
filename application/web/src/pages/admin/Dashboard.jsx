import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { detection, rag } from "../../api.js";
import { ChartCard, CHART, DailyBars, HorizontalBars } from "../../components/charts.jsx";
import { ErrorNote, Note, PageHeader, RefreshButton, Segmented, Spinner, Stat, useLoad } from "../../components/ui.jsx";
import { dayKey, formatNumber, formatPercent } from "../../lib/format.js";
import { ACTION_LABEL, SCOPE_LABEL, subjectLabel } from "../../lib/labels.js";

const RANGES = [
  { value: "7", label: "7 ngày" },
  { value: "30", label: "30 ngày" },
  { value: "all", label: "Tất cả" },
];

function sinceOf(range) {
  if (range === "all") return undefined;
  const date = new Date();
  date.setHours(0, 0, 0, 0);
  date.setDate(date.getDate() - Number(range) + 1);
  return date.toISOString();
}

function countBy(items, keyOf, labelOf = (key) => key) {
  const counts = new Map();
  for (const item of items) {
    const key = keyOf(item);
    if (key === null || key === undefined || key === "") continue;
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return [...counts.entries()].map(([key, value]) => ({ key, label: labelOf(key), value })).sort((a, b) => b.value - a.value);
}

function dailyRows(items, range) {
  const counts = countBy(items, (item) => dayKey(item.created_at));
  const byDay = new Map(counts.map((row) => [row.key, row.value]));
  const days = range === "all"
    ? counts.map((row) => row.key).sort()
    : Array.from({ length: Number(range) }, (_, index) => {
      const date = new Date();
      date.setDate(date.getDate() - (Number(range) - 1 - index));
      return dayKey(date);
    });
  if (range === "all" && days.length > 1) {
    // Điền cả những ngày không có lượt nào, để trục thời gian liên tục.
    const filled = [];
    const cursor = new Date(`${days[0]}T00:00:00`);
    const end = new Date(`${days[days.length - 1]}T00:00:00`);
    while (cursor <= end && filled.length < 400) {
      filled.push(dayKey(cursor));
      cursor.setDate(cursor.getDate() + 1);
    }
    days.splice(0, days.length, ...filled);
  }
  return days.map((key) => ({ label: `${key.slice(8, 10)}/${key.slice(5, 7)}`, value: byDay.get(key) || 0 }));
}

async function loadAll(range) {
  const [reviews, overview, llm, health] = await Promise.allSettled([
    detection.reviews({ all: true, from: sinceOf(range) }),
    rag.overview(),
    rag.llm(true),
    detection.health(),
  ]);
  return { reviews, overview, llm, health };
}

export default function Dashboard() {
  const [range, setRange] = useState("7");
  const { data, loading, reload } = useLoad(() => loadAll(range), [range]);

  const records = data?.reviews.status === "fulfilled" ? data.reviews.value : [];
  const stats = useMemo(() => {
    const sessions = new Set(records.map((item) => item.session_id)).size;
    const withGrounding = records.filter((item) => item.metadata?.grounded !== null && item.metadata?.grounded !== undefined);
    const grounded = withGrounding.filter((item) => item.metadata.grounded).length;
    const likes = records.filter((item) => item.user_feedback?.rating === "like").length;
    const unlikes = records.filter((item) => item.user_feedback?.rating === "unlike").length;
    return {
      answers: records.length, sessions, grounded, groundedBase: withGrounding.length, likes, unlikes,
      daily: dailyRows(records, range),
      actions: countBy(records, (item) => item.action || item.metadata?.action, (key) => ACTION_LABEL[key] || key),
      diseases: countBy(records.filter((item) => item.metadata?.disease),
        (item) => subjectLabel(item.metadata.plant, item.metadata.disease)).slice(0, 10),
      scopes: countBy(records, (item) => item.metadata?.scope_status, (key) => SCOPE_LABEL[key] || key),
      ratings: [
        { key: "like", label: "Hữu ích", value: likes },
        { key: "unlike", label: "Chưa tốt", value: unlikes },
        { key: "none", label: "Chưa đánh giá", value: records.length - likes - unlikes },
      ],
      reasons: countBy(records.flatMap((item) => item.user_feedback?.reasons || []), (reason) => reason),
    };
  }, [records, range]);

  const overview = data?.overview.status === "fulfilled" ? data.overview.value : null;
  const llm = data?.llm.status === "fulfilled" ? data.llm.value : null;
  const defaultIndex = overview?.indexes.find((item) => item.name === overview.default_index);
  const staleIndexes = overview?.indexes.filter((item) => item.matches_data === false) || [];
  const servedModel = llm?.served?.[0];

  return (
    <>
      <PageHeader
        title="Tổng quan"
        description="Mỗi câu trả lời của bot là một bản ghi. Số liệu tính trên trình duyệt từ API của detection và RAG."
        actions={(
          <>
            <Segmented label="Khoảng thời gian" options={RANGES} value={range} onChange={setRange} />
            <RefreshButton onClick={reload} loading={loading} />
          </>
        )}
      />

      {data?.health.status === "rejected" ? (
        <ErrorNote error={data.health.reason}>Không kết nối được detection-server: {data.health.reason.message}</ErrorNote>
      ) : null}
      {llm && llm.reachable === false ? (
        <Note tone="warn">
          <strong>LLM không kết nối được.</strong> SSH tunnel tới Vast có thể đã đứt hoặc máy Vast đã tắt. {llm.detail}
        </Note>
      ) : null}
      {staleIndexes.map((item) => (
        <Note key={item.name} tone={item.default ? "warn" : "info"}>
          Index <strong>{item.name}</strong> không khớp dữ liệu trong <code>data/</code>
          {item.default ? " (index đang dùng để trả lời)" : " (chỉ dùng khi so sánh)"}: chạy{" "}
          <code>python main.py index --strategy {item.name}</code>.
        </Note>
      ))}

      {data?.reviews.status === "rejected" ? (
        <ErrorNote error={data.reviews.reason} onRetry={reload}>
          Không tải được lượt chat: {data.reviews.reason.message}
        </ErrorNote>
      ) : null}

      {loading && !data ? <Spinner /> : null}

      <div className="grid grid-stats">
        <Stat label="Câu trả lời" value={formatNumber(stats.answers)} hint="mỗi lượt bot trả lời" />
        <Stat label="Cuộc trò chuyện" value={formatNumber(stats.sessions)} />
        <Stat
          label="Có tài liệu"
          value={formatPercent(stats.groundedBase ? stats.grounded / stats.groundedBase : null)}
          hint={`${stats.grounded}/${stats.groundedBase} lượt gọi RAG`}
        />
        <Stat
          label="Được thích"
          value={formatPercent(stats.likes + stats.unlikes ? stats.likes / (stats.likes + stats.unlikes) : null)}
          hint={`${stats.likes} thích · ${stats.unlikes} không thích`}
        />
      </div>

      <ChartCard title="Lượt dùng theo ngày" rows={stats.daily} labelHeader="Ngày" valueHeader="Câu trả lời">
        <DailyBars rows={stats.daily} />
      </ChartCard>

      <div className="grid grid-2">
        <ChartCard title="Cách hệ thống xử lý câu hỏi" description="Bước ⑤: trả lời, xin ảnh, hỏi lại hay ngoài phạm vi" rows={stats.actions}>
          <HorizontalBars rows={stats.actions} />
        </ChartCard>
        <ChartCard title="Kết quả tra cứu" description="Bước ⑦: RAG tìm trong phạm vi nào" rows={stats.scopes}>
          <HorizontalBars rows={stats.scopes} colorOf={(row) => (["unsupported_disease", "unknown_disease"].includes(row.key) ? CHART.bad : CHART.bar)} />
        </ChartCard>
        <ChartCard title="Bệnh được hỏi nhiều nhất" description="10 bệnh đầu, theo cây và bệnh đã xác định ở bước ④" rows={stats.diseases} labelHeader="Cây · bệnh">
          <HorizontalBars rows={stats.diseases} />
        </ChartCard>
        <ChartCard title="Đánh giá của người dùng" rows={[...stats.ratings, ...stats.reasons.map((row) => ({ ...row, label: `Lý do: ${row.label}` }))]}>
          <HorizontalBars
            rows={stats.ratings}
            colorOf={(row) => (row.key === "like" ? CHART.good : row.key === "unlike" ? CHART.bad : CHART.neutral)}
          />
          {stats.reasons.length ? (
            <>
              <h3 className="small muted" style={{ marginTop: "1rem" }}>Lý do không thích</h3>
              <HorizontalBars rows={stats.reasons} colorOf={() => CHART.bad} />
            </>
          ) : null}
        </ChartCard>
      </div>

      <div className="grid grid-2">
        <section className="card">
          <div className="card-header">
            <h2>Kho tri thức</h2>
            <Link to="/admin/kb">Chi tiết</Link>
          </div>
          {data?.overview.status === "rejected" ? (
            <ErrorNote error={data.overview.reason} />
          ) : defaultIndex ? (
            <div className="grid grid-stats" style={{ gridTemplateColumns: "repeat(3, minmax(0,1fr))" }}>
              <Stat label="Tài liệu" value={formatNumber(overview.data.documents)} />
              <Stat label="Chunk" value={formatNumber(defaultIndex.chunks)} hint={`index ${defaultIndex.name}`} />
              <Stat label="Khớp dữ liệu" value={defaultIndex.matches_data ? "Có" : defaultIndex.matches_data === false ? "Không" : "—"} tone={defaultIndex.matches_data === false ? "warn" : undefined} />
            </div>
          ) : <Spinner />}
        </section>
        <section className="card">
          <div className="card-header">
            <h2>Model trả lời</h2>
            <Link to="/admin/system">Hệ thống</Link>
          </div>
          {data?.llm.status === "rejected" ? (
            <ErrorNote error={data.llm.reason} />
          ) : llm ? (
            <div className="grid" style={{ gap: "0.5rem" }}>
              <div><span className="muted small">Provider</span><div><strong>{llm.provider}</strong> · {llm.model}</div></div>
              <div>
                <span className="muted small">Model thật đang chạy</span>
                <div>{servedModel?.root || servedModel?.id || (llm.reachable === false ? "không kết nối được" : "—")}</div>
              </div>
            </div>
          ) : <Spinner />}
        </section>
      </div>
    </>
  );
}
