import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatNumber } from "../lib/format.js";

// Màu cố định (recharts cần giá trị màu thật, không nhận biến CSS trong thuộc tính SVG).
export const CHART = {
  bar: "#2e6e3e",       // xanh lá của hệ Garden (--color-accent), dùng cho biểu đồ một chuỗi
  good: "#2e6e3e",      // trạng thái tốt (được thích)
  bad: "#b5452a",       // trạng thái xấu (không thích) — luôn đi kèm nhãn chữ
  neutral: "#b9bca8",   // chưa đánh giá
  grid: "#e2e3d6",
  ink: "#56604f",
};

const tooltipStyle = { borderRadius: 8, border: `1px solid ${CHART.grid}`, fontSize: 13 };

function DataTable({ rows, labelHeader, valueHeader }) {
  return (
    <table className="table">
      <thead>
        <tr><th>{labelHeader}</th><th className="num">{valueHeader}</th></tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.label}><td>{row.label}</td><td className="num">{formatNumber(row.value)}</td></tr>
        ))}
      </tbody>
    </table>
  );
}

// Thẻ biểu đồ có nút "Xem bảng": trên điện thoại đọc số trong biểu đồ khó.
export function ChartCard({ title, description, rows, labelHeader = "Mục", valueHeader = "Số lượt", children }) {
  const [table, setTable] = useState(false);
  return (
    <section className="card">
      <div className="card-header">
        <div>
          <h2>{title}</h2>
          {description ? <p className="small muted" style={{ margin: 0 }}>{description}</p> : null}
        </div>
        <button type="button" className="btn btn-small btn-quiet" onClick={() => setTable((value) => !value)} aria-pressed={table}>
          {table ? "Xem biểu đồ" : "Xem bảng"}
        </button>
      </div>
      {table ? <DataTable rows={rows} labelHeader={labelHeader} valueHeader={valueHeader} /> : children}
    </section>
  );
}

// Cột ngang cho dữ liệu dạng hạng mục (hành động, bệnh, phạm vi...). Một chuỗi nên không cần chú thích.
export function HorizontalBars({ rows, colorOf, height }) {
  if (!rows.length) return <div className="chart-box"><div className="chart-empty">Chưa có dữ liệu</div></div>;
  const boxHeight = height || Math.max(160, rows.length * 34 + 30);
  return (
    <div className="chart-box" style={{ height: boxHeight }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 36, bottom: 4, left: 4 }}>
          <CartesianGrid horizontal={false} stroke={CHART.grid} />
          <XAxis type="number" allowDecimals={false} tick={{ fill: CHART.ink, fontSize: 12 }} stroke={CHART.grid} />
          <YAxis type="category" dataKey="label" width={132} tick={{ fill: CHART.ink, fontSize: 12 }} stroke={CHART.grid} interval={0} />
          <Tooltip cursor={{ fill: "rgba(46,110,62,0.08)" }} contentStyle={tooltipStyle} formatter={(value) => [formatNumber(value), "Số lượt"]} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={22} label={{ position: "right", fill: CHART.ink, fontSize: 12 }}>
            {rows.map((row) => <Cell key={row.label} fill={colorOf ? colorOf(row) : CHART.bar} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// Cột đứng theo ngày.
export function DailyBars({ rows }) {
  if (!rows.some((row) => row.value > 0)) return <div className="chart-box"><div className="chart-empty">Chưa có lượt nào trong khoảng này</div></div>;
  return (
    <div className="chart-box">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 4, left: -16 }}>
          <CartesianGrid vertical={false} stroke={CHART.grid} />
          <XAxis dataKey="label" tick={{ fill: CHART.ink, fontSize: 11 }} stroke={CHART.grid} interval="preserveStartEnd" minTickGap={12} />
          <YAxis allowDecimals={false} tick={{ fill: CHART.ink, fontSize: 12 }} stroke={CHART.grid} />
          <Tooltip cursor={{ fill: "rgba(46,110,62,0.08)" }} contentStyle={tooltipStyle} formatter={(value) => [formatNumber(value), "Câu trả lời"]} />
          <Bar dataKey="value" fill={CHART.bar} radius={[4, 4, 0, 0]} maxBarSize={28} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
