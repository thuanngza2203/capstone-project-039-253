import { useCallback, useEffect, useRef, useState } from "react";
import Icon from "./Icon.jsx";

// Gọi API khi mở trang và khi bấm "Tải lại"; bỏ kết quả của lần gọi cũ nếu đã có lần gọi mới.
export function useLoad(loader, deps = []) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const counter = useRef(0);
  const run = useCallback(async () => {
    const id = ++counter.current;
    setState((previous) => ({ ...previous, loading: true, error: null }));
    try {
      const data = await loader();
      if (id === counter.current) setState({ data, error: null, loading: false });
    } catch (error) {
      if (id === counter.current) setState((previous) => ({ ...previous, error, loading: false }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  useEffect(() => {
    run();
  }, [run]);
  return { ...state, reload: run };
}

export function Spinner({ label = "Đang tải…" }) {
  return (
    <div className="spinner" role="status">
      <span className="spinner-dot" aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}

export function ErrorNote({ error, onRetry, children }) {
  if (!error && !children) return null;
  return (
    <div className="note note-error" role="alert">
      <Icon name="alert" />
      <div className="note-body">
        <div>{children || error?.message || String(error)}</div>
        {onRetry ? (
          <button type="button" className="btn btn-small" onClick={onRetry}>
            <Icon name="refresh" size={16} /> Thử lại
          </button>
        ) : null}
      </div>
    </div>
  );
}

export function Note({ tone = "info", icon, children }) {
  return (
    <div className={`note note-${tone}`}>
      <Icon name={icon || (tone === "warn" ? "alert" : tone === "ok" ? "check" : "info")} />
      {/* Bọc một lớp: note-body xếp dọc, chữ và <code> bên trong phải nằm chung một dòng chảy. */}
      <div className="note-body"><div>{children}</div></div>
    </div>
  );
}

export function Badge({ tone = "neutral", children, title }) {
  return (
    <span className={`badge badge-${tone}`} title={title}>
      {children}
    </span>
  );
}

export function Stat({ label, value, hint, tone }) {
  return (
    <div className={`stat ${tone ? `stat-${tone}` : ""}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {hint ? <div className="stat-hint">{hint}</div> : null}
    </div>
  );
}

export function PageHeader({ title, description, actions }) {
  return (
    <header className="page-header">
      <div>
        <h1>{title}</h1>
        {description ? <p className="page-description">{description}</p> : null}
      </div>
      {actions ? <div className="page-actions">{actions}</div> : null}
    </header>
  );
}

export function Tabs({ tabs, value, onChange }) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((tab) => (
        <button
          key={tab.value}
          type="button"
          role="tab"
          aria-selected={value === tab.value}
          className={`tab ${value === tab.value ? "is-active" : ""}`}
          onClick={() => onChange(tab.value)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

export function Segmented({ options, value, onChange, label }) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={value === option.value}
          className={value === option.value ? "is-active" : ""}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

// Bảng key → value gọn cho metadata, debug.
export function KeyValue({ rows }) {
  const visible = rows.filter(([, value]) => value !== undefined);
  return (
    <dl className="kv">
      {visible.map(([key, value]) => (
        <div className="kv-row" key={key}>
          <dt>{key}</dt>
          <dd>{value === null || value === "" ? <span className="muted">—</span> : value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function RefreshButton({ onClick, loading }) {
  return (
    <button type="button" className="btn btn-quiet" onClick={onClick} disabled={loading}>
      <Icon name="refresh" size={18} className={loading ? "spin" : ""} /> Tải lại
    </button>
  );
}
