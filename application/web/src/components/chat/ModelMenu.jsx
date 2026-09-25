import { useEffect, useRef, useState } from "react";
import Icon from "../Icon.jsx";
import { PROVIDER_DESCRIPTION, providerName } from "../../lib/labels.js";

// Nút chọn model ở đầu trang chat, như ChatGPT / Gemini: bấm để mở danh sách, mỗi model có mô tả,
// model đang dùng có dấu tích. Mũi tên lên/xuống để di chuyển, Esc để đóng.
export default function ModelMenu({ providers, value, onChange, answerBackend, webSearch, disabled }) {
  const [open, setOpen] = useState(false);
  const root = useRef(null);
  const button = useRef(null);
  const items = useRef([]);

  useEffect(() => {
    if (!open) return undefined;
    const onPointer = (event) => {
      if (!root.current?.contains(event.target)) setOpen(false);
    };
    const onKey = (event) => {
      if (event.key === "Escape") {
        setOpen(false);
        button.current?.focus();
      }
    };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    // Mở ra thì đặt focus vào model đang chọn.
    const selected = Math.max(0, providers.indexOf(value));
    items.current[selected]?.focus();
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, providers, value]);

  // Backend groq của detection: không có model nào để chọn, chỉ hiện tên.
  const label = webSearch ? "Tìm trên web" : answerBackend === "groq" ? "Groq" : providerName(value);
  if (!label) return null;
  const choosable = providers.length > 1;

  if (!choosable) {
    return (
      <div className="model-menu">
        <span className="model-button is-static">{webSearch ? <Icon name="globe" size={18} /> : null}{label}</span>
      </div>
    );
  }

  function onMenuKey(event) {
    const index = items.current.indexOf(document.activeElement);
    const last = providers.length - 1;
    const next = { ArrowDown: index >= last ? 0 : index + 1, ArrowUp: index <= 0 ? last : index - 1, Home: 0, End: last }[event.key];
    if (next !== undefined) {
      event.preventDefault();
      items.current[next]?.focus();
    } else if (event.key === "Tab") {
      setOpen(false);
    }
  }

  function choose(provider) {
    onChange(provider);
    setOpen(false);
    button.current?.focus();
  }

  return (
    <div className="model-menu" ref={root}>
      <button
        ref={button}
        type="button"
        className="model-button"
        aria-haspopup="menu"
        aria-expanded={open}
        disabled={disabled}
        onClick={() => setOpen((value) => !value)}
      >
        {webSearch ? <Icon name="globe" size={18} /> : null}
        <span>{label}</span>
        <Icon name="chevronDown" size={16} />
      </button>
      {open ? (
        <div className="model-popover" role="menu" aria-label="Chọn model trả lời" onKeyDown={onMenuKey}>
          <div className="model-popover-title">Model trả lời</div>
          {webSearch ? (
            <p className="model-popover-note">
              Đang bật Tìm trên web: câu trả lời lấy từ Groq. Tắt Tìm trên web để dùng model đã chọn dưới đây.
            </p>
          ) : null}
          {providers.map((provider, index) => (
            <button
              key={provider}
              ref={(element) => { items.current[index] = element; }}
              type="button"
              role="menuitemradio"
              aria-checked={provider === value}
              className="model-option"
              onClick={() => choose(provider)}
            >
              <span className="model-option-text">
                <span className="model-option-name">{providerName(provider)}</span>
                {PROVIDER_DESCRIPTION[provider] ? (
                  <span className="model-option-description">{PROVIDER_DESCRIPTION[provider]}</span>
                ) : null}
              </span>
              {provider === value ? <Icon name="check" size={18} className="model-option-check" /> : null}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
