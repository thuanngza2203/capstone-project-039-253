import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import Icon from "../Icon.jsx";
import { shrinkImage } from "../../lib/image.js";

const coarsePointer = () => window.matchMedia?.("(pointer: coarse)").matches;

// Ô nhập: chữ, ảnh hoặc cả hai. Ảnh được thu nhỏ ngay khi chọn để xem trước và gửi nhanh.
// `webSearch`/`onToggleWebSearch`: nút "Tìm web"; không truyền `onToggleWebSearch` thì không hiện nút.
const Composer = forwardRef(function Composer({ disabled, onSend, webSearch = false, onToggleWebSearch }, ref) {
  const [text, setText] = useState("");
  const [image, setImage] = useState(null);
  const [preparing, setPreparing] = useState(false);
  const fileInput = useRef(null);
  const textArea = useRef(null);

  useImperativeHandle(ref, () => ({
    pickImage: () => fileInput.current?.click(),
    setText: (value) => {
      setText(value);
      textArea.current?.focus();
    },
  }));

  useEffect(() => {
    const element = textArea.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 160)}px`;
  }, [text]);

  function removeImage() {
    if (image?.previewUrl) URL.revokeObjectURL(image.previewUrl);
    setImage(null);
  }

  async function choose(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (image?.previewUrl) URL.revokeObjectURL(image.previewUrl);
    setPreparing(true);
    const prepared = await shrinkImage(file);
    setPreparing(false);
    setImage({ ...prepared, previewUrl: URL.createObjectURL(prepared.blob), originalSize: file.size });
  }

  function submit(event) {
    event?.preventDefault();
    const message = text.trim();
    if (disabled || preparing || (!message && !image)) return;
    // Không revoke previewUrl: bong bóng tin nhắn vừa gửi vẫn hiện ảnh này.
    onSend({ text: message, image });
    setText("");
    setImage(null);
  }

  function onKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing && !coarsePointer()) {
      submit(event);
    }
  }

  const canSend = !disabled && !preparing && (text.trim() || image);

  return (
    <form className="composer" onSubmit={submit}>
      {image || preparing ? (
        <div className="composer-attachment">
          {preparing ? (
            <span className="muted">Đang chuẩn bị ảnh…</span>
          ) : (
            <>
              <img src={image.previewUrl} alt="Ảnh sẽ gửi" />
              <div className="attachment-meta">
                <span>{image.name}</span>
                <span className="muted">
                  {(image.blob.size / 1024).toFixed(0)} KB
                  {image.shrunk ? ` (gốc ${(image.originalSize / 1024 / 1024).toFixed(1)} MB)` : ""}
                </span>
              </div>
              <button type="button" className="icon-btn" onClick={removeImage} aria-label="Bỏ ảnh">
                <Icon name="close" />
              </button>
            </>
          )}
        </div>
      ) : null}
      <div className="composer-row">
        <button
          type="button"
          className="icon-btn composer-image"
          onClick={() => fileInput.current?.click()}
          aria-label="Chọn hoặc chụp ảnh lá"
          disabled={disabled}
        >
          <Icon name="camera" />
        </button>
        <input ref={fileInput} type="file" accept="image/*" hidden onChange={choose} />
        {onToggleWebSearch ? (
          <button
            type="button"
            className={`web-search-toggle ${webSearch ? "is-on" : ""}`}
            onClick={onToggleWebSearch}
            aria-pressed={webSearch}
            aria-label="Tìm trên web"
            title={webSearch
              ? "Đang bật: Groq tìm trên web và trả lời, không dùng kho tài liệu. Bấm để tắt."
              : "Tìm trên web: Groq tìm web và trả lời, không dùng kho tài liệu."}
          >
            <Icon name="globe" size={18} />
            <span className="web-search-label">Tìm web</span>
          </button>
        ) : null}
        <label className="visually-hidden" htmlFor="chat-input">Câu hỏi</label>
        <textarea
          id="chat-input"
          ref={textArea}
          rows={1}
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={onKeyDown}
          // Ngắn để vừa một dòng trên điện thoại 360 px (bên cạnh nút ảnh, Tìm web, Gửi).
          placeholder={webSearch ? "Hỏi và tìm trên web…" : image ? "Hỏi thêm về ảnh…" : "Hỏi hoặc gửi ảnh lá…"}
          maxLength={2000}
        />
        <button type="submit" className="send-btn" disabled={!canSend} aria-label="Gửi">
          <Icon name="send" />
        </button>
      </div>
    </form>
  );
});

export default Composer;
