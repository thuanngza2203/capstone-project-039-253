import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Chạy local: web gọi /detection/... và /rag/..., Vite chuyển tiếp tới hai backend.
// Nhờ vậy điện thoại trong cùng Wi-Fi chỉ cần một địa chỉ (http://<IP máy host>:5173).
// Deploy: đặt VITE_DETECTION_URL / VITE_RAG_URL trong .env.production, proxy không còn dùng.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const detection = env.DETECTION_TARGET || "http://127.0.0.1:8005";
  const rag = env.RAG_TARGET || "http://127.0.0.1:8010";

  const proxy = {
    "/detection": {
      target: detection,
      rewrite: (path) => path.replace(/^\/detection/, ""),
      // Model 27B trên Vast có thể mất 10–30 giây; lượt đầu còn chờ nạp model nhận diện.
      timeout: 180000,
      proxyTimeout: 180000,
    },
    "/rag": {
      target: rag,
      rewrite: (path) => path.replace(/^\/rag/, ""),
      timeout: 180000,
      proxyTimeout: 180000,
    },
  };

  return {
    plugins: [react()],
    server: { host: true, port: 5173, proxy },
    preview: { host: true, port: 4173, proxy },
  };
});
