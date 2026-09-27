# Web: build bằng Vite rồi phục vụ bằng nginx (cổng 80 trong container).
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
# Cache npm nằm trong cache mount: build bị ngắt thì lần sau không tải lại gói đã có.
RUN --mount=type=cache,target=/root/.npm npm ci
COPY . .
# Không đặt VITE_DETECTION_URL / VITE_RAG_URL: web gọi /detection, /rag cùng địa chỉ, nginx chuyển tiếp.
RUN npm run build

FROM nginx:1.27-alpine
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
