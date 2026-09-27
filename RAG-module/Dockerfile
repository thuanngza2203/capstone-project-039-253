# RAG API (python -m server) chạy embedding trên GPU. Build và chạy qua docker-compose.yml ở thư mục gốc.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# torch bản CUDA 13.2 giống .venv trên máy; nặng nhất nên cài trước để layer được cache lâu nhất.
RUN pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu132

COPY requirements.txt docker/constraints.txt /tmp/deps/
RUN pip install -r /tmp/deps/requirements.txt -c /tmp/deps/constraints.txt

COPY . .
# File sửa trên Windows có thể mang CRLF, làm /bin/sh báo lỗi khó hiểu.
RUN sed -i 's/\r$//' docker/entrypoint.sh

EXPOSE 8010
ENTRYPOINT ["/bin/sh", "docker/entrypoint.sh"]
CMD ["python", "-m", "server"]
