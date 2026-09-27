# Detection server (FastAPI, cổng 8005). Build và chạy qua docker-compose.yml ở thư mục gốc.
FROM python:3.11-slim

# Cache của pip nằm trong cache mount của BuildKit (không vào image): build bị ngắt giữa chừng
# thì lần sau dùng lại gói đã tải. Timeout dài cho mạng chậm.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

# opencv-python (ultralytics kéo theo) cần libGL và glib.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch bản CPU giống .venv trên máy; cài trước requirements để pip không kéo bản CUDA của PyPI.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt docker/constraints.txt /tmp/deps/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/deps/requirements.txt -c /tmp/deps/constraints.txt \
    && python -c "import cv2, ultralytics, timm"

COPY . .

EXPOSE 8005
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8005"]
