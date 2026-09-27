#!/bin/sh
# SSH tunnel tới vLLM trên Vast.ai: cổng 8000 của container -> 127.0.0.1:$VAST_LLM_PORT trên Vast.
# Container rag gọi http://vast-tunnel:8000/v1. Rớt kết nối thì ssh thoát và Docker tự chạy lại.
set -eu

if [ -z "${VAST_SSH_HOST:-}" ] || [ -z "${VAST_SSH_PORT:-}" ]; then
    echo "Chưa có VAST_SSH_HOST/VAST_SSH_PORT trong .env gốc: không mở tunnel."
    echo "RAG chỉ dùng được Gemini hoặc Ollama (LLM_PROVIDER trong RAG-module/.env)."
    exec tail -f /dev/null
fi

# Key mount từ Windows có quyền 0777 nên ssh từ chối dùng; chép ra bản chỉ chủ sở hữu đọc được.
install -m 600 /ssh/key /tmp/key

echo "Mở tunnel tới ${VAST_SSH_USER:-root}@${VAST_SSH_HOST}:${VAST_SSH_PORT} (vLLM cổng ${VAST_LLM_PORT:-8000})"
# Mỗi instance Vast có host key mới nên chấp nhận key lần đầu gặp (accept-new).
exec ssh -i /tmp/key -N \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/known_hosts \
    -p "$VAST_SSH_PORT" \
    -L "0.0.0.0:8000:127.0.0.1:${VAST_LLM_PORT:-8000}" \
    "${VAST_SSH_USER:-root}@${VAST_SSH_HOST}"
