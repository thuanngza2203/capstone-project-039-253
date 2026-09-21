#!/usr/bin/env bash
# Chạy một lần trong instance Linux; không cài bất kỳ dependency RAG nào.
set -euo pipefail

upgrade_args=()
case "${1:-}" in
  "") ;;
  --upgrade) upgrade_args=(--upgrade) ;;
  --help|-h)
    echo "Usage: bash install.sh [--upgrade]"
    echo "LLM_PYTHON_BIN=python3.12 bash install.sh  # Chon Python tao venv"
    exit 0
    ;;
  *) echo "Usage: bash install.sh [--upgrade]" >&2; exit 1 ;;
esac
if (( $# > 1 )); then
  echo "Usage: bash install.sh [--upgrade]" >&2
  exit 1
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Chay script nay tren server Linux, khong phai Windows." >&2
  exit 1
fi

module_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$module_dir"
venv_python="$module_dir/.venv/bin/python"

if [[ -e .venv && ! -x "$venv_python" ]]; then
  echo ".venv khong phai venv Linux hop le. Khong copy venv Windows sang server." >&2
  exit 1
fi

# Kiem tra truoc khi tai Torch/vLLM. Python 3.12 la lua chon trong README.
check_python() {
  "$1" -c 'import sys
if not (3, 10) <= sys.version_info[:2] < (3, 15):
    sys.exit("Can Python 3.10-3.14; dung LLM_PYTHON_BIN=python3.12 de chon interpreter.")
print("Python:", sys.executable, sys.version.split()[0])'
}

if [[ ! -d .venv ]]; then
  python_bin="${LLM_PYTHON_BIN:-python3}"
  if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Khong tim thay Python: $python_bin. Cai Python 3.12 va python3-venv theo README." >&2
    exit 1
  fi
  check_python "$python_bin"
  nvidia-smi
  "$python_bin" -m venv .venv
else
  check_python "$venv_python"
  nvidia-smi
fi
if [[ ! -x .venv/bin/python ]]; then
  echo ".venv khong phai venv Linux hop le. Khong copy venv Windows sang server." >&2
  exit 1
fi

# Chi dung venv cua module, ke ca khi shell dang activate venv khac hoac co UV_PYTHON.
"$venv_python" -m pip install --upgrade pip uv
"$venv_python" -m uv pip install --python "$venv_python" -r requirements.txt --torch-backend=auto "${upgrade_args[@]}"
"$venv_python" -m pip check
"$venv_python" -c 'import torch; assert torch.cuda.is_available(), "Torch khong truy cap duoc CUDA"; print(torch.cuda.get_device_name(0))'
mkdir -p runtime
"$venv_python" -m pip freeze > runtime/requirements.freeze.txt
echo "Cai xong. Tao .env theo README, sau do: source .venv/bin/activate && python serve.py"
