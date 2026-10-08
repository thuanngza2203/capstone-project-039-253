"""Chạy installer với công cụ giả trong thư mục tạm; không cài hay tải package."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]


def find_bash() -> str | None:
    if os.name == "nt":
        # Windows bash.exe có thể là launcher WSL; test này chỉ cần Git Bash.
        git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        return str(git_bash) if git_bash.is_file() else None
    return shutil.which("bash")


BASH = find_bash()
PYTHON_STUB = r'''#!/usr/bin/env bash
set -euo pipefail
printf '%s\0' "$0" "$@" '__END_CALL__' >> "$INSTALL_TEST_LOG"
if [[ "${1:-}" == "-c" && "${2:-}" == *sys.version_info* ]]; then
  if [[ "${INSTALL_TEST_OLD_PYTHON:-0}" == "1" ]]; then
    echo "Can Python 3.10-3.14" >&2
    exit 1
  fi
elif [[ "${1:-}" == "-c" && "${2:-}" == *secrets* ]]; then
  echo "generated-test-key"
elif [[ "${1:-}" == "-m" && "${2:-}" == "venv" ]]; then
  mkdir -p "$3/bin"
  cp "$0" "$3/bin/python"
  chmod +x "$3/bin/python"
elif [[ "${1:-}" == "-m" && "${2:-}" == "pip" && "${3:-}" == "freeze" ]]; then
  echo 'vllm==0.29.0'
elif [[ "${1:-}" == "-m" && "${2:-}" == "pip" && "${3:-}" == "config" ]]; then
  # Giả lập mirror ghi trong pip.conf; không có thì pip báo lỗi như thật.
  if [[ -n "${INSTALL_TEST_PIP_CONFIG_INDEX:-}" ]]; then
    echo "$INSTALL_TEST_PIP_CONFIG_INDEX"
  else
    echo "ERROR: No such key - global.index-url" >&2
    exit 1
  fi
elif [[ "${1:-}" == "-m" && "${2:-}" == "uv" ]]; then
  printf '%s' "${UV_DEFAULT_INDEX:-}" > "$INSTALL_TEST_LOG.uv-index"
  printf '%s' "${UV_SYSTEM_CERTS:-}" > "$INSTALL_TEST_LOG.uv-certs"
fi
'''


@unittest.skipUnless(BASH, "Cần Bash (Git Bash trên Windows) để kiểm tra installer.")
class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # Có khoảng trắng để kiểm tra quoting đường dẫn trong shell.
        self.module = self.root / "LLM module"
        self.module.mkdir()
        self.bin_dir = self.root / "fake-bin"
        self.bin_dir.mkdir()
        self.log = self.root / "calls.log"
        for name in ("install.sh", "requirements.txt", ".env.example"):
            shutil.copyfile(MODULE_DIR / name, self.module / name)
        # Kho chứng chỉ giả: test không phụ thuộc /etc/ssl của máy đang chạy.
        self.ca_bundle = self.root / "system-ca.crt"
        self.ca_bundle.write_text("-----BEGIN CERTIFICATE-----\n", encoding="utf-8")
        self.write_tool("python3", PYTHON_STUB)
        self.write_tool("chosen-python", PYTHON_STUB)
        self.write_tool("uname", '#!/usr/bin/env bash\necho "${INSTALL_TEST_OS:-Linux}"\n')
        self.write_tool("nvidia-smi", "#!/usr/bin/env bash\nexit 0\n")
        self.env = os.environ.copy()
        # Không nhận config installer từ máy đang chạy test.
        for key in ("LLM_PYTHON_BIN", "LLM_TORCH_BACKEND", "INSTALL_TEST_OLD_PYTHON", "INSTALL_TEST_OS",
                    "INSTALL_TEST_PIP_CONFIG_INDEX", "PIP_INDEX_URL", "UV_DEFAULT_INDEX", "UV_INDEX_URL",
                    "UV_SYSTEM_CERTS", "LLM_API_KEY"):
            self.env.pop(key, None)
        self.env.update({
            "INSTALL_TEST_BIN": self.bin_dir.as_posix(),
            "INSTALL_TEST_LOG": self.log.as_posix(),
            "LLM_CA_BUNDLE_CANDIDATES": f"{self.root.as_posix()}/missing-ca.crt {self.ca_bundle.as_posix()}",
            "UV_PYTHON": "unrelated-python",
            "VIRTUAL_ENV": str(self.root / "unrelated-venv"),
        })

    def write_tool(self, name: str, content: str):
        path = self.bin_dir / name
        path.write_text(content, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def run_installer(self, *arguments: str):
        # Git Bash chỉnh PATH khi khởi động; thêm stub sau bước đó.
        bootstrap = '''
stub_bin="$INSTALL_TEST_BIN"
if command -v cygpath >/dev/null 2>&1; then
  stub_bin="$(cygpath -u "$stub_bin")"
fi
export PATH="$stub_bin:$PATH"
exec bash install.sh "$@"
'''
        return subprocess.run(
            [BASH, "-c", bootstrap, "test_install", *arguments], cwd=self.module, env=self.env,
            text=True, encoding="utf-8", capture_output=True, timeout=30,
        )

    def calls(self) -> list[list[str]]:
        if not self.log.exists():
            return []
        entries = self.log.read_bytes().decode("utf-8").split("__END_CALL__\0")
        return [entry.rstrip("\0").split("\0") for entry in entries if entry]

    def test_fresh_install_uses_chosen_python_and_targets_own_venv(self):
        self.env["LLM_PYTHON_BIN"] = (self.bin_dir / "chosen-python").as_posix()
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        creation = next(call for call in calls if call[1:3] == ["-m", "venv"])
        self.assertTrue(creation[0].endswith("chosen-python"))
        package_calls = [call for call in calls if call[1:3] in (["-m", "pip"], ["-m", "uv"])]
        self.assertTrue(package_calls)
        for call in package_calls:
            self.assertTrue(call[0].endswith("/LLM module/.venv/bin/python"), call)
        install = next(call for call in package_calls if call[2] == "uv")
        self.assertEqual(install[install.index("--python") + 1], install[0])
        self.assertNotIn("--upgrade", install)
        self.assertIn("--torch-backend=auto", install)
        self.assertEqual(
            (self.module / "runtime/requirements.freeze.txt").read_text().strip(),
            "vllm==0.29.0",
        )

    def uv_install(self) -> list[str]:
        return next(call for call in self.calls() if call[1:3] == ["-m", "uv"])

    def uv_index(self) -> str:
        return Path(f"{self.log}.uv-index").read_text(encoding="utf-8")

    def test_uv_uses_pypi_when_pip_has_no_mirror(self):
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.uv_index(), "")

    def test_uv_reuses_pip_mirror_from_environment(self):
        self.env["PIP_INDEX_URL"] = "https://repo.huaweicloud.com/repository/pypi/simple"
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.uv_index(), "https://repo.huaweicloud.com/repository/pypi/simple")

    def test_uv_reuses_pip_mirror_from_pip_config(self):
        self.env["INSTALL_TEST_PIP_CONFIG_INDEX"] = "https://mirror.example/simple"
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.uv_index(), "https://mirror.example/simple")
        self.assertIn("uv dung chung mirror voi pip", result.stdout)

    def test_explicit_uv_index_is_not_overridden_by_pip_mirror(self):
        self.env["PIP_INDEX_URL"] = "https://mirror.example/simple"
        self.env["UV_DEFAULT_INDEX"] = "https://chosen.example/simple"
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.uv_index(), "https://chosen.example/simple")

    def uv_certs(self) -> str:
        return Path(f"{self.log}.uv-certs").read_text(encoding="utf-8")

    def test_uv_trusts_operating_system_certificates(self):
        # Máy Vast có proxy HTTPS: CA chỉ nằm trong kho của hệ điều hành ("UnknownIssuer" với uv).
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.uv_certs(), "1")
        self.assertIn("kho chung chi cua he dieu hanh", result.stdout)

    def test_uv_system_certs_choice_is_respected(self):
        self.env["UV_SYSTEM_CERTS"] = "0"
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.uv_certs(), "0")

    def test_uv_keeps_bundled_certificates_without_os_store(self):
        self.ca_bundle.unlink()
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.uv_certs(), "")

    def env_lines(self) -> list[str]:
        return (self.module / ".env").read_text(encoding="utf-8").splitlines()

    def test_first_install_creates_env_with_given_key(self):
        # Key có ký tự đặc biệt với sed/awk -v: phải giữ nguyên từng ký tự.
        self.env["LLM_API_KEY"] = r"key/&\n.*$-1"
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = self.env_lines()
        self.assertIn(r"LLM_API_KEY=key/&\n.*$-1", lines)
        expected = (self.module / ".env.example").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), len(expected))
        self.assertEqual([line for line in lines if not line.startswith("LLM_API_KEY=")],
                         [line for line in expected if not line.startswith("LLM_API_KEY=")])
        self.assertIn("VLLM_API_KEY", result.stdout)

    def test_first_install_generates_key_when_not_given(self):
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("LLM_API_KEY=generated-test-key", self.env_lines())

    def test_existing_env_is_kept(self):
        (self.module / ".env").write_text("LLM_API_KEY=dang-dung\n", encoding="utf-8")
        self.env["LLM_API_KEY"] = "key-moi"
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.env_lines(), ["LLM_API_KEY=dang-dung"])

    def test_key_with_whitespace_is_rejected(self):
        self.env["LLM_API_KEY"] = "hai tu"
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("khoang trang", result.stderr)
        self.assertFalse((self.module / ".env").exists())

    def test_torch_backend_pypi_drops_pytorch_index(self):
        self.env["LLM_TORCH_BACKEND"] = "pypi"
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse([arg for arg in self.uv_install() if arg.startswith("--torch-backend")])

    def test_torch_backend_can_be_pinned(self):
        self.env["LLM_TORCH_BACKEND"] = "cu128"
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--torch-backend=cu128", self.uv_install())

    def test_upgrade_reuses_venv_and_explicitly_upgrades_dependencies(self):
        initial = self.run_installer()
        self.assertEqual(initial.returncode, 0, initial.stderr)
        self.log.write_bytes(b"")
        result = self.run_installer("--upgrade")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertFalse(any(call[1:3] == ["-m", "venv"] for call in calls))
        install = next(call for call in calls if call[1:3] == ["-m", "uv"])
        self.assertIn("--upgrade", install)

    def test_windows_venv_is_rejected_without_installing(self):
        (self.module / ".venv/Scripts").mkdir(parents=True)
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Khong copy venv Windows", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_unsupported_python_stops_before_creating_venv_or_installing(self):
        self.env["INSTALL_TEST_OLD_PYTHON"] = "1"
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Python 3.10-3.14", result.stderr)
        self.assertFalse((self.module / ".venv").exists())
        self.assertTrue(all(call[1] == "-c" for call in self.calls()))

    def test_missing_python_reports_interpreter_before_installing(self):
        self.env["LLM_PYTHON_BIN"] = "python-that-is-not-installed-for-this-test"
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Khong tim thay Python", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_non_linux_is_rejected_before_running_python(self):
        self.env["INSTALL_TEST_OS"] = "MINGW64_NT"
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("server Linux", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_unknown_option_is_rejected_without_installing(self):
        result = self.run_installer("--upgarde")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Usage:", result.stderr)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
