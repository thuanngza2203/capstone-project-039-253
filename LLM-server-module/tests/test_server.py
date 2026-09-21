"""Test offline: config, khởi động process và HTTP contract; không tải model."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_api
import serve


@contextmanager
def local_http_server(handler):
    """HTTP thật trên loopback để kiểm tra cả hành vi urllib, không chỉ mock."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        with patch.dict(os.environ, {"NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}):
            yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class ServerTests(unittest.TestCase):
    def test_environment_overrides_file_and_does_not_load_rag_env(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("LLM_MODEL_ID=file-model\nLLM_API_KEY=file-key\n", encoding="utf-8")
            with patch.dict(os.environ, {"LLM_MODEL_ID": "shell-model"}, clear=True):
                settings = serve.ServerSettings.from_environment(serve.read_environment(env_file))
        self.assertEqual(settings.model_id, "shell-model")
        self.assertEqual(settings.api_key, "file-key")
        self.assertEqual(settings.port, 8000)

    def test_invalid_config_fails_before_starting_vllm(self):
        cases = [
            ("LLM_API_KEY", ""), ("LLM_API_KEY", "two words"),
            ("LLM_PORT", "65536"), ("LLM_PORT", "0"),
            ("LLM_MAX_MODEL_LEN", ""), ("LLM_MAX_NUM_SEQS", "-1"),
            ("LLM_TENSOR_PARALLEL_SIZE", "abc"),
            ("LLM_GPU_MEMORY_UTILIZATION", "nan"),
            ("LLM_GPU_MEMORY_UTILIZATION", "inf"),
            ("LLM_GPU_MEMORY_UTILIZATION", "1.1"),
            ("LLM_LANGUAGE_MODEL_ONLY", "maybe"), ("LLM_MODEL_ID", " "),
        ]
        for name, value in cases:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                serve.ServerSettings.from_environment({"LLM_API_KEY": "test-key", name: value})

    def test_launch_passes_key_in_environment_and_keeps_backend_private(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            env = {
                "LLM_API_KEY": "secret-not-in-argv",
                "VLLM_API_KEY": "stale-key",
                "LLM_PORT": "8123",
                "LLM_CACHE_DIR": str(cache),
            }
            with patch.object(serve, "read_environment", return_value=env), \
                 patch.object(serve.sys, "platform", "linux"), \
                 patch.object(serve.shutil, "which", return_value="/venv/bin/vllm") as lookup, \
                 patch.object(serve.os, "execvpe") as execute, redirect_stdout(io.StringIO()) as output:
                self.assertEqual(serve.main([]), 0)
            executable, command, child_env = execute.call_args.args
            self.assertEqual(executable, "/venv/bin/vllm")
            self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
            self.assertEqual(command[command.index("--port") + 1], "8123")
            self.assertEqual(child_env["VLLM_API_KEY"], "secret-not-in-argv")
            self.assertEqual(child_env["HF_HOME"], str(cache.resolve()))
            lookup.assert_called_once_with("vllm", path=str(Path(serve.sys.executable).parent))
            self.assertNotIn("secret-not-in-argv", " ".join(command) + output.getvalue())
            self.assertTrue(cache.is_dir())

    def test_dry_run_does_not_create_cache_or_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "not-created"
            with patch.object(serve, "read_environment", return_value={
                "LLM_API_KEY": "test-key", "LLM_CACHE_DIR": str(cache),
                "LLM_REASONING_PARSER": "", "LLM_LANGUAGE_MODEL_ONLY": "false",
            }), patch.object(serve.os, "execvpe") as execute, redirect_stdout(io.StringIO()) as output:
                self.assertEqual(serve.main(["--dry-run"]), 0)
            execute.assert_not_called()
            self.assertFalse(cache.exists())
            self.assertNotIn("--reasoning-parser", output.getvalue())
            self.assertNotIn("--language-model-only", output.getvalue())
            self.assertNotIn("test-key", output.getvalue())

    def test_missing_executable_reports_error_without_loading_model(self):
        with patch.object(serve, "read_environment", return_value={"LLM_API_KEY": "test-key"}), \
             patch.object(serve.sys, "platform", "linux"), \
             patch.object(serve.shutil, "which", return_value=None), \
             redirect_stderr(io.StringIO()) as error, redirect_stdout(io.StringIO()):
            self.assertEqual(serve.main([]), 1)
        self.assertIn("install.sh", error.getvalue())


class ApiTests(unittest.TestCase):
    def test_real_request_encoding_uses_remote_endpoint_and_bearer_key(self):
        responses = [
            {"data": [{"id": "plant-chat"}]},
            {"choices": [{"message": {"content": "Sẵn sàng."}}]},
        ]
        with patch.object(check_api, "open_request", side_effect=[
            io.BytesIO(json.dumps(response).encode()) for response in responses
        ]) as send:
            answer = check_api.check_api("https://llm.example.com:31443/v1/", "plant-chat", "test-key")
        self.assertEqual(answer, "Sẵn sàng.")
        get_request = send.call_args_list[0].args[0]
        post_request = send.call_args_list[1].args[0]
        self.assertEqual(get_request.full_url, "https://llm.example.com:31443/v1/models")
        self.assertEqual(get_request.get_method(), "GET")
        self.assertEqual(post_request.full_url, "https://llm.example.com:31443/v1/chat/completions")
        self.assertEqual(post_request.get_method(), "POST")
        self.assertEqual(post_request.get_header("Authorization"), "Bearer test-key")
        payload = json.loads(post_request.data)
        self.assertEqual(payload["model"], "plant-chat")
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_wrong_model_is_detected_before_chat_request(self):
        with patch.object(check_api, "request_json", return_value={"data": [{"id": "another-model"}]}) as send:
            with self.assertRaisesRegex(ValueError, "another-model"):
                check_api.check_api("http://localhost:8000/v1", "plant-chat", "test-key")
        self.assertEqual(send.call_count, 1)

    def test_empty_or_malformed_answer_is_not_reported_as_success(self):
        for response in [{}, {"choices": []}, {"choices": [{"message": {"content": ""}}]}]:
            with self.subTest(response=response), patch.object(check_api, "request_json", side_effect=[
                {"data": [{"id": "plant-chat"}]}, response,
            ]), self.assertRaises(ValueError):
                check_api.check_api("http://localhost:8000/v1", "plant-chat", "test-key")

    def test_rag_env_selects_endpoint_model_and_key_without_loading_rag(self):
        with patch.object(check_api, "read_environment", return_value={
            "VLLM_BASE_URL": "https://llm.example.com/v1",
            "VLLM_MODEL": "remote-model", "VLLM_API_KEY": "remote-key",
        }), patch.object(check_api, "check_api", return_value="OK") as check, redirect_stdout(io.StringIO()):
            self.assertEqual(check_api.main(["--env-file", "host.env"]), 0)
        check.assert_called_once_with("https://llm.example.com/v1", "remote-model", "remote-key", timeout=120)

    def test_unauthorized_response_has_clear_error_without_leaking_body(self):
        with patch.object(check_api, "read_environment", return_value={"LLM_API_KEY": "secret-key"}), \
             patch.object(check_api, "check_api", side_effect=HTTPError(
                 "http://localhost:8000/v1/models", 401, "Unauthorized", {}, io.BytesIO(b"secret-key")
             )), redirect_stderr(io.StringIO()) as error:
            self.assertEqual(check_api.main([]), 1)
        self.assertIn("HTTP 401", error.getvalue())
        self.assertNotIn("secret-key", error.getvalue())

    def test_invalid_url_does_not_send_credentials(self):
        for url in ["file:///v1", "https://example.com", "https://user:pass@example.com/v1"]:
            with self.subTest(url=url), patch.object(check_api, "open_request") as send, self.assertRaises(ValueError):
                check_api.check_api(url, "plant-chat", "test-key")
            send.assert_not_called()

    def test_bad_env_path_reports_error_before_any_request(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(check_api, "open_request") as send, \
             redirect_stderr(io.StringIO()) as error:
            result = check_api.main(["--env-file", str(Path(directory) / "missing.env")])
        self.assertEqual(result, 1)
        self.assertIn("--env-file", error.getvalue())
        send.assert_not_called()

    def test_windows_utf8_bom_does_not_hide_first_setting(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / ".env"
            path.write_text("VLLM_MODEL=custom-model\nVLLM_API_KEY=test-key\n", encoding="utf-8-sig")
            env = serve.read_environment(path)
        self.assertEqual(env["VLLM_MODEL"], "custom-model")

    def test_invalid_key_is_rejected_without_exposing_it(self):
        with patch.object(check_api, "open_request") as send:
            with self.assertRaises(ValueError) as error:
                check_api.check_api("http://localhost:8000/v1", "plant-chat", "secret\nvalue")
        self.assertNotIn("secret", str(error.exception))
        send.assert_not_called()


class HttpIntegrationTests(unittest.TestCase):
    def test_checker_sends_real_http_and_handles_authentication(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(("GET", self.path, self.headers.get("Authorization")))
                if self.headers.get("Authorization") != "Bearer test-key":
                    self.reply(401, {"error": "Unauthorized"})
                elif self.path == "/v1/models":
                    self.reply(200, {"data": [{"id": "plant-chat"}]})
                else:
                    self.reply(404, {})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(("POST", self.path, payload))
                self.reply(200, {"choices": [{"message": {"content": "Sẵn sàng."}}]})

            def reply(self, status, payload):
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        with local_http_server(Handler) as url:
            self.assertEqual(check_api.check_api(f"{url}/v1", "plant-chat", "test-key"), "Sẵn sàng.")
            with self.assertRaises(HTTPError) as error:
                check_api.check_api(f"{url}/v1", "plant-chat", "wrong-key")
        self.assertEqual(error.exception.code, 401)
        self.assertEqual(requests[0], ("GET", "/v1/models", "Bearer test-key"))
        self.assertEqual(requests[1][:2], ("POST", "/v1/chat/completions"))
        self.assertEqual(requests[1][2]["model"], "plant-chat")
        self.assertEqual(len(requests), 3)  # Sai key dừng trước bước sinh câu trả lời.

    def test_redirect_does_not_forward_token_to_another_endpoint(self):
        destination_requests = []

        class Destination(BaseHTTPRequestHandler):
            def do_GET(self):
                destination_requests.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        with local_http_server(Destination) as destination_url:
            class Origin(BaseHTTPRequestHandler):
                redirect_status = 302

                def do_GET(self):
                    self.send_response(self.redirect_status)
                    self.send_header("Location", f"{destination_url}/v1/models")
                    self.send_header("Content-Length", "0")
                    self.end_headers()

                def log_message(self, *args):
                    pass

            with local_http_server(Origin) as origin_url:
                for status in (301, 302, 303, 307, 308):
                    Origin.redirect_status = status
                    with self.subTest(status=status), self.assertRaises(HTTPError) as error:
                        check_api.request_json(f"{origin_url}/v1/models", "test-key")
                    self.assertEqual(error.exception.code, status)
        self.assertEqual(destination_requests, [])


if __name__ == "__main__":
    unittest.main()
