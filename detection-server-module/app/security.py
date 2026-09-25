import base64
import hashlib
import hmac
import logging
import secrets
import time
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBasic,
    HTTPBasicCredentials,
    HTTPBearer,
)

from app.config import Settings, get_settings


logger = logging.getLogger(__name__)

# auto_error=False: khi chưa đặt ADMIN_PASSWORD thì không bắt trình duyệt đăng nhập.
_basic = HTTPBasic(auto_error=False, realm="PlantGPT admin")
_bearer = HTTPBearer(auto_error=False, description="Token của POST /admin/login (trang admin của web).")


def admin_auth_enabled() -> bool:
    return bool(get_settings().admin_password)


def credentials_ok(username: str, password: str, settings: Settings) -> bool:
    user_ok = secrets.compare_digest(username.encode(), settings.admin_username.encode())
    password_ok = secrets.compare_digest(password.encode(), settings.admin_password.encode())
    return user_ok and password_ok


# --- Token cho web: "<payload>.<chữ ký>", payload = "tên:hết hạn (epoch giây)" ---------------

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _signing_key(settings: Settings) -> bytes:
    # Ký bằng chính tên + mật khẩu admin: đổi mật khẩu là mọi phiên cũ hết hiệu lực,
    # không cần thêm secret hay lưu phiên ở đâu.
    return hashlib.sha256(
        f"plantgpt-admin-token\0{settings.admin_username}\0{settings.admin_password}".encode()
    ).digest()


def issue_admin_token(settings: Settings, now: float | None = None) -> tuple[str, int]:
    expires = int((time.time() if now is None else now) + settings.admin_session_hours * 3600)
    payload = f"{settings.admin_username}:{expires}".encode()
    signature = hmac.new(_signing_key(settings), payload, hashlib.sha256).digest()
    return f"{_b64(payload)}.{_b64(signature)}", expires


def verify_admin_token(token: str, settings: Settings, now: float | None = None) -> bool:
    try:
        payload_text, signature_text = token.split(".")
        payload, signature = _unb64(payload_text), _unb64(signature_text)
    except ValueError:  # gồm cả lỗi base64 (binascii.Error)
        return False
    expected = hmac.new(_signing_key(settings), payload, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        return False
    username, _, expires = payload.decode(errors="replace").rpartition(":")
    current = time.time() if now is None else now
    return username == settings.admin_username and expires.isdigit() and int(expires) > current


def require_admin(
    basic: Annotated[HTTPBasicCredentials | None, Depends(_basic)],
    bearer: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    """Bảo vệ trang /admin và mọi API /admin/*.

    Web gửi token (`Authorization: Bearer`); trang admin cũ dùng HTTP Basic, trình duyệt tự
    hỏi và gửi lại cho các fetch cùng origin. ADMIN_PASSWORD trống thì bỏ qua (giữ hành vi
    cũ); api.py ghi cảnh báo lúc khởi động.
    """
    settings = get_settings()
    if not settings.admin_password:
        return

    if bearer is not None:
        if verify_admin_token(bearer.credentials, settings):
            return
        # Không trả "Basic": trình duyệt sẽ bật hộp đăng nhập riêng đè lên trang đăng nhập của web.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Phiên đăng nhập admin đã hết hạn. Đăng nhập lại.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if basic is not None and credentials_ok(basic.username, basic.password, settings):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Cần đăng nhập admin.",
        headers={"WWW-Authenticate": 'Basic realm="PlantGPT admin"'},
    )
