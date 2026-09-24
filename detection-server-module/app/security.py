import logging
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.config import get_settings


logger = logging.getLogger(__name__)

# auto_error=False: khi chưa đặt ADMIN_PASSWORD thì không bắt trình duyệt đăng nhập.
_basic = HTTPBasic(auto_error=False, realm="PlantGPT admin")


def admin_auth_enabled() -> bool:
    return bool(get_settings().admin_password)


def require_admin(
    credentials: Annotated[HTTPBasicCredentials | None, Depends(_basic)],
) -> None:
    """Bảo vệ trang /admin và mọi API /admin/* bằng HTTP Basic.

    ADMIN_PASSWORD trống thì bỏ qua (giữ hành vi cũ); api.py ghi cảnh báo lúc khởi động.
    Trình duyệt tự gửi lại thông tin đăng nhập cho các fetch cùng origin của trang admin.
    """
    settings = get_settings()
    if not settings.admin_password:
        return

    given_user = credentials.username if credentials else ""
    given_password = credentials.password if credentials else ""
    user_ok = secrets.compare_digest(given_user.encode(), settings.admin_username.encode())
    password_ok = secrets.compare_digest(given_password.encode(), settings.admin_password.encode())
    if not (user_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Cần đăng nhập admin.",
            headers={"WWW-Authenticate": 'Basic realm="PlantGPT admin"'},
        )
