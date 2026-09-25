from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.schemas import AdminSession
from app.security import credentials_ok, issue_admin_token

# Không gắn require_admin: đây là nơi lấy token để gọi các route /admin/* còn lại.
router = APIRouter(prefix="/admin", tags=["Admin"])


class AdminLogin(BaseModel):
    username: str = Field(max_length=200)
    password: str = Field(max_length=200)


@router.post("/login", response_model=AdminSession)
async def login(data: AdminLogin):
    """Đăng nhập trang admin của web; trả token dùng trong `Authorization: Bearer`."""
    settings = get_settings()
    if not settings.admin_password:
        raise HTTPException(
            status_code=503,
            detail="Chưa đặt ADMIN_PASSWORD trong .env của detection-server nên chưa đăng nhập được.",
        )
    # 401 không kèm WWW-Authenticate: trình duyệt không bật hộp đăng nhập riêng.
    if not credentials_ok(data.username, data.password, settings):
        raise HTTPException(status_code=401, detail="Sai tên đăng nhập hoặc mật khẩu.")
    token, expires = issue_admin_token(settings)
    return AdminSession(
        token=token,
        username=settings.admin_username,
        expires_at=datetime.fromtimestamp(expires, timezone.utc),
    )
