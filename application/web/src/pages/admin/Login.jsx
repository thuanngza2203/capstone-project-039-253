import { useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { detection } from "../../api.js";
import Icon from "../../components/Icon.jsx";
import { ErrorNote, Note } from "../../components/ui.jsx";
import { adminSession, saveSession } from "../../lib/auth.js";

// Tài khoản là ADMIN_USERNAME / ADMIN_PASSWORD trong .env của detection-server.
export default function Login() {
  const location = useLocation();
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const next = location.state?.from || "/admin";

  if (adminSession()) return <Navigate to={next} replace />;

  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      saveSession(await detection.login(username.trim(), password));
      navigate(next, { replace: true });
    } catch (failure) {
      setError(failure);
      setBusy(false);
    }
  }

  return (
    <main className="login-page">
      <form className="card login-card" onSubmit={submit}>
        <div className="sidebar-brand">
          <Icon name="leaf" size={22} />
          <span>Quản trị</span>
        </div>
        <h1>Đăng nhập</h1>
        {location.state?.expired ? <Note tone="warn">Phiên đăng nhập đã hết hạn. Đăng nhập lại để tiếp tục.</Note> : null}
        <ErrorNote error={error} />
        <label className="field">
          <span>Tên đăng nhập</span>
          <input className="input" autoComplete="username" value={username}
            onChange={(event) => setUsername(event.target.value)} required autoFocus />
        </label>
        <label className="field">
          <span>Mật khẩu</span>
          <input className="input" type="password" autoComplete="current-password" value={password}
            onChange={(event) => setPassword(event.target.value)} required />
        </label>
        <button type="submit" className="btn btn-accent btn-block" disabled={busy}>
          {busy ? "Đang đăng nhập…" : "Đăng nhập"}
        </button>
      </form>
    </main>
  );
}
