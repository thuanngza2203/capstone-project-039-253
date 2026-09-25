import { useEffect, useState } from "react";
import { Link, Navigate, NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import Icon from "../../components/Icon.jsx";
import { LOGOUT_EVENT, adminSession, clearSession } from "../../lib/auth.js";

const NAV = [
  { to: "/admin", label: "Tổng quan", icon: "chart", end: true },
  { to: "/admin/conversations", label: "Hội thoại", icon: "message" },
  { to: "/admin/feedback", label: "Feedback", icon: "thumbUp" },
  { to: "/admin/kb", label: "Kho tri thức", icon: "database", end: true },
  { to: "/admin/kb/playground", label: "Thử truy vấn", icon: "flask" },
  { to: "/admin/system", label: "Hệ thống", icon: "pulse" },
];

export default function AdminLayout() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const from = location.pathname + location.search;

  useEffect(() => setOpen(false), [location.pathname]);

  // Đăng xuất, hoặc detection trả 401 (phiên hết hạn / đổi mật khẩu): về trang đăng nhập rồi quay lại đúng trang này.
  useEffect(() => {
    const onLogout = (event) => navigate("/admin/login", {
      replace: true, state: { from, expired: event.detail === "expired" },
    });
    window.addEventListener(LOGOUT_EVENT, onLogout);
    return () => window.removeEventListener(LOGOUT_EVENT, onLogout);
  }, [navigate, from]);

  const session = adminSession();
  if (!session) return <Navigate to="/admin/login" replace state={{ from }} />;

  const current = NAV.slice().reverse().find((item) => (
    item.end ? location.pathname === item.to : location.pathname.startsWith(item.to)
  ));

  return (
    <div className="admin-app">
      <div className={`scrim ${open ? "is-open" : ""}`} onClick={() => setOpen(false)} aria-hidden="true" />
      <aside className={`admin-sidebar ${open ? "is-open" : ""}`} aria-label="Menu quản trị">
        <div className="sidebar-brand">
          <Icon name="leaf" size={22} />
          <span>Quản trị</span>
          <button type="button" className="icon-btn only-mobile" onClick={() => setOpen(false)} aria-label="Đóng menu">
            <Icon name="close" />
          </button>
        </div>
        <nav className="admin-nav">
          {NAV.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.end} className={({ isActive }) => (isActive ? "is-active" : "")}>
              <Icon name={item.icon} size={18} /> {item.label}
            </NavLink>
          ))}
        </nav>
        <nav className="sidebar-footer">
          <Link to="/"><Icon name="chat" size={18} /> Về trang chat</Link>
          <button type="button" className="sidebar-logout" onClick={() => clearSession("logout")}>
            <Icon name="close" size={18} /> Đăng xuất ({session.username})
          </button>
        </nav>
      </aside>
      <div className="admin-main">
        <header className="admin-topbar only-mobile">
          <button type="button" className="icon-btn" onClick={() => setOpen(true)} aria-label="Mở menu">
            <Icon name="menu" />
          </button>
          <span className="admin-topbar-title">{current?.label || "Quản trị"}</span>
        </header>
        <div className="admin-content">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
