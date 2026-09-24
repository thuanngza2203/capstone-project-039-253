import { useEffect, useState } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import Icon from "../../components/Icon.jsx";

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

  useEffect(() => setOpen(false), [location.pathname]);

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
