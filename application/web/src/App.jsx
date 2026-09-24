import { Suspense, lazy } from "react";
import { Link, Route, Routes } from "react-router-dom";
import Chat from "./pages/Chat.jsx";
import { Spinner } from "./components/ui.jsx";

// Trang admin tải riêng (có thư viện biểu đồ): người dùng chat trên điện thoại không phải tải phần này.
const AdminLayout = lazy(() => import("./pages/admin/AdminLayout.jsx"));
const Dashboard = lazy(() => import("./pages/admin/Dashboard.jsx"));
const Conversations = lazy(() => import("./pages/admin/Conversations.jsx"));
const ConversationDetail = lazy(() => import("./pages/admin/ConversationDetail.jsx"));
const Feedback = lazy(() => import("./pages/admin/Feedback.jsx"));
const KnowledgeBase = lazy(() => import("./pages/admin/KnowledgeBase.jsx"));
const DocumentPage = lazy(() => import("./pages/admin/Document.jsx"));
const ChunkPage = lazy(() => import("./pages/admin/Chunk.jsx"));
const Playground = lazy(() => import("./pages/admin/Playground.jsx"));
const System = lazy(() => import("./pages/admin/System.jsx"));

const page = (element) => <Suspense fallback={<div className="spinner-center"><Spinner /></div>}>{element}</Suspense>;

function NotFound() {
  return (
    <div className="not-found">
      <h1>Không có trang này</h1>
      <Link to="/">Về trang chat</Link>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Chat />} />
      <Route path="/admin" element={page(<AdminLayout />)}>
        <Route index element={page(<Dashboard />)} />
        <Route path="conversations" element={page(<Conversations />)} />
        <Route path="conversations/:id" element={page(<ConversationDetail />)} />
        <Route path="feedback" element={page(<Feedback />)} />
        <Route path="kb" element={page(<KnowledgeBase />)} />
        <Route path="kb/doc/*" element={page(<DocumentPage />)} />
        <Route path="kb/chunk/:id" element={page(<ChunkPage />)} />
        <Route path="kb/playground" element={page(<Playground />)} />
        <Route path="system" element={page(<System />)} />
      </Route>
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}
