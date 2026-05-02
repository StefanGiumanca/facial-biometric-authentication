import type { ReactElement } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { useAdminAuth } from './auth';
import { LoginPage } from './pages/LoginPage';
import { SessionsPage } from './pages/SessionsPage';
import { SessionDetailPage } from './pages/SessionDetailPage';

function RequireAdminKey({ children }: { children: ReactElement }) {
  const { adminKey } = useAdminAuth();
  if (!adminKey) {
    return <Navigate to="/login" replace />;
  }

  return children;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/"
        element={
          <RequireAdminKey>
            <SessionsPage />
          </RequireAdminKey>
        }
      />
      <Route
        path="/sessions/:sessionId"
        element={
          <RequireAdminKey>
            <SessionDetailPage />
          </RequireAdminKey>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
