import type { ReactElement } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { useAdminAuth } from './auth';
import { LoginPage } from './pages/LoginPage';
import { SessionsPage } from './pages/SessionsPage';
import { SessionDetailPage } from './pages/SessionDetailPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { AuditLogsPage } from './pages/AuditLogsPage';
import { SettingsPage } from './pages/SettingsPage';
import { ThemeProvider } from './theme';

function RequireAdminKey({ children }: { children: ReactElement }) {
  const { adminKey } = useAdminAuth();
  if (!adminKey) {
    return <Navigate to="/login" replace />;
  }

  return children;
}

export default function App() {
  return (
    <ThemeProvider>
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
          path="/sessions"
          element={
            <RequireAdminKey>
              <SessionsPage />
            </RequireAdminKey>
          }
        />
        <Route
          path="/manual-reviews"
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
        <Route
          path="/audit-logs"
          element={
            <RequireAdminKey>
              <AuditLogsPage />
            </RequireAdminKey>
          }
        />
        <Route
          path="/analytics"
          element={
            <RequireAdminKey>
              <AnalyticsPage />
            </RequireAdminKey>
          }
        />
        <Route
          path="/settings"
          element={
            <RequireAdminKey>
              <SettingsPage />
            </RequireAdminKey>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </ThemeProvider>
  );
}
