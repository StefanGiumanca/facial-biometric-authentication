import { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { BrandHeader, SectionCard, Shell, StatusBadge, Timeline, TopNav } from '../components';
import { useAdminAuth } from '../auth';
import { getAdminSessionLogs, getAdminSessions } from '../lib/api';
import { formatDateTime, getDisplayName, normalizeDecision, sessionMatchesSearch, shortSessionId } from '../lib/utils';
import type { AdminSession, AuditLogEntry } from '../types';

export function AuditLogsPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { adminKey, clearAdminKey } = useAdminAuth();
  const [sessions, setSessions] = useState<AdminSession[]>([]);
  const [logs, setLogs] = useState<AuditLogEntry[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoadingSessions, setIsLoadingSessions] = useState(true);
  const [isLoadingLogs, setIsLoadingLogs] = useState(false);
  const [error, setError] = useState('');
  const selectedSessionId = searchParams.get('session') ?? '';

  useEffect(() => {
    let cancelled = false;

    async function loadSessions() {
      try {
        setError('');
        const response = await getAdminSessions(adminKey, 50, 0);
        if (!cancelled) {
          setSessions(response.sessions);
          if (!selectedSessionId && response.sessions[0]) {
            setSearchParams({ session: response.sessions[0].session_id }, { replace: true });
          }
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load audit sessions.';
        if (!cancelled) {
          setError(message);
          if (message.toLowerCase().includes('invalid') || message.includes('403')) {
            clearAdminKey();
            navigate('/login');
          }
        }
      } finally {
        if (!cancelled) {
          setIsLoadingSessions(false);
        }
      }
    }

    loadSessions();

    return () => {
      cancelled = true;
    };
  }, [adminKey, clearAdminKey, navigate, selectedSessionId, setSearchParams]);

  useEffect(() => {
    let cancelled = false;

    async function loadLogs() {
      if (!selectedSessionId) {
        setLogs([]);
        return;
      }

      try {
        setIsLoadingLogs(true);
        setError('');
        const response = await getAdminSessionLogs(adminKey, selectedSessionId);
        if (!cancelled) {
          setLogs(response.logs);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load audit timeline.';
        if (!cancelled) {
          setError(message);
        }
      } finally {
        if (!cancelled) {
          setIsLoadingLogs(false);
        }
      }
    }

    loadLogs();

    return () => {
      cancelled = true;
    };
  }, [adminKey, selectedSessionId]);

  const visibleSessions = useMemo(
    () => sessions.filter((session) => sessionMatchesSearch(session, searchTerm)),
    [searchTerm, sessions],
  );
  const selectedSession = sessions.find((session) => session.session_id === selectedSessionId);
  const visibleLogs = useMemo(() => {
    const normalizedSearch = searchTerm.trim().toLowerCase();
    if (!normalizedSearch || (selectedSession && sessionMatchesSearch(selectedSession, searchTerm))) {
      return logs;
    }

    return logs.filter((log) => [log.event_type, log.message].some((value) => value.toLowerCase().includes(normalizedSearch)));
  }, [logs, searchTerm, selectedSession]);

  return (
    <Shell>
      <TopNav
        searchTerm={searchTerm}
        onSearchChange={setSearchTerm}
        searchPlaceholder="Search sessions or audit messages..."
        sessionCount={visibleSessions.length}
        onSignOut={() => {
          clearAdminKey();
          navigate('/login');
        }}
      />

      <BrandHeader
        title="Audit Logs"
        subtitle="Select a session to inspect the chronological admin audit timeline from existing session logs."
      />

      {error ? <div className="error-banner">{error}</div> : null}

      <div className="audit-layout">
        <SectionCard title="Recent Sessions" hint="Choose the session whose events you need to inspect.">
          {isLoadingSessions ? <div className="empty-state">Loading sessions...</div> : null}
          {!isLoadingSessions && visibleSessions.length === 0 ? <div className="empty-state">No sessions match the audit search.</div> : null}
          <div className="audit-session-list">
            {visibleSessions.map((session) => (
              <button
                key={session.session_id}
                type="button"
                className={session.session_id === selectedSessionId ? 'audit-session audit-session--active' : 'audit-session'}
                onClick={() => setSearchParams({ session: session.session_id })}>
                <span>
                  <strong>{getDisplayName(session.first_name, session.last_name)}</strong>
                  <small>{shortSessionId(session.session_id)} · {formatDateTime(session.updated_at)}</small>
                </span>
                <StatusBadge label={normalizeDecision(session)} />
              </button>
            ))}
          </div>
        </SectionCard>

        <SectionCard
          title={selectedSession ? `Timeline: ${getDisplayName(selectedSession.first_name, selectedSession.last_name)}` : 'Session Timeline'}
          hint="Log messages are filtered by the top search when audit event text matches.">
          {!selectedSessionId ? (
            <div className="empty-state">Select a session to load its audit trail.</div>
          ) : isLoadingLogs ? (
            <div className="empty-state">Loading audit timeline...</div>
          ) : (
            <>
              <div className="audit-meta">
                <code>{selectedSessionId}</code>
                <Link className="button button--ghost" to={`/sessions/${selectedSessionId}`}>
                  Open Detail
                </Link>
              </div>
              <Timeline items={visibleLogs} />
            </>
          )}
        </SectionCard>
      </div>
    </Shell>
  );
}
