import { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { BrandHeader, MetricCard, SectionCard, Shell, StatusBadge, TopNav } from '../components';
import { useAdminAuth } from '../auth';
import { getAdminSessions } from '../lib/api';
import { formatDateTime, getDisplayName, matchesFilter, normalizeDecision, shortSessionId, type SessionFilter } from '../lib/utils';
import type { AdminSession } from '../types';

const FILTERS: SessionFilter[] = ['ALL', 'VERIFIED', 'REJECTED', 'MANUAL_REVIEW', 'IN_PROGRESS'];

export function SessionsPage() {
  const navigate = useNavigate();
  const { adminKey, clearAdminKey } = useAdminAuth();
  const [sessions, setSessions] = useState<AdminSession[]>([]);
  const [activeFilter, setActiveFilter] = useState<SessionFilter>('ALL');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;

    async function loadSessions() {
      try {
        setError('');
        const response = await getAdminSessions(adminKey, 50, 0);
        if (!cancelled) {
          setSessions(response.sessions);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load sessions.';
        if (!cancelled) {
          setError(message);
          if (message.toLowerCase().includes('invalid') || message.includes('403')) {
            clearAdminKey();
            navigate('/login');
          }
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    loadSessions();

    return () => {
      cancelled = true;
    };
  }, [adminKey, clearAdminKey, navigate]);

  const filteredSessions = useMemo(
    () => sessions.filter((session) => matchesFilter(session, activeFilter)),
    [activeFilter, sessions],
  );

  const metrics = useMemo(() => {
    const verified = sessions.filter((session) => matchesFilter(session, 'VERIFIED')).length;
    const rejected = sessions.filter((session) => matchesFilter(session, 'REJECTED')).length;
    const review = sessions.filter((session) => matchesFilter(session, 'MANUAL_REVIEW')).length;
    return { verified, rejected, review };
  }, [sessions]);

  return (
    <Shell>
      <TopNav onSignOut={() => {
        clearAdminKey();
        navigate('/login');
      }} />

      <BrandHeader
        title="Review Dashboard"
        subtitle="FastAPI-backed session monitoring for manual verification and audit inspection."
        action={
          <button type="button" className="button button--secondary" onClick={() => window.location.reload()}>
            Refresh Data
          </button>
        }
      />

      <div className="metric-grid">
        <MetricCard label="Total sessions" value={String(sessions.length)} helper="Latest 50 records from /admin/sessions" />
        <MetricCard label="Verified" value={String(metrics.verified)} helper="Accepted or approved outcomes" />
        <MetricCard label="Rejected" value={String(metrics.rejected)} helper="Failed or security-locked sessions" />
        <MetricCard label="Manual review" value={String(metrics.review)} helper="Needs human decision endpoint" />
      </div>

      <SectionCard title="Sessions" hint="Filter the queue and drill into a single review case.">
        <div className="filter-row">
          {FILTERS.map((filter) => (
            <button
              key={filter}
              type="button"
              className={filter === activeFilter ? 'filter-pill filter-pill--active' : 'filter-pill'}
              onClick={() => setActiveFilter(filter)}>
              {filter.replace('_', ' ')}
            </button>
          ))}
        </div>

        {error ? <div className="error-banner">{error}</div> : null}

        {isLoading ? <div className="empty-state">Loading admin sessions...</div> : null}

        {!isLoading && filteredSessions.length === 0 ? (
          <div className="empty-state">No sessions match the current filter.</div>
        ) : null}

        {!isLoading && filteredSessions.length > 0 ? (
          <div className="sessions-table-wrapper">
            <table className="sessions-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Session</th>
                  <th>Decision</th>
                  <th>Liveness</th>
                  <th>Face Match</th>
                  <th>Updated</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {filteredSessions.map((session) => (
                  <tr key={session.session_id}>
                    <td>
                      <div className="table-primary">{getDisplayName(session.first_name, session.last_name)}</div>
                      <div className="table-secondary">{session.cnp || 'No CNP extracted'}</div>
                    </td>
                    <td>
                      <div className="table-primary">{shortSessionId(session.session_id)}</div>
                      <div className="table-secondary">{session.series_number || 'No series number'}</div>
                    </td>
                    <td>
                      <div className="table-badge-stack">
                        <StatusBadge label={normalizeDecision(session)} />
                        <span className="table-secondary">Status: {session.status}</span>
                      </div>
                    </td>
                    <td>
                      <span className={session.liveness_passed ? 'text-success' : session.liveness_passed === false ? 'text-danger' : 'text-muted'}>
                        {session.liveness_passed ? 'Passed' : session.liveness_passed === false ? 'Failed' : 'Pending'}
                      </span>
                    </td>
                    <td>
                      <div className="table-primary">{session.final_face_match_decision || session.face_match_decision || 'Pending'}</div>
                      <div className="table-secondary">
                        {typeof session.final_face_match_distance === 'number'
                          ? session.final_face_match_distance.toFixed(3)
                          : typeof session.face_match_distance === 'number'
                            ? session.face_match_distance.toFixed(3)
                            : 'No distance'}
                      </div>
                    </td>
                    <td>{formatDateTime(session.updated_at)}</td>
                    <td className="table-action">
                      <Link className="button button--ghost" to={`/sessions/${session.session_id}`}>
                        Inspect
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </SectionCard>
    </Shell>
  );
}
