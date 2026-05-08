import { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { BrandHeader, MetricCard, SectionCard, Shell, StatusBadge, TopNav } from '../components';
import { useAdminAuth } from '../auth';
import { getAdminSessions } from '../lib/api';
import {
  formatDateTime,
  getDisplayName,
  getFaceMatchDistance,
  matchesFilter,
  normalizeDecision,
  sessionMatchesSearch,
  shortSessionId,
  sortSessions,
  type SessionFilter,
  type SessionSortKey,
  type SortDirection,
} from '../lib/utils';
import type { AdminSession } from '../types';

const FILTERS: SessionFilter[] = ['ALL', 'VERIFIED', 'REJECTED', 'MANUAL_REVIEW', 'IN_PROGRESS'];
const SORT_OPTIONS: Array<{ value: SessionSortKey; label: string }> = [
  { value: 'updated_at', label: 'Updated date' },
  { value: 'created_at', label: 'Created date' },
  { value: 'name', label: 'Name' },
  { value: 'final_decision', label: 'Final decision' },
  { value: 'face_match_distance', label: 'Face match distance' },
];

export function SessionsPage() {
  const navigate = useNavigate();
  const { adminKey, clearAdminKey } = useAdminAuth();
  const [sessions, setSessions] = useState<AdminSession[]>([]);
  const [activeFilter, setActiveFilter] = useState<SessionFilter>('ALL');
  const [searchTerm, setSearchTerm] = useState('');
  const [sortKey, setSortKey] = useState<SessionSortKey>('updated_at');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
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

  const visibleSessions = useMemo(
    () =>
      sortSessions(
        sessions.filter((session) => matchesFilter(session, activeFilter) && sessionMatchesSearch(session, searchTerm)),
        sortKey,
        sortDirection,
      ),
    [activeFilter, searchTerm, sessions, sortDirection, sortKey],
  );

  const metrics = useMemo(() => {
    const verified = sessions.filter((session) => matchesFilter(session, 'VERIFIED')).length;
    const rejected = sessions.filter((session) => matchesFilter(session, 'REJECTED')).length;
    const review = sessions.filter((session) => matchesFilter(session, 'MANUAL_REVIEW')).length;
    const inProgress = sessions.filter((session) => matchesFilter(session, 'IN_PROGRESS')).length;
    return { verified, rejected, review, inProgress };
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

      <div className="metric-grid metric-grid--five">
        <MetricCard label="Total sessions" value={String(sessions.length)} helper="Latest 50 records from /admin/sessions" />
        <MetricCard label="Verified" value={String(metrics.verified)} helper="Accepted or approved outcomes" />
        <MetricCard label="Rejected" value={String(metrics.rejected)} helper="Failed or security-locked sessions" />
        <MetricCard label="Manual review" value={String(metrics.review)} helper="Manual decision required" />
        <MetricCard label="In progress" value={String(metrics.inProgress)} helper="Capture, OCR, or liveness still pending" />
      </div>

      <SectionCard title="Sessions" hint="Filter the queue and drill into a single review case.">
        <div className="toolbar">
          <label className="field field--search">
            <span>Search sessions</span>
            <input
              type="search"
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
              placeholder="Name, CNP, session ID, decision..."
            />
          </label>

          <label className="field field--compact">
            <span>Sort by</span>
            <select value={sortKey} onChange={(event) => setSortKey(event.target.value as SessionSortKey)}>
              {SORT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <button
            type="button"
            className="button button--ghost button--direction"
            onClick={() => setSortDirection((current) => (current === 'asc' ? 'desc' : 'asc'))}>
            {sortDirection === 'asc' ? 'Ascending' : 'Descending'}
          </button>
        </div>

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

        {!isLoading && visibleSessions.length === 0 ? (
          <div className="empty-state">No sessions match the current filter.</div>
        ) : null}

        {!isLoading && visibleSessions.length > 0 ? (
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
                {visibleSessions.map((session) => (
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
                        {typeof getFaceMatchDistance(session) === 'number' ? getFaceMatchDistance(session)?.toFixed(3) : 'No distance'}
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
