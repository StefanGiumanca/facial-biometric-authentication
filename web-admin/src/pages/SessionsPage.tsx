import { useEffect, useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
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
    const flagged = sessions.filter((session) => session.security_fail_count > 0 || session.liveness_passed === false).length;
    const scoredSessions = sessions
      .map((session) => getFaceMatchDistance(session))
      .filter((distance): distance is number => typeof distance === 'number');
    const averageConfidence =
      scoredSessions.length === 0
        ? null
        : Math.round(
            scoredSessions.reduce((total, distance) => total + Math.max(0, Math.min(1, 1 - distance / 0.6)), 0) /
              scoredSessions.length *
              100,
          );
    return { verified, rejected, review, inProgress, flagged, averageConfidence, total: sessions.length };
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
        <MetricCard label="Total sessions" value={String(sessions.length)} helper="Latest 50 records from /admin/sessions" tone="neutral" progress={100} />
        <MetricCard label="Verified" value={String(metrics.verified)} helper="Accepted or approved outcomes" tone="success" progress={asPercent(metrics.verified, metrics.total)} />
        <MetricCard label="Rejected" value={String(metrics.rejected)} helper="Failed or security-locked sessions" tone="danger" progress={asPercent(metrics.rejected, metrics.total)} />
        <MetricCard label="Manual review" value={String(metrics.review)} helper="Manual decision required" tone="warning" progress={asPercent(metrics.review, metrics.total)} />
        <MetricCard label="In progress" value={String(metrics.inProgress)} helper="Capture, OCR, or liveness still pending" tone="info" progress={asPercent(metrics.inProgress, metrics.total)} />
      </div>

      <motion.div
        className="command-panel"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.42, delay: 0.08 }}>
        <div>
          <span className="command-panel__label">Queue intelligence</span>
          <strong>{visibleSessions.length} visible cases</strong>
          <p>Live filtering, biometric status, and manual review triage for the latest verification sessions.</p>
        </div>
        <div className="decision-mix">
          <DecisionBar label="Verified" tone="success" value={metrics.verified} total={metrics.total} />
          <DecisionBar label="Rejected" tone="danger" value={metrics.rejected} total={metrics.total} />
          <DecisionBar label="Review" tone="warning" value={metrics.review} total={metrics.total} />
          <DecisionBar label="Progress" tone="info" value={metrics.inProgress} total={metrics.total} />
        </div>
      </motion.div>

      <div className="insight-grid">
        <InsightCard
          label="Review focus"
          value={getReviewFocus(metrics)}
          helper="Derived from rejected, review, and in-progress queues"
          tone={metrics.review + metrics.rejected > 0 ? 'warning' : 'success'}
        />
        <InsightCard
          label="Average confidence"
          value={metrics.averageConfidence === null ? 'Pending' : `${metrics.averageConfidence}%`}
          helper="Approximate confidence from stored face-match distances"
          tone={metrics.averageConfidence !== null && metrics.averageConfidence >= 70 ? 'success' : 'info'}
        />
        <InsightCard
          label="Security watch"
          value={`${metrics.flagged} flagged`}
          helper="Sessions with security failures or failed liveness"
          tone={metrics.flagged > 0 ? 'danger' : 'success'}
        />
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
                  <th>Priority</th>
                  <th>Liveness</th>
                  <th>Face Match</th>
                  <th>Updated</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                <AnimatePresence initial={false}>
                {visibleSessions.map((session, index) => (
                  <motion.tr
                    key={session.session_id}
                    layout
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, x: -18 }}
                    transition={{ duration: 0.22, delay: Math.min(index * 0.015, 0.16) }}>
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
                      <PriorityPill session={session} />
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
                  </motion.tr>
                ))}
                </AnimatePresence>
              </tbody>
            </table>
          </div>
        ) : null}
      </SectionCard>
    </Shell>
  );
}

function getReviewFocus(metrics: {
  rejected: number;
  review: number;
  inProgress: number;
  total: number;
}) {
  if (metrics.rejected > 0) {
    return 'Rejected cases';
  }

  if (metrics.review > 0) {
    return 'Manual review';
  }

  if (metrics.inProgress > 0) {
    return 'In progress';
  }

  if (metrics.total === 0) {
    return 'No sessions';
  }

  return 'Healthy queue';
}

function InsightCard({
  label,
  value,
  helper,
  tone,
}: {
  label: string;
  value: string;
  helper: string;
  tone: 'success' | 'danger' | 'warning' | 'info';
}) {
  return (
    <motion.article
      className={`insight-card insight-card--${tone}`}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -3 }}
      transition={{ duration: 0.3 }}>
      <span>{label}</span>
      <strong>{value}</strong>
      <p>{helper}</p>
    </motion.article>
  );
}

function PriorityPill({ session }: { session: AdminSession }) {
  const priority = getReviewPriority(session);
  return (
    <span className={`priority-pill priority-pill--${priority.tone}`}>
      <span className="priority-pill__pulse" />
      {priority.label}
    </span>
  );
}

function getReviewPriority(session: AdminSession) {
  const decision = normalizeDecision(session);

  if (session.security_fail_count > 0 || session.liveness_passed === false || decision === 'REJECTED') {
    return { label: 'High risk', tone: 'danger' };
  }

  if (decision === 'MANUAL_REVIEW') {
    return { label: 'Review', tone: 'warning' };
  }

  if (!session.liveness_passed || !getFaceMatchDistance(session)) {
    return { label: 'Pending', tone: 'info' };
  }

  return { label: 'Normal', tone: 'success' };
}

function asPercent(value: number, total: number) {
  if (total <= 0) {
    return 0;
  }

  return Math.round((value / total) * 100);
}

function DecisionBar({
  label,
  tone,
  value,
  total,
}: {
  label: string;
  tone: 'success' | 'danger' | 'warning' | 'info';
  value: number;
  total: number;
}) {
  const percent = asPercent(value, total);

  return (
    <div className="decision-mix__row">
      <span>{label}</span>
      <div className="decision-mix__track" aria-hidden="true">
        <motion.span
          className={`decision-mix__fill decision-mix__fill--${tone}`}
          initial={{ width: 0 }}
          animate={{ width: `${percent}%` }}
          transition={{ duration: 0.62, ease: 'easeOut' }}
        />
      </div>
      <strong>{percent}%</strong>
    </div>
  );
}
