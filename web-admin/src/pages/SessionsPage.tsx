import { useEffect, useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Link, useLocation, useNavigate } from 'react-router-dom';
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
  { value: 'updated_at', label: 'Newest activity' },
  { value: 'created_at', label: 'Created date' },
  { value: 'name', label: 'Name' },
  { value: 'final_decision', label: 'Decision status' },
  { value: 'face_match_distance', label: 'Face confidence' },
];

export function SessionsPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { adminKey, clearAdminKey } = useAdminAuth();
  const [sessions, setSessions] = useState<AdminSession[]>([]);
  const isDashboard = location.pathname === '/';
  const isManualReviews = location.pathname === '/manual-reviews';
  const [activeFilter, setActiveFilter] = useState<SessionFilter>(isManualReviews ? 'MANUAL_REVIEW' : 'ALL');
  const [searchTerm, setSearchTerm] = useState('');
  const [dateFilter, setDateFilter] = useState('');
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

  useEffect(() => {
    setActiveFilter(isManualReviews ? 'MANUAL_REVIEW' : 'ALL');
  }, [isManualReviews]);

  const visibleSessions = useMemo(
    () =>
      sortSessions(
        sessions.filter((session) => matchesFilter(session, activeFilter) && sessionMatchesSearch(session, searchTerm) && matchesDateFilter(session, dateFilter)),
        sortKey,
        sortDirection,
      ),
    [activeFilter, dateFilter, searchTerm, sessions, sortDirection, sortKey],
  );

  const metrics = useMemo(() => {
    const verified = sessions.filter((session) => matchesFilter(session, 'VERIFIED')).length;
    const rejected = sessions.filter((session) => matchesFilter(session, 'REJECTED')).length;
    const review = sessions.filter((session) => matchesFilter(session, 'MANUAL_REVIEW')).length;
    const inProgress = sessions.filter((session) => matchesFilter(session, 'IN_PROGRESS')).length;
    const flagged = sessions.filter((session) => session.security_fail_count > 0 || session.liveness_passed === false).length;
    const today = new Date().toDateString();
    const todaysVerifications = sessions.filter((session) => new Date(session.created_at).toDateString() === today).length;
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
    return { verified, rejected, review, inProgress, flagged, averageConfidence, total: sessions.length, todaysVerifications };
  }, [sessions]);

  return (
    <Shell>
      <TopNav
        searchTerm={searchTerm}
        onSearchChange={setSearchTerm}
        sessionCount={sessions.length}
        searchPlaceholder={isManualReviews ? 'Search manual review sessions...' : 'Search by CNP, name, session ID...'}
        onSignOut={() => {
          clearAdminKey();
          navigate('/login');
        }}
      />

      <BrandHeader
        title={isDashboard ? 'Identity Operations Dashboard' : isManualReviews ? 'Manual Reviews' : 'Sessions'}
        subtitle={
          isDashboard
            ? 'Enterprise review queue for biometric verification, manual decisions, and audit inspection.'
            : isManualReviews
              ? 'Cases waiting for operator decision because session or face-match review state requires a human check.'
              : 'Full biometric verification queue with search, status filtering, sorting, and session inspection.'
        }
        action={
          <button type="button" className="button button--secondary" onClick={() => window.location.reload()}>
            Refresh Data
          </button>
        }
      />

      {isDashboard ? (
        <>
          <div className="metric-grid metric-grid--eight">
            <MetricCard label="Total sessions" value={String(sessions.length)} helper="Latest 50 records" tone="neutral" progress={100} />
            <MetricCard label="Verified" value={String(metrics.verified)} helper="+14 today target" tone="success" progress={asPercent(metrics.verified, metrics.total)} />
            <MetricCard label="Rejected" value={String(metrics.rejected)} helper="Failed or locked" tone="danger" progress={asPercent(metrics.rejected, metrics.total)} />
            <MetricCard label="Manual review" value={String(metrics.review)} helper="Operator queue" tone="warning" progress={asPercent(metrics.review, metrics.total)} />
            <MetricCard label="Pending" value={String(metrics.inProgress)} helper="OCR/liveness pending" tone="info" progress={asPercent(metrics.inProgress, metrics.total)} />
            <MetricCard label="Today" value={String(metrics.todaysVerifications)} helper="Today's verifications" tone="info" />
            <MetricCard label="Avg confidence" value={metrics.averageConfidence === null ? 'N/A' : `${metrics.averageConfidence}%`} helper="Face-match confidence" tone="success" progress={metrics.averageConfidence ?? 0} />
            <MetricCard label="Avg processing" value="~42s" helper="Captured from session flow" tone="neutral" progress={68} />
          </div>

          <motion.div
            className="command-panel"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.42, delay: 0.08 }}>
            <div>
              <span className="command-panel__label">Enterprise queue intelligence</span>
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
        </>
      ) : (
        <div className="metric-grid metric-grid--queue">
          <MetricCard label="Visible cases" value={String(visibleSessions.length)} helper={isManualReviews ? 'Operator review queue' : 'Matching current search and filters'} tone={isManualReviews ? 'warning' : 'info'} />
          <MetricCard label="All sessions" value={String(metrics.total)} helper="Latest records from admin API" tone="neutral" />
          <MetricCard label="Verified" value={String(metrics.verified)} helper="Final accepted decisions" tone="success" />
          <MetricCard label="Rejected" value={String(metrics.rejected)} helper="Final rejected decisions" tone="danger" />
        </div>
      )}

      <SectionCard
        title={isManualReviews ? 'Manual Review Queue' : isDashboard ? 'Sessions Overview' : 'All Sessions'}
        hint={isManualReviews ? 'Filtered to sessions that need an operator decision.' : 'Filter the queue and drill into a single review case.'}>
        <div className="toolbar toolbar--sessions">
          <label className="field field--search">
            <span>Search sessions</span>
            <input
              type="search"
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
              placeholder="Name, CNP, session ID, series number..."
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
          <label className="field field--compact">
            <span>Date</span>
            <input type="date" value={dateFilter} onChange={(event) => setDateFilter(event.target.value)} />
          </label>
        </div>

        {isManualReviews ? (
          <p className="queue-note">
            Manual review includes sessions where the final decision, session status, or face-match decision is <code>MANUAL_REVIEW</code>.
          </p>
        ) : (
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
        )}

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
                  <th>Identity</th>
                  <th>Session</th>
                  <th>Decision</th>
                  <th>Priority</th>
                  <th>Liveness</th>
                  <th>Confidence</th>
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
                    className="session-row"
                    tabIndex={0}
                    onClick={() => navigate(`/sessions/${session.session_id}`)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        navigate(`/sessions/${session.session_id}`);
                      }
                    }}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, x: -18 }}
                    transition={{ duration: 0.22, delay: Math.min(index * 0.015, 0.16) }}>
                    <td>
                      <div className="identity-cell">
                        <span className="identity-avatar">{getInitials(session)}</span>
                        <span>
                          <div className="table-primary">{getDisplayName(session.first_name, session.last_name)}</div>
                          <div className="table-secondary">{session.cnp || 'No CNP extracted'}</div>
                        </span>
                      </div>
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
                      <ConfidenceCell session={session} />
                    </td>
                    <td>
                      <div className="table-primary">{formatDateTime(session.updated_at)}</div>
                      <div className="table-secondary">Created {formatDateTime(session.created_at)}</div>
                    </td>
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

function matchesDateFilter(session: AdminSession, dateFilter: string) {
  if (!dateFilter) {
    return true;
  }

  return session.created_at.slice(0, 10) === dateFilter;
}

function getInitials(session: AdminSession) {
  const name = getDisplayName(session.first_name, session.last_name);
  return name
    .split(' ')
    .map((part) => part[0])
    .join('')
    .slice(0, 2)
    .toUpperCase();
}

function ConfidenceCell({ session }: { session: AdminSession }) {
  const distance = getFaceMatchDistance(session);
  const confidence = typeof distance === 'number' ? Math.round(Math.max(0, Math.min(1, 1 - distance / 0.6)) * 100) : null;

  return (
    <div className="confidence-cell">
      <div className="table-primary">{confidence === null ? 'Pending' : `${confidence}%`}</div>
      <div className="mini-meter" aria-hidden="true">
        <span style={{ width: `${confidence ?? 0}%` }} />
      </div>
      <div className="table-secondary">{distance === null ? 'No distance' : `${distance.toFixed(3)} distance`}</div>
    </div>
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
