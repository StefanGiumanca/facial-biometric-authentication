import { useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useNavigate } from 'react-router-dom';
import { BrandHeader, MetricCard, SectionCard, Shell, TopNav } from '../components';
import { useAdminAuth } from '../auth';
import { getAdminSessions } from '../lib/api';
import { getFaceMatchDistance, matchesFilter, sessionMatchesSearch } from '../lib/utils';
import type { AdminSession } from '../types';

const chartColors = ['#60a5fa', '#22c55e', '#f59e0b', '#ef4444', '#38bdf8'];

export function AnalyticsPage() {
  const navigate = useNavigate();
  const { adminKey, clearAdminKey } = useAdminAuth();
  const [sessions, setSessions] = useState<AdminSession[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setError('');
        const response = await getAdminSessions(adminKey, 50, 0);
        if (!cancelled) {
          setSessions(response.sessions);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load analytics.';
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

    load();

    return () => {
      cancelled = true;
    };
  }, [adminKey, clearAdminKey, navigate]);

  const filteredSessions = useMemo(() => sessions.filter((session) => sessionMatchesSearch(session, searchTerm)), [searchTerm, sessions]);
  const analytics = useMemo(() => buildAnalytics(filteredSessions), [filteredSessions]);

  return (
    <Shell>
      <TopNav
        sessionCount={filteredSessions.length}
        searchTerm={searchTerm}
        onSearchChange={setSearchTerm}
        searchPlaceholder="Filter analytics sessions..."
        onSignOut={() => {
          clearAdminKey();
          navigate('/login');
        }}
      />

      <BrandHeader
        title="Analytics"
        subtitle="Operational health, pass rates, rejection reasons, and biometric confidence distribution."
      />

      {error ? <div className="error-banner">{error}</div> : null}
      {isLoading ? <div className="empty-state">Loading analytics...</div> : null}

      {!isLoading ? (
        <>
          <div className="metric-grid metric-grid--eight">
            <MetricCard label="Total sessions" value={String(analytics.total)} helper="Filtered analytics set" tone="neutral" />
            <MetricCard label="Verified" value={String(analytics.verified)} helper="Final accepted" tone="success" />
            <MetricCard label="Rejected" value={String(analytics.rejected)} helper="Final rejected" tone="danger" />
            <MetricCard label="Manual review" value={String(analytics.manualReview)} helper="Sessions requiring operator action" tone="warning" />
            <MetricCard label="Pending" value={String(analytics.pending)} helper="Still in progress" tone="info" />
            <MetricCard label="Acceptance rate" value={`${analytics.passRate}%`} helper="Verified / total sessions" tone="success" progress={analytics.passRate} />
            <MetricCard label="Rejection rate" value={`${analytics.rejectionRate}%`} helper="Rejected / total sessions" tone="danger" progress={analytics.rejectionRate} />
            <MetricCard label="Avg face distance" value={analytics.averageDistance === null ? 'N/A' : analytics.averageDistance.toFixed(3)} helper={`${analytics.averageConfidence}% confidence approx.`} tone="info" progress={analytics.averageConfidence} />
          </div>

          <div className="analytics-grid">
            <SectionCard title="Sessions per day" hint="Recent verification volume from stored sessions.">
              <ChartFrame>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={analytics.sessionsPerDay}>
                    <CartesianGrid stroke="rgba(148,163,184,0.12)" vertical={false} />
                    <XAxis dataKey="date" stroke="#94a3b8" tickLine={false} axisLine={false} />
                    <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} allowDecimals={false} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Line type="monotone" dataKey="sessions" stroke="#60a5fa" strokeWidth={3} dot={{ fill: '#60a5fa', r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              </ChartFrame>
            </SectionCard>

            <SectionCard title="Pass rate mix" hint="Current distribution of final decisions.">
              <ChartFrame>
                <ResponsiveContainer width="100%" height={280}>
                  <PieChart>
                    <Pie data={analytics.decisionMix} dataKey="value" nameKey="name" innerRadius={68} outerRadius={102} paddingAngle={3}>
                      {analytics.decisionMix.map((entry, index) => (
                        <Cell key={entry.name} fill={chartColors[index % chartColors.length]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={tooltipStyle} />
                  </PieChart>
                </ResponsiveContainer>
              </ChartFrame>
            </SectionCard>

            <SectionCard title="Rejection reasons" hint="Security and threshold indicators.">
              <ChartFrame>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={analytics.rejectionReasons}>
                    <CartesianGrid stroke="rgba(148,163,184,0.12)" vertical={false} />
                    <XAxis dataKey="reason" stroke="#94a3b8" tickLine={false} axisLine={false} />
                    <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} allowDecimals={false} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Bar dataKey="count" fill="#ef4444" radius={[10, 10, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartFrame>
            </SectionCard>

            <SectionCard title="Confidence distribution" hint="Face-match confidence buckets.">
              <ChartFrame>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={analytics.confidenceDistribution}>
                    <CartesianGrid stroke="rgba(148,163,184,0.12)" vertical={false} />
                    <XAxis dataKey="bucket" stroke="#94a3b8" tickLine={false} axisLine={false} />
                    <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} allowDecimals={false} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Bar dataKey="count" fill="#60a5fa" radius={[10, 10, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartFrame>
            </SectionCard>
          </div>
        </>
      ) : null}
    </Shell>
  );
}

function ChartFrame({ children }: { children: React.ReactNode }) {
  return <div className="chart-frame">{children}</div>;
}

const tooltipStyle = {
  background: 'rgba(15, 23, 42, 0.96)',
  border: '1px solid rgba(148, 163, 184, 0.18)',
  borderRadius: '14px',
  color: '#e5eefc',
};

function buildAnalytics(sessions: AdminSession[]) {
  const verified = sessions.filter((session) => matchesFilter(session, 'VERIFIED')).length;
  const rejected = sessions.filter((session) => matchesFilter(session, 'REJECTED')).length;
  const manualReview = sessions.filter((session) => matchesFilter(session, 'MANUAL_REVIEW')).length;
  const pending = sessions.filter((session) => matchesFilter(session, 'IN_PROGRESS')).length;
  const passRate = sessions.length ? Math.round((verified / sessions.length) * 100) : 0;
  const rejectionRate = sessions.length ? Math.round((rejected / sessions.length) * 100) : 0;
  const faceDistances = sessions
    .map((session) => getFaceMatchDistance(session))
    .filter((distance): distance is number => typeof distance === 'number');
  const confidenceValues = sessions
    .flatMap((session) => {
      const distance = getFaceMatchDistance(session);
      return typeof distance === 'number' ? [distance] : [];
    })
    .map((distance) => Math.round(Math.max(0, Math.min(1, 1 - distance / 0.6)) * 100));
  const averageConfidence = confidenceValues.length
    ? Math.round(confidenceValues.reduce((total, value) => total + value, 0) / confidenceValues.length)
    : 0;

  return {
    passRate,
    rejectionRate,
    total: sessions.length,
    verified,
    rejected,
    manualReview,
    pending,
    averageDistance: faceDistances.length
      ? faceDistances.reduce((total, distance) => total + distance, 0) / faceDistances.length
      : null,
    averageConfidence,
    sessionsPerDay: buildSessionsPerDay(sessions),
    decisionMix: [
      { name: 'Verified', value: verified },
      { name: 'Manual', value: manualReview },
      { name: 'Rejected', value: rejected },
      { name: 'Pending', value: Math.max(0, sessions.length - verified - rejected - manualReview) },
    ].filter((item) => item.value > 0),
    rejectionReasons: buildRejectionReasons(sessions),
    confidenceDistribution: buildConfidenceDistribution(confidenceValues),
  };
}

function buildSessionsPerDay(sessions: AdminSession[]) {
  const buckets = new Map<string, number>();
  sessions.forEach((session) => {
    const label = new Intl.DateTimeFormat('en-GB', { month: 'short', day: '2-digit' }).format(new Date(session.created_at));
    buckets.set(label, (buckets.get(label) ?? 0) + 1);
  });

  return Array.from(buckets, ([date, sessionsCount]) => ({ date, sessions: sessionsCount })).slice(-8);
}

function buildRejectionReasons(sessions: AdminSession[]) {
  const reasons = [
    { reason: 'Security', count: sessions.filter((session) => session.security_fail_count > 0).length },
    { reason: 'Liveness', count: sessions.filter((session) => session.liveness_passed === false).length },
    { reason: 'Face', count: sessions.filter((session) => String(session.final_face_match_decision || session.face_match_decision).toUpperCase() === 'REJECTED').length },
    { reason: 'Manual', count: sessions.filter((session) => matchesFilter(session, 'REJECTED') && session.reject_reason).length },
  ];

  return reasons.filter((item) => item.count > 0).length ? reasons : [{ reason: 'None', count: 0 }];
}

function buildConfidenceDistribution(values: number[]) {
  const buckets = [
    { bucket: '0-40', count: 0 },
    { bucket: '41-60', count: 0 },
    { bucket: '61-80', count: 0 },
    { bucket: '81-100', count: 0 },
  ];

  values.forEach((value) => {
    if (value <= 40) buckets[0].count += 1;
    else if (value <= 60) buckets[1].count += 1;
    else if (value <= 80) buckets[2].count += 1;
    else buckets[3].count += 1;
  });

  return buckets;
}
