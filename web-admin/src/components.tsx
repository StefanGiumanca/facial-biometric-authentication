import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { formatDateTime, getStatusTone } from './lib/utils';

export function Shell({ children }: { children: ReactNode }) {
  return (
    <div className="shell">
      <div className="shell__backdrop shell__backdrop--one" />
      <div className="shell__backdrop shell__backdrop--two" />
      <div className="shell__content">{children}</div>
    </div>
  );
}

export function BrandHeader({
  title,
  subtitle,
  action,
}: {
  title: string;
  subtitle: string;
  action?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <p className="eyebrow">VisionAuth eKYC Admin</p>
        <h1>{title}</h1>
        <p className="page-header__subtitle">{subtitle}</p>
      </div>
      {action}
    </header>
  );
}

export function StatusBadge({ label }: { label: string | null | undefined }) {
  const tone = getStatusTone(label);
  return <span className={`badge badge--${tone}`}>{label || 'UNKNOWN'}</span>;
}

export function MetricCard({
  label,
  value,
  helper,
}: {
  label: string;
  value: string;
  helper?: string;
}) {
  return (
    <div className="metric-card">
      <span className="metric-card__label">{label}</span>
      <strong className="metric-card__value">{value}</strong>
      {helper ? <span className="metric-card__helper">{helper}</span> : null}
    </div>
  );
}

export function SectionCard({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <section className="section-card">
      <div className="section-card__header">
        <div>
          <h2>{title}</h2>
          {hint ? <p>{hint}</p> : null}
        </div>
      </div>
      {children}
    </section>
  );
}

export function DetailGrid({
  rows,
}: {
  rows: Array<{ label: string; value: React.ReactNode; subtle?: boolean }>;
}) {
  return (
    <div className="detail-grid">
      {rows.map((row) => (
        <div key={row.label} className="detail-grid__row">
          <span className="detail-grid__label">{row.label}</span>
          <span className={row.subtle ? 'detail-grid__value detail-grid__value--subtle' : 'detail-grid__value'}>
            {row.value}
          </span>
        </div>
      ))}
    </div>
  );
}

export function Timeline({
  items,
}: {
  items: Array<{ id: number; event_type: string; message: string; created_at: string }>;
}) {
  if (items.length === 0) {
    return <div className="empty-state empty-state--compact">No audit events were returned for this session.</div>;
  }

  return (
    <div className="timeline">
      {items.map((item) => (
        <article key={item.id} className="timeline__item">
          <div className="timeline__dot" />
          <div className="timeline__content">
            <div className="timeline__meta">
              <strong>{item.event_type}</strong>
              <span>{formatDateTime(item.created_at)}</span>
            </div>
            <p>{item.message}</p>
          </div>
        </article>
      ))}
    </div>
  );
}

export function TopNav({ onSignOut }: { onSignOut: () => void }) {
  return (
    <nav className="top-nav">
      <Link to="/" className="top-nav__brand">
        <span className="top-nav__brand-mark">VA</span>
        <span>VisionAuth Admin</span>
      </Link>
      <button type="button" className="button button--ghost" onClick={onSignOut}>
        Clear Admin Key
      </button>
    </nav>
  );
}
