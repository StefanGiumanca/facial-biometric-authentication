import type { ReactNode } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Link, NavLink, useLocation } from 'react-router-dom';
import { formatDateTime, getStatusTone } from './lib/utils';
import { useTheme } from './theme';
import { AnimatedDashboardBackground } from './components/AnimatedDashboardBackground';

const cardMotion = {
  initial: { opacity: 0, y: 18, scale: 0.985 },
  animate: { opacity: 1, y: 0, scale: 1 },
  transition: { duration: 0.42, ease: 'easeOut' },
} as const;

export function Shell({ children }: { children: ReactNode }) {
  const location = useLocation();

  return (
    <div className="shell">
      <AnimatedDashboardBackground />
      <aside className="sidebar">
        <Link to="/" className="sidebar__brand">
          <span className="sidebar__brand-mark">VA</span>
          <span>
            <strong>VisionAuth</strong>
            <small>Identity Ops</small>
          </span>
        </Link>
        <nav className="sidebar__nav" aria-label="Dashboard navigation">
          <SidebarLink to="/" icon="◇" label="Dashboard" active={location.pathname === '/'} />
          <SidebarLink to="/sessions" icon="▦" label="Sessions" active={location.pathname.startsWith('/sessions')} />
          <SidebarLink to="/manual-reviews" icon="◎" label="Manual Reviews" active={location.pathname === '/manual-reviews'} />
          <SidebarLink to="/audit-logs" icon="⌁" label="Audit Logs" active={location.pathname === '/audit-logs'} />
          <SidebarLink to="/analytics" icon="⌬" label="Analytics" active={location.pathname === '/analytics'} />
          <SidebarLink to="/settings" icon="⚙" label="Settings" active={location.pathname === '/settings'} />
        </nav>
        <div className="sidebar__profile">
          <span className="admin-avatar">A</span>
          <span>
            <strong>Admin Console</strong>
            <small>dev-admin-key</small>
          </span>
        </div>
      </aside>
      <main className="workspace">
        <motion.div
          className="shell__content"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.35 }}>
          {children}
        </motion.div>
      </main>
    </div>
  );
}

function SidebarLink({
  to,
  icon,
  label,
  active,
}: {
  to: string;
  icon: string;
  label: string;
  active?: boolean;
}) {
  return (
    <NavLink to={to} className={active ? 'sidebar-link sidebar-link--active' : 'sidebar-link'}>
      <span className="sidebar-link__icon">{icon}</span>
      <span>{label}</span>
    </NavLink>
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
    <motion.header
      className="page-header"
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: 'easeOut' }}>
      <div>
        <p className="eyebrow">VisionAuth eKYC Admin</p>
        <h1>{title}</h1>
        <p className="page-header__subtitle">{subtitle}</p>
      </div>
      {action}
    </motion.header>
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
  tone = 'info',
  progress,
}: {
  label: string;
  value: string;
  helper?: string;
  tone?: 'info' | 'success' | 'danger' | 'warning' | 'neutral';
  progress?: number;
}) {
  const safeProgress = typeof progress === 'number' ? Math.max(0, Math.min(progress, 100)) : null;

  return (
    <motion.article className={`metric-card metric-card--${tone}`} whileHover={{ y: -4 }} {...cardMotion}>
      <span className="metric-card__label">{label}</span>
      <strong className="metric-card__value">{value}</strong>
      {safeProgress !== null ? (
        <div className="metric-card__meter" aria-hidden="true">
          <span style={{ width: `${safeProgress}%` }} />
        </div>
      ) : null}
      {helper ? <span className="metric-card__helper">{helper}</span> : null}
    </motion.article>
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
    <motion.section className="section-card" {...cardMotion}>
      <div className="section-card__header">
        <div>
          <h2>{title}</h2>
          {hint ? <p>{hint}</p> : null}
        </div>
      </div>
      {children}
    </motion.section>
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
    <motion.div className="timeline" initial="hidden" animate="show" variants={{ show: { transition: { staggerChildren: 0.055 } } }}>
      <AnimatePresence initial={false}>
        {items.map((item) => (
        <motion.article
          key={item.id}
          className="timeline__item"
          variants={{
            hidden: { opacity: 0, x: -12 },
            show: { opacity: 1, x: 0 },
          }}
          exit={{ opacity: 0, x: -12 }}
          transition={{ duration: 0.26 }}>
          <div className="timeline__dot" />
          <div className="timeline__content">
            <div className="timeline__meta">
              <strong>{item.event_type}</strong>
              <span>{formatDateTime(item.created_at)}</span>
            </div>
            <p>{item.message}</p>
          </div>
        </motion.article>
        ))}
      </AnimatePresence>
    </motion.div>
  );
}

export function TopNav({
  onSignOut,
  searchTerm,
  onSearchChange,
  sessionCount,
  searchPlaceholder = 'Search by CNP, name, session ID...',
}: {
  onSignOut: () => void;
  searchTerm?: string;
  onSearchChange?: (value: string) => void;
  sessionCount?: number;
  searchPlaceholder?: string;
}) {
  const now = new Date();
  const { theme, toggleTheme } = useTheme();

  return (
    <motion.nav
      className="top-nav"
      initial={{ opacity: 0, y: -12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.34 }}>
      <label className="top-nav__search">
        <span>⌕</span>
        <input
          value={searchTerm ?? ''}
          onChange={(event) => onSearchChange?.(event.target.value)}
          placeholder={searchPlaceholder}
          disabled={!onSearchChange}
        />
      </label>
      <div className="top-nav__actions">
        {typeof sessionCount === 'number' ? <span className="top-nav__counter">{sessionCount} sessions</span> : null}
        <span className="top-nav__live">
          <span className="top-nav__live-dot" />
          Live review console
        </span>
        <span className="top-nav__time">{now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })} · {now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}</span>
        <button
          type="button"
          className="icon-button"
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
          onClick={toggleTheme}>
          {theme === 'dark' ? '☼' : '☾'}
        </button>
        <span className="admin-avatar">A</span>
        <button type="button" className="button button--ghost" onClick={onSignOut}>
          Clear Admin Key
        </button>
      </div>
    </motion.nav>
  );
}
