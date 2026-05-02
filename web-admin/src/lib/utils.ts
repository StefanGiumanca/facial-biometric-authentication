import type { AdminSession } from '../types';

export type SessionFilter = 'ALL' | 'VERIFIED' | 'REJECTED' | 'MANUAL_REVIEW' | 'IN_PROGRESS';

export function formatDateTime(value: string | null | undefined) {
  if (!value) {
    return 'N/A';
  }

  return new Intl.DateTimeFormat('en-GB', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value));
}

export function shortSessionId(sessionId: string) {
  return `${sessionId.slice(0, 8)}...${sessionId.slice(-4)}`;
}

export function getDisplayName(firstName: string | null, lastName: string | null) {
  const fullName = [firstName, lastName].filter(Boolean).join(' ').trim();
  return fullName || 'Unknown identity';
}

export function normalizeDecision(session: Pick<AdminSession, 'status' | 'final_decision'>) {
  return (session.final_decision || session.status || 'IN_PROGRESS').toUpperCase();
}

export function matchesFilter(session: AdminSession, filter: SessionFilter) {
  const decision = normalizeDecision(session);

  if (filter === 'ALL') {
    return true;
  }

  if (filter === 'VERIFIED') {
    return ['VERIFIED', 'ACCEPTED', 'APPROVED'].includes(decision);
  }

  if (filter === 'REJECTED') {
    return decision === 'REJECTED';
  }

  if (filter === 'MANUAL_REVIEW') {
    return decision === 'MANUAL_REVIEW';
  }

  return !['VERIFIED', 'ACCEPTED', 'APPROVED', 'REJECTED', 'MANUAL_REVIEW'].includes(decision);
}

export function getStatusTone(status: string | null | undefined) {
  const normalized = (status || 'UNKNOWN').toUpperCase();

  if (['VERIFIED', 'ACCEPTED', 'APPROVED'].includes(normalized)) {
    return 'success';
  }

  if (normalized === 'REJECTED') {
    return 'danger';
  }

  if (normalized === 'MANUAL_REVIEW') {
    return 'warning';
  }

  return 'info';
}

export function formatDecisionWithDistance(decision?: string | null, distance?: number | null) {
  if (!decision) {
    return 'N/A';
  }

  if (typeof distance === 'number') {
    return `${decision} (${distance.toFixed(3)})`;
  }

  return decision;
}
