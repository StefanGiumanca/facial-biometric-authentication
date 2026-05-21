import type { AdminSession } from '../types';

export type SessionFilter = 'ALL' | 'VERIFIED' | 'REJECTED' | 'MANUAL_REVIEW' | 'IN_PROGRESS';
export type SessionSortKey = 'updated_at' | 'created_at' | 'name' | 'final_decision' | 'face_match_distance';
export type SortDirection = 'asc' | 'desc';

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
    return [
      decision,
      session.status,
      session.face_match_decision,
      session.final_face_match_decision,
    ].some((value) => String(value || '').toUpperCase() === 'MANUAL_REVIEW');
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

export function getFaceMatchDistance(session: AdminSession) {
  if (typeof session.final_face_match_distance === 'number') {
    return session.final_face_match_distance;
  }

  if (typeof session.face_match_distance === 'number') {
    return session.face_match_distance;
  }

  return null;
}

export function sessionMatchesSearch(session: AdminSession, searchTerm: string) {
  const normalizedSearch = searchTerm.trim().toLowerCase();
  if (!normalizedSearch) {
    return true;
  }

  const searchableValues = [
    session.first_name,
    session.last_name,
    session.cnp,
    session.session_id,
    session.series_number,
    session.final_decision,
    session.face_match_decision,
    session.final_face_match_decision,
  ];

  return searchableValues.some((value) => String(value ?? '').toLowerCase().includes(normalizedSearch));
}

export function sortSessions(sessions: AdminSession[], sortKey: SessionSortKey, direction: SortDirection) {
  const multiplier = direction === 'asc' ? 1 : -1;

  return [...sessions].sort((a, b) => {
    if (sortKey === 'updated_at' || sortKey === 'created_at') {
      return (new Date(a[sortKey]).getTime() - new Date(b[sortKey]).getTime()) * multiplier;
    }

    if (sortKey === 'name') {
      return getDisplayName(a.first_name, a.last_name).localeCompare(getDisplayName(b.first_name, b.last_name)) * multiplier;
    }

    if (sortKey === 'final_decision') {
      return normalizeDecision(a).localeCompare(normalizeDecision(b)) * multiplier;
    }

    const rawDistanceA = getFaceMatchDistance(a);
    const rawDistanceB = getFaceMatchDistance(b);
    if (rawDistanceA === null && rawDistanceB === null) {
      return 0;
    }
    if (rawDistanceA === null) {
      return 1;
    }
    if (rawDistanceB === null) {
      return -1;
    }

    const distanceA = rawDistanceA;
    const distanceB = rawDistanceB;
    return (distanceA - distanceB) * multiplier;
  });
}
