import type {
  AdminSessionDetailResponse,
  AdminSessionLogsResponse,
  AdminSessionsResponse,
  ApiError,
} from '../types';

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() || 'http://127.0.0.1:8000';

type JsonRecord = {
  ok?: boolean;
  error?: string;
  detail?: string;
  [key: string]: unknown;
};

async function readJsonResponse<T>(response: Response): Promise<T> {
  const data = (await response.json()) as JsonRecord;

  if (!response.ok || data.ok === false) {
    const error = new Error(data.error || data.detail || `Request failed with status ${response.status}`) as ApiError;
    error.status = response.status;
    error.detail = typeof data.detail === 'string' ? data.detail : undefined;
    throw error;
  }

  return data as T;
}

async function adminGet<T>(endpoint: string, adminKey: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    headers: {
      'X-Admin-Key': adminKey,
    },
  });

  return readJsonResponse<T>(response);
}

export function getAdminSessions(adminKey: string, limit = 50, offset = 0) {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });

  return adminGet<AdminSessionsResponse>(`/admin/sessions?${params.toString()}`, adminKey);
}

export function getAdminSessionDetail(adminKey: string, sessionId: string) {
  return adminGet<AdminSessionDetailResponse>(`/admin/sessions/${sessionId}`, adminKey);
}

export function getAdminSessionLogs(adminKey: string, sessionId: string) {
  return adminGet<AdminSessionLogsResponse>(`/admin/sessions/${sessionId}/logs`, adminKey);
}

export function buildAdminMediaUrl(sessionId: string, mediaKind: 'document' | 'id_face' | 'selfie' | 'liveness_video') {
  return `${API_BASE_URL}/admin/sessions/${sessionId}/media/${mediaKind}`;
}
