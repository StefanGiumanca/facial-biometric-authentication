import type {
  AdminDecision,
  AdminDecisionResponse,
  AdminDeleteSessionResponse,
  AdminSessionDetailResponse,
  AdminSessionLogsResponse,
  AdminSessionsResponse,
  ApiError,
} from '../types';

function getDefaultApiBaseUrl() {
  const { hostname, protocol } = window.location;

  if (hostname && hostname !== 'localhost' && hostname !== '127.0.0.1') {
    return `${protocol}//${hostname}:8000`;
  }

  return 'http://127.0.0.1:8000';
}

function getConfiguredApiBaseUrl() {
  const explicitApiBaseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim();
  if (explicitApiBaseUrl) {
    return explicitApiBaseUrl;
  }

  const connectionMode = (import.meta.env.VITE_CONNECTION_MODE as string | undefined)?.trim();
  const wifiApiBaseUrl = (import.meta.env.VITE_WIFI_API_BASE_URL as string | undefined)?.trim();
  const ngrokApiBaseUrl = (import.meta.env.VITE_NGROK_API_BASE_URL as string | undefined)?.trim();

  if (connectionMode === 'ngrok' && ngrokApiBaseUrl) {
    return ngrokApiBaseUrl;
  }

  if (connectionMode === 'wifi' && wifiApiBaseUrl) {
    return wifiApiBaseUrl;
  }

  return getDefaultApiBaseUrl();
}

export const API_BASE_URL = getConfiguredApiBaseUrl();
export const NGROK_SKIP_WARNING_HEADER = 'ngrok-skip-browser-warning';

export function buildAdminHeaders(adminKey: string, extraHeaders: HeadersInit = {}): HeadersInit {
  return {
    ...extraHeaders,
    'X-Admin-Key': adminKey,
    [NGROK_SKIP_WARNING_HEADER]: 'true',
  };
}

type JsonRecord = {
  ok?: boolean;
  error?: string;
  detail?: unknown;
  [key: string]: unknown;
};

async function readJsonResponse<T>(response: Response): Promise<T> {
  const data = (await response.json()) as JsonRecord;

  if (!response.ok || data.ok === false) {
    const detail = typeof data.detail === 'string' ? data.detail : undefined;
    const error = new Error(data.error || detail || `Request failed with status ${response.status}`) as ApiError;
    error.status = response.status;
    error.detail = detail;
    throw error;
  }

  return data as T;
}

async function adminGet<T>(endpoint: string, adminKey: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    headers: buildAdminHeaders(adminKey),
  });

  return readJsonResponse<T>(response);
}

async function adminPost<T>(endpoint: string, adminKey: string, body: Record<string, unknown>): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: 'POST',
    headers: buildAdminHeaders(adminKey, {
      'Content-Type': 'application/json',
    }),
    body: JSON.stringify(body),
  });

  return readJsonResponse<T>(response);
}

async function adminDelete<T>(endpoint: string, adminKey: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: 'DELETE',
    headers: buildAdminHeaders(adminKey),
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

export function saveAdminDecision(adminKey: string, sessionId: string, decision: AdminDecision, adminNote?: string) {
  return adminPost<AdminDecisionResponse>(`/admin/sessions/${sessionId}/decision`, adminKey, {
    decision,
    admin_note: adminNote?.trim() || undefined,
  });
}

export function deleteAdminSession(adminKey: string, sessionId: string) {
  return adminDelete<AdminDeleteSessionResponse>(`/admin/sessions/${sessionId}`, adminKey);
}

export function buildAdminMediaUrl(sessionId: string, mediaKind: 'document' | 'id_face' | 'selfie' | 'liveness_video') {
  return `${API_BASE_URL}/admin/sessions/${sessionId}/media/${mediaKind}`;
}
