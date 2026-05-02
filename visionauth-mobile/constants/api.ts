export const WIFI_API_BASE_URL = 'http://192.168.1.131:8000';
export const NGROK_API_BASE_URL = 'https://reward-botanist-tag.ngrok-free.dev';

export const API_BASE_URL = WIFI_API_BASE_URL; // Change this to switch between local and ngrok backend

export type UploadAsset = {
  uri: string;
  fileName?: string | null;
  mimeType?: string | null;
};

export type ApiResponse = {
  ok?: boolean;
  error?: string;
  detail?: string | {
    code?: string;
    reason?: string;
    message?: string;
    security_fail_count?: number;
  };
  [key: string]: unknown;
};

export type ApiError = Error & {
  data?: ApiResponse;
};

// ========== ADMIN API TYPES ==========

export type AdminSession = {
  session_id: string;
  created_at: string;
  updated_at: string;
  first_name: string | null;
  last_name: string | null;
  cnp: string | null;
  series_number: string | null;
  liveness_passed: boolean | null;
  selfie_gate_distance: number | null;
  selfie_gate_decision: string | null;
  final_face_match_distance: number | null;
  final_face_match_decision: string | null;
  face_match_distance: number | null;
  face_match_decision: string | null;
  final_decision: string | null;
  status: string;
  security_fail_count: number;
  reject_reason: string | null;
  locked_at: string | null;
};

export type EmbeddingMetadata = {
  id: number;
  embedding_type: string;
  vector_length: number | null;
  vector_preview: number[] | null;
  created_at: string;
};

export type SessionDetail = {
  session_id: string;
  created_at: string;
  updated_at: string;
  status: string;
  first_name: string | null;
  last_name: string | null;
  cnp: string | null;
  series_number: string | null;
  document_path: string | null;
  id_face_path: string | null;
  selfie_path: string | null;
  liveness_video_path: string | null;
  liveness_passed: boolean | null;
  selfie_gate_distance: number | null;
  selfie_gate_decision: string | null;
  final_face_match_distance: number | null;
  final_face_match_decision: string | null;
  face_match_distance: number | null;
  face_match_decision: string | null;
  final_decision: string | null;
  raw_ocr_text: string | null;
  security_fail_count: number;
  reject_reason: string | null;
  locked_at: string | null;
  embeddings: EmbeddingMetadata[];
};

export type AuditLogEntry = {
  id: number;
  event_type: string;
  message: string;
  created_at: string;
};

export type AdminSessionsResponse = {
  ok: boolean;
  count: number;
  limit: number;
  offset: number;
  sessions: AdminSession[];
};

export type AdminSessionDetailResponse = {
  ok: boolean;
  session: SessionDetail;
};

export type AdminSessionLogsResponse = {
  ok: boolean;
  session_id: string;
  logs: AuditLogEntry[];
};

async function readJsonResponse(response: Response): Promise<ApiResponse> {
  const data = (await response.json()) as ApiResponse;

  if (!response.ok || data.ok === false) {
    const detailMessage = typeof data.detail === 'object' ? data.detail?.message : data.detail;
    const error = new Error(data.error || detailMessage || `Request failed with status ${response.status}`) as ApiError;
    error.data = data;
    throw error;
  }

  return data;
}

function fileNameFromUri(uri: string, fallbackName: string) {
  const name = uri.split('/').pop();
  return name && name.includes('.') ? name : fallbackName;
}

export async function postKyc(endpoint: string) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: 'POST',
  });

  return readJsonResponse(response);
}

export async function postKycJson(endpoint: string, body: Record<string, unknown>) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  return readJsonResponse(response);
}

export async function getKyc(endpoint: string) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`);
  return readJsonResponse(response);
}

export async function uploadKycFile(
  endpoint: string,
  asset: UploadAsset,
  fallbackName: string,
  fallbackType: string,
) {
  const formData = new FormData();
  formData.append('file', {
    uri: asset.uri,
    name: asset.fileName || fileNameFromUri(asset.uri, fallbackName),
    type: asset.mimeType || fallbackType,
  } as unknown as Blob);

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: 'POST',
    body: formData,
  });

  return readJsonResponse(response);
}

// ========== ADMIN ENDPOINTS ==========

export async function getAdminSessions(
  adminKey: string,
  limit: number = 50,
  offset: number = 0,
): Promise<AdminSessionsResponse> {
  const params = new URLSearchParams({ limit: limit.toString(), offset: offset.toString() });
  const response = await fetch(`${API_BASE_URL}/admin/sessions?${params}`, {
    headers: {
      'X-Admin-Key': adminKey,
    },
  });
  return (await readJsonResponse(response)) as AdminSessionsResponse;
}

export async function getAdminSessionDetail(adminKey: string, sessionId: string): Promise<AdminSessionDetailResponse> {
  const response = await fetch(`${API_BASE_URL}/admin/sessions/${sessionId}`, {
    headers: {
      'X-Admin-Key': adminKey,
    },
  });
  return (await readJsonResponse(response)) as AdminSessionDetailResponse;
}

export async function getAdminSessionLogs(adminKey: string, sessionId: string): Promise<AdminSessionLogsResponse> {
  const response = await fetch(`${API_BASE_URL}/admin/sessions/${sessionId}/logs`, {
    headers: {
      'X-Admin-Key': adminKey,
    },
  });
  return (await readJsonResponse(response)) as AdminSessionLogsResponse;
}
