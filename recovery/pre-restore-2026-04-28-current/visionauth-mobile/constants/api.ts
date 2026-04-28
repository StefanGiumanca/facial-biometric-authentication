export const WIFI_API_BASE_URL = 'http://192.168.1.131:8000';
export const NGROK_API_BASE_URL = 'https://reward-botanist-tag.ngrok-free.dev';

export const API_BASE_URL = NGROK_API_BASE_URL;

export type UploadAsset = {
  uri: string;
  fileName?: string | null;
  mimeType?: string | null;
};

export type ApiResponse = {
  ok?: boolean;
  error?: string;
  detail?: string;
  [key: string]: unknown;
};

export type ApiError = Error & {
  data?: ApiResponse;
};

async function readJsonResponse(response: Response): Promise<ApiResponse> {
  const data = (await response.json()) as ApiResponse;

  if (!response.ok || data.ok === false) {
    const error = new Error(data.error || data.detail || `Request failed with status ${response.status}`) as ApiError;
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

export async function getAdminSessions(adminKey: string, limit: number = 50, offset: number = 0) {
  const params = new URLSearchParams({ limit: limit.toString(), offset: offset.toString() });
  const response = await fetch(`${API_BASE_URL}/admin/sessions?${params}`, {
    headers: {
      'X-Admin-Key': adminKey,
    },
  });
  return readJsonResponse(response);
}

export async function getAdminSessionDetail(adminKey: string, sessionId: string) {
  const response = await fetch(`${API_BASE_URL}/admin/sessions/${sessionId}`, {
    headers: {
      'X-Admin-Key': adminKey,
    },
  });
  return readJsonResponse(response);
}

export async function getAdminSessionLogs(adminKey: string, sessionId: string) {
  const response = await fetch(`${API_BASE_URL}/admin/sessions/${sessionId}/logs`, {
    headers: {
      'X-Admin-Key': adminKey,
    },
  });
  return readJsonResponse(response);
}
