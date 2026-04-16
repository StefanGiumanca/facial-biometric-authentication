export const WIFI_API_BASE_URL = 'http://192.168.68.53:8000';
export const NGROK_API_BASE_URL = 'https://reward-botanist-tag.ngrok-free.dev';

// Switch this one line depending on how the iPhone reaches the backend:
// - use WIFI_API_BASE_URL when phone and laptop are on the same Wi-Fi
// - use NGROK_API_BASE_URL when phone is on mobile data
export const API_BASE_URL = NGROK_API_BASE_URL;

export type UploadAsset = {
  uri: string;
  fileName?: string | null;
  mimeType?: string | null;
};

type ApiResponse = {
  ok?: boolean;
  error?: string;
  [key: string]: unknown;
};

async function readJsonResponse(response: Response): Promise<ApiResponse> {
  const data = (await response.json()) as ApiResponse;

  if (!response.ok || data.ok === false) {
    throw new Error(data.error || `Request failed with status ${response.status}`);
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
