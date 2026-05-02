const ADMIN_KEY_STORAGE = 'visionauth.adminKey';

export function getStoredAdminKey() {
  return window.localStorage.getItem(ADMIN_KEY_STORAGE) ?? '';
}

export function setStoredAdminKey(adminKey: string) {
  window.localStorage.setItem(ADMIN_KEY_STORAGE, adminKey);
}

export function clearStoredAdminKey() {
  window.localStorage.removeItem(ADMIN_KEY_STORAGE);
}
