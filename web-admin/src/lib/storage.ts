const ADMIN_KEY_STORAGE = 'visionauth.adminKey';
const THEME_STORAGE = 'visionauth.adminTheme';

export type AdminTheme = 'dark' | 'light';

export function getStoredAdminKey() {
  return window.localStorage.getItem(ADMIN_KEY_STORAGE) ?? '';
}

export function setStoredAdminKey(adminKey: string) {
  window.localStorage.setItem(ADMIN_KEY_STORAGE, adminKey);
}

export function clearStoredAdminKey() {
  window.localStorage.removeItem(ADMIN_KEY_STORAGE);
}

export function getStoredTheme(): AdminTheme {
  return window.localStorage.getItem(THEME_STORAGE) === 'light' ? 'light' : 'dark';
}

export function setStoredTheme(theme: AdminTheme) {
  window.localStorage.setItem(THEME_STORAGE, theme);
}
