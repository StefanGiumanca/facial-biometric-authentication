import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { getStoredTheme, setStoredTheme, type AdminTheme } from './lib/storage';

type ThemeContextValue = {
  theme: AdminTheme;
  setTheme: (theme: AdminTheme) => void;
  toggleTheme: () => void;
};

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<AdminTheme>(() => getStoredTheme());

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    setStoredTheme(theme);
  }, [theme]);

  const value = useMemo(
    () => ({
      theme,
      setTheme: setThemeState,
      toggleTheme: () => setThemeState((current) => (current === 'dark' ? 'light' : 'dark')),
    }),
    [theme],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }

  return context;
}
