import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { clearStoredAdminKey, getStoredAdminKey, setStoredAdminKey } from './lib/storage';

type AuthContextValue = {
  adminKey: string;
  setAdminKey: (value: string) => void;
  clearAdminKey: () => void;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [adminKey, setAdminKeyState] = useState('');

  useEffect(() => {
    setAdminKeyState(getStoredAdminKey());
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      adminKey,
      setAdminKey: (value: string) => {
        setStoredAdminKey(value);
        setAdminKeyState(value);
      },
      clearAdminKey: () => {
        clearStoredAdminKey();
        setAdminKeyState('');
      },
    }),
    [adminKey],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAdminAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAdminAuth must be used within AuthProvider');
  }

  return context;
}
