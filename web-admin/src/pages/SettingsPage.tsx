import { useNavigate } from 'react-router-dom';
import { BrandHeader, DetailGrid, SectionCard, Shell, TopNav } from '../components';
import { useAdminAuth } from '../auth';
import { API_BASE_URL } from '../lib/api';
import { useTheme } from '../theme';

export function SettingsPage() {
  const navigate = useNavigate();
  const { adminKey, clearAdminKey } = useAdminAuth();
  const { theme, setTheme } = useTheme();

  function clearAndReturnToLogin() {
    clearAdminKey();
    navigate('/login');
  }

  return (
    <Shell>
      <TopNav
        searchPlaceholder="Search is unavailable in settings"
        onSignOut={clearAndReturnToLogin}
      />

      <BrandHeader
        title="Settings"
        subtitle="Admin console configuration, API visibility, and local dashboard preferences."
      />

      <div className="settings-grid">
        <SectionCard title="Admin Access" hint="Authentication state is stored locally for this browser session.">
          <DetailGrid
            rows={[
              { label: 'Admin key status', value: adminKey ? 'Configured' : 'Missing' },
              { label: 'Key fingerprint', value: adminKey ? `${adminKey.slice(0, 4)}...${adminKey.slice(-4)}` : 'No admin key stored', subtle: !adminKey },
              { label: 'API base URL', value: <code>{API_BASE_URL}</code> },
            ]}
          />
          <div className="settings-actions">
            <button type="button" className="button button--danger" onClick={clearAndReturnToLogin}>
              Clear Admin Key
            </button>
          </div>
        </SectionCard>

        <SectionCard title="Theme Preference" hint="Theme choice is saved in localStorage for this browser.">
          <div className="theme-picker" role="group" aria-label="Admin theme">
            <button
              type="button"
              className={theme === 'dark' ? 'theme-choice theme-choice--active' : 'theme-choice'}
              onClick={() => setTheme('dark')}>
              <strong>Dark</strong>
              <span>Premium navy operations view</span>
            </button>
            <button
              type="button"
              className={theme === 'light' ? 'theme-choice theme-choice--active' : 'theme-choice'}
              onClick={() => setTheme('light')}>
              <strong>Light</strong>
              <span>Bright review workspace</span>
            </button>
          </div>
        </SectionCard>

        <SectionCard title="Dashboard Preferences" hint="Local preferences for future admin workflows.">
          <div className="preference-placeholder">
            Queue density, default filters, and notification preferences can live here as the operator workflow grows.
          </div>
        </SectionCard>
      </div>
    </Shell>
  );
}
