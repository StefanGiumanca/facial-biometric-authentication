import { FormEvent, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Shell } from '../components';
import { getAdminSessions } from '../lib/api';
import { useAdminAuth } from '../auth';

export function LoginPage() {
  const navigate = useNavigate();
  const { setAdminKey } = useAdminAuth();
  const [value, setValue] = useState('dev-admin-key');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      await getAdminSessions(value, 1, 0);
      setAdminKey(value);
      navigate('/');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unable to validate the admin key.';
      setError(message.includes('403') ? 'Invalid admin key. Please check X-Admin-Key and try again.' : message);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <Shell>
      <div className="auth-layout">
        <section className="auth-panel auth-panel--hero">
          <p className="eyebrow">Fintech Review Workspace</p>
          <h1>Manual review for VisionAuth eKYC sessions.</h1>
          <p className="auth-panel__lead">
            Inspect OCR output, face verification results, protected preview media, and audit logs from a single web dashboard.
          </p>
          <div className="hero-points">
            <div className="hero-point">
              <span className="hero-point__label">Review queue</span>
              <strong>Verified, rejected, and manual review sessions</strong>
            </div>
            <div className="hero-point">
              <span className="hero-point__label">Protected access</span>
              <strong>`X-Admin-Key` stored locally in your browser for development</strong>
            </div>
            <div className="hero-point">
              <span className="hero-point__label">Audit visibility</span>
              <strong>OCR, liveness, face match, and timeline inspection in one place</strong>
            </div>
          </div>
        </section>

        <section className="auth-panel auth-panel--form">
          <div className="auth-card">
            <p className="eyebrow">Admin Access</p>
            <h2>Enter your admin key</h2>
            <p className="auth-card__text">
              The key is kept only in browser state and `localStorage` to make thesis demos easier.
            </p>

            <form className="auth-form" onSubmit={handleSubmit}>
              <label className="field">
                <span>Admin key</span>
                <input
                  type="password"
                  value={value}
                  onChange={(event) => setValue(event.target.value)}
                  placeholder="dev-admin-key"
                  autoFocus
                />
              </label>

              {error ? <div className="error-banner">{error}</div> : null}

              <button type="submit" className="button button--primary" disabled={isLoading || !value.trim()}>
                {isLoading ? 'Validating...' : 'Open Dashboard'}
              </button>
            </form>
          </div>
        </section>
      </div>
    </Shell>
  );
}
