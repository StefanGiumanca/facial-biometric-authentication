import { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  BrandHeader,
  DetailGrid,
  MetricCard,
  SectionCard,
  Shell,
  StatusBadge,
  Timeline,
  TopNav,
} from '../components';
import { useAdminAuth } from '../auth';
import { API_BASE_URL, buildAdminMediaUrl, getAdminSessionDetail, getAdminSessionLogs } from '../lib/api';
import { formatDateTime, formatDecisionWithDistance } from '../lib/utils';
import type { AuditLogEntry, SessionDetail } from '../types';

const TOOL_LABELS = ['EasyOCR', 'OpenCV', 'MediaPipe', 'face_recognition'];

export function SessionDetailPage() {
  const { sessionId = '' } = useParams();
  const navigate = useNavigate();
  const { adminKey, clearAdminKey } = useAdminAuth();
  const [session, setSession] = useState<SessionDetail | null>(null);
  const [logs, setLogs] = useState<AuditLogEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;

    async function loadData() {
      try {
        setError('');
        const [detailResponse, logsResponse] = await Promise.all([
          getAdminSessionDetail(adminKey, sessionId),
          getAdminSessionLogs(adminKey, sessionId),
        ]);

        if (!cancelled) {
          setSession(detailResponse.session);
          setLogs(logsResponse.logs);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load session detail.';
        if (!cancelled) {
          setError(message);
          if (message.toLowerCase().includes('invalid') || message.includes('403')) {
            clearAdminKey();
            navigate('/login');
          }
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    loadData();

    return () => {
      cancelled = true;
    };
  }, [adminKey, clearAdminKey, navigate, sessionId]);

  return (
    <Shell>
      <TopNav onSignOut={() => {
        clearAdminKey();
        navigate('/login');
      }} />

      <BrandHeader
        title="Session Detail"
        subtitle={session ? `Full audit view for ${session.session_id}` : 'Loading full audit view'}
        action={
          <Link className="button button--secondary" to="/">
            Back to Sessions
          </Link>
        }
      />

      {error ? <div className="error-banner">{error}</div> : null}
      {isLoading ? <div className="empty-state">Loading session details...</div> : null}

      {session ? (
        <>
          <div className="metric-grid">
            <MetricCard label="Final decision" value={session.final_decision || 'Pending'} helper={`Status: ${session.status}`} />
            <MetricCard
              label="Liveness"
              value={session.liveness_passed ? 'Passed' : session.liveness_passed === false ? 'Failed' : 'Pending'}
              helper="Bound to stored session review state"
            />
            <MetricCard
              label="Face match"
              value={formatDecisionWithDistance(session.final_face_match_decision, session.final_face_match_distance)}
              helper="Final threshold decision"
            />
            <MetricCard
              label="Security fails"
              value={`${session.security_fail_count ?? 0}/3`}
              helper={session.reject_reason || 'No reject reason stored'}
            />
          </div>

          <div className="detail-layout">
            <div className="detail-layout__main">
              <SectionCard title="Identity Data" hint="Fields captured by the current backend schema and review flow.">
                <DetailGrid
                  rows={[
                    { label: 'First Name', value: session.first_name || 'N/A' },
                    { label: 'Last Name', value: session.last_name || 'N/A' },
                    { label: 'CNP', value: session.cnp || 'N/A' },
                    { label: 'Series Number', value: session.series_number || 'N/A' },
                    { label: 'Sex', value: 'Not stored in current backend schema', subtle: true },
                    { label: 'Nationality', value: 'Not stored in current backend schema', subtle: true },
                    { label: 'Address', value: 'Not stored in current backend schema', subtle: true },
                    { label: 'Valid From', value: 'Not stored in current backend schema', subtle: true },
                    { label: 'Valid Until', value: 'Not stored in current backend schema', subtle: true },
                  ]}
                />
              </SectionCard>

              <SectionCard title="Verification Results" hint="Core liveness, matching, and session decision data.">
                <DetailGrid
                  rows={[
                    { label: 'Final Decision', value: <StatusBadge label={session.final_decision || 'PENDING'} /> },
                    { label: 'Status', value: <StatusBadge label={session.status} /> },
                    {
                      label: 'Liveness Passed',
                      value: session.liveness_passed ? 'Yes' : session.liveness_passed === false ? 'No' : 'Pending',
                    },
                    {
                      label: 'Face Match Distance',
                      value:
                        typeof session.final_face_match_distance === 'number'
                          ? session.final_face_match_distance.toFixed(3)
                          : typeof session.face_match_distance === 'number'
                            ? session.face_match_distance.toFixed(3)
                            : 'N/A',
                    },
                    {
                      label: 'Face Match Decision',
                      value:
                        formatDecisionWithDistance(session.final_face_match_decision, session.final_face_match_distance) ||
                        formatDecisionWithDistance(session.face_match_decision, session.face_match_distance),
                    },
                    {
                      label: 'Selfie Gate',
                      value: formatDecisionWithDistance(session.selfie_gate_decision, session.selfie_gate_distance),
                    },
                  ]}
                />
              </SectionCard>

              <SectionCard title="Visual Comparison" hint="Protected previews from FastAPI admin endpoints.">
                <div className="media-grid">
                  <div className="media-card">
                    <div className="media-card__title">ID Face Image</div>
                    {session.id_face_path ? (
                      <ProtectedMediaPreview
                        adminKey={adminKey}
                        url={buildAdminMediaUrl(session.session_id, 'id_face')}
                        alt="ID face"
                      />
                    ) : (
                      <div className="media-fallback">No ID face image available.</div>
                    )}
                    <code>{session.id_face_path || 'TODO: backend did not return an ID face path.'}</code>
                  </div>

                  <div className="media-card">
                    <div className="media-card__title">Selfie Image</div>
                    {session.selfie_path ? (
                      <ProtectedMediaPreview
                        adminKey={adminKey}
                        url={buildAdminMediaUrl(session.session_id, 'selfie')}
                        alt="Selfie"
                      />
                    ) : (
                      <div className="media-fallback">No selfie image available.</div>
                    )}
                    <code>{session.selfie_path || 'TODO: backend did not return a selfie path.'}</code>
                  </div>
                </div>
                <p className="note-text">
                  Preview URLs come from protected admin routes under <code>{API_BASE_URL}/admin/sessions/:id/media/:kind</code>. If an image
                  does not render in your browser, the stored file may be missing on disk.
                </p>
              </SectionCard>

              <SectionCard title="Audit Timeline" hint="Chronological session activity from /admin/sessions/{session_id}/logs.">
                <Timeline items={logs} />
              </SectionCard>
            </div>

            <aside className="detail-layout__side">
              <SectionCard title="Manual Review Controls" hint="UI is present, but decision endpoints are not implemented yet.">
                <div className="review-actions">
                  <button type="button" className="button button--success" disabled>
                    Approve
                  </button>
                  <button type="button" className="button button--danger" disabled>
                    Reject
                  </button>
                  <button type="button" className="button button--warning" disabled>
                    Mark Manual Review
                  </button>
                </div>
                <p className="note-text">
                  Coming soon. A dedicated backend endpoint is required before these actions can safely change session outcomes.
                </p>
              </SectionCard>

              <SectionCard title="Technical Details" hint="Useful for thesis demos and implementation walkthroughs.">
                <DetailGrid
                  rows={[
                    { label: 'Session ID', value: session.session_id },
                    { label: 'Created', value: formatDateTime(session.created_at) },
                    { label: 'Updated', value: formatDateTime(session.updated_at) },
                    { label: 'Reject Reason', value: session.reject_reason || 'N/A' },
                    { label: 'Locked At', value: formatDateTime(session.locked_at) },
                    { label: 'Libraries', value: TOOL_LABELS.join(', ') },
                  ]}
                />
              </SectionCard>

              <SectionCard title="OCR Raw Text" hint="Preview of the raw OCR text stored with the session.">
                <pre className="ocr-preview">{session.raw_ocr_text || 'No OCR raw text was stored for this session.'}</pre>
              </SectionCard>

              <SectionCard title="Embedding Metadata" hint="Admin detail now includes metadata without exposing full vectors.">
                {session.embeddings.length === 0 ? (
                  <div className="empty-state empty-state--compact">No embedding metadata returned for this session.</div>
                ) : (
                  <div className="embedding-list">
                    {session.embeddings.map((embedding) => (
                      <div key={embedding.id} className="embedding-card">
                        <strong>{embedding.embedding_type}</strong>
                        <span>Length: {embedding.vector_length ?? 'Unknown'}</span>
                        <span>
                          Preview:{' '}
                          {embedding.vector_preview?.length
                            ? `[${embedding.vector_preview.map((value) => value.toFixed(3)).join(', ')}]`
                            : 'N/A'}
                        </span>
                        <span>Created: {formatDateTime(embedding.created_at)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </SectionCard>
            </aside>
          </div>
        </>
      ) : null}
    </Shell>
  );
}

function ProtectedMediaPreview({
  adminKey,
  url,
  alt,
}: {
  adminKey: string;
  url: string;
  alt: string;
}) {
  const [src, setSrc] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    let isCancelled = false;
    let objectUrl = '';

    async function loadMedia() {
      try {
        setError('');
        const response = await fetch(url, {
          headers: {
            'X-Admin-Key': adminKey,
          },
        });

        if (!response.ok) {
          throw new Error(`Preview unavailable (${response.status})`);
        }

        const blob = await response.blob();
        objectUrl = URL.createObjectURL(blob);

        if (!isCancelled) {
          setSrc(objectUrl);
        }
      } catch (err) {
        if (!isCancelled) {
          setError(err instanceof Error ? err.message : 'Preview unavailable');
        }
      }
    }

    loadMedia();

    return () => {
      isCancelled = true;
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [adminKey, url]);

  if (error) {
    return <div className="media-fallback">{error}</div>;
  }

  if (!src) {
    return <div className="media-fallback">Loading preview...</div>;
  }

  return <img src={src} alt={alt} className="media-preview" />;
}
