import { useCallback, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
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
import {
  API_BASE_URL,
  buildAdminHeaders,
  buildAdminMediaUrl,
  deleteAdminSession,
  getAdminSessionDetail,
  getAdminSessionLogs,
  saveAdminDecision,
} from '../lib/api';
import { formatDateTime, formatDecisionWithDistance } from '../lib/utils';
import type { AdminDecision, AuditLogEntry, SessionDetail } from '../types';

const TOOL_LABELS = ['EasyOCR', 'OpenCV', 'MediaPipe', 'face_recognition'];
const EMPTY_VALUE = 'No value recorded for this session';

export function SessionDetailPage() {
  const { sessionId = '' } = useParams();
  const navigate = useNavigate();
  const { adminKey, clearAdminKey } = useAdminAuth();
  const [session, setSession] = useState<SessionDetail | null>(null);
  const [logs, setLogs] = useState<AuditLogEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isSavingDecision, setIsSavingDecision] = useState(false);
  const [isDeletingSession, setIsDeletingSession] = useState(false);
  const [adminNote, setAdminNote] = useState('');
  const [error, setError] = useState('');
  const [decisionMessage, setDecisionMessage] = useState('');
  const [enlargedMedia, setEnlargedMedia] = useState<{ url: string; alt: string; kind: 'image' | 'video' } | null>(null);

  const loadData = useCallback(
    async (showLoading = true) => {
      if (showLoading) {
        setIsLoading(true);
      }

      try {
        setError('');
        const [detailResponse, logsResponse] = await Promise.all([
          getAdminSessionDetail(adminKey, sessionId),
          getAdminSessionLogs(adminKey, sessionId),
        ]);

        setSession(detailResponse.session);
        setLogs(logsResponse.logs);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load session detail.';
        setError(message);
        if (message.toLowerCase().includes('invalid') || message.includes('403')) {
          clearAdminKey();
          navigate('/login');
        }
      } finally {
        setIsLoading(false);
      }
    },
    [adminKey, clearAdminKey, navigate, sessionId],
  );

  useEffect(() => {
    loadData();
  }, [loadData]);

  async function handleDecision(decision: AdminDecision) {
    const actionLabel = decision === 'ACCEPTED' ? 'approve' : 'reject';
    const confirmed = window.confirm(`Confirm that you want to ${actionLabel} this KYC session?`);
    if (!confirmed) {
      return;
    }

    try {
      setIsSavingDecision(true);
      setError('');
      setDecisionMessage('');
      const response = await saveAdminDecision(adminKey, sessionId, decision, adminNote);
      setDecisionMessage(response.message || 'Decision saved successfully');
      setAdminNote('');
      await loadData(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not save admin decision.');
    } finally {
      setIsSavingDecision(false);
    }
  }

  async function handleDeleteSession() {
    const confirmed = window.confirm(
      'Permanently delete this KYC session from the database? This will also remove its audit logs and embedding records.',
    );
    if (!confirmed) {
      return;
    }

    const confirmedAgain = window.confirm('This cannot be undone. Delete this session permanently?');
    if (!confirmedAgain) {
      return;
    }

    try {
      setIsDeletingSession(true);
      setError('');
      await deleteAdminSession(adminKey, sessionId);
      navigate('/');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not delete session.');
    } finally {
      setIsDeletingSession(false);
    }
  }

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
          <Link className="button button--secondary" to="/sessions">
            Back to Sessions
          </Link>
        }
      />

      {error ? <div className="error-banner">{error}</div> : null}
      {decisionMessage ? <div className="success-banner">{decisionMessage}</div> : null}
      {isLoading ? <div className="empty-state">Loading session details...</div> : null}

      {session ? (
        <>
          <div className="metric-grid">
            <MetricCard label="Final decision" value={session.final_decision || 'Manual decision required'} helper={`Status: ${session.status}`} />
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
              helper={session.reject_reason || 'No rejection note recorded'}
            />
          </div>

          <motion.div
            className="evidence-strip"
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.42, ease: 'easeOut' }}>
            <EvidencePill label="Identity" value={session.first_name || session.last_name ? 'Captured' : 'Incomplete'} tone={session.first_name || session.last_name ? 'success' : 'warning'} />
            <EvidencePill label="Document face" value={session.id_face_path ? 'Available' : 'Missing'} tone={session.id_face_path ? 'success' : 'danger'} />
            <EvidencePill label="Selfie" value={session.selfie_path ? 'Available' : 'Missing'} tone={session.selfie_path ? 'success' : 'danger'} />
            <EvidencePill label="Match score" value={formatMatchScore(session)} tone={getMatchScorePercent(session) >= 70 ? 'success' : getMatchScorePercent(session) > 0 ? 'warning' : 'info'} />
          </motion.div>

          <VerificationPipeline session={session} />

          <div className="detail-layout">
            <div className="detail-layout__main">
              <SectionCard title="Identity Data" hint="Identity values captured from OCR and operator review.">
                <DetailGrid
                  rows={[
                    { label: 'First Name', value: session.first_name || EMPTY_VALUE, subtle: !session.first_name },
                    { label: 'Last Name', value: session.last_name || EMPTY_VALUE, subtle: !session.last_name },
                    { label: 'CNP', value: session.cnp || EMPTY_VALUE, subtle: !session.cnp },
                    { label: 'Series Number', value: session.series_number || EMPTY_VALUE, subtle: !session.series_number },
                    { label: 'Sex', value: session.sex || EMPTY_VALUE, subtle: !session.sex },
                    { label: 'Nationality', value: session.nationality || EMPTY_VALUE, subtle: !session.nationality },
                    { label: 'Address', value: session.address || EMPTY_VALUE, subtle: !session.address },
                    { label: 'Valid From', value: session.valid_from || EMPTY_VALUE, subtle: !session.valid_from },
                    { label: 'Valid Until', value: session.valid_until || EMPTY_VALUE, subtle: !session.valid_until },
                  ]}
                />
              </SectionCard>

              <SectionCard title="Verification Results" hint="Core liveness, matching, and session decision data.">
                <DetailGrid
                  rows={[
                    { label: 'Final Decision', value: <StatusBadge label={session.final_decision || 'MANUAL_REVIEW'} /> },
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
                          : EMPTY_VALUE,
                    },
                    {
                      label: 'Face Match Decision',
                      value: session.final_face_match_decision
                        ? formatDecisionWithDistance(session.final_face_match_decision, session.final_face_match_distance)
                        : session.face_match_decision
                          ? formatDecisionWithDistance(session.face_match_decision, session.face_match_distance)
                          : EMPTY_VALUE,
                    },
                    {
                      label: 'Selfie Gate',
                      value: formatDecisionWithDistance(session.selfie_gate_decision, session.selfie_gate_distance),
                    },
                  ]}
                />
              </SectionCard>

              <SectionCard title="Evidence Gallery" hint="ID image, extracted face, selfie, and liveness evidence from protected admin endpoints.">
                <div className="media-grid media-grid--four">
                  <EvidenceMediaCard
                    title="ID Document"
                    path={session.document_path}
                    adminKey={adminKey}
                    url={buildAdminMediaUrl(session.session_id, 'document')}
                    alt="ID document"
                    onEnlarge={(url) => setEnlargedMedia({ url, alt: 'ID document', kind: 'image' })}
                  />
                  <EvidenceMediaCard
                    title="ID Face Crop"
                    path={session.id_face_path}
                    adminKey={adminKey}
                    url={buildAdminMediaUrl(session.session_id, 'id_face')}
                    alt="ID face"
                    onEnlarge={(url) => setEnlargedMedia({ url, alt: 'ID face', kind: 'image' })}
                  />
                  <EvidenceMediaCard
                    title="Selfie"
                    path={session.selfie_path}
                    adminKey={adminKey}
                    url={buildAdminMediaUrl(session.session_id, 'selfie')}
                    alt="Selfie"
                    onEnlarge={(url) => setEnlargedMedia({ url, alt: 'Selfie', kind: 'image' })}
                  />
                  <EvidenceMediaCard
                    title="Liveness Video"
                    path={session.liveness_video_path}
                    adminKey={adminKey}
                    url={buildAdminMediaUrl(session.session_id, 'liveness_video')}
                    alt="Liveness video"
                    kind="video"
                    onEnlarge={(url) => setEnlargedMedia({ url, alt: 'Liveness video', kind: 'video' })}
                  />
                </div>
                <motion.div
                  className="match-score-panel"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.34 }}>
                  <div>
                    <span className="match-score-panel__label">Face match score</span>
                    <strong>{formatMatchScore(session)}</strong>
                  </div>
                  <div className="match-score-panel__meter" aria-hidden="true">
                    <motion.span
                      initial={{ width: 0 }}
                      animate={{ width: `${getMatchScorePercent(session)}%` }}
                      transition={{ duration: 0.7, ease: 'easeOut' }}
                    />
                  </div>
                </motion.div>
                <p className="note-text">
                  Preview images are served through protected admin routes under <code>{API_BASE_URL}/admin/sessions/:id/media/:kind</code>.
                  Review the biometric evidence before approving or rejecting.
                </p>
              </SectionCard>

              <SectionCard title="Audit Timeline" hint="Chronological session activity from /admin/sessions/{session_id}/logs.">
                <Timeline items={logs} />
              </SectionCard>
            </div>

            <aside className="detail-layout__side">
              <SectionCard title="Manual Decision" hint="Review the biometric evidence before approving or rejecting.">
                <div className="decision-panel-title">
                  <strong>Operator decision controls</strong>
                  <span>Final action is saved to the audit record.</span>
                </div>
                <label className="field">
                  <span>Admin note</span>
                  <textarea
                    value={adminNote}
                    onChange={(event) => setAdminNote(event.target.value)}
                    placeholder="Optional note for the audit log"
                    rows={4}
                    maxLength={500}
                  />
                </label>
                <div className="review-actions">
                  <button
                    type="button"
                    className="button button--success"
                    disabled={isSavingDecision}
                    onClick={() => handleDecision('ACCEPTED')}>
                    {isSavingDecision ? 'Saving...' : 'Approve'}
                  </button>
                  <button
                    type="button"
                    className="button button--danger"
                    disabled={isSavingDecision}
                    onClick={() => handleDecision('REJECTED')}>
                    {isSavingDecision ? 'Saving...' : 'Reject'}
                  </button>
                </div>
                <p className="note-text">
                  Approved sessions are marked <code>ACCEPTED</code>; rejected sessions are marked <code>REJECTED</code> and the note is retained in the audit log.
                </p>
              </SectionCard>

              <SectionCard title="Danger Zone" hint="Remove this session from the admin database.">
                <button
                  type="button"
                  className="button button--danger"
                  disabled={isDeletingSession}
                  onClick={handleDeleteSession}>
                  {isDeletingSession ? 'Deleting...' : 'Delete session permanently'}
                </button>
                <p className="note-text">
                  This removes the session record, audit timeline, and embedding metadata from PostgreSQL.
                </p>
              </SectionCard>

              <SectionCard title="Technical Details" hint="Useful for thesis demos and implementation walkthroughs.">
                <DetailGrid
                  rows={[
                    { label: 'Session ID', value: session.session_id },
                    { label: 'Created', value: formatDateTime(session.created_at) },
                    { label: 'Updated', value: formatDateTime(session.updated_at) },
                    { label: 'Reject Reason', value: session.reject_reason || EMPTY_VALUE, subtle: !session.reject_reason },
                    { label: 'Locked At', value: formatDateTime(session.locked_at) },
                    { label: 'Libraries', value: TOOL_LABELS.join(', ') },
                  ]}
                />
              </SectionCard>

              <SectionCard title="OCR Raw Text" hint="Preview of the raw OCR text stored with the session.">
                <pre className="ocr-preview">{session.raw_ocr_text || 'No value recorded for this session'}</pre>
              </SectionCard>

              <SectionCard title="Embedding Metadata" hint="Admin detail now includes metadata without exposing full vectors.">
                {session.embeddings.length === 0 ? (
                  <div className="empty-state empty-state--compact">No value recorded for this session.</div>
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
                            : EMPTY_VALUE}
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
      {enlargedMedia ? (
        <div className="media-modal" role="dialog" aria-modal="true" onClick={() => setEnlargedMedia(null)}>
          <button type="button" className="media-modal__close" onClick={() => setEnlargedMedia(null)}>
            Close
          </button>
          {enlargedMedia.kind === 'video' ? (
            <video src={enlargedMedia.url} controls className="media-modal__asset" />
          ) : (
            <img src={enlargedMedia.url} alt={enlargedMedia.alt} className="media-modal__asset" />
          )}
        </div>
      ) : null}
    </Shell>
  );
}

function EvidenceMediaCard({
  title,
  path,
  adminKey,
  url,
  alt,
  kind = 'image',
  onEnlarge,
}: {
  title: string;
  path: string | null;
  adminKey: string;
  url: string;
  alt: string;
  kind?: 'image' | 'video';
  onEnlarge: (url: string) => void;
}) {
  return (
    <motion.div className="media-card" whileHover={{ y: -4, scale: 1.01 }} transition={{ duration: 0.18 }}>
      <div className="media-card__title">{title}</div>
      {path ? (
        <ProtectedMediaPreview adminKey={adminKey} url={url} alt={alt} kind={kind} onEnlarge={onEnlarge} />
      ) : (
        <div className="media-fallback">No {title.toLowerCase()} available.</div>
      )}
      <code>{path || EMPTY_VALUE}</code>
    </motion.div>
  );
}

function ProtectedMediaPreview({
  adminKey,
  url,
  alt,
  kind = 'image',
  onEnlarge,
}: {
  adminKey: string;
  url: string;
  alt: string;
  kind?: 'image' | 'video';
  onEnlarge?: (url: string) => void;
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
          headers: buildAdminHeaders(adminKey),
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

  return (
    <button type="button" className="media-preview-button" onClick={() => onEnlarge?.(src)}>
      {kind === 'video' ? (
        <video src={src} className="media-preview" muted playsInline />
      ) : (
        <img src={src} alt={alt} className="media-preview" />
      )}
      <span>Click to enlarge</span>
    </button>
  );
}

function getSessionFaceDistance(session: SessionDetail) {
  if (typeof session.final_face_match_distance === 'number') {
    return session.final_face_match_distance;
  }

  if (typeof session.face_match_distance === 'number') {
    return session.face_match_distance;
  }

  return null;
}

function getMatchScorePercent(session: SessionDetail) {
  const distance = getSessionFaceDistance(session);
  if (distance === null) {
    return 0;
  }

  return Math.round(Math.max(0, Math.min(1, 1 - distance / 0.6)) * 100);
}

function formatMatchScore(session: SessionDetail) {
  const distance = getSessionFaceDistance(session);
  if (distance === null) {
    return 'No score yet';
  }

  return `${getMatchScorePercent(session)}% confidence (${distance.toFixed(3)} distance)`;
}

function EvidencePill({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: 'success' | 'danger' | 'warning' | 'info';
}) {
  return (
    <div className={`evidence-pill evidence-pill--${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function VerificationPipeline({ session }: { session: SessionDetail }) {
  const steps = [
    {
      label: 'Document',
      value: session.document_path ? 'Captured' : 'Missing',
      tone: session.document_path ? 'success' : 'warning',
    },
    {
      label: 'OCR review',
      value: session.first_name || session.cnp || session.series_number ? 'Data ready' : 'Incomplete',
      tone: session.first_name || session.cnp || session.series_number ? 'success' : 'warning',
    },
    {
      label: 'Selfie',
      value: session.selfie_path ? 'Captured' : 'Missing',
      tone: session.selfie_path ? 'success' : 'warning',
    },
    {
      label: 'Liveness',
      value: session.liveness_passed ? 'Passed' : session.liveness_passed === false ? 'Failed' : 'Pending',
      tone: session.liveness_passed ? 'success' : session.liveness_passed === false ? 'danger' : 'info',
    },
    {
      label: 'Face match',
      value: session.final_face_match_decision || session.face_match_decision || 'Pending',
      tone: getFaceMatchPipelineTone(session),
    },
    {
      label: 'Decision',
      value: session.final_decision || 'Manual queue',
      tone: getDecisionPipelineTone(session.final_decision),
    },
  ] as const;

  return (
    <motion.div
      className="verification-pipeline"
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.42, ease: 'easeOut' }}>
      {steps.map((step, index) => (
        <motion.div
          key={step.label}
          className={`verification-pipeline__step verification-pipeline__step--${step.tone}`}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.24, delay: index * 0.04 }}>
          <span className="verification-pipeline__index">{String(index + 1).padStart(2, '0')}</span>
          <span className="verification-pipeline__label">{step.label}</span>
          <strong>{step.value}</strong>
        </motion.div>
      ))}
    </motion.div>
  );
}

function getFaceMatchPipelineTone(session: SessionDetail) {
  const decision = String(session.final_face_match_decision || session.face_match_decision || '').toUpperCase();

  if (['ACCEPTED', 'VERIFIED', 'APPROVED', 'PASS', 'PASSED'].includes(decision)) {
    return 'success';
  }

  if (['REJECTED', 'FAILED', 'FAIL'].includes(decision)) {
    return 'danger';
  }

  if (decision === 'MANUAL_REVIEW') {
    return 'warning';
  }

  return 'info';
}

function getDecisionPipelineTone(decision?: string | null) {
  const normalized = String(decision || '').toUpperCase();

  if (['ACCEPTED', 'VERIFIED', 'APPROVED'].includes(normalized)) {
    return 'success';
  }

  if (normalized === 'REJECTED') {
    return 'danger';
  }

  if (normalized === 'MANUAL_REVIEW') {
    return 'warning';
  }

  return 'info';
}
