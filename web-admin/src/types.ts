export type AdminSession = {
  session_id: string;
  created_at: string;
  updated_at: string;
  first_name: string | null;
  last_name: string | null;
  cnp: string | null;
  series_number: string | null;
  sex: string | null;
  nationality: string | null;
  address: string | null;
  valid_from: string | null;
  valid_until: string | null;
  liveness_passed: boolean | null;
  selfie_gate_distance: number | null;
  selfie_gate_decision: string | null;
  final_face_match_distance: number | null;
  final_face_match_decision: string | null;
  face_match_distance: number | null;
  face_match_decision: string | null;
  final_decision: string | null;
  status: string;
  security_fail_count: number;
  reject_reason: string | null;
  locked_at: string | null;
};

export type EmbeddingMetadata = {
  id: number;
  embedding_type: string;
  vector_length: number | null;
  vector_preview: number[] | null;
  created_at: string;
};

export type SessionDetail = {
  session_id: string;
  created_at: string;
  updated_at: string;
  status: string;
  first_name: string | null;
  last_name: string | null;
  cnp: string | null;
  series_number: string | null;
  sex: string | null;
  nationality: string | null;
  address: string | null;
  valid_from: string | null;
  valid_until: string | null;
  document_path: string | null;
  id_face_path: string | null;
  selfie_path: string | null;
  liveness_video_path: string | null;
  liveness_passed: boolean | null;
  selfie_gate_distance: number | null;
  selfie_gate_decision: string | null;
  final_face_match_distance: number | null;
  final_face_match_decision: string | null;
  face_match_distance: number | null;
  face_match_decision: string | null;
  final_decision: string | null;
  raw_ocr_text: string | null;
  security_fail_count: number;
  reject_reason: string | null;
  locked_at: string | null;
  embeddings: EmbeddingMetadata[];
};

export type AuditLogEntry = {
  id: number;
  event_type: string;
  message: string;
  created_at: string;
};

export type AdminSessionsResponse = {
  ok: boolean;
  count: number;
  limit: number;
  offset: number;
  sessions: AdminSession[];
};

export type AdminSessionDetailResponse = {
  ok: boolean;
  session: SessionDetail;
};

export type AdminSessionLogsResponse = {
  ok: boolean;
  session_id: string;
  logs: AuditLogEntry[];
};

export type ApiError = Error & {
  status?: number;
  detail?: string;
};

export type AdminDecision = 'ACCEPTED' | 'REJECTED';

export type AdminDecisionResponse = {
  ok: boolean;
  message: string;
  session: {
    session_id: string;
    status: string;
    final_decision: AdminDecision;
    reject_reason: string | null;
  };
};

export type AdminDeleteSessionResponse = {
  ok: boolean;
  message: string;
  session_id: string;
};
