from collections.abc import Callable
from typing import Any
from datetime import datetime

from sqlalchemy import desc
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.db.database import SessionLocal
from backend.models.database_models import AuditLog, EmbeddingRecord, KycSession


def _run_db_write(operation: Callable[[Session], None]):
    try:
        with SessionLocal() as db:
            operation(db)
            db.commit()
    except SQLAlchemyError as error:
        print(f"[DB] Write failed: {error}")


def create_session_record(session_id: str):
    def operation(db: Session):
        session = db.get(KycSession, session_id)
        if session is None:
            db.add(KycSession(id=session_id, status="STARTED"))
        else:
            session.status = "STARTED"

    _run_db_write(operation)


def update_session_record(session_id: str, **fields: Any):
    def operation(db: Session):
        session = db.get(KycSession, session_id)
        if session is None:
            session = KycSession(id=session_id)
            db.add(session)

        for field, value in fields.items():
            if hasattr(session, field):
                setattr(session, field, value)

    _run_db_write(operation)


def log_audit_event(session_id: str | None, event_type: str, message: str):
    def operation(db: Session):
        db.add(AuditLog(session_id=session_id, event_type=event_type, message=message))

    _run_db_write(operation)


def store_embedding(session_id: str, embedding_type: str, embedding_vector: dict | list):
    def operation(db: Session):
        db.add(
            EmbeddingRecord(
                session_id=session_id,
                embedding_type=embedding_type,
                embedding_vector=embedding_vector,
            )
        )

    _run_db_write(operation)


# ========== ADMIN/AUDIT ENDPOINTS ==========

def get_admin_sessions(limit: int = 50, offset: int = 0) -> list[dict]:
    """Get recent KYC sessions for admin view, ordered by updated_at DESC."""
    try:
        with SessionLocal() as db:
            sessions = db.query(KycSession)\
                .order_by(desc(KycSession.updated_at))\
                .offset(offset)\
                .limit(limit)\
                .all()
            
            result = []
            for session in sessions:
                result.append({
                    "session_id": session.id,
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                    "first_name": session.first_name,
                    "last_name": session.last_name,
                    "cnp": session.cnp,
                    "series_number": session.series_number,
                    "liveness_passed": session.liveness_passed,
                    "face_match_distance": session.face_match_distance,
                    "face_match_decision": session.face_match_decision,
                    "final_decision": session.final_decision,
                    "status": session.status,
                })
            return result
    except SQLAlchemyError as error:
        print(f"[DB] Admin get_sessions failed: {error}")
        return []


def get_admin_session_detail(session_id: str) -> dict | None:
    """Get full details of a specific KYC session."""
    try:
        with SessionLocal() as db:
            session = db.get(KycSession, session_id)
            if session is None:
                return None
            
            return {
                "session_id": session.id,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "status": session.status,
                "first_name": session.first_name,
                "last_name": session.last_name,
                "cnp": session.cnp,
                "series_number": session.series_number,
                "document_path": session.document_path,
                "id_face_path": session.id_face_path,
                "selfie_path": session.selfie_path,
                "liveness_video_path": session.liveness_video_path,
                "liveness_passed": session.liveness_passed,
                "face_match_distance": session.face_match_distance,
                "face_match_decision": session.face_match_decision,
                "final_decision": session.final_decision,
                "raw_ocr_text": session.raw_ocr_text,
            }
    except SQLAlchemyError as error:
        print(f"[DB] Admin get_session_detail failed: {error}")
        return None


def get_admin_session_logs(session_id: str) -> list[dict]:
    """Get audit logs for a specific session."""
    try:
        with SessionLocal() as db:
            logs = db.query(AuditLog)\
                .filter(AuditLog.session_id == session_id)\
                .order_by(AuditLog.created_at)\
                .all()
            
            result = []
            for log in logs:
                result.append({
                    "id": log.id,
                    "event_type": log.event_type,
                    "message": log.message,
                    "created_at": log.created_at,
                })
            return result
    except SQLAlchemyError as error:
        print(f"[DB] Admin get_session_logs failed: {error}")
        return []


def get_admin_embedding_metadata(session_id: str) -> list[dict]:
    """Get embedding metadata for a session (without full vectors)."""
    try:
        with SessionLocal() as db:
            embeddings = db.query(EmbeddingRecord)\
                .filter(EmbeddingRecord.session_id == session_id)\
                .all()
            
            result = []
            for emb in embeddings:
                vector_preview = None
                vector_length = None
                if emb.embedding_vector:
                    if isinstance(emb.embedding_vector, list):
                        vector_length = len(emb.embedding_vector)
                        vector_preview = emb.embedding_vector[:5] if len(emb.embedding_vector) > 5 else emb.embedding_vector
                    elif isinstance(emb.embedding_vector, dict) and "embedding" in emb.embedding_vector:
                        vec = emb.embedding_vector["embedding"]
                        if isinstance(vec, list):
                            vector_length = len(vec)
                            vector_preview = vec[:5] if len(vec) > 5 else vec
                
                result.append({
                    "id": emb.id,
                    "embedding_type": emb.embedding_type,
                    "vector_length": vector_length,
                    "vector_preview": vector_preview,
                    "created_at": emb.created_at,
                })
            return result
    except SQLAlchemyError as error:
        print(f"[DB] Admin get_embedding_metadata failed: {error}")
        return []
