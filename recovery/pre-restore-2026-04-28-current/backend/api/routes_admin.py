"""
Admin and Audit endpoints for thesis MVP.

Protected with simple X-Admin-Key header (environment variable: ADMIN_KEY).
Read-only access to:
  - Sessions list with summarized verification data
  - Full session details (identity, verification results, media paths)
  - Audit logs timeline
  - Embedding metadata (without full vectors)
"""

import os
from fastapi import APIRouter, Header, HTTPException, Query
from typing import Annotated

from backend.services.db_service import (
    get_admin_sessions,
    get_admin_session_detail,
    get_admin_session_logs,
    get_admin_embedding_metadata,
)

router = APIRouter(prefix="/admin", tags=["admin"])

# Load admin key from environment; use dev default if not set
ADMIN_KEY = os.getenv("ADMIN_KEY", "dev-admin-key")


def verify_admin_key(x_admin_key: Annotated[str | None, Header()] = None):
    """Dependency: verify X-Admin-Key header."""
    if x_admin_key is None or x_admin_key != ADMIN_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing X-Admin-Key header",
        )


@router.get("/sessions")
def admin_get_sessions(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    x_admin_key: Annotated[str | None, Header()] = None,
):
    """
    Get list of recent KYC sessions.
    
    Query params:
    - limit: number of sessions to return (default 50, max 500)
    - offset: pagination offset (default 0)
    
    Headers:
    - X-Admin-Key: admin key value
    
    Returns list of sessions with summary data:
    - session_id, created_at, updated_at
    - first_name, last_name, cnp, series_number
    - liveness_passed, face_match_distance, face_match_decision, final_decision
    - status
    """
    verify_admin_key(x_admin_key)
    sessions = get_admin_sessions(limit=limit, offset=offset)
    return {
        "ok": True,
        "count": len(sessions),
        "limit": limit,
        "offset": offset,
        "sessions": sessions,
    }


@router.get("/sessions/{session_id}")
def admin_get_session_detail(
    session_id: str,
    x_admin_key: Annotated[str | None, Header()] = None,
):
    """
    Get full details of a specific KYC session.
    
    Path params:
    - session_id: UUID of the session
    
    Headers:
    - X-Admin-Key: admin key value
    
    Returns session details:
    - Identity/OCR fields
    - Document, ID face, selfie, liveness video paths
    - Verification results (liveness, face match, final decision)
    - Raw OCR text
    - Timestamps
    """
    verify_admin_key(x_admin_key)
    
    session = get_admin_session_detail(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get embedding metadata if available
    embeddings = get_admin_embedding_metadata(session_id)
    session["embeddings"] = embeddings
    
    return {
        "ok": True,
        "session": session,
    }


@router.get("/sessions/{session_id}/logs")
def admin_get_session_logs(
    session_id: str,
    x_admin_key: Annotated[str | None, Header()] = None,
):
    """
    Get audit log timeline for a specific session.
    
    Path params:
    - session_id: UUID of the session
    
    Headers:
    - X-Admin-Key: admin key value
    
    Returns list of audit events chronologically:
    - event_type, message, created_at
    """
    verify_admin_key(x_admin_key)
    
    logs = get_admin_session_logs(session_id)
    
    return {
        "ok": True,
        "session_id": session_id,
        "logs": logs,
    }
