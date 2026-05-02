import os
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.services.db_service import (
    delete_admin_session,
    get_admin_session_detail,
    get_admin_session_logs,
    get_admin_sessions,
    set_admin_session_decision,
)


router = APIRouter(prefix="/admin", tags=["admin"])
BACKEND_DIR = Path(__file__).resolve().parents[1]
ALLOWED_MEDIA_FIELDS = {
    "document": "document_path",
    "id_face": "id_face_path",
    "selfie": "selfie_path",
    "liveness_video": "liveness_video_path",
}


class AdminDecisionPayload(BaseModel):
    decision: str = Field(pattern="^(ACCEPTED|REJECTED)$")
    admin_note: str | None = Field(default=None, max_length=500)


def verify_admin_key(x_admin_key: Annotated[str | None, Header()] = None):
    admin_key = os.getenv("ADMIN_KEY", "dev-admin-key")
    if x_admin_key != admin_key:
        raise HTTPException(status_code=403, detail="Invalid or missing X-Admin-Key header")


@router.get("/sessions")
def admin_get_sessions(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    x_admin_key: Annotated[str | None, Header()] = None,
):
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
    verify_admin_key(x_admin_key)
    session = get_admin_session_detail(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "ok": True,
        "session": session,
    }


@router.get("/sessions/{session_id}/logs")
def admin_get_session_logs(
    session_id: str,
    x_admin_key: Annotated[str | None, Header()] = None,
):
    verify_admin_key(x_admin_key)
    logs = get_admin_session_logs(session_id)
    return {
        "ok": True,
        "session_id": session_id,
        "logs": logs,
    }


@router.post("/sessions/{session_id}/decision")
def admin_set_session_decision(
    session_id: str,
    payload: AdminDecisionPayload,
    x_admin_key: Annotated[str | None, Header()] = None,
):
    verify_admin_key(x_admin_key)
    admin_note = payload.admin_note.strip() if payload.admin_note else None

    try:
        result = set_admin_session_decision(
            session_id=session_id,
            decision=payload.decision,
            admin_note=admin_note,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Could not save admin decision")

    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "ok": True,
        "message": "Decision saved successfully",
        "session": result,
    }


@router.delete("/sessions/{session_id}")
def admin_delete_session(
    session_id: str,
    x_admin_key: Annotated[str | None, Header()] = None,
):
    verify_admin_key(x_admin_key)

    try:
        deleted = delete_admin_session(session_id)
    except Exception:
        raise HTTPException(status_code=500, detail="Could not delete session")

    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "ok": True,
        "message": "Session deleted permanently",
        "session_id": session_id,
    }


@router.get("/sessions/{session_id}/media/{media_kind}")
def admin_get_session_media(
    session_id: str,
    media_kind: str,
    x_admin_key: Annotated[str | None, Header()] = None,
):
    verify_admin_key(x_admin_key)
    session = get_admin_session_detail(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    media_field = ALLOWED_MEDIA_FIELDS.get(media_kind)
    if media_field is None:
        raise HTTPException(status_code=404, detail="Unsupported media type")

    media_path = session.get(media_field)
    if not media_path:
        raise HTTPException(status_code=404, detail="Media not available")

    file_path = (BACKEND_DIR / media_path).resolve()
    uploads_root = (BACKEND_DIR / "data" / "uploads").resolve()

    # Restrict media access to the known uploads area to avoid exposing arbitrary files.
    if uploads_root not in file_path.parents:
        raise HTTPException(status_code=403, detail="Media path is not allowed")

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Media file not found")

    return FileResponse(file_path)
