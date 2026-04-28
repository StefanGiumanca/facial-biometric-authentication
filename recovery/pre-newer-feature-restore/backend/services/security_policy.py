"""
Security Policy Engine for eKYC sessions.

Handles:
- Security failure registration (increment counter, lock on 3+ strikes)
- Session lock checking
- Audit logging for security events
"""

from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session
from fastapi import HTTPException

from backend.db.database import SessionLocal
from backend.models.database_models import KycSession, AuditLog


# Constants
MAX_SECURITY_FAILURES = 3

FACE_MATCH_THRESHOLD = 0.60 

# Face match thresholds (distance metric): higher distance = more permissive
# SELFIE_GATE_ACCEPT_DISTANCE: used for the selfie "security gate". This is more permissive
# than the final face-match decision because selfies may differ from ID photos (age, lighting).
# FINAL_FACE_MATCH_ACCEPT_DISTANCE: strict acceptance threshold used at the final check.
# For both we also expose a review threshold used to trigger MANUAL_REVIEW when distance
# is between accept and review thresholds.
SELFIE_GATE_ACCEPT_DISTANCE = 0.55
SELFIE_GATE_REVIEW_DISTANCE = 0.70

FINAL_FACE_MATCH_ACCEPT_DISTANCE = 0.50
FINAL_FACE_MATCH_REVIEW_DISTANCE = 0.60


def register_security_failure(
    session_id: str,
    reason: str,
    details: Optional[dict] = None,
) -> dict:
    """
    Register a security failure for a session.
    
    - Increments security_fail_count
    - Locks session if count >= 3
    - Logs audit events
    - Uses database transaction to avoid race conditions
    
    Args:
        session_id: UUID of the KYC session
        reason: Short code for failure reason (e.g., "FACE_MISMATCH_SELFIE_ID")
        details: Optional dict with additional context (e.g., {"distance": 0.75, "threshold": 0.60})
    
    Returns:
        dict with:
        - ok: bool (always true, this function doesn't throw)
        - session_locked: bool (true if session just locked at 3 failures)
        - security_fail_count: int (current count)
        - remaining_attempts: int (3 - count, min 0)
        - reject_reason: str or None (if locked)
    """
    try:
        with SessionLocal() as db:
            # Load session with lock to prevent race conditions
            session = db.query(KycSession).filter(KycSession.id == session_id).with_for_update().first()
            
            if not session:
                return {
                    "ok": False,
                    "error": "Session not found",
                }
            
            # If already locked, short-circuit
            if session.status == "REJECTED":
                return {
                    "ok": True,
                    "session_locked": True,
                    "security_fail_count": session.security_fail_count,
                    "remaining_attempts": 0,
                    "reject_reason": session.reject_reason,
                }
            
            # Increment counter
            session.security_fail_count += 1
            session.updated_at = datetime.utcnow()
            
            # Build audit message with details
            audit_message = reason
            if details:
                detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
                audit_message = f"{reason}: {detail_str}"
            
            # Log security failure
            db.add(AuditLog(
                session_id=session_id,
                event_type="SECURITY_FAIL",
                message=audit_message,
            ))
            
            session_locked = False
            
            # Check if we've hit the limit
            if session.security_fail_count >= MAX_SECURITY_FAILURES:
                session.status = "REJECTED"
                session.reject_reason = "TOO_MANY_FAILED_SECURITY_CHECKS"
                session.locked_at = datetime.utcnow()
                session_locked = True
                
                # Log the lock event
                db.add(AuditLog(
                    session_id=session_id,
                    event_type="SESSION_LOCKED",
                    message=f"Session locked after {session.security_fail_count} security failures",
                ))
            
            db.commit()
            
            return {
                "ok": True,
                "session_locked": session_locked,
                "security_fail_count": session.security_fail_count,
                "remaining_attempts": max(0, MAX_SECURITY_FAILURES - session.security_fail_count),
                "reject_reason": session.reject_reason,
            }
    
    except Exception as e:
        print(f"[SECURITY] Error registering security failure: {e}")
        # Don't throw - infrastructure error, not a security failure
        return {
            "ok": False,
            "error": "Failed to process security check",
        }


def ensure_session_not_locked(session_id: str) -> None:
    """
    Check if session is locked. Raises HTTPException if locked.
    
    Args:
        session_id: UUID of the KYC session
    
    Raises:
        HTTPException(409): If session is locked/rejected
    
    Returns:
        None if session is not locked
    """
    try:
        with SessionLocal() as db:
            session = db.query(KycSession).filter(KycSession.id == session_id).first()
            
            if not session:
                raise HTTPException(
                    status_code=404,
                    detail="Session not found",
                )
            
            if session.status == "REJECTED":
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "SESSION_LOCKED",
                        "message": "This session has been rejected due to security concerns.",
                        "reason": session.reject_reason,
                        "security_fail_count": session.security_fail_count,
                    },
                )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[SECURITY] Error checking session lock: {e}")
        # Don't block on infrastructure error
        pass


def get_session_security_status(session_id: str) -> dict:
    """
    Get current security status of a session.
    
    Returns:
        dict with security-related fields
    """
    try:
        with SessionLocal() as db:
            session = db.query(KycSession).filter(KycSession.id == session_id).first()
            
            if not session:
                return {}
            
            return {
                "status": session.status,
                "security_fail_count": session.security_fail_count,
                "reject_reason": session.reject_reason,
                "locked_at": session.locked_at,
                "remaining_attempts": max(0, MAX_SECURITY_FAILURES - session.security_fail_count),
                "is_locked": session.status == "REJECTED",
            }
    
    except Exception as e:
        print(f"[SECURITY] Error getting security status: {e}")
        return {}
