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

# ============================================================================
# FACE MATCHING THRESHOLDS (Euclidean distance metric)
# ============================================================================
# Distance metric: lower = closer match, higher = more permissive
# 
# NOTE: face_recognition uses Euclidean distance between 128-D embeddings.
# Typical ranges: 0.0-0.6 (same person), 0.6-0.8 (possibly same), >0.8 (different)
# ============================================================================

# SELFIE SECURITY GATE (Selfie endpoint)
# More permissive because selfies may differ from ID photos (age, lighting, angle).
# Used ONLY at the selfie upload step to give user multiple attempts.
# If a user fails this gate, they're allowed to retry (up to 3 times before lock).
SELFIE_GATE_ACCEPT_DISTANCE = 0.60      # Accept: distance <= 0.60 (more permissive)
SELFIE_GATE_REVIEW_DISTANCE = 0.70      # Review: 0.60 < distance <= 0.70 (currently rejected at selfie gate)

# FINAL FACE-MATCH CHECK (Face-match endpoint)
# Stricter because this is the final verification step before approval.
# If this fails, session goes to MANUAL_REVIEW (not a security strike unless 3 strikes already).
FINAL_FACE_MATCH_ACCEPT_DISTANCE = 0.50     # Accept: distance <= 0.50 (strict)
FINAL_FACE_MATCH_REVIEW_DISTANCE = 0.60     # Review: 0.50 < distance <= 0.60

# Legacy alias for backward compatibility
FACE_MATCH_THRESHOLD = FINAL_FACE_MATCH_ACCEPT_DISTANCE


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


def handle_final_face_match_failure(
    session_id: str,
    distance: float,
    details: Optional[dict] = None,
) -> dict:
    """
    Handle final face-match failure (non-strike path).
    
    At the final check, if face-match fails, we route to MANUAL_REVIEW instead of
    auto-counting as a security strike. This allows operators to review borderline cases.
    
    Args:
        session_id: UUID of the KYC session
        distance: face match distance for audit trail
        details: additional context
    
    Returns:
        dict with:
        - ok: bool
        - session_status: "MANUAL_REVIEW" or "REJECTED" (if already locked)
        - security_fail_count: int
        - reject_reason: str or None
    """
    try:
        with SessionLocal() as db:
            session = db.query(KycSession).filter(KycSession.id == session_id).with_for_update().first()
            
            if not session:
                return {"ok": False, "error": "Session not found"}
            
            # If session is already locked (3 strikes), keep REJECTED status
            if session.status == "REJECTED":
                return {
                    "ok": True,
                    "session_status": "REJECTED",
                    "security_fail_count": session.security_fail_count,
                    "reject_reason": session.reject_reason,
                }
            
            # Otherwise, mark for manual review (not a security strike)
            session.status = "MANUAL_REVIEW"
            session.final_face_match_decision = "REJECTED"
            session.final_face_match_distance = distance
            session.updated_at = datetime.utcnow()
            
            # Log as informational, NOT a security failure
            audit_message = f"Final face-match failed: distance={distance:.3f}, threshold={FINAL_FACE_MATCH_ACCEPT_DISTANCE}"
            if details:
                detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
                audit_message += f" ({detail_str})"
            
            db.add(AuditLog(
                session_id=session_id,
                event_type="MANUAL_REVIEW_REQUIRED",
                message=audit_message,
            ))
            
            db.commit()
            
            return {
                "ok": True,
                "session_status": "MANUAL_REVIEW",
                "security_fail_count": session.security_fail_count,
                "reject_reason": None,  # Not a rejection, just needs review
            }
    
    except Exception as e:
        print(f"[SECURITY] Error handling final face-match failure: {e}")
        return {"ok": False, "error": "Failed to process face-match result"}
