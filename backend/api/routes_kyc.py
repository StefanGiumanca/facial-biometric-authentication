import os

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import cv2
import uuid
from pathlib import Path
import numpy as np
import face_recognition
from backend.core.vision import load_haar_face_detector, extract_id_face
from backend.services.document_parser import parse_romanian_id, validate_reviewed_document_fields
from backend.services.ocr_engine import build_reader, ocr_full_text, ocr_series_text_dynamic
from backend.services.matching import match_faces
from backend.services.liveness import analyze_liveness_challenge_video, create_random_liveness_challenge
from backend.services.db_service import create_session_record, log_audit_event, update_session_record
from backend.services.security_policy import (
    register_security_failure,
    ensure_session_not_locked,
    get_session_security_status,
    handle_final_face_match_failure,
    SELFIE_GATE_ACCEPT_DISTANCE,
    SELFIE_GATE_REVIEW_DISTANCE,
    FINAL_FACE_MATCH_ACCEPT_DISTANCE,
    FINAL_FACE_MATCH_REVIEW_DISTANCE,
    FACE_MATCH_THRESHOLD,
)


router = APIRouter()
reader = build_reader(gpu=os.getenv("OCR_USE_GPU", "1") == "1")
detector = load_haar_face_detector()

BACKEND_DIR = Path(__file__).resolve().parents[1]
UPLOADS_DIR = BACKEND_DIR / "data" / "uploads"
ID_CARDS_DIR = UPLOADS_DIR / "id_cards"
ID_FACES_DIR = UPLOADS_DIR / "id_faces"
SELFIES_DIR = UPLOADS_DIR / "selfies"
OUTPUTS_DIR = BACKEND_DIR / "data" / "outputs"

for folder in (ID_CARDS_DIR, ID_FACES_DIR, SELFIES_DIR, OUTPUTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

sessions = {}
current_session_id = None

class DocumentReviewPayload(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    cnp: str | None = None
    series_number: str | None = None
    series: str | None = None
    number: str | None = None
    sex: str | None = None
    nationality: str | None = None
    address: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None

@router.post("/kyc/session/start")                  # endpoint for logging the current session
def start_session():
    global current_session_id

    session_id = str(uuid.uuid4())

    sessions[session_id] = {
        "document_path": None,
        "id_face_path": None,
        "selfie_path": None,
        "liveness_video_path": None,
        "liveness": None,
        "liveness_challenge": None,
        "liveness_challenge_passed": False,
        "liveness_challenge_details": None,
        "document_fields": None,
        "reviewed_document_fields": None,
        "document_review_passed": False,
    }

    current_session_id = session_id
    create_session_record(session_id)
    log_audit_event(session_id, "SESSION_STARTED", "KYC session started")

    return {
        "ok": True,
        "session_id": session_id
    }

@router.post("/kyc/document")          # endpoint for extracting document OCR + cropping ID face
async def extract_document(file: UploadFile = File(...)):

    if current_session_id is None:
        return {"ok": False, "error": "No active session"}

    session_id = current_session_id

    print(f"[DOCUMENT] session={session_id}")

    # Security: Check if session is locked
    ensure_session_not_locked(session_id)

    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return {"ok" : False, "error" : "Could not decode image"}

    document_filename = f"{uuid.uuid4()}.jpg"
    document_abs_path = ID_CARDS_DIR / document_filename
    document_rel_path = f"data/uploads/id_cards/{document_filename}"
    cv2.imwrite(str(document_abs_path), img_bgr)
    update_session_record(
        session_id,
        status="DOCUMENT_UPLOADED",
        document_path=document_rel_path,
    )
    log_audit_event(session_id, "DOCUMENT_UPLOADED", "ID document uploaded")

    # OCR + parsing
    full_text = ocr_full_text(reader, img_bgr)
    series_text = ocr_series_text_dynamic(reader, img_bgr, outputs_dir=OUTPUTS_DIR)
    parsed = parse_romanian_id(full_text, series_text)
    update_session_record(
        session_id,
        status="OCR_COMPLETED",
        first_name=parsed.get("first_name"),
        last_name=parsed.get("last_name"),
        cnp=parsed.get("cnp"),
        series_number=parsed.get("series_number"),
        sex=parsed.get("sex"),
        nationality=parsed.get("nationality"),
        address=parsed.get("address"),
        valid_from=parsed.get("valid_from"),
        valid_until=parsed.get("valid_until"),
        raw_ocr_text=full_text,
    )
    log_audit_event(session_id, "OCR_COMPLETED", "OCR extraction completed")

    # Extract and save cropped face from ID image
    id_face_crop = extract_id_face(detector, img_bgr)
    id_face_rel_path = None
    if id_face_crop is not None and id_face_crop.size > 0:
        id_face_filename = f"{uuid.uuid4()}.jpg"
        id_face_abs_path = ID_FACES_DIR / id_face_filename
        id_face_rel_path = f"data/uploads/id_faces/{id_face_filename}"
        cv2.imwrite(str(id_face_abs_path), id_face_crop)

    if id_face_rel_path is None:
        log_audit_event(session_id, "VALIDATION_FAILED", "No face detected on ID document")
        return {
            "ok": False,
            "error": "No face detected on ID"
        }

    sessions[session_id]["document_path"] = document_rel_path
    sessions[session_id]["id_face_path"] = id_face_rel_path
    sessions[session_id]["document_fields"] = parsed
    sessions[session_id]["reviewed_document_fields"] = None
    sessions[session_id]["document_review_passed"] = False
    update_session_record(
        session_id,
        status="ID_FACE_EXTRACTED",
        id_face_path=id_face_rel_path,
    )
    log_audit_event(session_id, "ID_FACE_EXTRACTED", "Face extracted from ID document")

    return {
        "ok": True,
        "filename": file.filename,
        **parsed,
        "document_path": document_rel_path,
        "id_face_path": id_face_rel_path,
        "series_roi_text": series_text,
    }

@router.post("/kyc/review/validate")
def validate_document_review(payload: DocumentReviewPayload):

    if current_session_id is None:
        return {"ok": False, "error": "No active session"}

    session_id = current_session_id

    # Security: Check if session is locked
    ensure_session_not_locked(session_id)

    session = sessions[session_id]
    ocr_fields = session.get("document_fields")

    if not ocr_fields:
        return {
            "ok": False,
            "error": "No OCR document data found for the current session"
        }

    reviewed_fields = payload.model_dump()
    result = validate_reviewed_document_fields(ocr_fields, reviewed_fields)

    session["reviewed_document_fields"] = reviewed_fields
    session["document_review_passed"] = result["ok"]
    update_session_record(
        session_id,
        status="DOCUMENT_REVIEW_PASSED" if result["ok"] else "DOCUMENT_REVIEW_FAILED",
        first_name=reviewed_fields.get("first_name"),
        last_name=reviewed_fields.get("last_name"),
        cnp=reviewed_fields.get("cnp"),
        series_number=reviewed_fields.get("series_number"),
        sex=reviewed_fields.get("sex"),
        nationality=reviewed_fields.get("nationality"),
        address=reviewed_fields.get("address"),
        valid_from=reviewed_fields.get("valid_from"),
        valid_until=reviewed_fields.get("valid_until"),
    )
    if not result["ok"]:
        log_audit_event(session_id, "VALIDATION_FAILED", "OCR review validation failed")

    return {
        "session_id": session_id,
        **result
    }

@router.post("/kyc/selfie")         # endpoint for saving the selfie path
async def capture_selfie(file: UploadFile = File(...)):

    if current_session_id is None:
        return {"ok": False, "error": "No active session"}

    session_id = current_session_id
    session = sessions[session_id]

    print(f"[SELFIE] session={session_id}")

    # Security: Check if session is locked
    ensure_session_not_locked(session_id)

    # Security: Check session order - selfie can only be uploaded after ID face is extracted
    if not session.get("id_face_path"):
        log_audit_event(session_id, "VALIDATION_FAILED", "Selfie uploaded before ID face extraction")
        return {
            "ok": False,
            "error": "Please upload and review the ID document first"
        }

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img_br = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_br is None:
        return {
            "ok" : False,
            "error" : "Could not decode image"
        }

    rgb = cv2.cvtColor(img_br, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)

    # Security: Exactly one face must be detected
    if len(faces) == 0:
        log_audit_event(session_id, "SELFIE_VALIDATION_FAILED", "No face detected in selfie")
        return {
            "ok": False,
            "error": "No face detected. Please take a clear selfie of your face."
        }

    if len(faces) > 1:
        log_audit_event(session_id, "SELFIE_VALIDATION_FAILED", f"Multiple faces detected in selfie: {len(faces)}")
        return {
            "ok": False,
            "error": "Multiple faces detected. Please take the selfie alone."
        }

    # Security Gate: Compare selfie face to ID document face
    try:
        id_face_path = session.get("id_face_path")
        if not id_face_path:
            log_audit_event(session_id, "VALIDATION_FAILED", "No ID face available for comparison")
            return {
                "ok": False,
                "error": "ID face data is missing. Please re-upload the document."
            }

        id_face_abs = BACKEND_DIR / id_face_path
        id_face_img = cv2.imread(str(id_face_abs))

        if id_face_img is None:
            log_audit_event(session_id, "VALIDATION_FAILED", "Could not load ID face for comparison")
            return {
                "ok": False,
                "error": "Could not verify against ID. Please try again."
            }

        # Compare selfie to ID face
        id_rgb = cv2.cvtColor(id_face_img, cv2.COLOR_BGR2RGB)
        match_result = match_faces(
            rgb,
            id_rgb,
            accept_threshold=SELFIE_GATE_ACCEPT_DISTANCE,
            review_threshold=SELFIE_GATE_REVIEW_DISTANCE,
            step="selfie_gate"
        )

        if not match_result.get("ok") or match_result.get("decision") != "ACCEPTED":
            # Security failure: selfie doesn't match ID at gate threshold
            failure_info = register_security_failure(
                session_id,
                "FACE_MISMATCH_SELFIE_ID_GATE",
                {
                    "distance": match_result.get("distance", 0),
                    "accept_threshold": SELFIE_GATE_ACCEPT_DISTANCE,
                    "review_threshold": SELFIE_GATE_REVIEW_DISTANCE,
                    "decision": match_result.get("decision"),
                    "step": "selfie_gate"
                }
            )

            log_audit_event(
                session_id,
                "SECURITY_FAIL",
                f"Selfie face does not match ID face at security gate (distance: {match_result.get('distance', 0):.3f}, accept_threshold: {SELFIE_GATE_ACCEPT_DISTANCE}, decision: {match_result.get('decision')})"
            )

            if failure_info.get("session_locked"):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "SESSION_LOCKED",
                        "reason": "TOO_MANY_FAILED_SECURITY_CHECKS",
                        "message": "This session has been rejected after too many failed security checks.",
                        "security_fail_count": failure_info.get("security_fail_count", 0),
                    },
                )

            return {
                "ok": False,
                "error": "Your face does not match the ID document. Please try again.",
                "security_fail_count": failure_info.get("security_fail_count", 0),
                "remaining_attempts": failure_info.get("remaining_attempts", 0),
                "session_locked": failure_info.get("session_locked", False),
            }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[SELFIE] Error during face comparison: {e}")
        log_audit_event(session_id, "VALIDATION_FAILED", f"Face comparison error: {str(e)}")
        # Don't treat infrastructure errors as security failures
        return {
            "ok": False,
            "error": "Technical issue during verification. Please try again."
        }

    filename = f"{uuid.uuid4()}.jpg"
    selfie_abs_path = SELFIES_DIR / filename
    selfie_rel_path = f"data/uploads/selfies/{filename}"

    cv2.imwrite(str(selfie_abs_path), img_br)

    sessions[session_id]["selfie_path"] = selfie_rel_path
    update_session_record(
        session_id,
        status="SELFIE_UPLOADED",
        selfie_path=selfie_rel_path,
        selfie_gate_distance=match_result.get("distance"),
        selfie_gate_decision="ACCEPTED",
    )
    log_audit_event(session_id, "SELFIE_UPLOADED", "Selfie uploaded with exactly 1 face detected and verified against ID at security gate")

    return {
        "ok": True,
        "selfie_path": selfie_rel_path,
        "faces_detected": len(faces)
    }

@router.post("/kyc/face-match")         # endpoint for face matching
async def face_match():

    if current_session_id is None:
        return {"ok": False, "error": "No active session"}

    session_id = current_session_id
    session = sessions[session_id]

    print(f"[FACE MATCH] session={session_id}")

    # Security: Check if session is locked
    ensure_session_not_locked(session_id)

    # Security: Check session order - face-match requires liveness to have passed
    if session.get("liveness") is not True:
        log_audit_event(session_id, "VALIDATION_FAILED", "Face match requested before successful liveness")
        return {
            "ok": False,
            "error": "Please complete the liveness check first"
        }

    if session.get("liveness_challenge_passed") is not True:
        log_audit_event(session_id, "VALIDATION_FAILED", "Face match requested before liveness challenge passed")
        return {
            "ok": False,
            "error": "Please complete the liveness challenge first"
        }

    selfie_path = session["selfie_path"]
    id_face_path = session["id_face_path"]

    if not selfie_path or not id_face_path:
        log_audit_event(session_id, "VALIDATION_FAILED", "Face match requested with missing data")
        return {
            "ok": False,
            "error": "Missing data for face match"
        }

    selfie_abs = BACKEND_DIR / selfie_path
    id_face_abs = BACKEND_DIR / id_face_path

    selfie_img = cv2.imread(str(selfie_abs))
    id_face_img = cv2.imread(str(id_face_abs))

    if selfie_img is None or id_face_img is None:
        log_audit_event(session_id, "VALIDATION_FAILED", "Face match images could not be loaded")
        return {
            "ok": False,
            "error": "Could not load images"
        }

    selfie_rgb = cv2.cvtColor(selfie_img, cv2.COLOR_BGR2RGB)
    id_rgb = cv2.cvtColor(id_face_img, cv2.COLOR_BGR2RGB)

    # Final face-match check using strict thresholds
    result = match_faces(
        selfie_rgb,
        id_rgb,
        accept_threshold=FINAL_FACE_MATCH_ACCEPT_DISTANCE,
        review_threshold=FINAL_FACE_MATCH_REVIEW_DISTANCE,
        step="final"
    )

    if not result.get("ok"):
        # Technical error during encoding - treat as processing error, not security failure
        log_audit_event(
            session_id,
            "VALIDATION_FAILED",
            f"Face encoding error: {result.get('error', 'Face match encoding failed')}"
        )

        return {
            "session_id": session_id,
            "ok": False,
            "error": result.get("error", "Face match failed"),
        }

    # Check face-match decision
    if result.get("decision") == "ACCEPTED":
        # Success path - face match accepted
        update_session_record(
            session_id,
            status="FACE_MATCH_COMPLETED",
            final_face_match_distance=result.get("distance"),
            final_face_match_decision="ACCEPTED",
            face_match_distance=result.get("distance"),
            face_match_decision="ACCEPTED",
            final_decision="APPROVED",
        )
        log_audit_event(
            session_id,
            "FACE_MATCH_COMPLETED",
            f"Final face-match ACCEPTED (distance: {result.get('distance', 0):.3f}, threshold: {FINAL_FACE_MATCH_ACCEPT_DISTANCE})"
        )
        log_audit_event(session_id, "FINAL_RESULT_GENERATED", "Final KYC result generated: APPROVED")

        return {
            "session_id": session_id,
            "ok": True,
            "passed": True,
            **result
        }

    else:
        # Final face-match failed or needs manual review
        # Route to MANUAL_REVIEW instead of auto-rejecting (unless session already locked)
        review_result = handle_final_face_match_failure(
            session_id,
            result.get("distance"),
            {
                "accept_threshold": FINAL_FACE_MATCH_ACCEPT_DISTANCE,
                "review_threshold": FINAL_FACE_MATCH_REVIEW_DISTANCE,
                "decision": result.get("decision"),
            }
        )

        if not review_result.get("ok"):
            log_audit_event(session_id, "ERROR", f"Error updating session for manual review: {review_result.get('error')}")
            return {
                "session_id": session_id,
                "ok": False,
                "error": "Error processing face-match result",
            }

        # Check if session is locked (3 strikes already) or just needs review
        session_status = review_result.get("session_status")

        if session_status == "REJECTED":
            # Session already locked from earlier failures - return as rejected
            log_audit_event(
                session_id,
                "FACE_MATCH_FAILED",
                f"Final face-match failed but session already locked (distance: {result.get('distance', 0):.3f})"
            )

            return {
                "session_id": session_id,
                "ok": False,
                "error": "This session has been rejected due to security concerns.",
                "security_fail_count": review_result.get("security_fail_count", 0),
                "session_locked": True,
                "reject_reason": review_result.get("reject_reason"),
            }

        else:
            # Session status = MANUAL_REVIEW (not a strike)
            log_audit_event(
                session_id,
                "FINAL_RESULT_GENERATED",
                f"Final face-match requires manual review (distance: {result.get('distance', 0):.3f}, threshold: {FINAL_FACE_MATCH_ACCEPT_DISTANCE})"
            )

            return {
                "session_id": session_id,
                "ok": True,
                "passed": False,
                "session_status": "MANUAL_REVIEW",
                "reason": f"Face-match distance ({result.get('distance', 0):.3f}) does not meet strict threshold. Please contact support for manual review.",
                **result
            }

@router.post("/kyc/liveness/challenge")
def create_liveness_challenge():
    if current_session_id is None:
        return {"ok": False, "error": "No active session"}

    session_id = current_session_id
    session = sessions[session_id]

    ensure_session_not_locked(session_id)

    if not session.get("selfie_path"):
        log_audit_event(session_id, "VALIDATION_FAILED", "Liveness challenge requested before selfie uploaded")
        return {
            "ok": False,
            "error": "Please take a selfie before the liveness challenge"
        }

    challenge = create_random_liveness_challenge()
    session["liveness_challenge"] = challenge
    session["liveness_challenge_passed"] = False
    session["liveness_challenge_details"] = None
    session["liveness"] = None

    update_session_record(
        session_id,
        status="LIVENESS_CHALLENGE_CREATED",
        liveness_passed=None,
        liveness_challenge_id=challenge["challenge_id"],
        liveness_challenge_type=challenge["challenge_type"],
        liveness_challenge_passed=False,
        liveness_challenge_details=challenge,
    )
    log_audit_event(
        session_id,
        "LIVENESS_CHALLENGE_CREATED",
        f"Challenge created: {challenge['challenge_type']} ({challenge['instruction']})"
    )

    return {
        "ok": True,
        **challenge,
    }


@router.post("/kyc/liveness")       # endpoint for liveness detection
async def liveness(file: UploadFile = File(...)):

    if current_session_id is None:
        return {"ok": False, "error": "No active session"}

    session_id = current_session_id
    session = sessions[session_id]

    print(f"[LIVENESS VIDEO] session={session_id}")

    # Security: Check if session is locked
    ensure_session_not_locked(session_id)

    # Security: Check session order - liveness requires selfie first
    if not session.get("selfie_path"):
        log_audit_event(session_id, "VALIDATION_FAILED", "Liveness attempted before selfie uploaded")
        return {
            "ok": False,
            "error": "Please take a selfie before the liveness check"
        }

    challenge = session.get("liveness_challenge")
    if not challenge:
        log_audit_event(session_id, "VALIDATION_FAILED", "Liveness attempted without an active challenge")
        return {
            "ok": False,
            "error": "Please request a liveness challenge before recording the video"
        }

    contents = await file.read()

    video_filename = f"{uuid.uuid4()}.mp4"
    video_path = OUTPUTS_DIR / video_filename
    video_rel_path = f"data/outputs/{video_filename}"

    with open(video_path, "wb") as f:
        f.write(contents)
    sessions[session_id]["liveness_video_path"] = video_rel_path
    update_session_record(
        session_id,
        status="LIVENESS_VIDEO_UPLOADED",
        liveness_video_path=video_rel_path,
    )

    # Load stored selfie for identity binding
    selfie_abs = BACKEND_DIR / session["selfie_path"]
    selfie_img = cv2.imread(str(selfie_abs))

    if selfie_img is None:
        log_audit_event(session_id, "VALIDATION_FAILED", "Could not load stored selfie for liveness identity binding")
        return {
            "ok": False,
            "error": "Error loading selfie. Please retry."
        }

    # Analyze randomized challenge with identity binding
    result = analyze_liveness_challenge_video(
        str(video_path),
        selfie_img,
        challenge,
        face_match_threshold=0.50
    )

    if not result.get("ok"):
        if not result.get("identity_match_passed"):
            log_audit_event(
                session_id,
                "LIVENESS_IDENTITY_MATCH_FAILED",
                f"Liveness face does not match selfie (distance: {result.get('identity_match_distance', 0):.3f})"
            )
            failure_reason = "LIVENESS_IDENTITY_MISMATCH"
        else:
            log_audit_event(
                session_id,
                "LIVENESS_CHALLENGE_FAILED",
                f"{challenge.get('challenge_type')} failed: {result.get('error', 'Challenge not completed')}"
            )
            failure_reason = f"LIVENESS_CHALLENGE_{challenge.get('challenge_type', 'UNKNOWN')}_FAILED"

        # Register security failure
        failure_info = register_security_failure(
            session_id,
            failure_reason,
            {
                "challenge_id": challenge.get("challenge_id"),
                "challenge_type": challenge.get("challenge_type"),
                "challenge_passed": result.get("passed"),
                "identity_match_passed": result.get("identity_match_passed"),
                "identity_match_distance": result.get("identity_match_distance", 0),
                "details": result.get("details"),
            }
        )

        sessions[session_id]["liveness"] = False
        sessions[session_id]["liveness_challenge_passed"] = False
        sessions[session_id]["liveness_challenge_details"] = result.get("details")
        update_session_record(
            session_id,
            status="LIVENESS_FAILED",
            liveness_passed=False,
            liveness_challenge_passed=False,
            liveness_challenge_details=result.get("details"),
        )

        if failure_info.get("session_locked"):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "SESSION_LOCKED",
                    "reason": "TOO_MANY_FAILED_SECURITY_CHECKS",
                    "message": "This session has been rejected after too many failed security checks.",
                    "security_fail_count": failure_info.get("security_fail_count", 0),
                },
            )

        return {
            "ok": False,
            "error": result.get("error", "Liveness check failed"),
            "message": result.get("message"),
            "challenge_id": challenge.get("challenge_id"),
            "challenge_type": challenge.get("challenge_type"),
            "instruction": challenge.get("instruction"),
            "challenge_passed": result.get("passed"),
            "identity_match_passed": result.get("identity_match_passed"),
            "details": result.get("details"),
            "security_fail_count": failure_info.get("security_fail_count", 0),
            "remaining_attempts": failure_info.get("remaining_attempts", 0),
            "session_locked": failure_info.get("session_locked", False),
        }

    # Liveness passed all checks
    sessions[session_id]["liveness"] = True
    sessions[session_id]["liveness_challenge_passed"] = True
    sessions[session_id]["liveness_challenge_details"] = result.get("details")
    update_session_record(
        session_id,
        status="LIVENESS_COMPLETED",
        liveness_passed=True,
        liveness_challenge_passed=True,
        liveness_challenge_details=result.get("details"),
    )

    log_audit_event(
        session_id,
        "LIVENESS_CHALLENGE_PASSED",
        f"{challenge.get('challenge_type')} challenge passed"
    )
    log_audit_event(
        session_id,
        "LIVENESS_IDENTITY_MATCH_PASSED",
        f"Liveness face matches selfie (distance: {result.get('identity_match_distance', 0):.3f})"
    )

    return {
        "ok": True,
        "passed": True,
        "challenge_id": challenge.get("challenge_id"),
        "challenge_type": challenge.get("challenge_type"),
        "instruction": challenge.get("instruction"),
        "challenge_passed": result.get("passed"),
        "identity_match_passed": result.get("identity_match_passed"),
        "identity_match_distance": result.get("identity_match_distance"),
        "details": result.get("details"),
    }

@router.get("/kyc/session/status")
def session_status():
    if current_session_id is None:
        return {
            "ok": False,
            "error": "No active session"
        }

    session = sessions[current_session_id]

    return {
        "ok": True,
        "session_id": current_session_id,
        "document_uploaded": session["document_path"] is not None,
        "id_face_extracted": session["id_face_path"] is not None,
        "document_review_passed": session.get("document_review_passed") is True,
        "selfie_uploaded": session["selfie_path"] is not None,
        "liveness_passed": session["liveness"] is True,
        "liveness_challenge": session.get("liveness_challenge"),
        "liveness_challenge_passed": session.get("liveness_challenge_passed") is True,
        "ready_for_face_match": (
            session["id_face_path"] is not None
            and session["selfie_path"] is not None
            and session["liveness"] is True
            and session.get("liveness_challenge_passed") is True
        ),
        "session_data": session
    }
