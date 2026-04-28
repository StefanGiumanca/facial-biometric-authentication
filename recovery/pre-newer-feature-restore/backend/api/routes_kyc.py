from fastapi import APIRouter, UploadFile, File
from typing import Annotated
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
from backend.services.liveness import analyze_blink_sequence, analyze_blink_video, analyze_liveness_with_identity_binding
from backend.services.db_service import create_session_record, log_audit_event, update_session_record
from backend.services.security_policy import (
    register_security_failure,
    ensure_session_not_locked,
    get_session_security_status,
    FACE_MATCH_THRESHOLD,
)


router = APIRouter()
reader = build_reader(gpu=True)
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
        match_result = match_faces(rgb, id_rgb)
        
        if not match_result.get("ok") or match_result.get("decision") not in ["ACCEPTED", "ACCEPTED_MANUAL_REVIEW"]:
            # Security failure: selfie doesn't match ID
            failure_info = register_security_failure(
                session_id,
                "FACE_MISMATCH_SELFIE_ID",
                {
                    "distance": match_result.get("distance", 0),
                    "threshold": FACE_MATCH_THRESHOLD,
                    "decision": match_result.get("decision")
                }
            )
            
            log_audit_event(
                session_id,
                "SECURITY_FAIL",
                f"Selfie face does not match ID face (distance: {match_result.get('distance', 0):.3f}, threshold: {FACE_MATCH_THRESHOLD})"
            )
            
            return {
                "ok": False,
                "error": "Your face does not match the ID document. Please try again.",
                "security_fail_count": failure_info.get("security_fail_count", 0),
                "remaining_attempts": failure_info.get("remaining_attempts", 0),
                "session_locked": failure_info.get("session_locked", False),
            }
    
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
    )
    log_audit_event(session_id, "SELFIE_UPLOADED", "Selfie uploaded with exactly 1 face detected and verified against ID")

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

    result = match_faces(selfie_rgb, id_rgb)
    if not result.get("ok"):
        # Security failure: face match failed
        failure_info = register_security_failure(
            session_id,
            "FACE_MATCH_FAILED",
            {
                "distance": result.get("distance", 0),
                "threshold": FACE_MATCH_THRESHOLD,
                "decision": result.get("decision")
            }
        )
        
        update_session_record(session_id, status="FACE_MATCH_FAILED")
        log_audit_event(session_id, "VALIDATION_FAILED", result.get("error", "Face match failed"))
        
        return {
            "session_id": session_id,
            "ok": False,
            "error": result.get("error", "Face match failed"),
            "security_fail_count": failure_info.get("security_fail_count", 0),
            "remaining_attempts": failure_info.get("remaining_attempts", 0),
            "session_locked": failure_info.get("session_locked", False),
        }

    update_session_record(
        session_id,
        status="FACE_MATCH_COMPLETED",
        face_match_distance=result.get("distance"),
        face_match_decision=result.get("decision"),
        final_decision=result.get("decision"),
    )
    log_audit_event(session_id, "FACE_MATCH_COMPLETED", "Face match completed")
    log_audit_event(session_id, "FINAL_RESULT_GENERATED", "Final KYC result generated")

    return {
        "session_id": session_id,
        **result
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

    # Analyze liveness with identity binding
    result = analyze_liveness_with_identity_binding(
        str(video_path),
        selfie_img,
        face_match_threshold=0.50
    )

    if not result.get("ok"):
        # Liveness failed - log appropriate event and register security failure
        if not result.get("blink_passed"):
            log_audit_event(
                session_id,
                "LIVENESS_VALIDATION_FAILED",
                f"Blink detection failed: only {result.get('blink_details', {}).get('blink_count', 0)} blinks detected"
            )
            failure_reason = "LIVENESS_BLINK_FAILED"
        elif not result.get("identity_match_passed"):
            log_audit_event(
                session_id,
                "LIVENESS_IDENTITY_MATCH_FAILED",
                f"Liveness face does not match selfie (distance: {result.get('identity_match_distance', 0):.3f})"
            )
            failure_reason = "LIVENESS_IDENTITY_MISMATCH"
        else:
            log_audit_event(session_id, "LIVENESS_VALIDATION_FAILED", result.get("error", "Liveness validation failed"))
            failure_reason = "LIVENESS_VALIDATION_FAILED"
        
        # Register security failure
        failure_info = register_security_failure(
            session_id,
            failure_reason,
            {
                "blink_passed": result.get("blink_passed"),
                "identity_match_passed": result.get("identity_match_passed"),
                "blink_count": result.get("blink_details", {}).get("blink_count", 0),
                "identity_match_distance": result.get("identity_match_distance", 0),
            }
        )
        
        sessions[session_id]["liveness"] = False
        update_session_record(
            session_id,
            status="LIVENESS_FAILED",
            liveness_passed=False,
        )
        
        return {
            "ok": False,
            "error": result.get("error", "Liveness check failed"),
            "blink_passed": result.get("blink_passed"),
            "identity_match_passed": result.get("identity_match_passed"),
            "security_fail_count": failure_info.get("security_fail_count", 0),
            "remaining_attempts": failure_info.get("remaining_attempts", 0),
            "session_locked": failure_info.get("session_locked", False),
        }

    # Liveness passed all checks
    sessions[session_id]["liveness"] = True
    update_session_record(
        session_id,
        status="LIVENESS_COMPLETED",
        liveness_passed=True,
    )
    
    # Log both successful checks
    log_audit_event(
        session_id,
        "LIVENESS_COMPLETED",
        f"Blink check passed ({result.get('blink_details', {}).get('blink_count', 0)} blinks)"
    )
    log_audit_event(
        session_id,
        "LIVENESS_IDENTITY_MATCH_PASSED",
        f"Liveness face matches selfie (distance: {result.get('identity_match_distance', 0):.3f})"
    )

    return {
        "ok": True,
        "passed": True,
        "blink_passed": result.get("blink_passed"),
        "blink_count": result.get("blink_details", {}).get("blink_count"),
        "identity_match_passed": result.get("identity_match_passed"),
        "identity_match_distance": result.get("identity_match_distance"),
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
        "ready_for_face_match": (
            session["id_face_path"] is not None
            and session["selfie_path"] is not None
            and session["liveness"] is True
        ),
        "session_data": session
    }
