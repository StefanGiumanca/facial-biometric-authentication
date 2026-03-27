from fastapi import APIRouter, UploadFile, File
from typing import Annotated
import cv2
import uuid
from pathlib import Path
import numpy as np
import face_recognition
from backend.core.vision import load_haar_face_detector, extract_id_face
from backend.services.document_parser import parse_romanian_id
from backend.services.ocr_engine import build_reader, ocr_full_text, ocr_series_text_dynamic
from backend.services.matching import match_faces
from backend.services.liveness import analyze_blink_sequence, analyze_blink_video


router = APIRouter()
reader = build_reader(gpu=False)
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

@router.post("/kyc/session/start")                  # endpoint for logging the current session
def start_session():
    global current_session_id

    session_id = str(uuid.uuid4())

    sessions[session_id] = {
        "document_path": None,
        "id_face_path": None,
        "selfie_path": None,
        "liveness": None
    }

    current_session_id = session_id
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
        
    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return {"ok" : False, "error" : "Could not decode image"}
    
    document_filename = f"{uuid.uuid4()}.jpg"
    document_abs_path = ID_CARDS_DIR / document_filename
    document_rel_path = f"data/uploads/id_cards/{document_filename}"
    cv2.imwrite(str(document_abs_path), img_bgr)

    # OCR + parsing
    full_text = ocr_full_text(reader, img_bgr)
    series_text = ocr_series_text_dynamic(reader, img_bgr, outputs_dir=OUTPUTS_DIR)
    parsed = parse_romanian_id(full_text, series_text)

    # Extract and save cropped face from ID image
    id_face_crop = extract_id_face(detector, img_bgr)
    id_face_rel_path = None
    if id_face_crop is not None and id_face_crop.size > 0:
        id_face_filename = f"{uuid.uuid4()}.jpg"
        id_face_abs_path = ID_FACES_DIR / id_face_filename
        id_face_rel_path = f"data/uploads/id_faces/{id_face_filename}"
        cv2.imwrite(str(id_face_abs_path), id_face_crop)

    if id_face_rel_path is None:
        return {
            "ok": False,
            "error": "No face detected on ID"
        }

    sessions[session_id]["document_path"] = document_rel_path
    sessions[session_id]["id_face_path"] = id_face_rel_path

    return {
        "ok": True,
        "filename": file.filename,
        **parsed,
        "document_path": document_rel_path,
        "id_face_path": id_face_rel_path,
        "series_roi_text": series_text,
    }

@router.post("/kyc/selfie")         # endpoint for saving the selfie path 
async def capture_selfie(file: UploadFile = File(...)):

    if current_session_id is None:
        return {"ok": False, "error": "No active session"}

    session_id = current_session_id
    
    print(f"[SELFIE] session={session_id}")

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

    if len(faces) == 0:
        return {
            "ok" : False,
            "error" : "No face detected"
        }
    
    filename = f"{uuid.uuid4()}.jpg"
    selfie_abs_path = SELFIES_DIR / filename
    selfie_rel_path = f"data/uploads/selfies/{filename}"

    cv2.imwrite(str(selfie_abs_path), img_br)

    sessions[session_id]["selfie_path"] = selfie_rel_path

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
    
    print(f"[FACE MATCH] session={session_id}")

    session = sessions[session_id]

    if session.get("liveness") is not True:
        return {
            "ok" : False,
            "error" : "Liveness check failed"
        }

    selfie_path = session["selfie_path"]
    id_face_path = session["id_face_path"]

    if not selfie_path or not id_face_path:
        return {
            "ok": False,
            "error": "Missing data for face match"
        }

    selfie_abs = BACKEND_DIR / selfie_path
    id_face_abs = BACKEND_DIR / id_face_path

    selfie_img = cv2.imread(str(selfie_abs))
    id_face_img = cv2.imread(str(id_face_abs))

    if selfie_img is None or id_face_img is None:
        return {
            "ok": False,
            "error": "Could not load images"
        }

    selfie_rgb = cv2.cvtColor(selfie_img, cv2.COLOR_BGR2RGB)
    id_rgb = cv2.cvtColor(id_face_img, cv2.COLOR_BGR2RGB)

    result = match_faces(selfie_rgb, id_rgb)
    return {
        "session_id": session_id,
        **result
    }

@router.post("/kyc/liveness")       # endpoint for liveness detection
async def liveness(file: UploadFile = File(...)):

    if current_session_id is None:
        return {"ok": False, "error": "No active session"}

    session_id = current_session_id
    print(f"[LIVENESS VIDEO] session={session_id}")

    contents = await file.read()

    video_filename = f"{uuid.uuid4()}.mp4"
    video_path = OUTPUTS_DIR / video_filename

    with open(video_path, "wb") as f:
        f.write(contents)

    result = analyze_blink_video(str(video_path))

    if not result["ok"]:
        return result

    sessions[session_id]["liveness"] = result["passed"]

    return {
        "session_id": session_id,
        **result
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
        "selfie_uploaded": session["selfie_path"] is not None,
        "liveness_passed": session["liveness"] is True,
        "ready_for_face_match": (
            session["id_face_path"] is not None
            and session["selfie_path"] is not None
            and session["liveness"] is True
        ),
        "session_data": session
    }