import face_recognition
import numpy as np


def match_faces(selfie_img, id_img, accept_threshold=0.50, review_threshold=0.60, step="final"):
    """
    Compare two face images and return match decision.
    
    Args:
        selfie_img: Selfie image (BGR from OpenCV)
        id_img: ID face image (BGR from OpenCV)
        accept_threshold: Distance threshold for ACCEPTED decision (default: 0.50 for final check)
        review_threshold: Distance threshold for MANUAL_REVIEW decision
        step: "selfie_gate" or "final" - for logging/audit purposes
    
    Returns:
        dict with:
        - ok: bool (True if encoding succeeded)
        - distance: float (Euclidean distance between embeddings)
        - decision: "ACCEPTED" | "MANUAL_REVIEW" | "REJECTED"
        - accept_threshold: threshold used
        - review_threshold: threshold used
        - step: step identifier for audit trail
        - error: str (if ok=False)
    """
    selfie_encodings = face_recognition.face_encodings(selfie_img)
    id_encodings = face_recognition.face_encodings(id_img)

    if len(selfie_encodings) == 0:
        return {
            "ok": False,
            "error": "No face encoding found in selfie image.",
            "step": step,
        }

    if len(id_encodings) == 0:
        return {
            "ok": False,
            "error": "No face encoding found in ID image.",
            "step": step,
        }

    selfie_emb = selfie_encodings[0]
    id_emb = id_encodings[0]

    distance = float(np.linalg.norm(selfie_emb - id_emb))

    if distance < accept_threshold:
        decision = "ACCEPTED"
    elif distance < review_threshold:
        decision = "MANUAL_REVIEW"
    else:
        decision = "REJECTED"

    return {
        "ok": True,
        "distance": distance,
        "decision": decision,
        "accept_threshold": accept_threshold,
        "review_threshold": review_threshold,
        "step": step,
    }