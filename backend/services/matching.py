import face_recognition
import numpy as np

def match_faces(selfie_img, id_img, accept_threshold=0.45, review_threshold=0.60):
    selfie_encodings = face_recognition.face_encodings(selfie_img)
    id_encodings = face_recognition.face_encodings(id_img)

    if len(selfie_encodings) == 0:
        return {"ok": False, "error": "No face encoding found in selfie image."}

    if len(id_encodings) == 0:
        return {"ok": False, "error": "No face encoding found in ID image."}

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
    }