import cv2
import mediapipe as mp
import numpy as np
import face_recognition


def compute_eye_ear(eye_points):
    p1 = np.array(eye_points[0], dtype=np.float32)
    p2 = np.array(eye_points[1], dtype=np.float32)
    p3 = np.array(eye_points[2], dtype=np.float32)
    p4 = np.array(eye_points[3], dtype=np.float32)
    p5 = np.array(eye_points[4], dtype=np.float32)
    p6 = np.array(eye_points[5], dtype=np.float32)

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.0

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def analyze_single_frame(face_mesh, frame_bgr):
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        return {"ok": False, "error": "No face detected"}

    h, w, _ = frame_bgr.shape
    landmarks = results.multi_face_landmarks[0].landmark

    def pt(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h], dtype=np.float32)

    left_eye = [pt(33), pt(160), pt(158), pt(133), pt(153), pt(144)]
    right_eye = [pt(362), pt(385), pt(387), pt(263), pt(373), pt(380)]

    left_ear = compute_eye_ear(left_eye)
    right_ear = compute_eye_ear(right_eye)
    ear = (left_ear + right_ear) / 2.0

    return {
        "ok": True,
        "ear": float(ear),
        "left_ear": float(left_ear),
        "right_ear": float(right_ear),
    }


def analyze_blink_sequence(frames_bgr, ear_threshold=0.20, required_blinks=3):
    mp_face_mesh = mp.solutions.face_mesh

    blink_count = 0
    eye_closed = False
    analyzed_frames = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:

        for frame in frames_bgr:
            result = analyze_single_frame(face_mesh, frame)

            if not result["ok"]:
                continue

            analyzed_frames += 1

            left_ear = result["left_ear"]
            right_ear = result["right_ear"]

            if right_ear < ear_threshold and left_ear < ear_threshold and not eye_closed:
                eye_closed = True
            elif right_ear >= ear_threshold and left_ear >= ear_threshold and eye_closed:
                blink_count += 1
                eye_closed = False

    return {
        "ok": True,
        "blink_count": blink_count,
        "passed": blink_count >= required_blinks,
        "required_blinks": required_blinks,
        "ear_threshold": ear_threshold,
        "analyzed_frames": analyzed_frames,
    }

def extract_frames_from_video(video_path, max_frames=200):
    cap = cv2.VideoCapture(video_path)

    frames = []
    count = 0

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        count += 1

    cap.release()
    return frames


def extract_faces_from_frames(frames, sample_every_n=10):
    """
    Extract faces from video frames at regular intervals.
    
    Args:
        frames: list of BGR frames from video
        sample_every_n: sample every Nth frame to reduce processing
    
    Returns:
        dict with extracted faces, frame indices, and validation results
    """
    extracted_faces = []
    sampled_frame_indices = []
    
    # Sample frames at regular intervals
    for i in range(0, len(frames), sample_every_n):
        frame = frames[i]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces in this frame
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if face_locations:
            # Extract face encoding and crop
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            if len(face_locations) == 1 and len(face_encodings) == 1:
                # Exactly one face in this frame - this is good
                top, right, bottom, left = face_locations[0]
                face_crop = frame[top:bottom, left:right]
                
                extracted_faces.append({
                    "frame_index": i,
                    "face_crop": face_crop,
                    "face_encoding": face_encodings[0],
                    "face_location": face_locations[0],
                    "num_faces": 1,
                })
                sampled_frame_indices.append(i)
            else:
                # Multiple or no faces in this sampled frame
                return {
                    "ok": False,
                    "error": "multiple_faces_in_frame" if len(face_locations) > 1 else "no_face",
                    "frame_index": i,
                    "num_faces": len(face_locations),
                }
    
    if not extracted_faces:
        return {
            "ok": False,
            "error": "no_faces_extracted",
        }
    
    return {
        "ok": True,
        "extracted_faces": extracted_faces,
        "sampled_frame_indices": sampled_frame_indices,
        "total_frames_sampled": len(sampled_frame_indices),
    }

def analyze_blink_video(video_path):
    frames = extract_frames_from_video(video_path)

    if len(frames) == 0:
        return {"ok": False, "error": "No frames extracted from video"}

    return analyze_blink_sequence(frames)


def analyze_liveness_with_identity_binding(video_path, selfie_img_bgr, face_match_threshold=0.50):
    """
    Complete liveness analysis including:
    1. Blink detection from video
    2. Face detection in video frames
    3. Identity verification (liveness face matches selfie)
    
    Args:
        video_path: path to liveness video file
        selfie_img_bgr: BGR image of stored selfie
        face_match_threshold: distance threshold for face matching (lower = stricter)
    
    Returns:
        dict with comprehensive liveness validation results
    """
    # Extract frames from video
    frames = extract_frames_from_video(video_path)
    if len(frames) == 0:
        return {
            "ok": False,
            "error": "No frames extracted from video",
            "blink_passed": None,
            "face_detected": False,
            "identity_match_passed": None,
        }
    
    # Step 1: Analyze blinks
    blink_result = analyze_blink_sequence(frames)
    blink_passed = blink_result.get("passed", False)
    
    # Step 2: Extract faces from video frames
    face_extraction = extract_faces_from_frames(frames, sample_every_n=10)
    
    if not face_extraction.get("ok"):
        error = face_extraction.get("error", "unknown")
        if error == "multiple_faces_in_frame":
            return {
                "ok": False,
                "error": "Multiple faces detected in liveness video. Please ensure only your face is visible.",
                "blink_passed": blink_passed,
                "face_detected": False,
                "identity_match_passed": False,
                "details": f"Multiple faces found at frame {face_extraction.get('frame_index')}",
            }
        else:
            return {
                "ok": False,
                "error": "No face detected in liveness video.",
                "blink_passed": blink_passed,
                "face_detected": False,
                "identity_match_passed": False,
            }
    
    extracted_faces = face_extraction.get("extracted_faces", [])
    
    # Step 3: Compare each extracted liveness face with selfie
    selfie_rgb = cv2.cvtColor(selfie_img_bgr, cv2.COLOR_BGR2RGB)
    selfie_encodings = face_recognition.face_encodings(selfie_rgb)
    
    if not selfie_encodings:
        return {
            "ok": False,
            "error": "Could not extract face encoding from selfie",
            "blink_passed": blink_passed,
            "face_detected": True,
            "identity_match_passed": False,
        }
    
    selfie_encoding = selfie_encodings[0]
    
    # Check if any liveness face matches the selfie
    best_distance = float('inf')
    best_match_index = -1
    
    for idx, face_data in enumerate(extracted_faces):
        liveness_encoding = face_data["face_encoding"]
        distance = float(np.linalg.norm(selfie_encoding - liveness_encoding))
        
        if distance < best_distance:
            best_distance = distance
            best_match_index = idx
    
    identity_match_passed = best_distance < face_match_threshold
    
    return {
        "ok": blink_passed and identity_match_passed,
        "blink_passed": blink_passed,
        "blink_details": {
            "blink_count": blink_result.get("blink_count"),
            "required_blinks": blink_result.get("required_blinks"),
            "analyzed_frames": blink_result.get("analyzed_frames"),
        },
        "face_detected": True,
        "num_faces_extracted": len(extracted_faces),
        "identity_match_passed": identity_match_passed,
        "identity_match_distance": best_distance,
        "identity_match_threshold": face_match_threshold,
        "error": None if (blink_passed and identity_match_passed) else (
            "The person in the liveness video does not match the selfie." if not identity_match_passed else "Liveness check failed. Please ensure you blink during recording."
        ),
        "passed": blink_passed and identity_match_passed,
    }