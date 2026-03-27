import cv2
import mediapipe as mp
import numpy as np


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

def analyze_blink_video(video_path):
    frames = extract_frames_from_video(video_path)

    if len(frames) == 0:
        return {"ok": False, "error": "No frames extracted from video"}

    return analyze_blink_sequence(frames)