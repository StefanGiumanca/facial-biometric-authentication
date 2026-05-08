import os
import random
import uuid
from dataclasses import dataclass
from typing import Any

import cv2
import face_recognition
import numpy as np


BLINK_EAR_THRESHOLD = 0.18
BLINK_REQUIRED_COUNT = 3
BLINK_CLOSED_CONSECUTIVE_FRAMES = 2
BLINK_OPEN_CONSECUTIVE_FRAMES = 2
BASELINE_FACE_FRAMES = 6
MIN_FACE_CHALLENGE_HIT_FRAMES = 12
MIN_HAND_CHALLENGE_HIT_FRAMES = 10
MIN_ANALYZED_FACE_FRAMES = 18
MIN_VIDEO_FRAMES = 36
MIN_FACE_COVERAGE_RATIO = 0.50
MAX_VIDEO_FRAMES = 220
FRAME_SAMPLE_STEP = 2
ANALYSIS_FRAME_WIDTH = 480

SMILE_RATIO_THRESHOLD = 0.44
SMILE_RATIO_DELTA = 0.055
SMILE_MIN_MOUTH_OPEN_RATIO = 0.022
OPEN_MOUTH_RATIO_THRESHOLD = 0.085
OPEN_MOUTH_RATIO_DELTA = 0.04
HEAD_TURN_RATIO_THRESHOLD = 0.13
HEAD_TURN_RATIO_DELTA = 0.085
LOOK_UP_RATIO_THRESHOLD = 0.36
LOOK_DOWN_RATIO_THRESHOLD = 0.62
LOOK_VERTICAL_RATIO_DELTA = 0.07
NOD_VERTICAL_RANGE_DELTA = 0.13
SHAKE_HORIZONTAL_RANGE_DELTA = 0.17
MOVE_CLOSER_FACE_WIDTH_DELTA = 0.06
MOVE_BACK_FACE_WIDTH_DELTA = 0.045
HAND_RAISED_WRIST_Y_MARGIN = 0.12
FINGER_TIP_PROXIMITY_THRESHOLD = 0.16

RANDOM_CHALLENGE_TYPES = [
    "BLINK",
    "SMILE",
    "OPEN_MOUTH",
    "NOD_YES",
    "SHAKE_HEAD_NO",
    "MOVE_CLOSER",
    "MOVE_BACK",
    "RAISE_HAND",
    "SHOW_OPEN_PALM",
    "SHOW_TWO_FINGERS",
    "TOUCH_NOSE",
]

FACE_REQUIRED_CHALLENGES = {
    "BLINK",
    "SMILE",
    "TURN_HEAD_LEFT",
    "TURN_HEAD_RIGHT",
    "LOOK_UP",
    "LOOK_DOWN",
    "OPEN_MOUTH",
    "NOD_YES",
    "SHAKE_HEAD_NO",
    "MOVE_CLOSER",
    "MOVE_BACK",
    "RAISE_HAND",
    "SHOW_OPEN_PALM",
    "SHOW_TWO_FINGERS",
    "TOUCH_NOSE",
}
HAND_REQUIRED_CHALLENGES = {
    "RAISE_HAND",
    "SHOW_OPEN_PALM",
    "SHOW_TWO_FINGERS",
    "TOUCH_NOSE",
}


@dataclass(frozen=True)
class ChallengeDefinition:
    challenge_type: str
    instruction: str
    required_action: dict[str, Any]
    stable: bool = True


CHALLENGE_DEFINITIONS = {
    "BLINK": ChallengeDefinition(
        "BLINK",
        "Blink 3 times",
        {"required_blinks": BLINK_REQUIRED_COUNT},
    ),
    "SMILE": ChallengeDefinition(
        "SMILE",
        "Smile clearly",
        {"min_smile_frames": MIN_FACE_CHALLENGE_HIT_FRAMES},
    ),
    "TURN_HEAD_LEFT": ChallengeDefinition(
        "TURN_HEAD_LEFT",
        "Turn your head left",
        {"min_turn_frames": MIN_FACE_CHALLENGE_HIT_FRAMES},
    ),
    "TURN_HEAD_RIGHT": ChallengeDefinition(
        "TURN_HEAD_RIGHT",
        "Turn your head right",
        {"min_turn_frames": MIN_FACE_CHALLENGE_HIT_FRAMES},
    ),
    "LOOK_UP": ChallengeDefinition(
        "LOOK_UP",
        "Look up",
        {"min_pose_frames": MIN_FACE_CHALLENGE_HIT_FRAMES},
    ),
    "LOOK_DOWN": ChallengeDefinition(
        "LOOK_DOWN",
        "Look down",
        {"min_pose_frames": MIN_FACE_CHALLENGE_HIT_FRAMES},
    ),
    "OPEN_MOUTH": ChallengeDefinition(
        "OPEN_MOUTH",
        "Open your mouth",
        {"min_open_mouth_frames": MIN_FACE_CHALLENGE_HIT_FRAMES},
    ),
    "NOD_YES": ChallengeDefinition(
        "NOD_YES",
        "Nod yes",
        {"min_motion_frames": MIN_FACE_CHALLENGE_HIT_FRAMES},
    ),
    "SHAKE_HEAD_NO": ChallengeDefinition(
        "SHAKE_HEAD_NO",
        "Shake your head no",
        {"min_motion_frames": MIN_FACE_CHALLENGE_HIT_FRAMES},
    ),
    "MOVE_CLOSER": ChallengeDefinition(
        "MOVE_CLOSER",
        "Move your face closer to the camera",
        {"min_motion_frames": MIN_FACE_CHALLENGE_HIT_FRAMES},
    ),
    "MOVE_BACK": ChallengeDefinition(
        "MOVE_BACK",
        "Move your face farther from the camera",
        {"min_motion_frames": MIN_FACE_CHALLENGE_HIT_FRAMES},
    ),
    "RAISE_HAND": ChallengeDefinition(
        "RAISE_HAND",
        "Raise one hand into the frame",
        {"min_hand_frames": MIN_HAND_CHALLENGE_HIT_FRAMES},
    ),
    "SHOW_OPEN_PALM": ChallengeDefinition(
        "SHOW_OPEN_PALM",
        "Show an open palm",
        {"min_open_palm_frames": MIN_HAND_CHALLENGE_HIT_FRAMES},
    ),
    "SHOW_TWO_FINGERS": ChallengeDefinition(
        "SHOW_TWO_FINGERS",
        "Show two fingers",
        {"required_extended_fingers": 2, "min_two_finger_frames": MIN_HAND_CHALLENGE_HIT_FRAMES},
    ),
    "TOUCH_NOSE": ChallengeDefinition(
        "TOUCH_NOSE",
        "Touch your nose",
        {"min_touch_frames": MIN_HAND_CHALLENGE_HIT_FRAMES},
    ),
}


def create_random_liveness_challenge() -> dict[str, Any]:
    enabled_types = RANDOM_CHALLENGE_TYPES
    if os.getenv("VISIONAUTH_ENABLE_EXTRA_LIVENESS", "0") == "1":
        enabled_types = list(CHALLENGE_DEFINITIONS.keys())

    challenge_type = random.choice(enabled_types)
    definition = CHALLENGE_DEFINITIONS[challenge_type]

    return {
        "challenge_id": str(uuid.uuid4()),
        "challenge_type": definition.challenge_type,
        "instruction": definition.instruction,
        "required_action": definition.required_action,
        "stable": definition.stable,
    }


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


def extract_frames_from_video(video_path, max_frames=MAX_VIDEO_FRAMES):
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


def _resize_for_analysis(frame_bgr):
    height, width = frame_bgr.shape[:2]
    if width <= ANALYSIS_FRAME_WIDTH:
        return frame_bgr

    scale = ANALYSIS_FRAME_WIDTH / float(width)
    resized_height = max(int(height * scale), 1)
    return cv2.resize(frame_bgr, (ANALYSIS_FRAME_WIDTH, resized_height), interpolation=cv2.INTER_AREA)


def _face_recognition_frame_metrics(frame_bgr) -> tuple[str | None, dict[str, float] | None]:
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    if len(face_locations) > 1:
        return "multiple_faces", None
    if not face_locations:
        return "no_face", None

    landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)
    if not landmarks_list:
        return "no_face_landmarks", None

    top, right, bottom, left = face_locations[0]
    landmarks = landmarks_list[0]
    face_width = max(float(right - left), 1.0)
    face_height = max(float(bottom - top), 1.0)
    face_center_x = float((left + right) / 2.0)

    left_eye = landmarks.get("left_eye", [])
    right_eye = landmarks.get("right_eye", [])
    nose_tip = landmarks.get("nose_tip", [])
    top_lip = landmarks.get("top_lip", [])
    bottom_lip = landmarks.get("bottom_lip", [])

    if len(left_eye) < 6 or len(right_eye) < 6 or not nose_tip or not top_lip or not bottom_lip:
        return "no_face_landmarks", None

    left_ear = compute_eye_ear(left_eye[:6])
    right_ear = compute_eye_ear(right_eye[:6])
    mouth_points = [np.array(point, dtype=np.float32) for point in top_lip + bottom_lip]
    mouth_left = min(mouth_points, key=lambda point: point[0])
    mouth_right = max(mouth_points, key=lambda point: point[0])
    top_lip_y = float(np.mean([point[1] for point in top_lip]))
    bottom_lip_y = float(np.mean([point[1] for point in bottom_lip]))
    nose = np.mean([np.array(point, dtype=np.float32) for point in nose_tip], axis=0)

    return None, {
        "ear": float((left_ear + right_ear) / 2.0),
        "left_ear": float(left_ear),
        "right_ear": float(right_ear),
        "smile_ratio": float(np.linalg.norm(mouth_right - mouth_left)) / face_width,
        "mouth_open_ratio": max(bottom_lip_y - top_lip_y, 0.0) / face_width,
        "nose_x_ratio": (float(nose[0]) - face_center_x) / face_width,
        "nose_y_ratio": (float(nose[1]) - float(top)) / face_height,
        "nose_x": float(nose[0]),
        "nose_y": float(nose[1]),
        "face_width": face_width,
        "face_width_frame_ratio": face_width / max(float(frame_bgr.shape[1]), 1.0),
        "frame_width": float(frame_bgr.shape[1]),
        "frame_height": float(frame_bgr.shape[0]),
    }


def _hand_points(hand_landmarks, frame_shape) -> dict[int, np.ndarray]:
    height, width, _ = frame_shape
    return {
        index: np.array([landmark.x * width, landmark.y * height], dtype=np.float32)
        for index, landmark in enumerate(hand_landmarks.landmark)
    }


def _finger_count(hand_landmarks, frame_shape) -> int:
    points = _hand_points(hand_landmarks, frame_shape)
    extended = 0

    # Finger tips above their PIP joints are treated as extended. This is simple
    # and deterministic, so it works best when the palm faces the camera upright.
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if points[tip][1] < points[pip][1]:
            extended += 1

    thumb_tip = points[4]
    thumb_ip = points[3]
    wrist = points[0]
    if abs(float(thumb_tip[0] - wrist[0])) > abs(float(thumb_ip[0] - wrist[0])) + 8:
        extended += 1

    return extended


def _open_palm_detected(hand_landmarks, frame_shape) -> bool:
    return _finger_count(hand_landmarks, frame_shape) >= 4


def _two_fingers_detected(hand_landmarks, frame_shape) -> bool:
    points = _hand_points(hand_landmarks, frame_shape)

    index_up = points[8][1] < points[6][1]
    middle_up = points[12][1] < points[10][1]
    ring_down = points[16][1] >= points[14][1]
    pinky_down = points[20][1] >= points[18][1]

    return bool(index_up and middle_up and ring_down and pinky_down)


def _hand_raised_detected(hand_landmarks, frame_shape) -> bool:
    points = _hand_points(hand_landmarks, frame_shape)
    frame_height = frame_shape[0]
    wrist_y = float(points[0][1])
    fingertips_y = min(float(points[index][1]) for index in (4, 8, 12, 16, 20))
    return fingertips_y < frame_height * 0.55 and fingertips_y < wrist_y - (frame_height * HAND_RAISED_WRIST_Y_MARGIN)


def _touch_nose_detected(hand_landmarks, frame_shape, face_metrics: dict[str, float]) -> bool:
    points = _hand_points(hand_landmarks, frame_shape)
    nose = np.array([face_metrics["nose_x"], face_metrics["nose_y"]], dtype=np.float32)
    threshold = face_metrics["face_width"] * FINGER_TIP_PROXIMITY_THRESHOLD

    # The index fingertip is the most intentional point for a nose touch.
    # Thumb and middle fingertip are accepted as backup for slight camera angles.
    for fingertip_index in (8, 4, 12):
        if float(np.linalg.norm(points[fingertip_index] - nose)) <= threshold:
            return True

    return False


def _stable_hit_count(values: list[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _median_metric(metrics: list[dict[str, float]], key: str, fallback: float = 0.0) -> float:
    values = [metric[key] for metric in metrics if key in metric]
    return float(np.median(values)) if values else fallback


def _summarize_result(
    *,
    challenge: dict[str, Any],
    passed: bool,
    message: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": passed,
        "passed": passed,
        "challenge_id": challenge.get("challenge_id"),
        "challenge_type": challenge.get("challenge_type"),
        "instruction": challenge.get("instruction"),
        "error": None if passed else message,
        "message": "Challenge completed." if passed else message,
        "details": details,
    }


def _analyze_challenge_frames(frames: list[np.ndarray], challenge: dict[str, Any]) -> dict[str, Any]:
    challenge_type = challenge.get("challenge_type")
    needs_face = challenge_type in FACE_REQUIRED_CHALLENGES
    needs_hand = challenge_type in HAND_REQUIRED_CHALLENGES
    sampled_frame_count = len(frames[::FRAME_SAMPLE_STEP])

    face_frames = 0
    hand_frames = 0
    multiple_face_frame = None
    blink_count = 0
    eye_closed = False
    closed_streak = 0
    open_streak = 0
    baseline_face_metrics: list[dict[str, float]] = []
    face_width_frame_ratios: list[float] = []
    hits: list[bool] = []
    metric_samples: dict[str, list[float]] = {
        "ear": [],
        "smile_ratio": [],
        "mouth_open_ratio": [],
        "nose_x_ratio": [],
        "nose_y_ratio": [],
        "face_width_frame_ratio": [],
        "finger_count": [],
    }

    hands = None
    if needs_hand:
        try:
            if os.getenv("LIVENESS_USE_MEDIAPIPE_GPU", "0") != "1":
                os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
            import mediapipe as mp

            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.50,
            )
        except Exception as error:
            return _summarize_result(
                challenge=challenge,
                passed=False,
                message="Hand gesture detection is not available on this machine. Please try a face challenge.",
                details={
                    "mediapipe_error": str(error),
                    "mediapipe_error_type": type(error).__name__,
                },
            )

    try:
        for frame_index, frame in enumerate(frames[::FRAME_SAMPLE_STEP]):
            analysis_frame = _resize_for_analysis(frame)
            face_error, face_metrics = _face_recognition_frame_metrics(analysis_frame)

            if face_error == "multiple_faces":
                multiple_face_frame = frame_index * FRAME_SAMPLE_STEP
                break

            hand_result = None
            if hands is not None:
                rgb_frame = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
                hand_result = hands.process(rgb_frame)

            if face_metrics:
                face_frames += 1
                if len(baseline_face_metrics) < BASELINE_FACE_FRAMES:
                    baseline_face_metrics.append(face_metrics)
                metric_samples["smile_ratio"].append(face_metrics["smile_ratio"])
                metric_samples["mouth_open_ratio"].append(face_metrics["mouth_open_ratio"])
                metric_samples["nose_x_ratio"].append(face_metrics["nose_x_ratio"])
                metric_samples["nose_y_ratio"].append(face_metrics["nose_y_ratio"])
                metric_samples["face_width_frame_ratio"].append(face_metrics["face_width_frame_ratio"])
                metric_samples["ear"].append(face_metrics["ear"])
                face_width_frame_ratios.append(face_metrics["face_width_frame_ratio"])

            hand_landmarks = hand_result.multi_hand_landmarks if hand_result and hand_result.multi_hand_landmarks else []
            if hand_landmarks:
                hand_frames += 1

            hit = False
            baseline_ready = len(baseline_face_metrics) >= BASELINE_FACE_FRAMES
            baseline_smile = _median_metric(baseline_face_metrics, "smile_ratio")
            baseline_mouth_open = _median_metric(baseline_face_metrics, "mouth_open_ratio")
            baseline_nose_x = _median_metric(baseline_face_metrics, "nose_x_ratio")
            baseline_nose_y = _median_metric(baseline_face_metrics, "nose_y_ratio")
            baseline_face_width = _median_metric(baseline_face_metrics, "face_width_frame_ratio")

            if challenge_type == "BLINK" and face_metrics:
                eyes_closed = (
                    face_metrics["left_ear"] < BLINK_EAR_THRESHOLD
                    and face_metrics["right_ear"] < BLINK_EAR_THRESHOLD
                )
                eyes_open = (
                    face_metrics["left_ear"] >= BLINK_EAR_THRESHOLD + 0.035
                    and face_metrics["right_ear"] >= BLINK_EAR_THRESHOLD + 0.035
                )

                if eyes_closed:
                    closed_streak += 1
                    open_streak = 0
                elif eyes_open:
                    open_streak += 1
                    closed_streak = 0
                else:
                    closed_streak = 0
                    open_streak = 0

                if (
                    closed_streak >= BLINK_CLOSED_CONSECUTIVE_FRAMES
                    and not eye_closed
                ):
                    eye_closed = True

                if open_streak >= BLINK_OPEN_CONSECUTIVE_FRAMES and eye_closed:
                    blink_count += 1
                    eye_closed = False

                hit = blink_count >= BLINK_REQUIRED_COUNT
            elif challenge_type == "SMILE" and face_metrics and baseline_ready:
                hit = (
                    face_metrics["smile_ratio"] >= max(SMILE_RATIO_THRESHOLD, baseline_smile + SMILE_RATIO_DELTA)
                    and face_metrics["mouth_open_ratio"] >= SMILE_MIN_MOUTH_OPEN_RATIO
                )
            elif challenge_type == "TURN_HEAD_LEFT" and face_metrics and baseline_ready:
                hit = (
                    face_metrics["nose_x_ratio"] >= HEAD_TURN_RATIO_THRESHOLD
                    and face_metrics["nose_x_ratio"] >= baseline_nose_x + HEAD_TURN_RATIO_DELTA
                )
            elif challenge_type == "TURN_HEAD_RIGHT" and face_metrics and baseline_ready:
                hit = (
                    face_metrics["nose_x_ratio"] <= -HEAD_TURN_RATIO_THRESHOLD
                    and face_metrics["nose_x_ratio"] <= baseline_nose_x - HEAD_TURN_RATIO_DELTA
                )
            elif challenge_type == "LOOK_UP" and face_metrics and baseline_ready:
                hit = (
                    face_metrics["nose_y_ratio"] <= LOOK_UP_RATIO_THRESHOLD
                    and face_metrics["nose_y_ratio"] <= baseline_nose_y - LOOK_VERTICAL_RATIO_DELTA
                )
            elif challenge_type == "LOOK_DOWN" and face_metrics and baseline_ready:
                hit = (
                    face_metrics["nose_y_ratio"] >= LOOK_DOWN_RATIO_THRESHOLD
                    and face_metrics["nose_y_ratio"] >= baseline_nose_y + LOOK_VERTICAL_RATIO_DELTA
                )
            elif challenge_type == "OPEN_MOUTH" and face_metrics and baseline_ready:
                hit = face_metrics["mouth_open_ratio"] >= max(
                    OPEN_MOUTH_RATIO_THRESHOLD,
                    baseline_mouth_open + OPEN_MOUTH_RATIO_DELTA,
                )
            elif challenge_type == "NOD_YES" and face_metrics and baseline_ready:
                nose_y_values = metric_samples["nose_y_ratio"]
                vertical_range = max(nose_y_values) - min(nose_y_values)
                hit = vertical_range >= NOD_VERTICAL_RANGE_DELTA
            elif challenge_type == "SHAKE_HEAD_NO" and face_metrics and baseline_ready:
                nose_x_values = metric_samples["nose_x_ratio"]
                horizontal_range = max(nose_x_values) - min(nose_x_values)
                hit = horizontal_range >= SHAKE_HORIZONTAL_RANGE_DELTA
            elif challenge_type == "MOVE_CLOSER" and face_metrics and baseline_ready:
                hit = face_metrics["face_width_frame_ratio"] >= baseline_face_width + MOVE_CLOSER_FACE_WIDTH_DELTA
            elif challenge_type == "MOVE_BACK" and face_metrics and baseline_ready:
                hit = face_metrics["face_width_frame_ratio"] <= baseline_face_width - MOVE_BACK_FACE_WIDTH_DELTA
            elif challenge_type in {"RAISE_HAND", "SHOW_OPEN_PALM", "SHOW_TWO_FINGERS", "TOUCH_NOSE"}:
                for hand in hand_landmarks:
                    finger_count = _finger_count(hand, analysis_frame.shape)
                    metric_samples["finger_count"].append(float(finger_count))

                    if challenge_type == "RAISE_HAND" and _hand_raised_detected(hand, analysis_frame.shape):
                        hit = True
                    elif challenge_type == "SHOW_OPEN_PALM" and _open_palm_detected(hand, analysis_frame.shape):
                        hit = True
                    elif challenge_type == "SHOW_TWO_FINGERS" and _two_fingers_detected(hand, analysis_frame.shape):
                        hit = True
                    elif challenge_type == "TOUCH_NOSE" and face_metrics and _touch_nose_detected(hand, analysis_frame.shape, face_metrics):
                        hit = True

                    if hit:
                        break

            hits.append(hit)
    finally:
        if hands is not None:
            hands.close()

    if multiple_face_frame is not None:
        return _summarize_result(
            challenge=challenge,
            passed=False,
            message="Multiple faces were detected. Please record alone with only your face visible.",
            details={"multiple_face_frame": multiple_face_frame},
        )

    min_required_face_frames = max(MIN_ANALYZED_FACE_FRAMES, int(sampled_frame_count * MIN_FACE_COVERAGE_RATIO))
    if needs_face and face_frames < min_required_face_frames:
        return _summarize_result(
            challenge=challenge,
            passed=False,
            message="Your face was not visible clearly enough. Keep your face centered for the whole video.",
            details={
                "face_frames": face_frames,
                "sampled_frames": sampled_frame_count,
                "required_face_frames": min_required_face_frames,
                "required_face_coverage_ratio": MIN_FACE_COVERAGE_RATIO,
            },
        )

    if needs_hand and hand_frames < MIN_HAND_CHALLENGE_HIT_FRAMES:
        return _summarize_result(
            challenge=challenge,
            passed=False,
            message="No clear hand gesture was detected. Please keep your hand visible in the camera frame.",
            details={"hand_frames": hand_frames, "required_hand_frames": MIN_HAND_CHALLENGE_HIT_FRAMES},
        )

    consecutive_hits = _stable_hit_count(hits)
    required_hits = BLINK_REQUIRED_COUNT if challenge_type == "BLINK" else (
        MIN_HAND_CHALLENGE_HIT_FRAMES if needs_hand else MIN_FACE_CHALLENGE_HIT_FRAMES
    )
    passed = (
        blink_count >= BLINK_REQUIRED_COUNT
        if challenge_type == "BLINK"
        else consecutive_hits >= required_hits and (not needs_face or len(baseline_face_metrics) >= BASELINE_FACE_FRAMES)
    )

    baseline_details = {
        "smile_ratio": _median_metric(baseline_face_metrics, "smile_ratio"),
        "mouth_open_ratio": _median_metric(baseline_face_metrics, "mouth_open_ratio"),
        "nose_x_ratio": _median_metric(baseline_face_metrics, "nose_x_ratio"),
        "nose_y_ratio": _median_metric(baseline_face_metrics, "nose_y_ratio"),
        "face_width_frame_ratio": _median_metric(baseline_face_metrics, "face_width_frame_ratio"),
    }

    details = {
        "face_frames": face_frames,
        "hand_frames": hand_frames,
        "sampled_frames": sampled_frame_count,
        "face_coverage_ratio": round(face_frames / max(sampled_frame_count, 1), 3),
        "baseline_frames": len(baseline_face_metrics),
        "baseline": baseline_details,
        "consecutive_hit_frames": consecutive_hits,
        "required_hit_frames": required_hits,
        "blink_count": blink_count,
        "required_blinks": BLINK_REQUIRED_COUNT,
        "min_ear": min(metric_samples["ear"] or [0.0]),
        "max_smile_ratio": max(metric_samples["smile_ratio"] or [0.0]),
        "max_mouth_open_ratio": max(metric_samples["mouth_open_ratio"] or [0.0]),
        "min_nose_y_ratio": min(metric_samples["nose_y_ratio"] or [0.0]),
        "max_nose_y_ratio": max(metric_samples["nose_y_ratio"] or [0.0]),
        "nose_y_range": (max(metric_samples["nose_y_ratio"] or [0.0]) - min(metric_samples["nose_y_ratio"] or [0.0])),
        "min_nose_x_ratio": min(metric_samples["nose_x_ratio"] or [0.0]),
        "max_nose_x_ratio": max(metric_samples["nose_x_ratio"] or [0.0]),
        "nose_x_range": (max(metric_samples["nose_x_ratio"] or [0.0]) - min(metric_samples["nose_x_ratio"] or [0.0])),
        "min_face_width_frame_ratio": min(metric_samples["face_width_frame_ratio"] or [0.0]),
        "max_face_width_frame_ratio": max(metric_samples["face_width_frame_ratio"] or [0.0]),
        "max_finger_count": int(max(metric_samples["finger_count"] or [0])),
    }

    return _summarize_result(
        challenge=challenge,
        passed=passed,
        message="Challenge not completed. Please try again and make the requested action clear.",
        details=details,
    )


def extract_faces_from_frames(frames, sample_every_n=10):
    extracted_faces = []
    sampled_frame_indices = []

    for i in range(0, len(frames), sample_every_n):
        frame = frames[i]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if len(face_locations) == 1 and len(face_encodings) == 1:
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


def _analyze_identity_binding(frames: list[np.ndarray], selfie_img_bgr, face_match_threshold=0.50) -> dict[str, Any]:
    face_extraction = extract_faces_from_frames(frames, sample_every_n=10)

    if not face_extraction.get("ok"):
        error = face_extraction.get("error", "unknown")
        if error == "multiple_faces_in_frame":
            return {
                "ok": False,
                "error": "Multiple faces detected in liveness video. Please ensure only your face is visible.",
                "face_detected": False,
                "identity_match_passed": False,
                "details": f"Multiple faces found at frame {face_extraction.get('frame_index')}",
            }

        return {
            "ok": False,
            "error": "No face detected in liveness video.",
            "face_detected": False,
            "identity_match_passed": False,
        }

    extracted_faces = face_extraction.get("extracted_faces", [])
    selfie_rgb = cv2.cvtColor(selfie_img_bgr, cv2.COLOR_BGR2RGB)
    selfie_encodings = face_recognition.face_encodings(selfie_rgb)

    if not selfie_encodings:
        return {
            "ok": False,
            "error": "Could not extract face encoding from selfie",
            "face_detected": True,
            "identity_match_passed": False,
        }

    selfie_encoding = selfie_encodings[0]
    best_distance = float("inf")

    for face_data in extracted_faces:
        liveness_encoding = face_data["face_encoding"]
        distance = float(np.linalg.norm(selfie_encoding - liveness_encoding))
        best_distance = min(best_distance, distance)

    identity_match_passed = best_distance < face_match_threshold

    return {
        "ok": identity_match_passed,
        "face_detected": True,
        "num_faces_extracted": len(extracted_faces),
        "identity_match_passed": identity_match_passed,
        "identity_match_distance": best_distance,
        "identity_match_threshold": face_match_threshold,
        "error": None if identity_match_passed else "The person in the liveness video does not match the selfie.",
    }


def analyze_liveness_challenge_video(video_path, selfie_img_bgr, challenge: dict[str, Any], face_match_threshold=0.50):
    frames = extract_frames_from_video(video_path)
    if len(frames) == 0:
        return {
            "ok": False,
            "passed": False,
            "error": "No frames could be extracted from the video. Please record a new liveness video.",
            "challenge_id": challenge.get("challenge_id"),
            "challenge_type": challenge.get("challenge_type"),
            "identity_match_passed": None,
            "details": {"frames_extracted": 0},
        }

    if len(frames) < MIN_VIDEO_FRAMES:
        return {
            "ok": False,
            "passed": False,
            "error": "The liveness video is too short. Please record a longer video and complete the challenge clearly.",
            "challenge_id": challenge.get("challenge_id"),
            "challenge_type": challenge.get("challenge_type"),
            "instruction": challenge.get("instruction"),
            "identity_match_passed": None,
            "details": {
                "frames_extracted": len(frames),
                "required_frames": MIN_VIDEO_FRAMES,
            },
        }

    challenge_result = _analyze_challenge_frames(frames, challenge)
    identity_result = _analyze_identity_binding(frames, selfie_img_bgr, face_match_threshold=face_match_threshold)

    passed = bool(challenge_result.get("passed")) and bool(identity_result.get("identity_match_passed"))
    error = None
    if not challenge_result.get("passed"):
        error = challenge_result.get("error") or challenge_result.get("message")
    elif not identity_result.get("identity_match_passed"):
        error = identity_result.get("error")

    return {
        **challenge_result,
        "ok": passed,
        "passed": passed,
        "error": error,
        "message": "Liveness challenge passed." if passed else error,
        "identity_match_passed": identity_result.get("identity_match_passed"),
        "identity_match_distance": identity_result.get("identity_match_distance"),
        "identity_match_threshold": identity_result.get("identity_match_threshold"),
        "details": {
            "frames_extracted": len(frames),
            "challenge": challenge_result.get("details", {}),
            "identity": {
                key: value
                for key, value in identity_result.items()
                if key not in {"ok", "error"}
            },
        },
    }


def analyze_liveness_with_identity_binding(video_path, selfie_img_bgr, face_match_threshold=0.50):
    challenge = {
        "challenge_id": "legacy-blink",
        "challenge_type": "BLINK",
        "instruction": "Blink 3 times",
        "required_action": {"required_blinks": BLINK_REQUIRED_COUNT},
    }
    result = analyze_liveness_challenge_video(video_path, selfie_img_bgr, challenge, face_match_threshold=face_match_threshold)
    details = result.get("details", {}).get("challenge", {})

    return {
        **result,
        "blink_passed": result.get("passed"),
        "blink_details": {
            "blink_count": details.get("blink_count"),
            "required_blinks": details.get("required_blinks"),
            "analyzed_frames": details.get("face_frames"),
        },
    }
