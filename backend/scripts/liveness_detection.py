import cv2 
import mediapipe as mp
import numpy as np
import time

def main():
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Camera failed")
        return
    
    # loading the mediapipe face landmarks model
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode = False,
        max_num_faces = 1,
        refine_landmarks = True
    )

    # setting the time limit for the verification
    start_time = time.time()
    time_limit = 10

    blink_count = 0
    eye_closed = False  # state
    EAR_THRESHOLD = 0.20    # we consider the eye closed under this value
    REQUIRED_BLINKS = 3
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Frame can't be taken.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        elapsed = time.time() - start_time
        remaining = max(0, int(time_limit - elapsed))

        # Draw FaceMesh landmark points in green
        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = results.multi_face_landmarks[0].landmark

            def pt(i):
                return np.array([landmarks[i].x * w, landmarks[i].y * h], dtype=np.float32)

            # left eye indices
            l1 = pt(33)
            l2 = pt(160)
            l3 = pt(158)
            l4 = pt(133)
            l5 = pt(153)
            l6 = pt(144)

            # right eye indices
            r1 = pt(362)
            r2 = pt(385)
            r3 = pt(387)
            r4 = pt(263)
            r5 = pt(373)
            r6 = pt(380)

            # left eye EAR
            left_v1 = np.linalg.norm(l2 - l6)
            left_v2 = np.linalg.norm(l3 - l5)
            left_h = np.linalg.norm(l1 - l4)
            left_ear = (left_v1 + left_v2) / (2.0 * left_h) if left_h > 0 else 0.0

            # right eye EAR
            right_v1 = np.linalg.norm(r2 - r6)
            right_v2 = np.linalg.norm(r3 - r5)
            right_h = np.linalg.norm(r1 - r4)
            right_ear = (right_v1 + right_v2) / (2.0 * right_h) if right_h > 0 else 0.0

            # final EAR (average of both eyes)
            ear = (left_ear + right_ear) / 2.0

            # Blink detection (state machine)
            if right_ear < EAR_THRESHOLD and left_ear < EAR_THRESHOLD and not eye_closed:
                eye_closed = True
            elif right_ear >= EAR_THRESHOLD and left_ear >= EAR_THRESHOLD and eye_closed:
                blink_count += 1
                eye_closed = False
                print(f"Blink {blink_count}/{REQUIRED_BLINKS}")

            # Draw minimal debug points (green) for both eyes
            for p in [l1, l2, l3, l4, l5, l6, r1, r2, r3, r4, r5, r6]:
                cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)

            cv2.putText(
                frame,
                f"EAR: {ear:.3f}  Blinks: {blink_count}/{REQUIRED_BLINKS}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Time left: {remaining}s",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )
        cv2.imshow("Liveness Detection - Blink x 3 times", frame)

        if blink_count >= REQUIRED_BLINKS:
            print("LIVENESS PASSED")
            break

        if elapsed > time_limit:
            print("Time expired")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()            

if __name__ == "__main__":
    main()