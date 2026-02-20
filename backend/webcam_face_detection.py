import cv2
from vision import load_haar_face_detector, detect_faces, draw_boxes

def main():
    detector = load_haar_face_detector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera failed")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame can't be read.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(detector, gray)
        if len(faces) > 0:
            faces = [max(faces, key=lambda b: b[2] *b [3])] # largest rectangle area
        frame_with_boxes = draw_boxes(frame, faces)

        cv2.imshow("Webcam - Press Q to quit", frame_with_boxes)

        # s for taking a selfie
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            if len(faces) == 0:
                print("No face detected. Cannot save selfie.")
            else:
                (x, y, w, h) = faces[0]
                face_crop = frame[y:y+h, x:x+w]
                saved = cv2.imwrite("data/outputs/selfie_face.jpg", face_crop)
                print("Saved selfie face." if saved else "Failed to save selfie face.")

        # q for quiting camera
        if key == ord("q") or key == 27:  # 27 = ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()