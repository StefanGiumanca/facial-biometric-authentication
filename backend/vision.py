import cv2

#function for model loading
def load_haar_face_detector():
    face_detect = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return face_detect

#function for detecting faces
def detect_faces(detector, gray_image):
    faces = detector.detectMultiScale(
        gray_image,
        scaleFactor = 1.05,
        minNeighbors = 8,
        minSize = (150,150)
    )
    return faces


# function for drawing bounding boxes
def draw_boxes(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
    return image

# extract ROI - photo on the id
def extract_id_face(detector, image):
    h, w = image.shape[:2]
    roi_w = int(w * 0.6)
    roi = image[:, :roi_w]

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(detector, roi_gray)

    if len(faces) == 0:
        return None
    
    (x, y, w_box, h_box) = max(faces, key=lambda b: b[2] * b[3])

    # Add small margin around detected face
    margin_x = int(0.15 * w_box)
    margin_y = int(0.15 * h_box)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(roi.shape[1], x + w_box + margin_x)
    y2 = min(roi.shape[0], y + h_box + margin_y)

    face_crop = roi[y1:y2, x1:x2]

    return face_crop
