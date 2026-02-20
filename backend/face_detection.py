import cv2
from vision import load_haar_face_detector, detect_faces, draw_boxes

# Load detector once (clean reuse)
face_detector = load_haar_face_detector()

# Load image
img = cv2.imread("data/samples/face.png")

if img is None:
    print("Inexisting image.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces using vision.py
faces = detect_faces(face_detector, gray)

# Draw bounding boxes using vision.py
img_with_boxes = draw_boxes(img, faces)

# Show result
cv2.imshow("Face Detection - Haar Cascades", img_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()