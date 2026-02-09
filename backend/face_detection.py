import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

img = cv2.imread("data/faces.png")

if img is None:
    print("Inexisting image.")
    exit()

# grayscale conversion
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect the faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(60,60)
)

# drawing the rectangle
for (x,y,w,h) in faces:
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

#print the output
cv2.imshow("Face Detection - Haar Cascades", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

