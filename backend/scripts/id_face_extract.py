import cv2
from backend.core.vision import load_haar_face_detector, extract_id_face

def main():
    detector = load_haar_face_detector()

    input_path = "backend/data/private/id.jpg"
    output_path = "backend/data/outputs/id_face.jpg"

    img = cv2.imread(input_path)
    if img is None:
        print(f"Could not read ID image: {input_path}")
        return

    face_crop = extract_id_face(detector, img)
    if face_crop is None:
        print("No face detected on ID image.")
        return

    ok = cv2.imwrite(output_path, face_crop)
    if not ok:
        print(f"Failed to save cropped ID face to: {output_path}")
        return

    print(f"Saved cropped ID face to: {output_path}")


if __name__ == "__main__":
    main()