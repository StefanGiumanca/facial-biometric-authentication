import face_recognition
import numpy as np 
from pathlib import Path
from backend.services.matching import match_faces

def main():
    script_dir = Path(__file__).resolve().parent
    backend_dir = script_dir.parent

    selfie_path = backend_dir / "data" / "outputs" / "selfie_face.jpg"
    id_path = backend_dir / "data" / "outputs" / "id_face.jpg"

    if not selfie_path.exists():
        print(f"Missing file: {selfie_path}")
        return

    if not id_path.exists():
        print(f"Missing file: {id_path}")
        return

    # RGB array for embeddings
    selfie_img = face_recognition.load_image_file(str(selfie_path))
    id_img = face_recognition.load_image_file(str(id_path))

    result = match_faces(selfie_img, id_img)

    if not result["ok"]:
        print(result["error"])
        return

    print(f"Distance: {result['distance']:.4f}")
    print(f"Decision: {result['decision']}")
    print(f"Accept threshold: {result['accept_threshold']}")
    print(f"Review threshold: {result['review_threshold']}")

if __name__ == "__main__":
    main()
