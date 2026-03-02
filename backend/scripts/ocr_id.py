from pathlib import Path
import cv2
import re

from backend.services.ocr_engine import build_reader, ocr_full_text, ocr_series_text_dynamic
from backend.services.document_parser import extract_names_from_mrz, is_valid_cnp, extract_series_from_text, name_similarity

def main():
    script_dir = Path(__file__).resolve().parent
    backend_dir = script_dir.parent
    outputs_dir = backend_dir / "data" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    img_path = backend_dir / "data" / "private" / "id.jpg"

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"Could not read image: {img_path}")
        return

    reader = build_reader(gpu=False)
    full_text = ocr_full_text(reader, img_bgr)
    series_text = ocr_series_text_dynamic(reader, img_bgr, outputs_dir=outputs_dir)

    series_number = extract_series_from_text(series_text)

    last_name, first_name = extract_names_from_mrz(full_text)

    candidates = re.findall(r"\b\d{13}\b", full_text)
    valid_cnps = [c for c in candidates if is_valid_cnp(c)]
    cnp = valid_cnps[0] if len(valid_cnps) == 1 else None

    print("\n=== OCR DEMO RESULT ===")
    print("LAST NAME:", last_name)
    print("FIRST NAME:", first_name)
    expected_last = input("Enter expected LAST NAME: ")
    expected_first = input("Enter expected FIRST NAME: ")
    print("LAST NAME SIM:", name_similarity(expected_last, last_name))
    print("FIRST NAME SIM:", name_similarity(expected_first, first_name))
    print("CNP:", cnp)
    print("SERIES ROI TEXT:", series_text)
    print("SERIES NUMBER:", series_number)


if __name__ == "__main__":
    main()