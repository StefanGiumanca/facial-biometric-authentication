from pathlib import Path
import cv2
import easyocr
import re

def crop_series_roi(img_bgr):
    h, w = img_bgr.shape[:2]

    y1 = int(0.12 * h)
    y2 = int(0.28 * h)
    x1 = int(0.45 * w)
    x2 = int(0.85 * w)

    roi = img_bgr[y1:y2, x1:x2] 
    return roi


def crop_last_name_roi(img_bgr):
    h, w = img_bgr.shape[:2]

    y1 = int(0.30 * h)
    y2 = int(0.39 * h)

    x1 = int(0.25 * w)
    x2 = int(0.70 * w)

    return img_bgr[y1:y2, x1:x2]


def crop_first_name_roi(img_bgr):
    h, w = img_bgr.shape[:2]

    y1 = int(0.36 * h)
    y2 = int(0.45 * h)

    x1 = int(0.28 * w)
    x2 = int(0.75 * w)

    return img_bgr[y1:y2, x1:x2]


def preprocess_for_ocr(gray):
    """Preprocessing to improve OCR on small ID text (upscale + contrast + binarize)."""
    up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)


    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    up = clahe.apply(up)

    up = cv2.GaussianBlur(up, (3, 3), 0)

    thr = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)

    return up, thr


def ocr_roi_text(reader, roi_bgr, label: str, outputs_dir: Path):
    """Run OCR on a ROI using two preprocessed variants and return both joined texts."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outputs_dir / f"{label}.jpg"), roi_bgr)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    up, thr = preprocess_for_ocr(gray)

    ocr_up = reader.readtext(up, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz- ")
    up_text = " ".join([t for (_, t, conf) in ocr_up])

    ocr_thr = reader.readtext(thr, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz- ")
    thr_text = " ".join([t for (_, t, conf) in ocr_thr])

    return up_text, thr_text



def build_reader(gpu: bool = False):
    """Create and return an EasyOCR Reader instance."""
    return easyocr.Reader(["en"], gpu=gpu)


def ocr_full_text(reader, img_bgr) -> str:
    """Run OCR on the full ID image and return a single joined text string."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    results = reader.readtext(gray)
    texts = [t for (_, t, conf) in results]
    return " ".join(texts)


def ocr_series_text_dynamic(reader, img_bgr, outputs_dir: Path | None = None) -> str:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    results = reader.readtext(gray)

    # normalize tokens for matching
    def norm(s: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", s.upper())

    # anchor keywords (common OCR mistakes included)
    seria_variants = {"SERIA", "SEAIA", "SEPIA", "SERLA", "SERJA"}
    nr_variants = {"NR", "NA", "N", "H4", "HA"}  # we'll handle N R by normalization

    anchor = None
    anchor_bbox = None
    anchor_conf = 0.0

    for (bbox, text, conf) in results:
        t = norm(text)

        # treat "N R" etc. as NR because norm removes spaces/punct
        if t in seria_variants or t in nr_variants:
            if conf > anchor_conf:
                anchor = t
                anchor_bbox = bbox
                anchor_conf = conf

    if anchor_bbox is None:
        # fallback: old static ROI
        return ocr_series_text(reader, img_bgr, outputs_dir=outputs_dir)

    xs = [p[0] for p in anchor_bbox]
    ys = [p[1] for p in anchor_bbox]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    h, w = gray.shape[:2]

    pad_left = int(0.05 * w)
    pad_right = int(0.35 * w)   
    pad_up = int(0.03 * h)
    pad_down = int(0.06 * h)

    rx1 = max(0, x_min - pad_left)
    ry1 = max(0, y_min - pad_up)
    rx2 = min(w, x_max + pad_right)
    ry2 = min(h, y_max + pad_down)

    roi = img_bgr[ry1:ry2, rx1:rx2]

    if outputs_dir is not None:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(outputs_dir / "series_roi_dynamic.jpg"), roi)

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)

    roi_results = reader.readtext(roi_gray)
    texts = [t for (_, t, conf) in roi_results]
    return " ".join(texts)