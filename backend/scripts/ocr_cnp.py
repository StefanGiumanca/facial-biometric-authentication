from pathlib import Path
import re
import cv2
import easyocr

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

def is_valid_cnp(cnp: str) -> bool:
    if len(cnp) != 13:
        return False
    
    if not cnp.isdigit():
        return False
    
    control_key = "279146358279"

    total = 0
    for i in range(12):
        total += int(cnp[i]) * int(control_key[i])
    
    remainder = total % 11
    if remainder == 10:
        check_digit = 1
    else:
        check_digit = remainder

    return check_digit == int(cnp[12])


def preprocess_for_ocr(gray):
    """Preprocessing to improve OCR on small ID text (upscale + contrast + binarize)."""
    up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # Contrast boost (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    up = clahe.apply(up)

    # Light denoise
    up = cv2.GaussianBlur(up, (3, 3), 0)

    # Binarize
    thr = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Make text slightly bolder
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

# --- MRZ name extraction helper ---
def extract_names_from_mrz(text: str):
    """Extract (last_name, first_name) from Romanian ID MRZ if present.

    MRZ line for RO IDs often contains something like:
      IDROU<...<LAST<<FIRST<NAMES<<<<
    OCR may introduce spaces; we normalize them.
    """
    upper = text.upper().replace(" ", "")

    # Look for the MRZ document prefix.
    m = re.search(r"IDROU[0-9A-Z<]+", upper)
    if not m:
        return None, None

    mrz = m.group(0)

    # Names section begins after 'IDROU' and optional fillers.
    # Find the first occurrence of a surname pattern ending with '<<'.
    # Typical: IDROUCIopanoiU<<ALEXIA<ELENA<<<<
    # We'll take everything after IDROU and split by '<<'.
    after = mrz[5:]  # strip 'IDROU'

    parts = after.split("<<", 1)
    if len(parts) != 2:
        return None, None

    raw_last = parts[0]
    raw_first = parts[1]

    # Clean fillers
    raw_last = raw_last.replace("<", "")

    # First names end before the next filler run '<<<<' typically; take up to first occurrence of '<<<' if present.
    raw_first = raw_first.split("<<<", 1)[0]
    raw_first = raw_first.replace("<", " ").strip()

    # Keep only letters and hyphen/space
    last_name = re.sub(r"[^A-Z-]", "", raw_last)
    first_name = re.sub(r"[^A-Z- ]", "", raw_first)
    first_name = re.sub(r"\s+", " ", first_name).strip()

    if len(last_name) < 2 or len(first_name) < 2:
        return None, None

    return last_name, first_name

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

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    series_roi = crop_series_roi(img_bgr)
    cv2.imwrite(str(outputs_dir / "series_roi.jpg"), series_roi)
    series_gray = cv2.cvtColor(series_roi, cv2.COLOR_BGR2GRAY)
    series_gray = cv2.GaussianBlur(series_gray, (3, 3), 0)

    reader = easyocr.Reader(["en"], gpu=False)
    ocr_results = reader.readtext(gray)

    texts = [t for (_, t, conf) in ocr_results]
    joined = " ".join(texts)

    upper_full = joined.upper()

    # 1) Prefer ROI-based OCR for last/first name (more robust than full text parsing).
    ln_roi = crop_last_name_roi(img_bgr)
    fn_roi = crop_first_name_roi(img_bgr)

    ln_up, ln_thr = ocr_roi_text(reader, ln_roi, "last_name_roi", outputs_dir)
    fn_up, fn_thr = ocr_roi_text(reader, fn_roi, "first_name_roi", outputs_dir)

    def pick_name(text_a: str, text_b: str):
        label_words = {
            "NUME", "NOM", "LAST", "NAME",
            "PRENUME", "PRENOM", "FIRST",
            "IDENTITY", "IDENTITE", "CARTE", "CARD",
            "ROMANIA", "ROMANA", "ROU", "ROM", "CNP", "SERIA", "NR",
            "CETATENIE", "NATIONALITATE", "NATIONALITY", "NATIONALLLY", "NALIONALLTE", "NATLONALITY",
            "SEZ", "SERE", "SEX", "SEXE", "ROMANA"
        }
        bad_substrings = [
            "NOM", "PRENOM", "LAST", "FIRST", "NAME",
            "CETAT", "NATION", "NATIONAL", "NATL", "NALION", "NATLON"
        ]

        good = []
        for txt in (text_a, text_b):
            toks = re.findall(r"[A-Za-z-]+", txt)
            for tok in toks:
                up_tok = tok.upper()
                # Skip exact label tokens
                if up_tok in label_words:
                    continue
                # Skip glued/partial label tokens like NOMILAST, PRENOMIFIRST, etc.
                if any(s in up_tok for s in bad_substrings):
                    continue
                # Keep only plausible name-like tokens
                if len(up_tok) >= 2 and re.fullmatch(r"[A-Z-]+", up_tok):
                    good.append(up_tok)

        # Prefer the most name-like token: longest token, bonus if it contains '-'.
        if not good:
            return None

        def score(tok: str):
            return (len(tok), 1 if '-' in tok else 0)

        return sorted(good, key=score, reverse=True)[0]

    # Prefer MRZ parsing (most stable), then fallback to ROI OCR.
    mrz_last, mrz_first = extract_names_from_mrz(joined)

    last_name = mrz_last if mrz_last else pick_name(ln_up, ln_thr)
    first_name = mrz_first if mrz_first else pick_name(fn_up, fn_thr)

    # 2) Fallback to full-text parsing using English labels if ROI failed.
    if last_name is None:
        m_last = re.search(r"\bLAST\s*NAME\b[^A-Z0-9]+([A-Z-]{2,})", upper_full)
        if not m_last:
            m_last = re.search(r"\bNUME\b[^A-Z0-9]+([A-Z-]{2,})", upper_full)
        last_name = m_last.group(1) if m_last else None

    if first_name is None:
        m_first = re.search(r"\bFIRST\s*NAME\b[^A-Z0-9]+([A-Z-]{2,})", upper_full)
        if not m_first:
            m_first = re.search(r"\bPRENUME\b[^A-Z0-9]+([A-Z-]{2,})", upper_full)
        first_name = m_first.group(1) if m_first else None

    print("OCR Text:")
    print (joined)
    print("\n\n")

    print("EXTRACTED LAST NAME:", last_name)
    print("EXTRACTED FIRST NAME:", first_name)

    m_series_full = re.search(r"\bSERIA\b\s+([A-Z]{2})\s+(?:NR|NA|N\s*R|H[A4])\s+(\d{6})\b", upper_full)
    id_series_number_full = f"{m_series_full.group(1)}{m_series_full.group(2)}" if m_series_full else None
    print("BEST-EFFORT SERIES+NUMBER (FULL OCR):", id_series_number_full)

    print("SERIES OCR TEXT:")
    ocr_series = reader.readtext(series_gray)
    series_texts = [t for (_, t, conf) in ocr_series]
    series_joined = " ".join(series_texts)
    print(series_joined)

    upper = series_joined.upper()

    series_matches = re.findall(r"\b([A-Z]{2})\s*(?:NR|N\s*R|HA|H4)\s*(\d{5,7})\b", upper)

    if not series_matches:
        series_matches = re.findall(r"\b([A-Z]{2})\s*(?:NR\s*)?(\d{5,7})\b", upper)

    print("Series matches:", series_matches)

    if series_matches:
        series, number = series_matches[0]
        id_series_number_roi = f"{series}{number}"
        print("ID series + number (ROI):", id_series_number_roi)

        if len(number) != 6:
            print(f"WARNING: number length is {len(number)} (expected 6) -> MANUAL_REVIEW")
    else:
        print("No series + number found -> MANUAL_REVIEW")

    candidates = re.findall(r"\b\d{13}\b", joined)
    if not candidates:
        print("No 13-digit candidate found.")
        return
    
    print("CNP candidates: ", candidates)

    valid_cnps = [c for c in candidates if is_valid_cnp(c)]

    if not valid_cnps:
        print("No valid CNP found (checksum failed) -> MANUAL_REVIEW")
        return
    
    if len(valid_cnps) > 1:
        print("Multiple CNPs found -> MANUAL_REVIEW")
        print("Valid CNPs: ", valid_cnps)
        return 
    
    cnp = valid_cnps[0]
    print("VALID CNP:", cnp)

if __name__ == "__main__":
    main()
