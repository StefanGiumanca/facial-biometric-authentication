import re
from difflib import SequenceMatcher

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

# --- MRZ name extraction helper ---
def extract_names_from_mrz(text: str):
    """Extract (last_name, first_name) from Romanian ID MRZ if present.

    MRZ line for RO IDs often contains something like:
      IDROU<...<LAST<<FIRST<NAMES<<<<
    OCR may introduce spaces; we normalize them.
    """
    upper = text.upper().replace(" ", "")

    m = re.search(r"IDROU[0-9A-Z<]+", upper)
    if not m:
        return None, None

    mrz = m.group(0)

    after = mrz[5:]  #

    parts = after.split("<<", 1)
    if len(parts) != 2:
        return None, None

    raw_last = parts[0]
    raw_first = parts[1]


    raw_last = raw_last.replace("<", "")

    raw_first = raw_first.split("<<<", 1)[0]
    raw_first = raw_first.replace("<", " ").strip()

    last_name = re.sub(r"[^A-Z-]", "", raw_last)
    first_name = re.sub(r"[^A-Z- ]", "", raw_first)
    first_name = re.sub(r"\s+", " ", first_name).strip()

    if len(last_name) < 2 or len(first_name) < 2:
        return None, None

    return last_name, first_name

def extract_series_from_text(text: str):
    upper = text.upper()

    m = re.search(r"\b([A-Z]{2})\s*(?:N\s*R|NR|NA|HA|H4)\s*(\d{6})\b", upper)
    if m:
        return f"{m.group(1)}{m.group(2)}"

    m2 = re.search(r"\b([A-Z]{2})\s*(\d{6})\b", upper)
    if m2:
        return f"{m2.group(1)}{m2.group(2)}"

    return None

def normalize_name(s : str | None) -> str:
    if not s:
        return ""
    s = s.upper().strip()
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^A-Z ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def name_similarity(a: str | None, b: str | None) -> float:
    na = normalize_name(a)
    nb = normalize_name(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()

def evaluate_similarity(score: float, accept: float = 0.90, review: float = 0.75) -> str:
    if score >= accept:
        return "ACCEPT"
    if score >= review:
        return "REVIEW"
    return "REJECT"

def evaluate_names(expected_last: str, expected_first: str,
                   extracted_last: str | None, extracted_first: str | None,
                   accept: float = 0.90, review: float = 0.75) -> dict:
    last_score = name_similarity(expected_last, extracted_last)
    first_score = name_similarity(expected_first, extracted_first)
    combined = (last_score + first_score) / 2.0
    return {
        "last_name_score": round(last_score, 3),
        "first_name_score": round(first_score, 3),
        "combined_score": round(combined, 3),
        "decision": evaluate_similarity(combined, accept, review),
    }

def parse_romanian_id(full_text: str, series_text: str | None = None) -> dict:
    last_name, first_name = extract_names_from_mrz(full_text)

    candidates = re.findall(r"\b\d{13}\b", full_text)
    valid_cnps = [c for c in candidates if is_valid_cnp(c)]
    cnp = valid_cnps[0] if len(valid_cnps) == 1 else None

    series_number = extract_series_from_text(series_text) if series_text else None

    return {
        "last_name": last_name,
        "first_name": first_name,
        "cnp": cnp,
        "series_number": series_number,
    }


def validate_reviewed_document_fields(ocr_fields: dict, reviewed_fields: dict) -> dict:
    """Validate user-reviewed document fields against OCR-extracted data.

    Checks:
    - Required identity fields (first_name, last_name, cnp) are present.
    - CNP passes the Romanian checksum validation.
    - Reviewed names are similar enough to the OCR-extracted names.

    Returns dict with ``ok`` bool, human-readable ``warnings``, and a
    ``failed_fields`` list that the mobile review screen can map back to form
    inputs.
    """
    warnings: list[str] = []
    failed_fields: list[dict] = []

    # --- required fields ---
    first_name = reviewed_fields.get("first_name")
    last_name = reviewed_fields.get("last_name")
    cnp = reviewed_fields.get("cnp")

    if not first_name or not str(first_name).strip():
        warnings.append("First name is missing")
        failed_fields.append({"field": "first_name", "message": "First name is missing"})
    if not last_name or not str(last_name).strip():
        warnings.append("Last name is missing")
        failed_fields.append({"field": "last_name", "message": "Last name is missing"})
    if not cnp or not str(cnp).strip():
        warnings.append("CNP is missing")
        failed_fields.append({"field": "cnp", "message": "CNP is missing"})

    # --- CNP validity ---
    if cnp and str(cnp).strip():
        if not is_valid_cnp(str(cnp).strip()):
            warnings.append("CNP checksum is invalid")
            failed_fields.append({"field": "cnp", "message": "CNP checksum is invalid"})

    # --- name similarity (only when OCR extracted something) ---
    ocr_last = ocr_fields.get("last_name")
    ocr_first = ocr_fields.get("first_name")
    if ocr_last and ocr_first and first_name and last_name:
        names_result = evaluate_names(
            expected_last=last_name,
            expected_first=first_name,
            extracted_last=ocr_last,
            extracted_first=ocr_first,
        )
        if names_result["decision"] == "REJECT":
            warnings.append(
                f"Reviewed names differ significantly from OCR (score: {names_result['combined_score']:.2f})"
            )
            failed_fields.append({
                "field": "first_name",
                "message": "Reviewed first name differs significantly from OCR",
                "similarity": names_result["first_name_score"],
            })
            failed_fields.append({
                "field": "last_name",
                "message": "Reviewed last name differs significantly from OCR",
                "similarity": names_result["last_name_score"],
            })

    reviewed_series = reviewed_fields.get("series_number")
    ocr_series = ocr_fields.get("series_number")
    if reviewed_series and ocr_series and str(reviewed_series).strip().upper() != str(ocr_series).strip().upper():
        warnings.append("Series and number do not match the OCR result")
        failed_fields.append({
            "field": "series_number",
            "message": "Series and number do not match the OCR result",
        })

    ok = len(warnings) == 0
    return {
        "ok": ok,
        "warnings": warnings,
        "failed_fields": failed_fields,
    }
