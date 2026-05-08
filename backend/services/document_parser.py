import re
import unicodedata
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


def strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_ocr_text(text: str) -> str:
    text = strip_diacritics(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_sex_from_text(text: str, cnp: str | None = None) -> str | None:
    normalized = normalize_ocr_text(text).upper()

    match = re.search(r"\bSEX(?:E)?\b\s*[:\-]?\s*([MF])\b", normalized)
    if match:
        return match.group(1)

    if cnp:
        return "M" if cnp[0] in {"1", "3", "5", "7"} else "F" if cnp[0] in {"2", "4", "6", "8"} else None

    return None


def extract_nationality_from_text(text: str) -> str | None:
    normalized = normalize_ocr_text(text)
    upper = normalized.upper()
    label_match = re.search(r"\b(CETATENIE|NATIONALITE|NATIONALITY)\b", upper)

    if label_match:
        window = normalized[label_match.end():label_match.end() + 50]
        if re.search(r"\bROU\b", window, flags=re.IGNORECASE):
            return "ROU"
        romanian_match = re.search(r"\bROMAN[A]?\b", strip_diacritics(window), flags=re.IGNORECASE)
        if romanian_match:
            return romanian_match.group(0).capitalize()

    if re.search(r"\bROU\b", upper):
        return "ROU"
    if re.search(r"\bROMAN[A]?\b", upper):
        return "Romana"

    return None


ADDRESS_STOP_WORDS = (
    "EMISA",
    "EMIS",
    "DELIVREE",
    "ISSUED",
    "VALABILITATE",
    "VALABIL",
    "VALIDITE",
    "VALIDITY",
    "VALID",
    "SPCLEP",
    "CETATENIE",
    "NATIONALITE",
    "NATIONALITY",
    r"SEX(?:E)?",
)


def extract_validity_dates_from_text(text: str) -> tuple[str | None, str | None]:
    normalized = normalize_ocr_text(text)
    date = r"\d{2}[./]\d{2}[./]\d{2,4}"
    range_match = re.search(rf"({date})\s*[-–]\s*({date})", normalized)
    if range_match:
        return range_match.group(1), range_match.group(2)

    label_match = re.search(rf"\b(VALABILITATE|VALABIL|VALID|VALIDITY)\b[^0-9]*({date})", normalized, flags=re.IGNORECASE)
    if label_match:
        return label_match.group(2), None

    return None, None


def extract_address_from_text(text: str) -> str | None:
    normalized = normalize_ocr_text(text)
    upper = normalized.upper()
    label_match = re.search(r"\b(DOMICILIU|ADRESSE|ADDRESS|DOMICILE)\b", upper)
    if not label_match:
        return None

    after_label = normalized[label_match.end():]
    stop_match = re.search(
        rf"\b({'|'.join(ADDRESS_STOP_WORDS)})\b",
        after_label,
        flags=re.IGNORECASE,
    )
    address = after_label[:stop_match.start()] if stop_match else after_label
    address = re.sub(r"^(?:\s*/?\s*(ADRESSE|ADDRESS|DOMICILE|DOMICILIU)\s*/?\s*)+", "", address, flags=re.IGNORECASE)
    address = re.sub(r"\b(JUD|JUD\.|SECT|SECTOR)\b", lambda match: match.group(0).upper(), address, flags=re.IGNORECASE)
    address = re.sub(r"\b(nr\.?\s*\d+[A-Z]?)\s+\d{2,4}\s+[A-Z]{1,3}\s+[A-Z]{1,3}\b", r"\1", address, flags=re.IGNORECASE)
    address = re.sub(r"\s+", " ", address)
    address = address.strip(" :;-.,")

    return address or None

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
    sex = extract_sex_from_text(full_text, cnp)
    nationality = extract_nationality_from_text(full_text)
    address = extract_address_from_text(full_text)
    valid_from, valid_until = extract_validity_dates_from_text(full_text)

    return {
        "last_name": last_name,
        "first_name": first_name,
        "cnp": cnp,
        "series_number": series_number,
        "sex": sex,
        "nationality": nationality,
        "address": address,
        "valid_from": valid_from,
        "valid_until": valid_until,
    }


def validate_romanian_id_document(full_text: str, series_text: str | None, parsed_fields: dict) -> dict:
    """Decide whether OCR output looks like a real Romanian identity card.

    A face photo can pass face-crop extraction, so this guard requires document
    text evidence before the flow may continue to review. The checks are
    intentionally conservative: at least one strong identity signal plus enough
    Romanian ID keywords must be present.
    """
    normalized_full_text = normalize_ocr_text(full_text).upper()
    normalized_series_text = normalize_ocr_text(series_text or "").upper()
    combined_text = f"{normalized_full_text} {normalized_series_text}"

    keyword_patterns = {
        "romania": r"\b(ROMANIA|ROU|ROUMANIE|ROUMANIAN)\b",
        "identity_card": r"\b(CARTE\s+DE\s+IDENTITATE|CARTE|IDENTITATE|IDENTITY|IDROU)\b",
        "cnp_label": r"\b(CNP|COD\s+NUMERIC\s+PERSONAL)\b",
        "series_label": r"\b(SERIA|SERIE|NR|NUMAR|NUMBER)\b",
        "nationality": r"\b(CETATENIE|NATIONALITATE|NATIONALITY|NATIONALITE)\b",
        "address": r"\b(DOMICILIU|ADRESA|ADRESSE|ADDRESS)\b",
        "validity": r"\b(VALABILITATE|VALABIL|VALIDITY|VALID)\b",
    }
    matched_keywords = [
        name for name, pattern in keyword_patterns.items()
        if re.search(pattern, combined_text, flags=re.IGNORECASE)
    ]

    strong_fields = [
        field for field in ("cnp", "series_number")
        if parsed_fields.get(field)
    ]
    optional_fields = [
        field for field in ("first_name", "last_name", "sex", "nationality", "address", "valid_from", "valid_until")
        if parsed_fields.get(field)
    ]

    has_mrz = "IDROU" in combined_text
    has_valid_cnp = bool(parsed_fields.get("cnp"))
    has_series = bool(parsed_fields.get("series_number"))
    keyword_score = len(matched_keywords)
    field_score = len(strong_fields) * 2 + len(optional_fields)

    passed = (
        (has_valid_cnp and (has_series or keyword_score >= 2))
        or (has_mrz and keyword_score >= 1 and field_score >= 2)
        or (has_series and keyword_score >= 3 and field_score >= 3)
    )

    return {
        "ok": passed,
        "matched_keywords": matched_keywords,
        "strong_fields": strong_fields,
        "optional_fields": optional_fields,
        "keyword_score": keyword_score,
        "field_score": field_score,
        "has_mrz": has_mrz,
        "message": None if passed else "The uploaded image does not look like a Romanian ID card.",
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
