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
    last_score = name_similarity

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