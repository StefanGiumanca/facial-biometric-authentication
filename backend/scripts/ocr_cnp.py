

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

                if up_tok in label_words:
                    continue

                if any(s in up_tok for s in bad_substrings):
                    continue

                if len(up_tok) >= 2 and re.fullmatch(r"[A-Z-]+", up_tok):
                    good.append(up_tok)


        if not good:
            return None

        def score(tok: str):
            return (len(tok), 1 if '-' in tok else 0)

        return sorted(good, key=score, reverse=True)[0]


    mrz_last, mrz_first = extract_names_from_mrz(joined)

    last_name = mrz_last if mrz_last else pick_name(ln_up, ln_thr)
    first_name = mrz_first if mrz_first else pick_name(fn_up, fn_thr)


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
