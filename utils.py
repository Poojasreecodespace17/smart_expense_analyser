import cv2
import numpy as np
import pytesseract
import re
import pandas as pd
from classifier import classify_item

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ══════════════════════════════════════════════════════════════════════════════
# OCR PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _upscale(img):
    h, w = img.shape[:2]
    if min(h, w) < 1200:
        scale = 1200 / min(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)
    return img

def _deskew(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                             minLineLength=100, maxLineGap=10)
    if lines is None:
        return img
    angles = [np.degrees(np.arctan2(y2-y1, x2-x1))
              for x1,y1,x2,y2 in lines[:,0]
              if x2 != x1 and abs(np.degrees(np.arctan2(y2-y1,x2-x1))) < 10]
    if not angles:
        return img
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2),
                                 np.median(angles), 1.0)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

def _preprocess(img):
    img  = _upscale(img)
    img  = _deskew(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    blur = cv2.GaussianBlur(gray, (0,0), 3)
    gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 15)

def extract_text(image):
    """Try 9 combinations of image variant × PSM, return highest-confidence text."""
    proc = _preprocess(image)
    inv  = cv2.bitwise_not(proc)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, ot = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best, best_score = "", -1.0
    for variant in (proc, inv, ot):
        for psm in (6, 4, 11):
            try:
                raw = pytesseract.image_to_string(variant,
                                                   config=f"--oem 3 --psm {psm}",
                                                   lang="eng")
                df  = pytesseract.image_to_data(variant,
                                                 config=f"--oem 3 --psm {psm}",
                                                 lang="eng",
                                                 output_type=pytesseract.Output.DATAFRAME)
                score = df[df["conf"] >= 50]["conf"].mean() if not df.empty else 0.0
                if score > best_score:
                    best_score, best = score, raw
            except Exception:
                continue
    return _clean_raw_text(best)

def _clean_raw_text(text):
    text = text.replace("\x0c", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<=\d)[Oo](?=\d)", "0", text)
    text = re.sub(r"(?<=\d)[Il](?=\d)", "1", text)
    return "\n".join(l.strip() for l in text.splitlines()).strip()


# ══════════════════════════════════════════════════════════════════════════════
# ROW-WISE EXTRACTION (bounding-box grouping)
# ══════════════════════════════════════════════════════════════════════════════

def extract_rows(image):
    """Return one string per visual line using Tesseract word bounding boxes."""
    proc = _preprocess(image)
    try:
        df = pytesseract.image_to_data(proc, config="--oem 3 --psm 6",
                                        lang="eng",
                                        output_type=pytesseract.Output.DATAFRAME)
    except Exception:
        return []
    df = df[df["conf"] >= 30].copy()
    df = df[df["text"].notna()]
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    rows = []
    for _, grp in df.groupby(["block_num","par_num","line_num"], sort=True):
        line = " ".join(grp.sort_values("left")["text"].tolist()).strip()
        if line:
            rows.append(line)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# OCR CORRECTION  (thermal-printer abbreviation → real word)
# ══════════════════════════════════════════════════════════════════════════════

_CORRECTIONS = [
    # Milk / dairy
    (r'\bWILKMRID\b',      'MILKMAID'),
    (r'\bCOND\b',          'CONDENSED'),
    (r'\bENSED\b',         ''),
    (r'\bWILK\b',          'MILK'),
    (r'\bVILK\b',          'MILK'),
    (r'\bWlkske?\b',       'MILKSHAKE'),
    (r'\bBelgianChocol\b', 'BELGIAN CHOCOLATE'),
    # Chocolate
    (r'\bIALND\b',         'ISLAND'),
    (r'\bALND\b',          'ALMOND'),
    (r'\bCHOCLT\b',        'CHOCOLATE'),
    (r'\bCHOCL\b',         'CHOCOLATE'),
    (r'\bDAIRY MLK\b',     'DAIRY MILK'),
    (r'\bMLK\b',           'MILK'),
    # Vermicelli / staples
    (r'\bVERNICELLI\b',    'VERMICELLI'),
    (r'\bVERNICELL\b',     'VERMICELLI'),
    # Hing
    (r'\bCONPOUNDED\b',    'COMPOUNDED'),
    (r'\bHING P\b',        'HING'),
    (r'\bOWDER\b',         'POWDER'),
    # Soap
    (r'\bSOA\b',           'SOAP'),
    # Toothpaste
    (r'\bCOLGA\b',         'COLGATE'),
    (r'\bTHPST\b',         'TOOTHPASTE'),
    (r'\bDNTLCRN?\b',      ''),
    (r'\bSTRNG\b',         'STRONG'),
    (r'\bTTH\b',           'TEETH'),
    # Dishwash
    (r'\bVIN\b',           'VIM'),
    (r'\b0ISVASH\b',       'DISHWASH'),
    (r'\bDISAVPSH\b',      'DISHWASH'),
    (r'\bIQUID\b',         'LIQUID'),
    (r'\bPITANBARI\b',     'PITAMBARI'),
    (r'\bSHINLNODI\b',     'SHINING'),
    (r'\bSHWS\b',          'DISHWASH'),
    (r'\bPOVDER\b',        'POWDER'),
    # Handwash
    (r'\bG0DREJ\b',        'GODREJ'),
    (r'\bPRTKTMGCPML\b',   'PROTEKT'),
    (r'\bIO HW\b',         'HANDWASH'),
    (r'\bHW\b',            'HANDWASH'),
    (r'\bLME\b',           'LIME'),
    # Shampoo / hair colour
    (r'\bESY\b',           'EASY'),
    (r'\bSHNP\b',          'SHAMPOO'),
    (r'\bSHMP\b',          'SHAMPOO'),
    (r'\bNTRL\b',          'NATURAL'),
    (r'\bBLK\b',           'BLACK'),
    # Floor cleaner
    (r'\bB1GK\b',          'BLOCK'),
    (r'\bFNT SC\b',        'FRESH'),
    (r'\bWY HM\b',         'MY HOME'),
    (r'\bMOPZFRS?\b',      'MOPZ FRESH'),
    # Noise / packaging codes
    (r'\b1008\b',          ''),
    (r'\b1908\b',          '190G'),
    (r'\b18NL\b',          '18ML'),
    (r'\bC6ST\b',          ''),
    (r'\bS6ST\b',          ''),
    (r'\b2\)\b',           ''),
    (r'\b(CBD\d*|PCH|TPK|PET|PPZ|PP\b|PPS|PEJ|4P|RS\b)\b', ''),
]

def _ocr_correct(name: str) -> str:
    """Apply the abbreviation correction map to a raw OCR product name."""
    for pattern, replacement in _CORRECTIONS:
        name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
    return re.sub(r'\s{2,}', ' ', name).strip()


# ══════════════════════════════════════════════════════════════════════════════
# HSN-ANCHORED ITEM EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
# Why HSN anchoring?
#   Tesseract extracts columns independently — prices end up in one block,
#   names in another, quantities in a third, all as separate rows.
#   A simple line-by-line regex cannot pair them correctly.
#
# Solution:
#   1. Use the 6-digit HSN code that starts every product line as an anchor.
#   2. Grab that line + any following continuation lines (no HSN prefix,
#      has alphabetic content) to reconstruct the full product name.
#   3. Pull prices exclusively from the "Value" column block
#      (the second price column, after the "Value" header row).
#   4. Pair names[i] ↔ values[i] by sequence.

_HSN_RE    = re.compile(r'^(\d{6})\s+(.+)$')
_INLINE_RE = re.compile(r'^(\d{6})\s+(.+?)\s+(\d{1,5}\.\d{2})\s*$')
_IS_PRICE  = re.compile(r'^\s*\d{1,5}\.\d{1,2}\s*$')
_IS_QTY    = re.compile(r'^\s*\d{1,2}\s*$')
_SKIP_RE   = re.compile(
    r'\b(cgst|sgst|igst|gst|tax|invoice|total|discount|gross|net|amount|'
    r'paid|cash|upi|jiopay|bill|pos|store|cashier|date|place|supply|'
    r'customer|qty|oty|value|price|savings|receipt|inclusive|original|'
    r'recipient|state|code|type|urd|items|met price|wet |sross|iotal|'
    r'anount|komarapalayam|azaar|hoppine|azaar)\b',
    re.IGNORECASE
)

def _is_continuation(row: str) -> bool:
    """True if a row looks like the second line of a wrapped product name."""
    r = row.strip()
    if not r or _HSN_RE.match(r) or _IS_PRICE.match(r) or _SKIP_RE.search(r):
        return False
    if re.fullmatch(r'[\d\s.]+', r):
        return False
    return sum(c.isalpha() for c in r) / max(len(r), 1) > 0.25


def extract_items(text: str) -> list[dict]:
    """
    Parse items from raw OCR text using HSN anchoring + Value-column pairing.

    Works correctly even when Tesseract scrambles columns into separate rows.
    """
    rows = [r.strip() for r in text.split("\n") if r.strip()]

    # ── 1. Find the Value column price block ──────────────────────────────────
    val_start = None
    for i, r in enumerate(rows):
        if r.lower() == "value":
            val_start = i + 1
            break

    value_prices = []
    if val_start is not None:
        for r in rows[val_start:]:
            if _IS_PRICE.match(r):
                v = float(r.strip())
                if v < 500:               # >500 = subtotal / grand total
                    value_prices.append(v)
            elif re.match(r'0?ty:', r, re.I):
                break                     # "0ty:22" marks end of value column
            elif r == "-Amount (INR)":
                continue                  # noise row

    # ── 2. Find end of item-name section (before qty/value columns) ───────────
    section_end = next(
        (i for i, r in enumerate(rows) if r.lower() in ("oty", "qty", "value")),
        len(rows)
    )

    # ── 3. Reconstruct product names from HSN-anchored rows ──────────────────
    hsn_positions = [i for i, r in enumerate(rows[:section_end])
                     if _HSN_RE.match(r)]

    raw_names = []
    for pos in hsn_positions:
        row = rows[pos]

        # Case A: price already on the same row → inline item
        m_inline = _INLINE_RE.match(row)
        if m_inline:
            raw_names.append(m_inline.group(2).strip())
            continue

        # Case B: name only on this row, possibly continues on next line(s)
        m = _HSN_RE.match(row)
        name = m.group(2).strip() if m else row

        j = pos + 1
        while j < section_end and _is_continuation(rows[j]):
            if _HSN_RE.match(rows[j]):
                break
            name = name + " " + rows[j]
            j += 1

        raw_names.append(name.strip())

    # ── 4. Fallback: if no HSN codes found, use improved line-by-line parse ──
    if not raw_names:
        return _extract_items_fallback(text)

    # ── 5. Pair names ↔ prices ────────────────────────────────────────────────
    if not value_prices:
        # No Value column found — try to pull prices from inline or net-price col
        value_prices = _extract_fallback_prices(rows, section_end)

    items = []
    for name, price in zip(raw_names, value_prices):
        corrected = _ocr_correct(name)
        if len(corrected) >= 4:
            items.append({"name": corrected, "price": price})

    return items


def _extract_fallback_prices(rows, section_end):
    """Extract net prices (first price column) when Value column is absent."""
    prices = []
    inline_price_re = re.compile(r'^.*?\s+(\d{1,5}\.\d{2})\s*$')
    for r in rows[:section_end]:
        if _SKIP_RE.search(r):
            continue
        if _IS_PRICE.match(r):
            v = float(r.strip())
            if 0 < v < 500:
                prices.append(v)
            continue
        m = inline_price_re.match(r)
        if m and not _HSN_RE.match(r):
            v = float(m.group(1))
            if 0 < v < 500:
                prices.append(v)
    return prices


def _extract_items_fallback(text: str) -> list[dict]:
    """
    Line-by-line fallback for clean PDF text where columns are not scrambled.
    """
    _SKIP = re.compile(
        r'\b(cgst|sgst|igst|gst|hsn|tax|invoice|total|discount|gross|net|'
        r'amount|paid|cash|upi|jiopay|bill|pos|store|cashier|date|place|'
        r'supply|customer|qty|oty|value|price|savings|receipt|inclusive|'
        r'original|recipient|state|code|type|urd|items)\b',
        re.IGNORECASE
    )
    items   = []
    pending = ""
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line or _SKIP.search(line):
            pending = ""
            continue
        line = re.sub(r'^\d{6}\s*', '', line).strip()
        m = re.search(r'^(.+?)\s+(\d{1,5}(?:\.\d{1,2})?)\s*(?:\d+\s+[\d.]+)?\s*$', line)
        if m:
            name  = (pending + " " + m.group(1)).strip() if pending else m.group(1).strip()
            price = float(m.group(2))
            pending = ""
            name = _ocr_correct(re.sub(r'\s{2,}', ' ', name))
            if len(name) >= 4 and not re.fullmatch(r'[\d\s.]+', name) and 0 < price <= 10000:
                items.append({"name": name, "price": price})
        else:
            alpha = sum(c.isalpha() for c in line) / max(len(line), 1)
            pending = line if alpha > 0.4 and len(line) >= 4 else ""
    return items


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def analyze_expense_ml(text: str) -> dict:
    items = extract_items(text)
    for item in items:
        try:
            item["category"] = classify_item(item["name"])
        except Exception:
            item["category"] = "Uncategorised"
    total = round(sum(i["price"] for i in items), 2)
    return {"items": items, "total": total}


def save_expense(items: list) -> None:
    if not items:
        return
    required = {"name", "price", "category"}
    valid = [i for i in items if required.issubset(i.keys())]
    if not valid:
        return
    df = pd.DataFrame(valid)
    try:
        existing = pd.read_csv("expenses.csv")
        df = pd.concat([existing, df], ignore_index=True)
    except FileNotFoundError:
        pass
    except pd.errors.EmptyDataError:
        pass
    except Exception as e:
        print(f"[save_expense] {e}")
    df.to_csv("expenses.csv", index=False)


def clean_text(text: str) -> str:
    return text.lower().strip()