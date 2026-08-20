import re


# ============================================================
# DEVANAGARI
# ============================================================

DEVANAGARI_RANGE = r"\u0900-\u097F"

DEVANAGARI_DIGITS = str.maketrans(
    "०१२३४५६७८९",
    "0123456789"
)


# ============================================================
# BASIC NORMALIZATION
# ============================================================

def normalize_plate_text(text):

    if text is None:
        return ""

    text = str(text).strip()

    # Devanagari digits -> English digits
    text = text.translate(DEVANAGARI_DIGITS)

    # Remove unwanted punctuation
    text = re.sub(
        r"[^A-Za-z0-9\u0900-\u097F ]",
        " ",
        text
    )

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# ENGLISH PLATE
# ============================================================

def validate_english_plate(text):

    text = normalize_plate_text(text)

    compact = text.replace(" ", "").upper()

    if not compact:
        return False, 0.0

    # English plates only
    if not re.fullmatch(r"[A-Z0-9]+", compact):
        return False, 0.0

    letters = len(
        re.findall(r"[A-Z]", compact)
    )

    digits = len(
        re.findall(r"[0-9]", compact)
    )

    # Must contain both
    if letters < 2 or digits < 2:
        return False, 0.0

    # Too short
    if len(compact) < 5:
        return False, 0.0

    # Too long
    if len(compact) > 10:
        return False, 0.0

    # --------------------------------------------------------
    # Validation score
    # --------------------------------------------------------

    score = 0.50

    # More letters = stronger
    if letters >= 2:
        score += 0.15

    if letters >= 3:
        score += 0.10

    # More digits = stronger
    if digits >= 2:
        score += 0.10

    if digits >= 4:
        score += 0.05

    score = min(score, 1.0)

    return True, score


# ============================================================
# NEPALI PLATE
# ============================================================

def validate_nepali_plate(text):

    text = normalize_plate_text(text)

    if not text:
        return False, 0.0

    # --------------------------------------------------------
    # Must contain Devanagari
    # --------------------------------------------------------

    devanagari_chars = re.findall(
        rf"[{DEVANAGARI_RANGE}]",
        text
    )

    if len(devanagari_chars) < 2:
        return False, 0.0

    # --------------------------------------------------------
    # Must contain digits
    # --------------------------------------------------------

    digits = re.findall(
        r"[0-9]",
        text
    )

    if len(digits) < 3:
        return False, 0.0

    # --------------------------------------------------------
    # Prevent OCR sentences
    # --------------------------------------------------------

    compact = text.replace(" ", "")

    if len(compact) > 16:
        return False, 0.0

    # --------------------------------------------------------
    # Prevent extremely short garbage
    # --------------------------------------------------------

    if len(compact) < 5:
        return False, 0.0

    # --------------------------------------------------------
    # Validation score
    # --------------------------------------------------------

    score = 0.50

    # Devanagari characters
    if len(devanagari_chars) >= 2:
        score += 0.15

    if len(devanagari_chars) >= 3:
        score += 0.10

    # Digits
    if len(digits) >= 3:
        score += 0.10

    if len(digits) >= 4:
        score += 0.05

    score = min(score, 1.0)

    return True, score


# ============================================================
# MAIN VALIDATOR
# ============================================================

def validate_plate(
    text,
    plate_type=None
):
    """
    Returns:

        (is_valid, validation_score)

    Example:

        (True, 0.85)
        (False, 0.0)
    """

    text = normalize_plate_text(text)

    if not text:
        return False, 0.0

    # --------------------------------------------------------
    # Explicit Nepali
    # --------------------------------------------------------

    if plate_type == "nepali":

        return validate_nepali_plate(
            text
        )

    # --------------------------------------------------------
    # Explicit English
    # --------------------------------------------------------

    if plate_type == "english":

        return validate_english_plate(
            text
        )

    # --------------------------------------------------------
    # Automatic script detection
    # --------------------------------------------------------

    if re.search(
        rf"[{DEVANAGARI_RANGE}]",
        text
    ):

        return validate_nepali_plate(
            text
        )

    return validate_english_plate(
        text
    )