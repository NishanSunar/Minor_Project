import re
import unicodedata


# ============================================================
# DEVANAGARI DIGITS
# ============================================================

DEV_TO_ENG = str.maketrans(
    "०१२३४५६७८९",
    "0123456789"
)


# ============================================================
# OCR CONFUSION MAPS
# ============================================================

LETTER_TO_DIGIT = {
    "O": "0",
    "D": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
}

DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}


# ============================================================
# BASIC CLEANING
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = unicodedata.normalize(
        "NFC",
        text
    )

    # Remove OCR brackets
    text = re.sub(
        r"[\[\]\(\)\{\}\"'`]",
        "",
        text
    )

    # Remove punctuation but preserve
    # English + Devanagari + numbers + spaces
    text = re.sub(
        r"[^A-Za-z0-9\u0900-\u097F ]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# NORMALIZE DEVANAGARI DIGITS
# ============================================================

def normalize_digits(text):

    if not text:
        return ""

    return text.translate(
        DEV_TO_ENG
    )


# ============================================================
# ENGLISH PLATE
# ============================================================

def correct_english_plate(
    text,
    pattern="LLLDDDD"
):

    text = clean_text(text)

    text = text.replace(
        " ",
        ""
    ).upper()

    if not text:
        return ""

    # If OCR has extra characters,
    # don't try dangerous corrections.
    if len(text) != len(pattern):
        return text

    corrected = []

    for char, expected in zip(
        text,
        pattern
    ):

        # ----------------------------------------------------
        # Expected LETTER
        # ----------------------------------------------------

        if expected == "L":

            if char.isdigit():

                char = DIGIT_TO_LETTER.get(
                    char,
                    char
                )

        # ----------------------------------------------------
        # Expected DIGIT
        # ----------------------------------------------------

        elif expected == "D":

            if char.isalpha():

                char = LETTER_TO_DIGIT.get(
                    char,
                    char
                )

        corrected.append(
            char
        )

    return "".join(
        corrected
    )


# ============================================================
# NEPALI PLATE
# ============================================================

def correct_nepali_plate(text):

    """
    Nepali plate OCR correction.

    We normalize Devanagari digits internally:

        ०१२३४५६७८९
            ↓
        0123456789

    Devanagari letters are NOT blindly replaced.
    """

    text = clean_text(text)

    text = normalize_digits(
        text
    )

    # Remove spaces around groups
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SCRIPT DETECTION
# ============================================================

def contains_devanagari(text):

    if not text:
        return False

    return bool(
        re.search(
            r"[\u0900-\u097F]",
            text
        )
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def correct_plate(
    text,
    plate_type=None
):

    text = clean_text(text)

    if not text:
        return ""

    # Explicit Nepali
    if plate_type == "nepali":

        return correct_nepali_plate(
            text
        )

    # Explicit English
    if plate_type == "english":

        return correct_english_plate(
            text
        )

    # Automatic detection
    if contains_devanagari(text):

        return correct_nepali_plate(
            text
        )

    return correct_english_plate(
        text
    )