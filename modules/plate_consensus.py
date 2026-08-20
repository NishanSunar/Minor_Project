from difflib import SequenceMatcher

from modules.plate_validation import validate_plate


def normalize_for_compare(text):

    if not text:
        return ""

    return (
        str(text)
        .replace(" ", "")
        .upper()
    )


def similarity(a, b):

    a = normalize_for_compare(a)
    b = normalize_for_compare(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


def safe_validate(plate):

    try:

        result = validate_plate(plate)

        # New validation function:
        # (True, score)

        if isinstance(result, tuple):

            if len(result) >= 2:
                return (
                    bool(result[0]),
                    float(result[1])
                )

            if len(result) == 1:
                return (
                    bool(result[0]),
                    1.0 if result[0] else 0.0
                )

        # Old validation function:
        # True / False

        if isinstance(result, bool):

            return (
                result,
                1.0 if result else 0.0
            )

        return True, 0.5

    except Exception as e:

        print(
            f"[VALIDATION WARNING] "
            f"{type(e).__name__}: {e}"
        )

        # Do not crash ANPR because of validation.
        return True, 0.5


def add_reading(
    store,
    vehicle_id,
    plate,
    confidence,
    frame
):

    if not plate:
        return False

    try:
        vehicle_id = int(vehicle_id)
        confidence = float(confidence)

    except (
        TypeError,
        ValueError
    ):
        return False

    if confidence <= 0:
        return False

    is_valid, validation_score = safe_validate(
        plate
    )

    if not is_valid:

        print(
            f"[OCR REJECTED] "
            f"ID {vehicle_id} | "
            f"PLATE: {plate} | "
            f"CONF: {confidence:.2f}"
        )

        return False

    if vehicle_id not in store:
        store[vehicle_id] = []

    store[vehicle_id].append({

        "plate": str(
            plate
        ).strip(),

        "confidence": confidence,

        "validation_score": validation_score,

        "frame": int(frame)
    })

    return True


def choose_best_plate(readings):

    if not readings:
        return None

    candidates = []

    for current in readings:

        current_plate = current["plate"]
        current_conf = current["confidence"]

        validation_score = current.get(
            "validation_score",
            0.5
        )

        support_count = 0
        support_conf = 0.0

        for other in readings:

            sim = similarity(
                current_plate,
                other["plate"]
            )

            if sim >= 0.80:

                support_count += 1

                support_conf += (
                    other["confidence"]
                )

        if support_count:

            average_support = (
                support_conf /
                support_count
            )

        else:
            average_support = 0.0

        score = (
            0.50 * current_conf
            +
            0.30 * average_support
            +
            0.20 * validation_score
        )

        candidates.append({

            "plate": current_plate,

            "confidence": current_conf,

            "score": score,

            "frame": current["frame"],

            "support": support_count
        })

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda x: (
            x["score"],
            x["confidence"],
            x["support"]
        )
    )


def get_final_plate(
    store,
    vehicle_id
):

    try:
        vehicle_id = int(vehicle_id)
    except:
        return None

    readings = store.get(
        vehicle_id,
        []
    )

    if not readings:
        return None

    return choose_best_plate(
        readings
    )


def remove_vehicle_readings(
    store,
    vehicle_id
):

    try:
        vehicle_id = int(vehicle_id)
    except:
        return

    store.pop(
        vehicle_id,
        None
    )