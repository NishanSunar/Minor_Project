import cv2
import csv
import time
import queue
import threading
import os

from collections import defaultdict

from config import (
    VEHICLE_MODEL,
    PLATE_MODEL,
    SAMPLE_VIDEO
)

from modules.vehicle_detector import VehicleDetector
from modules.plate_detector import PlateDetector
from modules.ocr import OCRReader
from modules.image_preprocessing import ImagePreprocessor

from modules.plate_correction import correct_plate

from modules.plate_consensus import (
    add_reading,
    get_final_plate
)


# ============================================================
# CONFIG
# ============================================================

DETECT_EVERY = 2

ANPR_EVERY = 6

OCR_CONFIDENCE = 0.35

MAX_MISSED = 15

ANPR_QUEUE_SIZE = 8

PLATE_SMOOTH_ALPHA = 0.30

CSV_PATH = "output.csv"


# ============================================================
# MODELS
# ============================================================

detector = VehicleDetector(
    VEHICLE_MODEL
)

plate_detector = PlateDetector(
    PLATE_MODEL
)

ocr_reader = OCRReader()

preprocessor = ImagePreprocessor()


# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(
    SAMPLE_VIDEO
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

video_fps = cap.get(
    cv2.CAP_PROP_FPS
)

if video_fps <= 0:
    video_fps = 30.0


# ============================================================
# CSV
# ============================================================

csv_file = open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
)

csv_writer = csv.writer(
    csv_file
)

csv_writer.writerow([
    "Frame",
    "Vehicle ID",
    "Vehicle Box",
    "Plate Box",
    "Plate",
    "Confidence",
    "Consensus Score",
    "OCR Reads",
    "Result Type"
])

csv_file.flush()


def write_csv(
    frame,
    vehicle_id,
    vehicle_box,
    plate_box,
    plate,
    confidence,
    consensus_score,
    reads,
    result_type
):

    try:

        csv_writer.writerow([

            frame,

            vehicle_id,

            str(vehicle_box),

            str(plate_box),

            plate,

            round(
                float(confidence),
                4
            ),

            round(
                float(consensus_score),
                4
            ),

            reads,

            result_type
        ])

        csv_file.flush()

    except Exception as e:

        print(
            f"CSV error: "
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# SHARED DATA
# ============================================================

vehicles = {}

plate_readings = defaultdict(list)

data_lock = threading.Lock()

anpr_queue = queue.Queue(
    maxsize=ANPR_QUEUE_SIZE
)

running = True


# ============================================================
# SMOOTH ABSOLUTE PLATE BOX
# ============================================================

def smooth_box(
    old_box,
    new_box,
    alpha=PLATE_SMOOTH_ALPHA
):

    if old_box is None:
        return list(new_box)

    if new_box is None:
        return list(old_box)

    return [

        int(
            old_box[i]
            +
            alpha *
            (
                new_box[i]
                -
                old_box[i]
            )
        )

        for i in range(4)
    ]


# ============================================================
# FINAL RESULT
# ============================================================

def print_final_result(vehicle_id):

    try:

        result = get_final_plate(
            plate_readings,
            vehicle_id
        )

        if not result:
            return None

        readings = plate_readings.get(
            int(vehicle_id),
            []
        )

        vehicle = vehicles.get(
            vehicle_id
        )

        if vehicle:

            vehicle_box = vehicle.get(
                "box"
            )

            plate_box = vehicle.get(
                "plate"
            )

            last_frame = vehicle.get(
                "last_seen",
                0
            )

        else:

            vehicle_box = None
            plate_box = None
            last_frame = 0

        print(
            "\n"
            "====================================================\n"
            "FINAL VEHICLE RESULT\n"
            f"Vehicle ID       : {vehicle_id}\n"
            f"Plate            : {result['plate']}\n"
            f"Best Confidence  : "
            f"{result['confidence']:.3f}\n"
            f"Consensus Score  : "
            f"{result['score']:.3f}\n"
            f"OCR Reads        : "
            f"{len(readings)}\n"
            "===================================================="
        )

        write_csv(

            last_frame,

            vehicle_id,

            vehicle_box,

            plate_box,

            result["plate"],

            result["confidence"],

            result["score"],

            len(readings),

            "FINAL"
        )

        return result

    except Exception as e:

        print(
            f"Finalization error for ID "
            f"{vehicle_id}: "
            f"{type(e).__name__}: {e}"
        )

        return None


# ============================================================
# ANPR WORKER
# ============================================================

def anpr_worker():

    global running

    while running:

        try:

            job = anpr_queue.get(
                timeout=0.1
            )

        except queue.Empty:

            continue

        if job is None:

            anpr_queue.task_done()

            break

        track_id = job["id"]

        try:

            crop = job["crop"]

            vehicle_box = job[
                "vehicle_box"
            ]

            frame_number = job[
                "frame"
            ]

            # =================================================
            # PLATE DETECTION
            # =================================================

            result = plate_detector.detect(
                crop
            )

            best_plate = None

            best_plate_conf = 0.0

            for box in result.boxes:

                px1, py1, px2, py2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                px1 = max(
                    0,
                    px1
                )

                py1 = max(
                    0,
                    py1
                )

                px2 = min(
                    crop.shape[1],
                    px2
                )

                py2 = min(
                    crop.shape[0],
                    py2
                )

                if (
                    px2 <= px1
                    or
                    py2 <= py1
                ):
                    continue

                if (
                    px2 - px1 < 12
                    or
                    py2 - py1 < 6
                ):
                    continue

                conf = float(
                    box.conf[0]
                )

                if conf > best_plate_conf:

                    best_plate_conf = conf

                    best_plate = [
                        px1,
                        py1,
                        px2,
                        py2
                    ]

            # =================================================
            # NO PLATE
            # =================================================

            if best_plate is None:
                continue

            px1, py1, px2, py2 = best_plate

            # =================================================
            # CONVERT TO ABSOLUTE FRAME COORDINATES
            # =================================================

            vx1, vy1, vx2, vy2 = vehicle_box

            abs_plate = [

                vx1 + px1,

                vy1 + py1,

                vx1 + px2,

                vy1 + py2
            ]

            # =================================================
            # SAVE / SMOOTH PLATE BOX
            # =================================================

            with data_lock:

                vehicle = vehicles.get(
                    track_id
                )

                if vehicle is None:
                    continue

                old_plate = vehicle.get(
                    "plate"
                )

                smoothed = smooth_box(
                    old_plate,
                    abs_plate
                )

                vehicle["plate"] = smoothed

                vehicle["plate_conf"] = (
                    best_plate_conf
                )

            # =================================================
            # WRITE PLATE DETECTION TO CSV
            # =================================================

            write_csv(

                frame_number,

                track_id,

                vehicle_box,

                smoothed,

                "",

                best_plate_conf,

                0.0,

                0,

                "PLATE_DETECTION"
            )

            # =================================================
            # PLATE CROP
            # =================================================

            plate_crop = crop[
                py1:py2,
                px1:px2
            ]

            if plate_crop.size == 0:
                continue

            plate_crop = plate_crop.copy()

            # =================================================
            # PREPROCESS
            # =================================================

            processed = (
                preprocessor.preprocess(
                    plate_crop
                )
            )

            # =================================================
            # OCR
            # =================================================

            ocr_results = (
                ocr_reader.read_text(
                    processed
                )
            )

            if not ocr_results:
                continue

            # =================================================
            # SELECT BEST OCR RESULT
            # =================================================

            best_text = None

            best_ocr_conf = 0.0

            best_position = ""

            for item in ocr_results:

                try:

                    text = str(
                        item.get(
                            "text",
                            ""
                        )
                    ).strip()

                    confidence = float(
                        item.get(
                            "confidence",
                            0.0
                        )
                    )

                    position = item.get(
                        "position",
                        ""
                    )

                except Exception:
                    continue

                if not text:
                    continue

                if confidence < OCR_CONFIDENCE:
                    continue

                # Prefer valid-looking longer readings
                score = confidence

                if len(text) >= 4:
                    score += 0.05

                if (
                    position == "top"
                    and any(
                        "\u0900"
                        <= c
                        <= "\u097F"
                        for c in text
                    )
                ):
                    score += 0.05

                if (
                    position == "bottom"
                    and any(
                        c.isdigit()
                        for c in text
                    )
                ):
                    score += 0.03

                if (
                    best_text is None
                    or
                    score > best_ocr_conf
                ):

                    best_text = text

                    best_ocr_conf = (
                        confidence
                    )

                    best_position = (
                        position
                    )

            if not best_text:
                continue

            # =================================================
            # CORRECTION
            # =================================================

            corrected = correct_plate(
                best_text
            )

            if not corrected:
                continue

            # =================================================
            # CONSENSUS
            # =================================================

            with data_lock:

                accepted = add_reading(

                    plate_readings,

                    track_id,

                    corrected,

                    best_ocr_conf,

                    frame_number
                )

                current = (
                    get_final_plate(
                        plate_readings,
                        track_id
                    )
                )

                vehicle = vehicles.get(
                    track_id
                )

                if vehicle:

                    if current:

                        vehicle[
                            "plate_text"
                        ] = current[
                            "plate"
                        ]

                        vehicle[
                            "ocr_conf"
                        ] = current[
                            "confidence"
                        ]

                        vehicle[
                            "consensus_score"
                        ] = current[
                            "score"
                        ]

                    vehicle[
                        "plate_reads"
                    ] = len(
                        plate_readings[
                            track_id
                        ]
                    )

                    plate_box = vehicle.get(
                        "plate"
                    )

                else:

                    plate_box = smoothed

            # =================================================
            # ALWAYS WRITE OCR RESULT
            # =================================================

            write_csv(

                frame_number,

                track_id,

                vehicle_box,

                plate_box,

                corrected,

                best_ocr_conf,

                (
                    current["score"]
                    if current
                    else best_ocr_conf
                ),

                len(
                    plate_readings[
                        track_id
                    ]
                ),

                "OCR"
            )

            print(
                f"[OCR] "
                f"ID {track_id} | "
                f"RAW: {best_text} | "
                f"CORRECTED: {corrected} | "
                f"CONF: {best_ocr_conf:.2f} | "
                f"POS: {best_position}"
            )

        except Exception as e:

            print(
                f"ANPR error for ID "
                f"{track_id}: "
                f"{type(e).__name__}: {e}"
            )

        finally:

            anpr_queue.task_done()


# ============================================================
# START WORKER
# ============================================================

worker = threading.Thread(
    target=anpr_worker,
    daemon=True
)

worker.start()


# ============================================================
# FPS
# ============================================================

frame_number = 0

fps_count = 0

fps_start = time.time()

processing_fps = 0.0


# ============================================================
# DRAW VEHICLE
# ============================================================

def draw_vehicle(
    frame,
    track_id,
    data
):

    x1, y1, x2, y2 = data["box"]

    cv2.rectangle(

        frame,

        (x1, y1),

        (x2, y2),

        (0, 220, 0),

        2
    )

    label = (
        f"ID {track_id} "
        f"{data['class']} "
        f"{data['conf']:.2f}"
    )

    cv2.putText(

        frame,

        label,

        (
            x1,
            max(
                25,
                y1 - 8
            )
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (0, 255, 255),

        2,

        cv2.LINE_AA
    )


# ============================================================
# DRAW PLATE
# ============================================================

def draw_plate(
    frame,
    vehicle
):

    plate = vehicle.get(
        "plate"
    )

    if plate is None:
        return

    # IMPORTANT:
    # plate is now ABSOLUTE frame coordinates

    px1, py1, px2, py2 = map(
        int,
        plate
    )

    px1 = max(
        0,
        min(
            frame.shape[1] - 1,
            px1
        )
    )

    py1 = max(
        0,
        min(
            frame.shape[0] - 1,
            py1
        )
    )

    px2 = max(
        0,
        min(
            frame.shape[1] - 1,
            px2
        )
    )

    py2 = max(
        0,
        min(
            frame.shape[0] - 1,
            py2
        )
    )

    if (
        px2 <= px1
        or
        py2 <= py1
    ):
        return

    # ========================================================
    # PLATE BOX
    # ========================================================

    cv2.rectangle(

        frame,

        (px1, py1),

        (px2, py2),

        (255, 120, 0),

        2
    )

    # ========================================================
    # PLATE TEXT
    # ========================================================

    plate_text = vehicle.get(
        "plate_text"
    )

    if not plate_text:
        return

    conf = vehicle.get(
        "ocr_conf",
        0.0
    )

    reads = vehicle.get(
        "plate_reads",
        0
    )

    label = (
        f"{plate_text} "
        f"{conf:.2f}"
    )

    if reads > 1:
        label += f" [{reads}]"

    cv2.putText(

        frame,

        label,

        (
            px1,
            max(
                20,
                py1 - 8
            )
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 120, 0),

        2,

        cv2.LINE_AA
    )


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # ====================================================
        # VEHICLE DETECTION
        # ====================================================

        if (
            frame_number
            % DETECT_EVERY
            == 0
        ):

            results = (
                detector.detect_and_track(
                    frame
                )
            )

            with data_lock:

                for box in results.boxes:

                    if box.id is None:
                        continue

                    track_id = int(
                        box.id[0]
                    )

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[
                            0
                        ].tolist()
                    )

                    x1 = max(
                        0,
                        x1
                    )

                    y1 = max(
                        0,
                        y1
                    )

                    x2 = min(
                        frame.shape[1],
                        x2
                    )

                    y2 = min(
                        frame.shape[0],
                        y2
                    )

                    if (
                        x2 <= x1
                        or
                        y2 <= y1
                    ):
                        continue

                    confidence = float(
                        box.conf[0]
                    )

                    class_id = int(
                        box.cls[0]
                    )

                    class_name = (
                        detector.model.names[
                            class_id
                        ]
                    )

                    # ========================================
                    # NEW VEHICLE
                    # ========================================

                    if track_id not in vehicles:

                        vehicles[
                            track_id
                        ] = {

                            "box": [
                                x1,
                                y1,
                                x2,
                                y2
                            ],

                            "class":
                                class_name,

                            "conf":
                                confidence,

                            "plate":
                                None,

                            "plate_conf":
                                0.0,

                            "plate_text":
                                None,

                            "ocr_conf":
                                0.0,

                            "consensus_score":
                                0.0,

                            "plate_reads":
                                0,

                            "last_anpr":
                                frame_number
                                - ANPR_EVERY,

                            "last_seen":
                                frame_number
                        }

                    else:

                        vehicle = vehicles[
                            track_id
                        ]

                        vehicle["box"] = [
                            x1,
                            y1,
                            x2,
                            y2
                        ]

                        vehicle["class"] = (
                            class_name
                        )

                        vehicle["conf"] = (
                            confidence
                        )

                        vehicle["last_seen"] = (
                            frame_number
                        )

                    # ========================================
                    # ANPR
                    # ========================================

                    vehicle = vehicles[
                        track_id
                    ]

                    if (
                        frame_number
                        -
                        vehicle["last_anpr"]
                        >= ANPR_EVERY
                    ):

                        if not anpr_queue.full():

                            crop = frame[
                                y1:y2,
                                x1:x2
                            ]

                            if crop.size:

                                try:

                                    anpr_queue.put_nowait({

                                        "id":
                                            track_id,

                                        "crop":
                                            crop.copy(),

                                        "vehicle_box":
                                            [
                                                x1,
                                                y1,
                                                x2,
                                                y2
                                            ],

                                        "frame":
                                            frame_number
                                    })

                                    vehicle[
                                        "last_anpr"
                                    ] = (
                                        frame_number
                                    )

                                except queue.Full:
                                    pass

        # ====================================================
        # REMOVE OLD VEHICLES
        # ====================================================

        with data_lock:

            old_ids = []

            for track_id in list(
                vehicles.keys()
            ):

                if (
                    frame_number
                    -
                    vehicles[
                        track_id
                    ][
                        "last_seen"
                    ]
                    > MAX_MISSED
                ):

                    old_ids.append(
                        track_id
                    )

            for track_id in old_ids:

                print_final_result(
                    track_id
                )

                vehicles.pop(
                    track_id,
                    None
                )

        # ====================================================
        # DRAW
        # ====================================================

        with data_lock:

            for (
                track_id,
                vehicle
            ) in list(
                vehicles.items()
            ):

                draw_vehicle(
                    frame,
                    track_id,
                    vehicle
                )

                draw_plate(
                    frame,
                    vehicle
                )

        # ====================================================
        # FPS
        # ====================================================

        fps_count += 1

        elapsed = (
            time.time()
            -
            fps_start
        )

        if elapsed >= 1:

            processing_fps = (
                fps_count
                /
                elapsed
            )

            fps_count = 0

            fps_start = time.time()

        cv2.putText(

            frame,

            f"VIDEO FPS: "
            f"{video_fps:.1f}",

            (20, 30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (0, 255, 255),

            2
        )

        cv2.putText(

            frame,

            f"PROCESS FPS: "
            f"{processing_fps:.1f}",

            (20, 58),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (0, 255, 255),

            2
        )

        cv2.imshow(
            "Nepali ANPR",
            frame
        )

        if (
            cv2.waitKey(1)
            &
            0xFF
            ==
            27
        ):
            break


finally:

    # ========================================================
    # FINALIZE VEHICLES
    # ========================================================

    with data_lock:

        remaining = list(
            vehicles.keys()
        )

    for track_id in remaining:

        print_final_result(
            track_id
        )

    running = False

    try:

        anpr_queue.put_nowait(
            None
        )

    except queue.Full:

        # Let worker finish queued jobs
        pass

    worker.join(
        timeout=5
    )

    cap.release()

    cv2.destroyAllWindows()

    csv_file.flush()

    csv_file.close()

    print(
        "\nCSV saved to: "
        f"{os.path.abspath(CSV_PATH)}"
    )