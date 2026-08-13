import cv2
from utils.helper import draw_box
from config import VEHICLE_MODEL, SAMPLE_VIDEO
from modules.vehicle_detector import VehicleDetector
from modules.plate_detector import PlateDetector
from config import PLATE_MODEL
from modules.ocr import OCRReader
from modules.image_preprocessing import ImagePreprocessor
from utils.csv_writer import CSVWriter

preprocessor = ImagePreprocessor()
detector = VehicleDetector(VEHICLE_MODEL)
plate_detector = PlateDetector(PLATE_MODEL)
ocr_reader = OCRReader()
csv_writer = CSVWriter()

cap = cv2.VideoCapture("input/video/sample.mp4")

frame_number = 0

while True:

    ret, frame = cap.read()
    frame_number += 1

    if not ret:
        break

    results = detector.detect_and_track(frame)

    for box in results.boxes:

        x1, y1, x2, y2 = box.xyxy[0]

        vehicle_crop = frame[
            int(y1):int(y2),
            int(x1):int(x2)
        ]

        plate_results = plate_detector.detect(
            vehicle_crop
        )

        for plate_box in plate_results.boxes:

            confidence = float(
                plate_box.conf[0]
            )

            px1, py1, px2, py2 = plate_box.xyxy[0]

            plate_crop = vehicle_crop[
                int(py1):int(py2),
                int(px1):int(px2)
            ]

            h, w = plate_crop.shape[:2]

            if w < 20 or h < 10:
                continue

            plate_x1 = int(x1 + px1)
            plate_y1 = int(y1 + py1)
            plate_x2 = int(x1 + px2)
            plate_y2 = int(y1 + py2)

            cv2.rectangle(
                frame,
                (plate_x1, plate_y1),
                (plate_x2, plate_y2),
                (0, 255, 0),
                2
            )

            processed_plate = preprocessor.preprocess(
                plate_crop
            )

            ocr_results = ocr_reader.read_text(
                processed_plate
            )

            for result in ocr_results:

                text = result[1]
                confidence = result[2]

                if confidence > 0.4:

                    csv_writer.write(
                        frame_number,
                        int(box.id[0])
                        if box.id is not None else -1,
                        [
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2)
                        ],
                        [
                            plate_x1,
                            plate_y1,
                            plate_x2,
                            plate_y2
                        ],
                        text,
                        confidence
                    )

        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        track_id = (
            int(box.id[0])
            if box.id is not None
            else -1
        )

        label = (
            f"ID {track_id} | "
            f"{detector.model.names[class_id]} "
            f"{confidence:.2f}"
        )

        draw_box(
            frame,
            [x1, y1, x2, y2],
            label
        )

    cv2.imshow(
        "Vehicle Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

csv_writer.close()