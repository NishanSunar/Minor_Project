from ultralytics import YOLO
from config import VEHICLE_CLASSES


class VehicleDetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_and_track(self, frame):
        return self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,
            imgsz=640,
            verbose=False
        )[0]