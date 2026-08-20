from ultralytics import YOLO


class PlateDetector:

    def __init__(self, model_path):

        self.model = YOLO(model_path)

    def detect(self, image):

        if image is None or image.size == 0:
            return None

        return self.model.predict(
            source=image,
            conf=0.15,
            iou=0.45,
            imgsz=960,
            max_det=10,
            verbose=False
        )[0]