from ultralytics import YOLO


class PlateDetector:

    def __init__(self, model_path):
        """
        Load the license plate detection model.
        """
        self.model = YOLO(model_path)

    def detect(self, vehicle_image):
        """
        Detect license plate inside a cropped vehicle image.

        Returns:
            YOLO Results
        """
        results = self.model(
            vehicle_image,
            verbose=False
        )[0]

        return results