from ultralytics import YOLO
from config import VEHICLE_CLASSES
class VehicleDetector:
    
    def __init__(self,model_path):
        """_summary_
        Load the Yolo model once when the object is created.
        """
        self.model = YOLO(model_path)
        
        
    def detect_and_track(self,frame):
        """
        Detect all objects in a frame.
        Returns YOLO results.
        """
        results = self.model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=VEHICLE_CLASSES,
        verbose=False
        )[0]
        
        return results