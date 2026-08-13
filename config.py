import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "input")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
VIDEO_DIR = os.path.join(INPUT_DIR, "video")
SAMPLE_VIDEO = os.path.join(VIDEO_DIR, "sample.mp4")
VEHICLE_MODEL = os.path.join(MODEL_DIR, "yolov8n.pt")
VEHICLE_CLASSES = [2, 3, 5, 7]
PLATE_MODEL=os.path.join(MODEL_DIR, "license_plate_detector.pt")