# ...existing code...
from ultralytics import YOLO

try:
    _yolo_model = YOLO("yolov8n-face.pt")
    yolo_available = True
except Exception as e:
    print(f"YOLO model load error: {e}")
    _yolo_model = None
    yolo_available = False

def detect_faces(img_rgb):
    """img_rgb: numpy array (H,W,3) in RGB -> list of (x1,y1,x2,y2)"""
    if not yolo_available:
        return []
    results = _yolo_model(img_rgb)[0]
    boxes = []
    if hasattr(results, "boxes"):
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            boxes.append((x1, y1, x2, y2))
    return boxes
# ...existing code...
