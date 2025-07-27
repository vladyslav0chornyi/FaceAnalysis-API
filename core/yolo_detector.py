from ultralytics import YOLO
import numpy as np

class YoloDetector:
    def __init__(self, model_path="models/yolo/yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, image: np.ndarray, conf=0.25):
        """Повертає список детекцій: [{'bbox': [x1, y1, x2, y2], 'conf': score, 'cls': class_id}]"""
        results = self.model(image, conf=conf)
        detections = []
        for box, cls, score in zip(results[0].boxes.xyxy.cpu().numpy(),
                                   results[0].boxes.cls.cpu().numpy(),
                                   results[0].boxes.conf.cpu().numpy()):
            detections.append({
                "bbox": box.tolist(),
                "conf": float(score),
                "class_id": int(cls)
            })
        return detections