from ultralytics import YOLO
import numpy as np

class YoloDetector:
    def __init__(self, model_path="models/yolo/yolo11n.pt"):
        self.model = YOLO(model_path)

    def detect(self, image: np.ndarray, conf=0.25, task="detect"):
        """
        task: "detect", "segment", "pose"
        Повертає список детекцій: [{'bbox': [...], 'conf': ..., 'class_id': ...}]
        Для task="segment" додається "mask"
        Для task="pose" додається "keypoints"
        """
        results = self.model(image, conf=conf, task=task)
        detections = []
        for i in range(len(results[0].boxes)):
            box = results[0].boxes.xyxy[i].cpu().numpy()
            cls = int(results[0].boxes.cls[i].cpu().numpy())
            score = float(results[0].boxes.conf[i].cpu().numpy())
            det = {
                "bbox": box.tolist(),
                "conf": score,
                "class_id": cls
            }
            if task == "segment" and hasattr(results[0], "masks"):
                det["mask"] = results[0].masks.data[i].cpu().numpy().tolist()
            if task == "pose" and hasattr(results[0], "keypoints"):
                det["keypoints"] = results[0].keypoints[i].cpu().numpy().tolist()
            detections.append(det)
        return detections