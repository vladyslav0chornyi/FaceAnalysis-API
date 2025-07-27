# InsightFace Models

InsightFace is a state-of-the-art deep face analysis library providing face detection, recognition, and analysis capabilities.

## Available Models

### ArcFace (Face Recognition)
- **Description**: Deep face recognition model with excellent accuracy
- **Format**: ONNX
- **Download**: 
  ```bash
  # Download pre-trained ArcFace models
  wget -O arcface_r100_v1.onnx https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_r100_v1.onnx
  ```

### RetinaFace (Face Detection)  
- **Description**: Robust face detection model
- **Format**: ONNX
- **Download**:
  ```bash
  # Download RetinaFace model
  wget -O retinaface_r50_v1.onnx https://github.com/deepinsight/insightface/releases/download/v0.7/retinaface_r50_v1.onnx
  ```

### Gender & Age Recognition
- **Description**: Multi-task model for gender and age estimation
- **Format**: ONNX
- **Download**:
  ```bash
  # Download gender-age model
  wget -O genderage.onnx https://github.com/deepinsight/insightface/releases/download/v0.7/genderage.onnx
  ```

### Mask Detection
- **Description**: Face mask detection model
- **Format**: ONNX
- **Download**:
  ```bash
  # Download mask detection model
  wget -O mask_detector.onnx https://github.com/deepinsight/insightface/releases/download/v0.7/mask_detector.onnx
  ```

## Usage Example

```python
import insightface
from insightface.app import FaceAnalysis

# Initialize InsightFace
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Analyze faces in image
import cv2
img = cv2.imread('path/to/image.jpg')
faces = app.get(img)

for face in faces:
    print(f"Age: {face.age}, Gender: {face.gender}")
    print(f"Embedding shape: {face.embedding.shape}")
```

## Installation

InsightFace is already included in the project requirements. If you need to install it separately:

```bash
pip install insightface
```

## Official Resources

- **GitHub**: https://github.com/deepinsight/insightface
- **Documentation**: https://insightface.ai/
- **Paper**: https://arxiv.org/abs/1801.07698