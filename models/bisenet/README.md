# BiSeNet Models (Face Parsing)

BiSeNet (Bilateral Segmentation Network) is used for real-time semantic segmentation, particularly effective for face parsing tasks.

## Face Parsing with BiSeNet

### BiSeNet v2 Face Parsing
- **Description**: High-performance face parsing model for segmenting different facial regions
- **Format**: PyTorch (.pth), ONNX
- **Use Case**: Segment face into regions like skin, hair, eyes, nose, mouth, etc.

## Model Download

```bash
# Download BiSeNet face parsing model
mkdir -p models/bisenet/
cd models/bisenet/

# Option 1: Download from official repository
wget -O bisenet_v2_face_parsing.pth https://github.com/zllrunning/face-parsing.PyTorch/releases/download/v1.0/79999_iter.pth

# Option 2: Download ONNX version (if available)
wget -O bisenet_face_parsing.onnx https://github.com/onnx/models/raw/main/vision/body_analysis/face_parsing/face_parsing.onnx
```

## Usage Example

```python
import torch
import cv2
import numpy as np
from torchvision import transforms

# Load model (example for PyTorch)
model = torch.load('models/bisenet/bisenet_v2_face_parsing.pth')
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Face parsing
def parse_face(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    input_tensor = transform(img_rgb).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        parsing = output[0].argmax(0).cpu().numpy()
    
    return parsing

# Face regions mapping
FACE_REGIONS = {
    0: 'background',
    1: 'skin',
    2: 'l_brow',
    3: 'r_brow', 
    4: 'l_eye',
    5: 'r_eye',
    6: 'eye_g',
    7: 'l_ear',
    8: 'r_ear',
    9: 'ear_r',
    10: 'nose',
    11: 'mouth',
    12: 'u_lip',
    13: 'l_lip',
    14: 'neck',
    15: 'neck_l',
    16: 'cloth',
    17: 'hair',
    18: 'hat'
}
```

## Installation

Required dependencies are included in requirements.txt. For manual installation:

```bash
pip install torch torchvision
```

## Official Resources

- **GitHub**: https://github.com/zllrunning/face-parsing.PyTorch
- **Paper**: https://arxiv.org/abs/1808.00897
- **Demo**: https://github.com/zllrunning/face-parsing.PyTorch