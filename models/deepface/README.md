# DeepFace Models

DeepFace is a comprehensive face analysis library providing emotion recognition, age estimation, gender classification, and race detection.

## Available Analysis Types

### Emotion Recognition
- **Models**: FER (Facial Expression Recognition), VGG-Face, OpenFace
- **Emotions**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Accuracy**: State-of-the-art emotion classification

### Age Estimation  
- **Models**: Age-Net, VGG-Face
- **Range**: Continuous age estimation (0-100 years)
- **Use Case**: Demographic analysis, age verification

### Gender Classification
- **Models**: Gender-Net, VGG-Face
- **Classes**: Male, Female
- **Accuracy**: High-performance gender classification

### Race Detection
- **Models**: Race-Net, VGG-Face  
- **Classes**: Asian, Black, Indian, Latino Hispanic, Middle Eastern, White
- **Use Case**: Demographic analysis, bias detection

## Models Download

DeepFace automatically downloads models on first use. Models are stored in:

```bash
# Default model directory (automatically created)
~/.deepface/weights/

# Or check current location
python -c "from deepface import DeepFace; import os; print(os.path.expanduser('~/.deepface/weights/'))"
```

## Usage Examples

### Basic Analysis

```python
from deepface import DeepFace
import cv2

# Analyze single image
result = DeepFace.analyze(
    img_path="path/to/image.jpg",
    actions=['age', 'gender', 'emotion', 'race'],
    enforce_detection=False
)

print(f"Age: {result[0]['age']}")
print(f"Gender: {result[0]['dominant_gender']}")
print(f"Emotion: {result[0]['dominant_emotion']}")
print(f"Race: {result[0]['dominant_race']}")
```

### Advanced Analysis

```python
import numpy as np
from deepface import DeepFace

# Analyze cropped face directly
def analyze_cropped_face(cropped_face_array):
    """
    Analyze cropped face from numpy array
    Args:
        cropped_face_array: numpy array of cropped face (BGR format)
    """
    try:
        result = DeepFace.analyze(
            img_path=cropped_face_array,
            actions=['age', 'gender', 'emotion', 'race'],
            enforce_detection=False,
            detector_backend='skip'  # Skip detection since face is already cropped
        )
        
        return {
            'age': result[0]['age'],
            'gender': result[0]['dominant_gender'],
            'gender_confidence': result[0]['gender'][result[0]['dominant_gender']],
            'emotion': result[0]['dominant_emotion'],
            'emotion_confidence': result[0]['emotion'][result[0]['dominant_emotion']],
            'race': result[0]['dominant_race'],
            'race_confidence': result[0]['race'][result[0]['dominant_race']]
        }
    except Exception as e:
        print(f"DeepFace analysis error: {e}")
        return None

# Batch processing
def analyze_multiple_faces(image_paths):
    results = []
    for path in image_paths:
        try:
            result = DeepFace.analyze(
                img_path=path,
                actions=['emotion', 'age', 'gender'],
                enforce_detection=True
            )
            results.append(result)
        except Exception as e:
            print(f"Failed to analyze {path}: {e}")
            results.append(None)
    
    return results
```

### Face Recognition

```python
from deepface import DeepFace

# Face verification (1:1 matching)
result = DeepFace.verify(
    img1_path="person1.jpg",
    img2_path="person2.jpg",
    model_name="VGG-Face",
    distance_metric="cosine"
)

print(f"Same person: {result['verified']}")
print(f"Distance: {result['distance']}")

# Face recognition (1:N matching)
dfs = DeepFace.find(
    img_path="target.jpg",
    db_path="database/",
    model_name="VGG-Face"
)
```

### Custom Model Backends

```python
# Available models for each task
models = {
    'emotion': ['VGG-Face', 'OpenFace', 'Facenet', 'DeepFace'],
    'age': ['VGG-Face', 'OpenFace', 'Facenet', 'DeepFace'],
    'gender': ['VGG-Face', 'OpenFace', 'Facenet', 'DeepFace'],
    'race': ['VGG-Face', 'OpenFace', 'Facenet', 'DeepFace']
}

# Use specific model
result = DeepFace.analyze(
    img_path="image.jpg",
    actions=['emotion'],
    models={'emotion': 'VGG-Face'}
)
```

## Integration with Project

The current project already uses DeepFace in `main.py`:

```python
from deepface import DeepFace

def deepface_analyze_func(cropped_face):
    return DeepFace.analyze(
        cropped_face, 
        actions=['age', 'gender', 'emotion', 'race'], 
        enforce_detection=False
    )
```

## Performance Optimization

```python
# For better performance, use detector_backend='skip' for pre-cropped faces
result = DeepFace.analyze(
    img_path=cropped_face,
    actions=['emotion', 'age', 'gender'],
    detector_backend='skip',  # Skip face detection
    enforce_detection=False
)
```

## Installation

DeepFace is already included in the project requirements. For manual installation:

```bash
pip install deepface
```

## Official Resources

- **GitHub**: https://github.com/serengil/deepface
- **PyPI**: https://pypi.org/project/deepface/
- **Documentation**: https://github.com/serengil/deepface/blob/master/README.md
- **Paper**: https://arxiv.org/abs/2005.06881