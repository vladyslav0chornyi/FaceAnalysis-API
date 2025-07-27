# MediaPipe Models

MediaPipe provides robust machine learning solutions for face detection, face mesh, and face landmarks detection.

## Available Models

### FaceMesh
- **Description**: High-fidelity 3D face landmarks detection (468 landmarks)
- **Format**: TensorFlow Lite (.tflite)
- **Use Case**: Face mesh generation, facial expression analysis, AR/VR applications

### Face Detection
- **Description**: Fast and accurate face detection
- **Format**: TensorFlow Lite (.tflite)  
- **Use Case**: Real-time face detection in images and video

### Face Landmarks
- **Description**: 2D facial landmarks detection (68 key points)
- **Format**: TensorFlow Lite (.tflite)
- **Use Case**: Facial feature analysis, emotion recognition

## Models Download

MediaPipe models are automatically downloaded when using the library. For manual download:

```bash
# Create MediaPipe models directory
mkdir -p models/mediapipe/

# Models are automatically downloaded by MediaPipe
# But you can find them in MediaPipe installation directory
python -c "import mediapipe as mp; print(mp.__file__)"
```

## Usage Example

```python
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Face Mesh Detection
def detect_face_mesh(image_path):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                # Extract landmark coordinates
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = landmark.x * image.shape[1]
                    y = landmark.y * image.shape[0]
                    z = landmark.z
                    landmarks.append([x, y, z])
                
                return np.array(landmarks)
        
        return None

# Face Detection
def detect_faces(image_path):
    mp_face_detection = mp.solutions.face_detection
    
    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
        
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = face_detection.process(image_rgb)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                       int(bboxC.width * w), int(bboxC.height * h)
                faces.append(bbox)
        
        return faces
```

## Real-time Video Processing

```python
import cv2
import mediapipe as mp

def process_video_realtime():
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)  # Use webcam
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS)
            
            cv2.imshow('MediaPipe FaceMesh', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
```

## Installation

MediaPipe is included in the project requirements. For manual installation:

```bash
pip install mediapipe
```

## Official Resources

- **GitHub**: https://github.com/google/mediapipe
- **Documentation**: https://developers.google.com/mediapipe
- **Solutions**: https://developers.google.com/mediapipe/solutions
- **Models**: https://developers.google.com/mediapipe/models