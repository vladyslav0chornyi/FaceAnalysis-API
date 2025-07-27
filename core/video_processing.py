import cv2
import numpy as np
import os
from core.yolo_detector import YoloDetector  # Додаємо імпорт YOLO


def process_video(
        video_path,
        output_frames_dir,
        face_app,
        openvino_models,
        deepface_analyze_func,
        interval_sec=2,
        yolo_model_path="models/yolo/yolov8n.pt",
        yolo_conf=0.25
):
    """
    Обробляє відео, зберігає кадри, детектить обличчя, рахує атрибути, детектить об'єкти YOLOv8.
    Повертає frames_info (список даних по кадрах/обличчях/детекціях).
    """
    os.makedirs(output_frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Відео не відкрито! Перевір назву та шлях до файлу: {video_path}")
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(frame_rate * interval_sec) if frame_rate > 0 else 30
    frame_num = 0
    saved_num = 0
    frames_info = []

    attr_compiled = openvino_models["attr_compiled"]
    age_gender_compiled = openvino_models["age_gender_compiled"]
    emotion_compiled = openvino_models["emotion_compiled"]

    # --- Ініціалізація YOLO ---
    yolo_detector = YoloDetector(yolo_model_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % interval_frames == 0:
            fname = f"frame_{saved_num:03d}.jpg"
            frame_path = os.path.join(output_frames_dir, fname)
            cv2.imwrite(frame_path, frame)
            faces = face_app.get(frame)
            faces_info = []
            for face in faces:
                # bbox crop
                box = face.bbox.astype(int)
                y1, y2 = max(box[1], 0), min(box[3], frame.shape[0])
                x1, x2 = max(box[0], 0), min(box[2], frame.shape[1])
                cropped_face = frame[y1:y2, x1:x2]
                if cropped_face is None or cropped_face.size == 0:
                    continue

                # InsightFace
                sex = getattr(face, "sex", None)
                sex_score = getattr(face, "sex_score", None)
                age = int(face.age) if hasattr(face, "age") else None
                glasses = getattr(face, "glasses", None)
                glasses_score = getattr(face, "glasses_score", None)
                beard = getattr(face, "beard", None)
                beard_score = getattr(face, "beard_score", None)
                emotion = getattr(face, "emotion", None)
                liveness_score = getattr(face, "liveness_score", None)
                pose = getattr(face, "pose", None)
                mask = getattr(face, 'mask', False)

                # DeepFace
                deep = deepface_analyze_func(cropped_face)
                if isinstance(deep, list):
                    deep_attr = deep[0] if len(deep) > 0 else {}
                else:
                    deep_attr = deep if deep is not None else {}
                gender_deep = deep_attr.get("gender", None)
                age_deep = deep_attr.get("age", None)
                emotion_deep = deep_attr.get("dominant_emotion", None)
                race_deep = deep_attr.get("dominant_race", None)

                # OpenVINO person-attributes
                resized_person = cv2.resize(frame, (80, 160))
                input_blob = np.expand_dims(resized_person.transpose(2, 0, 1), axis=0)
                attr_results = attr_compiled([input_blob])
                keys = [
                    "стать (OpenVINO)", "сумка", "капелюх", "довгі рукави",
                    "довгі штани", "довге волосся", "куртка/піджак", "сонцезахисні окуляри"
                ]
                person_attr = {k: ("є" if v > 0.75 else "немає") for k, v in zip(keys, attr_results[0][0])}

                # OpenVINO age-gender
                if cropped_face is not None and cropped_face.size > 0:
                    try:
                        age_gender_input = cv2.resize(cropped_face, (62, 62))
                        age_gender_input = np.expand_dims(age_gender_input.transpose(2, 0, 1), axis=0)
                        ag_results = age_gender_compiled([age_gender_input])
                        openvino_age = int(ag_results["age_conv3"][0][0][0][0] * 100)
                        male_score, female_score = ag_results["prob"][0]
                        openvino_gender = "чоловік" if male_score > female_score else "жінка"
                    except Exception:
                        openvino_age = None
                        openvino_gender = None
                else:
                    openvino_age = None
                    openvino_gender = None

                # OpenVINO emotion
                if cropped_face is not None and cropped_face.size > 0:
                    try:
                        emotion_input = cv2.resize(cropped_face, (64, 64))
                        emotion_input = np.expand_dims(emotion_input.transpose(2, 0, 1), axis=0)
                        emotion_results = emotion_compiled([emotion_input])
                        emotions_list = ["neutral", "happy", "sad", "surprise", "anger"]
                        emotion_openvino = emotions_list[np.argmax(emotion_results[0][0])] if emotion_results[0][0].size == 5 else None
                    except Exception:
                        emotion_openvino = None
                else:
                    emotion_openvino = None

                face_attrs = {
                    "sex": sex,
                    "sex_score": sex_score,
                    "age": age,
                    "glasses": glasses,
                    "glasses_score": glasses_score,
                    "beard": beard,
                    "beard_score": beard_score,
                    "emotion": emotion,
                    "liveness_score": liveness_score,
                    "pose": pose,
                    "mask": mask,
                    "gender_deep": gender_deep,
                    "age_deep": age_deep,
                    "emotion_deep": emotion_deep,
                    "race_deep": race_deep,
                    "person_attr": person_attr,
                    "openvino_age": openvino_age,
                    "openvino_gender": openvino_gender,
                    "emotion_openvino": emotion_openvino
                }
                faces_info.append(face_attrs)

            # --- Детекція YOLO на кадрі ---
            yolo_detections = yolo_detector.detect(frame, conf=yolo_conf)

            frames_info.append({
                "frame_name": fname,
                "faces": faces_info,
                "yolo_objects": yolo_detections
            })
            saved_num += 1
        frame_num += 1
    cap.release()
    return frames_info