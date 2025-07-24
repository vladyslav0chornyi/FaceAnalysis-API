import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from deepface import DeepFace
from openvino.runtime import Core

# ----------- 1. Ініціалізація моделей ------------
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

core = Core()
age_gender_model = core.read_model(model="models/intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml")
age_gender_compiled = core.compile_model(model=age_gender_model, device_name="CPU")
attr_model = core.read_model(model="models/intel/person-attributes-recognition-crossroad-0230/FP32/person-attributes-recognition-crossroad-0230.xml")
attr_compiled = core.compile_model(model=attr_model, device_name="CPU")
emotion_model = core.read_model(model="models/intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml")
emotion_compiled = core.compile_model(model=emotion_model, device_name="CPU")

def confident_label(score, label, threshold=0.75):
    if score is None:
        return "невизначено"
    return label if score >= threshold else "невпевнено"

def fairface_predict(crop):
    return {"gender": None, "age": None, "race": None}
def deepfashion_predict(crop):
    return {"clothes_type": None, "clothes_color": None}
def beauty_score_predict(crop):
    return None
def hairnet_predict(crop):
    return {"hair_style": None, "hair_color": None}
def skin_status_predict(crop):
    return None
def jewellery_predict(crop):
    return None
def smile_predict(crop):
    return None
def makeup_predict(crop):
    return None
def height_weight_estimate(pose):
    return {"height": None, "weight": None}

# ----------- 2. Обробка відео ------------
video_path = "video.mp4"
output_frames_dir = "frames"
os.makedirs(output_frames_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Відео не відкрито! Перевір назву та шлях до файлу.")
    exit()
frame_rate = cap.get(cv2.CAP_PROP_FPS)
interval_sec = 2
interval_frames = int(frame_rate * interval_sec) if frame_rate > 0 else 30
frame_num = 0
saved_num = 0

frames_info = []

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
            box = face.bbox.astype(int)
            y1, y2 = max(box[1], 0), min(box[3], frame.shape[0])
            x1, x2 = max(box[0], 0), min(box[2], frame.shape[1])
            cropped_face = frame[y1:y2, x1:x2]
            if cropped_face is None or cropped_face.size == 0:
                continue

            sex = confident_label(getattr(face, "sex_score", None), "чоловік" if face.sex == 1 else "жінка")
            age = int(face.age)
            glasses = confident_label(getattr(face, "glasses_score", None), "є" if face.glasses else "немає")
            beard = confident_label(getattr(face, "beard_score", None), "є" if face.beard else "немає")
            emotion = getattr(face, "emotion", "невизначено")
            liveness = confident_label(getattr(face, 'liveness_score', 1.0), "живий")
            pose = getattr(face, "pose", "невідомо")
            mask = "є" if getattr(face, 'mask', False) else "немає"

            deep = DeepFace.analyze(cropped_face, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)
            if isinstance(deep, list):
                deep_attr = deep[0] if len(deep) > 0 else {}
            else:
                deep_attr = deep if deep is not None else {}
            gender_deep = deep_attr.get("gender", "невідомо")
            age_deep = deep_attr.get("age", "невідомо")
            emotion_deep = deep_attr.get("dominant_emotion", "невідомо")
            race_deep = deep_attr.get("dominant_race", "невідомо")

            fairface_attr = fairface_predict(cropped_face)
            gender_fairface = fairface_attr.get("gender")
            age_fairface = fairface_attr.get("age")
            race_fairface = fairface_attr.get("race")

            resized_person = cv2.resize(frame, (80, 160))
            input_blob = np.expand_dims(resized_person.transpose(2, 0, 1), axis=0)
            attr_results = attr_compiled([input_blob])
            keys = ["стать (OpenVINO)", "сумка", "капелюх", "довгі рукави", "довгі штани", "довге волосся", "куртка/піджак", "сонцезахисні окуляри"]
            person_attr = {k: ("є" if v > 0.75 else "немає") for k, v in zip(keys, attr_results[0][0])}

            if cropped_face is not None and cropped_face.size > 0:
                try:
                    age_gender_input = cv2.resize(cropped_face, (62, 62))
                    age_gender_input = np.expand_dims(age_gender_input.transpose(2, 0, 1), axis=0)
                    ag_results = age_gender_compiled([age_gender_input])
                    openvino_age = int(ag_results["age_conv3"][0][0][0][0] * 100)
                    male_score, female_score = ag_results["prob"][0]
                    openvino_gender = "чоловік" if male_score > female_score else "жінка"
                except Exception as e:
                    openvino_age = None
                    openvino_gender = "невідомо"
            else:
                openvino_age = None
                openvino_gender = "невідомо"

            if cropped_face is not None and cropped_face.size > 0:
                try:
                    emotion_input = cv2.resize(cropped_face, (64, 64))
                    emotion_input = np.expand_dims(emotion_input.transpose(2, 0, 1), axis=0)
                    emotion_results = emotion_compiled([emotion_input])
                    emotions_list = ["neutral", "happy", "sad", "surprise", "anger"]
                    emotion_openvino = emotions_list[np.argmax(emotion_results[0][0])] if emotion_results[0][0].size == 5 else "невідомо"
                except Exception as e:
                    emotion_openvino = "невідомо"
            else:
                emotion_openvino = "невідомо"

            deepfashion_attr = deepfashion_predict(frame)
            clothes_type = deepfashion_attr.get("clothes_type")
            clothes_color = deepfashion_attr.get("clothes_color")

            hair_attr = hairnet_predict(cropped_face)
            hair_style = hair_attr.get("hair_style")
            hair_color = hair_attr.get("hair_color")

            skin_status = skin_status_predict(cropped_face)
            beauty_score = beauty_score_predict(cropped_face)
            jewellery = jewellery_predict(cropped_face)
            smile = smile_predict(cropped_face)
            makeup = makeup_predict(cropped_face)
            height_weight = height_weight_estimate(pose)
            height_estimate = height_weight.get("height")
            weight_estimate = height_weight.get("weight")

            genders = [
                str(sex),
                str(gender_deep),
                str(person_attr.get("стать (OpenVINO)", "невідомо")),
                str(openvino_gender),
                str(gender_fairface) if gender_fairface else "",
            ]
            gender_final = max(set(genders), key=genders.count)

            face_attrs = {
                "Стать (ensemble)": gender_final,
                "Стать (InsightFace)": sex,
                "Стать (DeepFace)": gender_deep,
                "Стать (FairFace)": gender_fairface,
                "Стать (OpenVINO)": person_attr["стать (OpenVINO)"],
                "Стать (OpenVINO-AG)": openvino_gender,
                "Вік (InsightFace)": age,
                "Вік (DeepFace)": age_deep,
                "Вік (FairFace)": age_fairface,
                "Вік (OpenVINO)": openvino_age,
                "Раса (DeepFace)": race_deep,
                "Раса (FairFace)": race_fairface,
                "Окуляри": glasses,
                "Сонцезахисні окуляри": person_attr.get("сонцезахисні окуляри"),
                "Борода": beard,
                "Зачіска": hair_style,
                "Колір волосся": hair_color,
                "Емоція (InsightFace)": emotion,
                "Емоція (DeepFace)": emotion_deep,
                "Емоція (OpenVINO)": emotion_openvino,
                "Живість": liveness,
                "Посмішка": smile,
                "Макіяж": makeup,
                "Стан шкіри": skin_status,
                "Естетична оцінка": beauty_score,
                "Прикраси": jewellery,
                "Тип одягу": clothes_type,
                "Колір одягу": clothes_color,
                "Одяг та аксесуари (OpenVINO)": {k: v for k, v in person_attr.items() if k != "стать (OpenVINO)"},
                "Поза голови": pose,
                "Маска на обличчі": mask,
                "Зріст (оцінка)": height_estimate,
                "Вага (оцінка)": weight_estimate
            }
            faces_info.append(face_attrs)
        frames_info.append({
            "frame_name": fname,
            "faces": faces_info
        })
        saved_num += 1
    frame_num += 1
cap.release()

# ----------- 3. Генерація сучасного HTML-репорту з пропуском невизначених атрибутів ------------

def render_face_attrs(face):
    emoji_map = {
        "Стать (ensemble)": "👤",
        "Стать (InsightFace)": "👤",
        "Стать (DeepFace)": "👤",
        "Стать (FairFace)": "👤",
        "Стать (OpenVINO)": "👤",
        "Стать (OpenVINO-AG)": "👤",
        "Вік (InsightFace)": "🎂",
        "Вік (DeepFace)": "🎂",
        "Вік (FairFace)": "🎂",
        "Вік (OpenVINO)": "🎂",
        "Раса (DeepFace)": "🌎",
        "Раса (FairFace)": "🌎",
        "Окуляри": "👓",
        "Сонцезахисні окуляри": "🕶️",
        "Борода": "🧔",
        "Зачіска": "💇",
        "Колір волосся": "🎨",
        "Емоція (InsightFace)": "😊",
        "Емоція (DeepFace)": "😊",
        "Емоція (OpenVINO)": "😊",
        "Живість": "👤",
        "Посмішка": "😁",
        "Макіяж": "💄",
        "Стан шкіри": "🧴",
        "Естетична оцінка": "🏅",
        "Прикраси": "💍",
        "Тип одягу": "👕",
        "Колір одягу": "🎨",
        "Поза голови": "🧭",
        "Маска на обличчі": "😷",
        "Зріст (оцінка)": "📏",
        "Вага (оцінка)": "⚖️"
    }
    html = ""
    for key, value in face.items():
        # Спеціальна обробка для 'Одяг та аксесуари (OpenVINO)'
        if key == "Одяг та аксесуари (OpenVINO)":
            val_str = ", ".join([f"{k}: {v}" for k, v in value.items() if v not in [None, "невизначено"]])
            if val_str:
                html += f'<li><span class="emoji">👚</span>Одяг та аксесуари (OpenVINO): {val_str}</li>\n'
        else:
            # Якщо value — numpy масив або інший нестроковий тип, відображаємо тільки, якщо це строка або число й не "невизначено"
            skip = False
            if isinstance(value, np.ndarray):
                skip = True
            elif value is None:
                skip = True
            elif isinstance(value, str) and value == "невизначено":
                skip = True
            if not skip:
                emoji = emoji_map.get(key, "")
                html += f'<li><span class="emoji">{emoji}</span>{key}: {value}</li>\n'
    return html

html_head = """
<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="utf-8">
<title>Атрибути відео</title>
<style>
body {
  background: #f4f6f9;
  font-family: 'Segoe UI', Arial, sans-serif;
  margin: 0;
  padding: 0;
}
h1 {
  text-align: center;
  color: #222;
  margin-top: 2rem;
  letter-spacing: 2px;
}
.frame-card {
  background: #fff;
  border-radius: 14px;
  box-shadow: 0 2px 16px rgba(0,0,0,0.06);
  padding: 2rem;
  margin: 2rem auto;
  max-width: 900px;
}
.frame-title {
  font-size: 1.3rem;
  color: #2255a4;
  font-weight: 600;
  margin-bottom: 1rem;
  letter-spacing: 1px;
}
.frame-img {
  display: block;
  margin: 0 auto 1.2rem auto;
  border-radius: 10px;
  box-shadow: 0 2px 12px rgba(40,60,120,0.11);
  max-width: 400px;
}
.faces-grid {
  display: flex;
  gap: 2rem;
  flex-wrap: wrap;
  justify-content: flex-start;
  margin-top: 1.2rem;
}
.face-card {
  background: #eef2fa;
  border-radius: 12px;
  box-shadow: 0 1px 6px rgba(30,60,120,0.09);
  padding: 1.3rem;
  min-width: 250px;
  flex: 1 1 250px;
  margin-bottom: 1rem;
}
.face-title {
  color: #2b4d8c;
  font-weight: 600;
  margin-bottom: .7rem;
  font-size: 1.1rem;
  letter-spacing: .5px;
}
.face-attrs {
  list-style: none;
  padding: 0;
  margin: 0;
}
.face-attrs li {
  margin-bottom: .6rem;
  font-size: 1rem;
  color: #333;
  display: flex;
  align-items: center;
}
.face-attrs li span.emoji {
  font-size: 1.2em;
  margin-right: .5em;
}
.no-face {
  color: #b00;
  font-weight: 500;
  margin: 1rem 0;
  letter-spacing: 1px;
}
@media (max-width: 600px) {
  .frame-card {padding: 1rem;}
  .faces-grid {gap: 1rem;}
  .face-card {min-width: 180px;}
}
</style>
</head>
<body>
<h1>Кадри відео і їх атрибути</h1>
"""

html_body = ""
for frame in frames_info:
    html_body += f'<div class="frame-card">\n'
    html_body += f'  <div class="frame-title">{frame["frame_name"]}</div>\n'
    html_body += f'  <img src="frames/{frame["frame_name"]}" class="frame-img" alt="Кадр">\n'
    if frame["faces"]:
        html_body += f'  <div class="faces-grid">\n'
        for idx, face in enumerate(frame["faces"], 1):
            html_body += f'    <div class="face-card">\n'
            html_body += f'      <div class="face-title">Обличчя {idx}</div>\n'
            html_body += f'      <ul class="face-attrs">\n'
            html_body += render_face_attrs(face)
            html_body += f'      </ul>\n'
            html_body += f'    </div>\n'
        html_body += f'  </div>\n'
    else:
        html_body += f'  <div class="no-face">Обличчя не знайдено</div>\n'
    html_body += f'</div>\n'

html_tail = "</body></html>"

with open("video_report.html", "w", encoding="utf-8") as f:
    f.write(html_head + html_body + html_tail)

print("video_report.html згенеровано!")