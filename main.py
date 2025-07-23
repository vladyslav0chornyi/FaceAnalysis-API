import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from deepface import DeepFace
from openvino.runtime import Core

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

def pretty_attrs(res):
    lines = []
    lines.append(f"<ul>")
    lines.append(f"<li>📦 Область обличчя: {res.get('область_обличчя')}</li>")
    lines.append(f"<li>👤 Стать (ensemble, голосування): {res.get('стать (ensemble, голосування)')}</li>")
    lines.append(f"<li>InsightFace: {res.get('стать (InsightFace)')}</li>")
    lines.append(f"<li>DeepFace: {res.get('стать (DeepFace)')}</li>")
    lines.append(f"<li>FairFace: {res.get('стать (FairFace)')}</li>")
    lines.append(f"<li>OpenVINO (person): {res.get('стать (OpenVINO)')}</li>")
    lines.append(f"<li>OpenVINO-AG: {res.get('стать (OpenVINO-AG)')}</li>")
    lines.append(f"<li>🎂 Вік (InsightFace): {res.get('вік (InsightFace)')}</li>")
    lines.append(f"<li>DeepFace: {res.get('вік (DeepFace)')}</li>")
    lines.append(f"<li>FairFace: {res.get('вік (FairFace)')}</li>")
    lines.append(f"<li>OpenVINO: {res.get('вік (OpenVINO)')}</li>")
    lines.append(f"<li>🌎 Раса (DeepFace): {res.get('раса (DeepFace)')}</li>")
    lines.append(f"<li>FairFace: {res.get('раса (FairFace)')}</li>")
    lines.append(f"<li>👓 Окуляри: {res.get('окуляри')}</li>")
    lines.append(f"<li>🕶️ Сонцезахисні окуляри: {res.get('сонцезахисні окуляри')}</li>")
    lines.append(f"<li>🧔 Борода: {res.get('борода')}</li>")
    lines.append(f"<li>💇 Зачіска: {res.get('зачіска')}</li>")
    lines.append(f"<li>🎨 Колір волосся: {res.get('колір волосся')}</li>")
    lines.append(f"<li>😊 Емоція (InsightFace): {res.get('емоція (InsightFace)')}</li>")
    lines.append(f"<li>DeepFace: {res.get('емоція (DeepFace)')}</li>")
    lines.append(f"<li>OpenVINO: {res.get('емоція (OpenVINO)')}</li>")
    lines.append(f"<li>👤 Живість: {res.get('живість')}</li>")
    lines.append(f"<li>😁 Посмішка: {res.get('посмішка')}</li>")
    lines.append(f"<li>💄 Макіяж: {res.get('макіяж')}</li>")
    lines.append(f"<li>🧴 Стан шкіри: {res.get('стан шкіри')}</li>")
    lines.append(f"<li>🏅 Естетична оцінка: {res.get('естетична оцінка')}</li>")
    lines.append(f"<li>💍 Прикраси: {res.get('прикраси')}</li>")
    lines.append(f"<li>👕 Тип одягу: {res.get('тип одягу')}</li>")
    lines.append(f"<li>🎨 Колір одягу: {res.get('колір одягу')}</li>")
    lines.append(f"<li>👚 Одяг та аксесуари (OpenVINO):")
    for k, v in res.get('одяг та аксесуари (OpenVINO)', {}).items():
        lines.append(f"{k}: {v} ")
    lines.append(f"</li>")
    lines.append(f"<li>🧭 Поза голови: {res.get('поза голови')}</li>")
    lines.append(f"<li>😷 Маска на обличчі: {res.get('маска на обличчі')}</li>")
    lines.append(f"<li>📏 Зріст (оцінка): {res.get('зріст (оцінка)')}</li>")
    lines.append(f"<li>⚖️ Вага (оцінка): {res.get('вага (оцінка)')}</li>")
    lines.append(f"</ul>")
    return "\n".join(lines)

photo_dir = "photo"
supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

results_all = []

for filename in os.listdir(photo_dir):
    if not filename.lower().endswith(supported_ext):
        continue
    img_path = os.path.join(photo_dir, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Не вдалося відкрити {img_path}")
        continue

    faces = face_app.get(img)
    results = []
    for face in faces:
        box = face.bbox.astype(int)
        y1, y2 = max(box[1], 0), min(box[3], img.shape[0])
        x1, x2 = max(box[0], 0), min(box[2], img.shape[1])
        cropped_face = img[y1:y2, x1:x2]
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

        resized_person = cv2.resize(img, (80, 160))
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

        deepfashion_attr = deepfashion_predict(img)
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

        results.append({
            "область_обличчя": box.tolist(),
            "стать (ensemble, голосування)": gender_final,
            "стать (InsightFace)": sex,
            "стать (DeepFace)": gender_deep,
            "стать (FairFace)": gender_fairface,
            "стать (OpenVINO)": person_attr["стать (OpenVINO)"],
            "стать (OpenVINO-AG)": openvino_gender,
            "вік (InsightFace)": age,
            "вік (DeepFace)": age_deep,
            "вік (FairFace)": age_fairface,
            "вік (OpenVINO)": openvino_age,
            "раса (DeepFace)": race_deep,
            "раса (FairFace)": race_fairface,
            "окуляри": glasses,
            "сонцезахисні окуляри": person_attr.get("сонцезахисні окуляри"),
            "борода": beard,
            "зачіска": hair_style,
            "колір волосся": hair_color,
            "емоція (InsightFace)": emotion,
            "емоція (DeepFace)": emotion_deep,
            "емоція (OpenVINO)": emotion_openvino,
            "живість": liveness,
            "посмішка": smile,
            "макіяж": makeup,
            "стан шкіри": skin_status,
            "естетична оцінка": beauty_score,
            "прикраси": jewellery,
            "тип одягу": clothes_type,
            "колір одягу": clothes_color,
            "одяг та аксесуари (OpenVINO)": {k: v for k, v in person_attr.items() if k != "стать (OpenVINO)"},
            "поза голови": pose,
            "маска на обличчі": mask,
            "зріст (оцінка)": height_estimate,
            "вага (оцінка)": weight_estimate
        })
    # Додаємо результати по фото
    results_all.append({
        "filename": filename,
        "results": results
    })

# ----------- 4. Генерація HTML-репорту ------------

with open("report.html", "w", encoding="utf-8") as f:
    f.write("<!DOCTYPE html>\n<html>\n<head>\n<meta charset='utf-8'>\n<title>Атрибути фото</title>\n</head>\n<body>\n")
    f.write("<h1>Фото і їх атрибути</h1>\n")
    for photo in results_all:
        fname = photo["filename"]
        f.write(f"<h2>{fname}</h2>\n")
        # Вставляємо фото (шлях відносний, якщо report.html поруч із photo/)
        f.write(f"<img src='photo/{fname}' width='400' style='border:1px solid #ccc'><br>\n")
        if photo["results"]:
            for res in photo["results"]:
                f.write(pretty_attrs(res))
        else:
            f.write("<b>Обличчя не знайдено.</b><br>")
        f.write("<hr>")
    f.write("</body></html>\n")

print("report.html згенеровано!")