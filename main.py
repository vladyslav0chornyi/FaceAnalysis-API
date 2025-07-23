import cv2
import numpy as np
from insightface.app import FaceAnalysis
from deepface import DeepFace
from openvino.runtime import Core

# ----------- 1. Ініціалізація моделей для продакшн ------------

face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

core = Core()
age_gender_model = core.read_model(model="models/intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml")
age_gender_compiled = core.compile_model(model=age_gender_model, device_name="CPU")
attr_model = core.read_model(model="models/intel/person-attributes-recognition-crossroad-0230/FP32/person-attributes-recognition-crossroad-0230.xml")
attr_compiled = core.compile_model(model=attr_model, device_name="CPU")
emotion_model = core.read_model(model="models/intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml")
emotion_compiled = core.compile_model(model=emotion_model, device_name="CPU")

# Можливість підключення інших моделей через API, ONNX, PyTorch, etc (FairFace, DeepFashion, OpenPose, etc)
# Для демонстрації — залишаємо структуру, щоб легко додати нову модель

# ----------- 2. Функції для обробки атрибутів та ensemble -----------

def confident_label(score, label, threshold=0.75):
    if score is None:
        return "невизначено"
    return label if score >= threshold else "невпевнено"

def fairface_predict(crop):
    # Placeholder for FairFace API/model
    # Повертає dict: {"gender": ..., "age": ..., "race": ...}
    return {"gender": None, "age": None, "race": None}

def deepfashion_predict(crop):
    # Placeholder for DeepFashion API/model
    # Повертає dict: {"clothes_type": ..., "clothes_color": ...}
    return {"clothes_type": None, "clothes_color": None}

def beauty_score_predict(crop):
    # Placeholder for BeautyNet/FaceQuality
    return None

def hairnet_predict(crop):
    # Placeholder for HairNet (стиль та колір волосся)
    return {"hair_style": None, "hair_color": None}

def skin_status_predict(crop):
    # Placeholder for SkinNet (стан шкіри)
    return None

def jewellery_predict(crop):
    # Placeholder for JewelleryNet
    return None

def smile_predict(crop):
    # Placeholder for SmileNet
    return None

def makeup_predict(crop):
    # Placeholder for MakeupNet
    return None

def height_weight_estimate(pose):
    # Placeholder for estimate via skeleton/pose models
    return {"height": None, "weight": None}

# ----------- 3. Завантаження фото і обробка -----------

img = cv2.imread("photo.jpg")
faces = face_app.get(img)

results = []
for face in faces:
    box = face.bbox.astype(int)
    cropped_face = img[box[1]:box[3], box[0]:box[2]]

    # InsightFace
    sex = confident_label(getattr(face, "sex_score", None), "чоловік" if face.sex == 1 else "жінка")
    age = int(face.age)
    glasses = confident_label(getattr(face, "glasses_score", None), "є" if face.glasses else "немає")
    beard = confident_label(getattr(face, "beard_score", None), "є" if face.beard else "немає")
    emotion = getattr(face, "emotion", "невизначено")
    liveness = confident_label(getattr(face, 'liveness_score', 1.0), "живий")
    pose = getattr(face, "pose", "невідомо")
    mask = "є" if getattr(face, 'mask', False) else "немає"

    # DeepFace
    deep = DeepFace.analyze(cropped_face, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)
    deep_attr = deep[0] if isinstance(deep, list) else deep
    gender_deep = deep_attr.get("gender", "невідомо")
    age_deep = deep_attr.get("age", "невідомо")
    emotion_deep = deep_attr.get("dominant_emotion", "невідомо")
    race_deep = deep_attr.get("dominant_race", "невідомо")

    # FairFace
    fairface_attr = fairface_predict(cropped_face)
    gender_fairface = fairface_attr.get("gender")
    age_fairface = fairface_attr.get("age")
    race_fairface = fairface_attr.get("race")

    # OpenVINO: person-attributes
    resized_person = cv2.resize(img, (80, 160))
    input_blob = np.expand_dims(resized_person.transpose(2, 0, 1), axis=0)
    attr_results = attr_compiled([input_blob])
    keys = ["стать (OpenVINO)", "сумка", "капелюх", "довгі рукави", "довгі штани", "довге волосся", "куртка/піджак", "сонцезахисні окуляри"]
    person_attr = {k: ("є" if v > 0.75 else "немає") for k, v in zip(keys, attr_results[0][0])}

    # OpenVINO: age-gender-recognition
    age_gender_input = cv2.resize(cropped_face, (62, 62))
    age_gender_input = np.expand_dims(age_gender_input.transpose(2, 0, 1), axis=0)
    ag_results = age_gender_compiled([age_gender_input])
    openvino_age = int(ag_results["age_conv3"][0][0][0][0] * 100)
    male_score, female_score = ag_results["prob"][0]
    openvino_gender = "чоловік" if male_score > female_score else "жінка"

    # OpenVINO: emotion-recognition
    emotion_input = cv2.resize(cropped_face, (64, 64))
    emotion_input = np.expand_dims(emotion_input.transpose(2, 0, 1), axis=0)
    emotion_results = emotion_compiled([emotion_input])
    emotions_list = ["neutral", "happy", "sad", "surprise", "anger"]
    emotion_openvino = emotions_list[np.argmax(emotion_results[0][0])] if emotion_results[0][0].size == 5 else "невідомо"

    # DeepFashion
    deepfashion_attr = deepfashion_predict(img)
    clothes_type = deepfashion_attr.get("clothes_type")
    clothes_color = deepfashion_attr.get("clothes_color")

    # HairNet
    hair_attr = hairnet_predict(cropped_face)
    hair_style = hair_attr.get("hair_style")
    hair_color = hair_attr.get("hair_color")

    # Додаткові моделі
    skin_status = skin_status_predict(cropped_face)
    beauty_score = beauty_score_predict(cropped_face)
    jewellery = jewellery_predict(cropped_face)
    smile = smile_predict(cropped_face)
    makeup = makeup_predict(cropped_face)
    height_weight = height_weight_estimate(pose)
    height_estimate = height_weight.get("height")
    weight_estimate = height_weight.get("weight")

    # Ensemble gender
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

# ----------- 4. Красивий вивід -----------

def pretty_gender(g):
    if isinstance(g, dict):
        if not g:
            return "невідомо"
        key = max(g, key=lambda x: g[x])
        value = g[key]
        return f"{key} ({value:.2f}%)"
    return str(g)

def pretty_attrs(res):
    lines = []
    lines.append(f"🧑‍💼 Атрибути особи:")
    lines.append(f"  📦 Область обличчя: {res.get('область_обличчя')}")
    lines.append(f"  👤 Стать (ensemble, голосування): {pretty_gender(res.get('стать (ensemble, голосування)'))}")
    lines.append(f"  ├─ InsightFace: {pretty_gender(res.get('стать (InsightFace)'))}")
    lines.append(f"  ├─ DeepFace: {pretty_gender(res.get('стать (DeepFace)'))}")
    lines.append(f"  ├─ FairFace: {pretty_gender(res.get('стать (FairFace)'))}")
    lines.append(f"  ├─ OpenVINO (person): {pretty_gender(res.get('стать (OpenVINO)'))}")
    lines.append(f"  └─ OpenVINO-AG: {pretty_gender(res.get('стать (OpenVINO-AG)'))}")
    lines.append(f"  🎂 Вік (InsightFace): {res.get('вік (InsightFace)')}")
    lines.append(f"  ├─ DeepFace: {res.get('вік (DeepFace)')}")
    lines.append(f"  ├─ FairFace: {res.get('вік (FairFace)')}")
    lines.append(f"  └─ OpenVINO: {res.get('вік (OpenVINO)')}")
    lines.append(f"  🌎 Раса (DeepFace): {res.get('раса (DeepFace)')}")
    lines.append(f"  └─ FairFace: {res.get('раса (FairFace)')}")
    lines.append(f"  👓 Окуляри: {res.get('окуляри')}")
    lines.append(f"  🕶️ Сонцезахисні окуляри: {res.get('сонцезахисні окуляри')}")
    lines.append(f"  🧔 Борода: {res.get('борода')}")
    lines.append(f"  💇 Зачіска: {res.get('зачіска')}")
    lines.append(f"  🎨 Колір волосся: {res.get('колір волосся')}")
    lines.append(f"  😊 Емоція (InsightFace): {res.get('емоція (InsightFace)')}")
    lines.append(f"  ├─ DeepFace: {res.get('емоція (DeepFace)')}")
    lines.append(f"  └─ OpenVINO: {res.get('емоція (OpenVINO)')}")
    lines.append(f"  👤 Живість: {res.get('живість')}")
    lines.append(f"  😁 Посмішка: {res.get('посмішка')}")
    lines.append(f"  💄 Макіяж: {res.get('макіяж')}")
    lines.append(f"  🧴 Стан шкіри: {res.get('стан шкіри')}")
    lines.append(f"  🏅 Естетична оцінка: {res.get('естетична оцінка')}")
    lines.append(f"  💍 Прикраси: {res.get('прикраси')}")
    lines.append(f"  👕 Тип одягу: {res.get('тип одягу')}")
    lines.append(f"  🎨 Колір одягу: {res.get('колір одягу')}")
    lines.append(f"  👚 Одяг та аксесуари (OpenVINO):")
    for k, v in res.get('одяг та аксесуари (OpenVINO)', {}).items():
        lines.append(f"    └─ {k}: {v}")
    lines.append(f"  🧭 Поза голови: {res.get('поза голови')}")
    lines.append(f"  😷 Маска на обличчі: {res.get('маска на обличчі')}")
    lines.append(f"  📏 Зріст (оцінка): {res.get('зріст (оцінка)')}")
    lines.append(f"  ⚖️ Вага (оцінка): {res.get('вага (оцінка)')}")
    return "\n".join(lines)

for res in results:
    print(pretty_attrs(res))