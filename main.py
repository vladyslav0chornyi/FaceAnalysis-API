import cv2
import numpy as np
from insightface.app import FaceAnalysis
from deepface import DeepFace
from openvino.runtime import Core

# ----------- 1. Ініціалізація моделей для продакшн ------------

# InsightFace: топова модель для обличчя та атрибутів
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

# OpenVINO: моделі класифікації одягу (Gender/Age/Clothes)
core = Core()
# Абсолютний або відносний шлях до .xml
age_gender_model = core.read_model(model="models/intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml")
age_gender_compiled = core.compile_model(model=age_gender_model, device_name="CPU")
attr_model = core.read_model(model="models/intel/person-attributes-recognition-crossroad-0230/FP32/person-attributes-recognition-crossroad-0230.xml")
attr_compiled = core.compile_model(model=attr_model, device_name="CPU")

# ----------- 2. Функція для аналізу впевненості -----------

def confident_label(score, label, threshold=0.75):
    if score is None:
        return "невизначено"
    return label if score >= threshold else "невпевнено"

# ----------- 3. Завантаження фото -----------

img = cv2.imread("photo.jpg")
faces = face_app.get(img)

results = []
for face in faces:
    box = face.bbox.astype(int)
    cropped_face = img[box[1]:box[3], box[0]:box[2]]

    # InsightFace атрибути + впевненість
    sex = confident_label(getattr(face, "sex_score", None), "чоловік" if face.sex == 1 else "жінка")
    age = int(face.age)
    glasses = confident_label(getattr(face, "glasses_score", None), "є" if face.glasses else "немає")
    beard = confident_label(getattr(face, "beard_score", None), "є" if face.beard else "немає")
    emotion = getattr(face, "emotion", "невизначено")
    liveness = confident_label(getattr(face, 'liveness_score', 1.0), "живий")
    pose = getattr(face, "pose", "невідомо")
    mask = "є" if getattr(face, 'mask', False) else "немає"

    # DeepFace для ensemble
    deep = DeepFace.analyze(cropped_face, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)
    deep_attr = deep[0] if isinstance(deep, list) else deep
    gender_deep = deep_attr.get("gender", "невідомо")
    age_deep = deep_attr.get("age", "невідомо")
    emotion_deep = deep_attr.get("dominant_emotion", "невідомо")
    race_deep = deep_attr.get("dominant_race", "невідомо")

    # OpenVINO: атрибути людини (одяг, аксесуари, стать, вік)
    # person-attributes-recognition-crossroad-0230: shape=(1,3,160,80)
    resized_person = cv2.resize(img, (80, 160))  # width=80, height=160
    input_blob = np.expand_dims(resized_person.transpose(2, 0, 1), axis=0)  # [1, 3, 160, 80]
    attr_results = attr_compiled([input_blob])
    # Результат — масив ймовірностей
    keys = ["стать (OpenVINO)", "сумка", "капелюх", "довгі рукави", "довгі штани", "довге волосся", "куртка/піджак", "сонцезахисні окуляри"]
    person_attr = {k: ("є" if v > 0.75 else "немає") for k, v in zip(keys, attr_results[0][0])}

    # OpenVINO age-gender-recognition-retail-0013: shape=(1,3,62,62)
    age_gender_input = cv2.resize(cropped_face, (62, 62))
    age_gender_input = np.expand_dims(age_gender_input.transpose(2, 0, 1), axis=0)
    ag_results = age_gender_compiled([age_gender_input])
    # Вік — вихід age_conv3 (значення від 0 до 1, множимо на 100)
    openvino_age = int(ag_results["age_conv3"][0][0][0][0] * 100)
    # Стать — вихід prob: [male_score, female_score]
    male_score, female_score = ag_results["prob"][0]
    openvino_gender = "чоловік" if male_score > female_score else "жінка"

    # ----------- 4. Голосування моделей для статі -----------

    genders = [
        str(sex),
        str(gender_deep),
        str(person_attr.get("стать (OpenVINO)", "невідомо")),
        str(openvino_gender)
    ]
    gender_final = max(set(genders), key=genders.count)

    # ----------- 5. Збір усіх атрибутів -----------

    results.append({
        "область_обличчя": box.tolist(),
        "стать (ensemble, голосування)": gender_final,
        "стать (InsightFace)": sex,
        "стать (DeepFace)": gender_deep,
        "стать (OpenVINO)": person_attr["стать (OpenVINO)"],
        "стать (OpenVINO-AG)": openvino_gender,
        "вік (InsightFace)": age,
        "вік (DeepFace)": age_deep,
        "вік (OpenVINO)": openvino_age,
        "окуляри": glasses,
        "борода": beard,
        "емоція (InsightFace)": emotion,
        "емоція (DeepFace)": emotion_deep,
        "раса (DeepFace)": race_deep,
        "живість": liveness,
        "поза голови": pose,
        "маска на обличчі": mask,
        "одяг та аксесуари (OpenVINO)": {k: v for k, v in person_attr.items() if k != "стать (OpenVINO)"}
    })

# ----------- 6. Вивід атрибутів -----------

def pretty_gender(g):
    # Якщо це dict (ймовірності), то беремо ключ з найбільшою ймовірністю
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
    lines.append(f"  ├─ OpenVINO (person): {pretty_gender(res.get('стать (OpenVINO)'))}")
    lines.append(f"  └─ OpenVINO-AG: {pretty_gender(res.get('стать (OpenVINO-AG)'))}")
    lines.append(f"  🎂 Вік (InsightFace): {res.get('вік (InsightFace)')}")
    lines.append(f"  ├─ DeepFace: {res.get('вік (DeepFace)')}")
    lines.append(f"  └─ OpenVINO: {res.get('вік (OpenVINO)')}")
    lines.append(f"  👓 Окуляри: {res.get('окуляри')}")
    lines.append(f"  🧔 Борода: {res.get('борода')}")
    lines.append(f"  😊 Емоція (InsightFace): {res.get('емоція (InsightFace)')}")
    lines.append(f"  └─ DeepFace: {res.get('емоція (DeepFace)')}")
    lines.append(f"  🌎 Раса (DeepFace): {res.get('раса (DeepFace)')}")
    lines.append(f"  👤 Живість: {res.get('живість')}")
    lines.append(f"  🧭 Поза голови: {res.get('поза голови')}")
    lines.append(f"  😷 Маска на обличчі: {res.get('маска на обличчі')}")
    lines.append(f"  👕 Одяг та аксесуари (OpenVINO):")
    for k, v in res.get('одяг та аксесуари (OpenVINO)', {}).items():
        lines.append(f"    └─ {k}: {v}")
    return "\n".join(lines)

# Приклад використання:
print(pretty_attrs(results[0]))