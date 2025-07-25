import numpy as np

def confident_label(score, label, threshold=0.75):
    if score is None:
        return "невизначено"
    return label if score >= threshold else "невпевнено"

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
        if key == "Одяг та аксесуари (OpenVINO)":
            val_str = ", ".join([f"{k}: {v}" for k, v in value.items() if v not in [None, "невизначено"]])
            if val_str:
                html += f'<li><span class="emoji">👚</span>Одяг та аксесуари (OpenVINO): {val_str}</li>\n'
        else:
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

# Заглушки для моделей, якщо їх не буде (для швидкої заміни)
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