def get_most_probable_yolo_object(yolo_objects, class_id=None):
    """
    Returns the YOLO object with the highest confidence.
    If class_id is specified, returns the top object of that class.
    """
    filtered = [obj for obj in yolo_objects if class_id is None or obj.get("class_id") == class_id]
    if not filtered:
        return None
    return max(filtered, key=lambda obj: obj.get("conf", 0.0))

def summarize_face(face):
    """
    Returns a dict with the most probable value for each face parameter,
    automatically handling all present attributes except 'blurred'.
    """
    # Стать
    sex = face.get("sex")

    # Вік (DeepFace > age_deep > openvino_age)
    age = face.get("age") or face.get("age_deep") or face.get("openvino_age")

    # Емоція (DeepFace > emotion_deep > emotion_openvino)
    emotion = face.get("emotion_deep") or face.get("emotion_openvino") or face.get("emotion")

    # Раса (race_deep > None)
    race = face.get("race_deep")

    # Аксесуари (person_attr)
    person_attr = face.get("person_attr", {})
    accessories = []
    # Вибираємо все, що не "немає" і не "стать (OpenVINO)"
    for att, val in person_attr.items():
        if att.lower() == "стать (openvino)": continue
        if isinstance(val, str) and val.lower() != "немає":
            accessories.append(f"{att}: {val}")

    accessories = ", ".join(accessories) if accessories else "аксесуари відсутні"

    # Додати всі інші атрибути, якщо вони є (наприклад, gender_deep, openvino_gender, emotion_openvino, etc.)
    extra_attrs = {}
    for key in face:
        if key in {"sex", "age", "age_deep", "openvino_age", "emotion", "emotion_deep", "emotion_openvino", "race_deep", "person_attr", "blurred"}:
            continue
        extra_attrs[key] = face[key]

    summary = {
        "sex": sex,
        "age": age,
        "emotion": emotion,
        "race": race,
        "accessories": accessories
    }
    summary.update(extra_attrs)
    return summary

def summarize_analysis(frame):
    """
    Returns a summary for a frame, using only most probable values.
    Excludes any blurred faces.
    """
    yolo_objects = frame.get("yolo_objects", [])
    faces = frame.get("faces", [])

    # Найімовірніша людина
    person = get_most_probable_yolo_object(yolo_objects, class_id=0)
    # Найімовірніший інший об'єкт (не людина)
    other_obj = get_most_probable_yolo_object([o for o in yolo_objects if o.get("class_id") != 0])

    summary_lines = []

    if person:
        summary_lines.append(
            f"Людина: BBox: {person['bbox']}, Ймовірність: {person['conf']:.2f}"
        )
    if other_obj:
        class_name_uk = other_obj.get("class_name_uk", "")
        class_name_en = other_obj.get("class_name_en", "")
        # якщо назв немає — fallback на class_id
        if not class_name_uk or not class_name_en:
            class_id = other_obj.get("class_id")
            class_name_uk = f"Обʼєкт {class_id}"
            class_name_en = ""
        summary_lines.append(
            f"{class_name_uk} ({class_name_en}): BBox: {other_obj['bbox']}, Ймовірність: {other_obj['conf']:.2f}"
        )

    # Вибираємо перше не-розмите обличчя
    for face in faces:
        if face.get("blurred") is True:
            continue
        face_summary = summarize_face(face)
        summary_lines.append(
            "Обличчя: " +
            ", ".join([f"{k}: {v}" for k, v in face_summary.items() if v is not None])
        )
        break  # лише одне обличчя

    return "\n".join(summary_lines)