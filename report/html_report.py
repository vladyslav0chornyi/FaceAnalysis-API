from core.analysis import render_face_attrs
from core.yolo_coco_classes import YOLO_COCO_CLASSES
from core.analysis_summary import summarize_analysis  # Додаємо імпорт для підсумку

def render_yolo_objects(yolo_objects):
    if not yolo_objects:
        return '<div class="no-yolo">YOLO-обʼєкти не знайдено</div>\n'
    html = '<div class="yolo-objects">\n'
    for idx, obj in enumerate(yolo_objects, 1):
        class_id = obj.get("class_id")
        class_en, class_uk = YOLO_COCO_CLASSES.get(class_id, ["?", "?"])
        bbox = obj.get("bbox")
        conf = obj.get("conf", 0.0)
        html += f'''
        <div class="yolo-card">
            <div class="yolo-title">YOLO #{idx}: <b>{class_uk}</b> <span style="color:#888;">({class_en})</span></div>
            <div class="yolo-attr yolo-bbox">BBox: {bbox}</div>
            <div class="yolo-attr yolo-conf">Ймовірність: {conf:.2f}</div>
        '''
        if "mask" in obj and obj["mask"] is not None:
            html += f'<div class="yolo-attr yolo-mask">[Маска детектована]</div>'
        if "keypoints" in obj and obj["keypoints"] is not None:
            html += f'<div class="yolo-attr yolo-kp">[Ключові точки: {len(obj["keypoints"])}]</div>'
        html += '</div>\n'
    html += '</div>\n'
    return html

def generate_html_body(frames_info):
    html_body = ""
    for frame in frames_info:
        html_body += f'<div class="frame-card">\n'
        html_body += f'  <div class="frame-title">{frame["frame_name"]}</div>\n'
        html_body += f'  <img src="../data/frames/{frame["frame_name"]}" class="frame-img" alt="Кадр">\n'
        html_body += render_yolo_objects(frame.get("yolo_objects", []))
        # Додаємо короткий підсумок (найбільш ймовірний варіант кожного параметра, без згадки blurred face)
        html_body += f'  <div class="frame-summary">{summarize_analysis(frame)}</div>\n'
        if frame.get("faces"):
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
    return html_body

def build_report(template_path, body_path, out_path):
    with open(template_path, encoding="utf-8") as f:
        html_template = f.read()
    with open(body_path, encoding="utf-8") as f:
        html_body = f.read()
    final_html = html_template.replace("{%BODY%}", html_body)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_html)