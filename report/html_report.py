from core.analysis import render_face_attrs

def generate_html_body(frames_info):
    html_body = ""
    for frame in frames_info:
        html_body += f'<div class="frame-card">\n'
        html_body += f'  <div class="frame-title">{frame["frame_name"]}</div>\n'
        html_body += f'  <img src="../data/frames/{frame["frame_name"]}" class="frame-img" alt="Кадр">\n'
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
    return html_body

def build_report(template_path, body_path, out_path):
    with open(template_path, encoding="utf-8") as f:
        html_template = f.read()
    with open(body_path, encoding="utf-8") as f:
        html_body = f.read()
    final_html = html_template.replace("{%BODY%}", html_body)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_html)