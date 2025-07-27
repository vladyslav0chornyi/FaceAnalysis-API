from core.analysis import render_face_attrs

# Словник класів COCO (ID: [en, uk])
YOLO_COCO_CLASSES = {
    0:  ["person", "людина"],
    1:  ["bicycle", "велосипед"],
    2:  ["car", "автомобіль"],
    3:  ["motorcycle", "мотоцикл"],
    4:  ["airplane", "літак"],
    5:  ["bus", "автобус"],
    6:  ["train", "потяг"],
    7:  ["truck", "вантажівка"],
    8:  ["boat", "човен"],
    9:  ["traffic light", "світлофор"],
    10: ["fire hydrant", "пожежний гідрант"],
    11: ["stop sign", "знак STOP"],
    12: ["parking meter", "паркомат"],
    13: ["bench", "лавка"],
    14: ["bird", "птах"],
    15: ["cat", "кіт"],
    16: ["dog", "собака"],
    17: ["horse", "кінь"],
    18: ["sheep", "вівця"],
    19: ["cow", "корова"],
    20: ["elephant", "слон"],
    21: ["bear", "ведмідь"],
    22: ["zebra", "зебра"],
    23: ["giraffe", "жирафа"],
    24: ["backpack", "рюкзак"],
    25: ["umbrella", "парасоля"],
    26: ["handbag", "сумка"],
    27: ["tie", "краватка"],
    28: ["suitcase", "валіза"],
    29: ["frisbee", "фрісбі"],
    30: ["skis", "лижі"],
    31: ["snowboard", "сноуборд"],
    32: ["sports ball", "м’яч"],
    33: ["kite", "повітряний змій"],
    34: ["baseball bat", "бейсбольна бита"],
    35: ["baseball glove", "бейсбольна рукавиця"],
    36: ["skateboard", "скейтборд"],
    37: ["surfboard", "серфборд"],
    38: ["tennis racket", "тенісна ракетка"],
    39: ["bottle", "пляшка"],
    40: ["wine glass", "бокал"],
    41: ["cup", "чашка"],
    42: ["fork", "виделка"],
    43: ["knife", "ніж"],
    44: ["spoon", "ложка"],
    45: ["bowl", "миска"],
    46: ["banana", "банан"],
    47: ["apple", "яблуко"],
    48: ["sandwich", "бутерброд"],
    49: ["orange", "апельсин"],
    50: ["broccoli", "броколі"],
    51: ["carrot", "морква"],
    52: ["hot dog", "хот-дог"],
    53: ["pizza", "піца"],
    54: ["donut", "пончик"],
    55: ["cake", "торт"],
    56: ["chair", "стілець"],
    57: ["couch", "диван"],
    58: ["potted plant", "вазон"],
    59: ["bed", "ліжко"],
    60: ["dining table", "обідній стіл"],
    61: ["toilet", "унітаз"],
    62: ["tv", "телевізор"],
    63: ["laptop", "ноутбук"],
    64: ["mouse", "мишка"],
    65: ["remote", "пульт"],
    66: ["keyboard", "клавіатура"],
    67: ["cell phone", "мобільний телефон"],
    68: ["microwave", "мікрохвильовка"],
    69: ["oven", "духовка"],
    70: ["toaster", "тостер"],
    71: ["sink", "мийка"],
    72: ["refrigerator", "холодильник"],
    73: ["book", "книга"],
    74: ["clock", "годинник"],
    75: ["vase", "ваза"],
    76: ["scissors", "ножиці"],
    77: ["teddy bear", "плюшевий ведмідь"],
    78: ["hair drier", "фен"],
    79: ["toothbrush", "зубна щітка"],
}

def render_yolo_objects(yolo_objects):
    if not yolo_objects:
        return '<div class="no-yolo">YOLO-обʼєкти не знайдено</div>\n'
    html = '<div class="yolo-objects"><b>YOLO-детекції:</b>\n<ul class="yolo-list">\n'
    for idx, obj in enumerate(yolo_objects, 1):
        class_id = obj.get("class_id")
        class_en, class_uk = YOLO_COCO_CLASSES.get(class_id, ["?", "?"])
        html += (
            f'<li>'
            f'#{idx}: <b>{class_uk}</b> ({class_en}) '
            f'BBox: {obj.get("bbox")}, '
            f'Ймовірність: {obj.get("conf"):.2f}'
            f'</li>\n'
        )
    html += '</ul></div>\n'
    return html

def generate_html_body(frames_info):
    html_body = ""
    for frame in frames_info:
        html_body += f'<div class="frame-card">\n'
        html_body += f'  <div class="frame-title">{frame["frame_name"]}</div>\n'
        html_body += f'  <img src="../data/frames/{frame["frame_name"]}" class="frame-img" alt="Кадр">\n'
        # Додаємо YOLO-детекції (навіть якщо faces немає)
        html_body += render_yolo_objects(frame.get("yolo_objects", []))
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