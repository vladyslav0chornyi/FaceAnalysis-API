# FaceAnalysis-API

**FaceAnalysis-API** — це Python-система для автоматизованого аналізу облич на зображеннях та відео, зручного формування HTML-звітів, а також моніторингу використання ресурсів. Проєкт використовує сучасні AI-моделі (InsightFace, OpenVINO), має REST API та модульну архітектуру для гнучкого розширення.

---

## Основний функціонал

- **Детекція та розпізнавання облич** на фото та відео
- **Витяг атрибутів** (стать, вік, емоції) для кожного обличчя
- **Генерація HTML-звітів** з результатами аналізу
- **REST API** для інтеграції з іншими системами (FastAPI/Flask)
- **Моніторинг ресурсів** (CPU, RAM) в процесі аналізу
- Підтримка моделей у форматах Intel/OpenVINO та ONNX

---

## Структура проєкту

```
├── config.py
├── core/
│   ├── analysis.py
│   ├── models.py
│   ├── resource_logging.py
│   └── video_processing.py
├── data/
│   ├── frames/
│   └── video.mp4
├── main.py
├── models/
│   ├── intel/
│   └── onnx/
├── photo/         # Вхідні зображення для аналізу
├── report/
│   ├── html_report.py
│   ├── report.html
│   ├── video_report.html
│   ├── report-body.html
│   └── report.css
├── requirements.txt
├── resource_log.txt
└── venv/          # Віртуальне оточення (ігнорувати при деплої)
```

---

## Встановлення

1. **Клонувати репозиторій**
   ```bash
   git clone https://github.com/vladyslav0chornyi/FaceAnalysis-API.git
   cd FaceAnalysis-API
   ```

2. **Створити та активувати віртуальне оточення**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Встановити залежності**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   > **Додатково:** Для роботи з OpenVINO переконайтесь, що встановлено [OpenVINO™ Toolkit](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html).

---

## Швидкий старт

1. **Запуск аналізу**
   ```bash
   python3 main.py
   ```
   За замовчуванням API стартує на [http://localhost:8000](http://localhost:8000)

2. **Використання API (FastAPI Swagger UI)**
   - Перейдіть у браузері: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Перевірте доступні ендпоінти, наприклад, `/analyze-photo`, `/analyze-video`.

3. **Аналіз фото через curl:**
   ```bash
   curl -X POST "http://localhost:8000/analyze-photo" -F "file=@photo/photo.jpg"
   ```

4. **Результати**
   - Кадри з відео: `data/frames/`
   - HTML-звіти: `report/report.html`, `report/video_report.html`
   - Логи ресурсів: `resource_log.txt`

---

## Приклад основних сценаріїв

### Аналіз фотографії

1. Додаєш фото у папку `photo/`
2. Викликаєш ендпоінт або запускаєш скрипт
3. Отримуєш результат у вигляді HTML-звіту у `report/`

### Аналіз відео

1. Кладеш відеофайл у `data/video.mp4`
2. Запускаєш обробку через API або напряму через `core/video_processing.py`
3. Кадри автоматично зберігаються у `data/frames/`, а звіт — у `report/video_report.html`

---

## Основні залежності

- Python 3.9+
- [FastAPI](https://fastapi.tiangolo.com/) або [Flask](https://flask.palletsprojects.com/)
- [InsightFace](https://github.com/deepinsight/insightface)
- [OpenVINO](https://github.com/openvinotoolkit/openvino)
- opencv-python
- numpy
- psutil
- Jinja2

**Повний перелік — у [requirements.txt](requirements.txt)**

---

## Advanced AI models

FaceAnalysis-API підтримує сучасні моделі машинного зору для різних завдань аналізу облич. Усі моделі оптимізовані для продуктивності та точності.

### 🔬 InsightFace Models
- **ArcFace**: Розпізнавання облич з високою точністю
- **RetinaFace**: Надійна детекція облич
- **GenderAge**: Визначення статі та віку
- **Mask Detection**: Детекція медичних масок

**Встановлення моделей:**
```bash
cd models/insightface/
# Завантажити моделі (див. README у папці)
wget -O arcface_r100_v1.onnx https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_r100_v1.onnx
```
**Документація**: [models/insightface/README.md](models/insightface/README.md)

### 🎭 BiSeNet (Face Parsing)
- **Призначення**: Сегментація частин обличчя (шкіра, волосся, очі, ніс, рот)
- **Формат**: PyTorch (.pth), ONNX

**Встановлення моделей:**
```bash
cd models/bisenet/
wget -O bisenet_v2_face_parsing.pth https://github.com/zllrunning/face-parsing.PyTorch/releases/download/v1.0/79999_iter.pth
```
**Документація**: [models/bisenet/README.md](models/bisenet/README.md)

### 🕸️ MediaPipe
- **FaceMesh**: 3D лендмарки обличчя (468 точок)
- **Face Detection**: Швидка детекція облич
- **Face Landmarks**: 2D лендмарки (68 ключових точок)

**Встановлення:**
```bash
# Моделі завантажуються автоматично MediaPipe
pip install mediapipe
```
**Документація**: [models/mediapipe/README.md](models/mediapipe/README.md)

### 🧠 DeepFace (Multi-Modal)
- **Emotion Recognition**: 7 базових емоцій
- **Age Estimation**: Оцінка віку (0-100 років)
- **Gender Classification**: Класифікація статі
- **Race Detection**: Визначення етнічної приналежності

**Встановлення:**
```bash
# Моделі завантажуються автоматично DeepFace
pip install deepface
```
**Документація**: [models/deepface/README.md](models/deepface/README.md)

### 📋 Швидкий старт з новими моделями

1. **Встановити залежності:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Завантажити потрібні моделі:**
   ```bash
   # Тільки для InsightFace та BiSeNet (інші завантажуються автоматично)
   bash scripts/download_models.sh  # (створіть цей скрипт при потребі)
   ```

3. **Інтеграція в код:**
   ```python
   # Приклад використання (див. повні приклади в README кожної моделі)
   from insightface.app import FaceAnalysis
   from deepface import DeepFace
   import mediapipe as mp
   
   # InsightFace
   app = FaceAnalysis()
   app.prepare(ctx_id=0)
   
   # DeepFace (вже інтегровано в проект)
   result = DeepFace.analyze(img, actions=['emotion', 'age', 'gender'])
   
   # MediaPipe
   mp_face_mesh = mp.solutions.face_mesh
   ```

**⚠️ Примітка**: Файли моделей не включені в репозиторій. Завантажуйте їх окремо згідно з інструкціями в README кожної папки.

---

## Моніторинг ресурсів

- **resource_log.txt** містить детальну інформацію про використання ресурсів (CPU, RAM) під час аналізу.

---

## Розширення та внесок

- Всі pull-requests та issue вітаються!
- Для розширення функціоналу додавайте нові модулі у папки `core/` або `report/`
- Питання — у [issues](https://github.com/vladyslav0chornyi/FaceAnalysis-API/issues)

---

## Ліцензія

MIT (або додайте свою)

---

**Автор**: [vladyslav0chornyi](https://github.com/vladyslav0chornyi)