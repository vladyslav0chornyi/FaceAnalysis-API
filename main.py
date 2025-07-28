from config import (
    VIDEO_PATH, OUTPUT_FRAMES_DIR, REPORT_TEMPLATE,
    REPORT_BODY, REPORT_OUTPUT, RESOURCE_LOG, FRAME_INTERVAL_SEC,
    OPENVINO_DEVICE, INSIGHTFACE_PROVIDER, INSIGHTFACE_CTX_ID
)
from core.resource_logging import start_resource_monitor
from core.models import init_face_analysis, init_openvino_models
from deepface import DeepFace
from core.video_processing import process_video
from report.html_report import generate_html_body, build_report

def deepface_analyze_func(cropped_face):
    return DeepFace.analyze(cropped_face, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)

def main():
    # 1. Запускаємо монітор ресурсів
    monitor_thread = start_resource_monitor(RESOURCE_LOG, 1)

    # 2. Завантаження моделей
    face_app = init_face_analysis(provider=INSIGHTFACE_PROVIDER, ctx_id=INSIGHTFACE_CTX_ID)
    openvino_models = init_openvino_models(device_name=OPENVINO_DEVICE)

    # 3. Обробка відео (можна змінити yolo_task на "detect", "segment", "pose")
    frames_info = process_video(
        video_path=VIDEO_PATH,
        output_frames_dir=OUTPUT_FRAMES_DIR,
        face_app=face_app,
        openvino_models=openvino_models,
        deepface_analyze_func=deepface_analyze_func,
        interval_sec=FRAME_INTERVAL_SEC,
        yolo_model_path="models/yolo/yolo11n.pt",
        yolo_conf=0.25,
        yolo_task="detect"  # або "segment", "pose" для відповідних моделей
    )

    # 4. Формування html body
    html_body = generate_html_body(frames_info)
    with open(REPORT_BODY, "w", encoding="utf-8") as f:
        f.write(html_body)

    # 5. Збірка фінального звіту
    build_report(REPORT_TEMPLATE, REPORT_BODY, REPORT_OUTPUT)

    monitor_thread.join(timeout=2)
    print(f"Готово! Відкрийте файл {REPORT_OUTPUT} у браузері.")
    print(f"Лог споживання ресурсів збережено у {RESOURCE_LOG}")
    print(f"HTML body збережено окремо у {REPORT_BODY}")

if __name__ == "__main__":
    main()