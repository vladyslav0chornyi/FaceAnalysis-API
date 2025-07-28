import cv2
import csv
import os
import time
import numpy as np
import subprocess
from collections import deque
from datetime import datetime

from deepface import DeepFace
from insightface.app import FaceAnalysis

PHRASES_CSV = "phrases_and_videos.csv"
VIDEO_DIR = "videos"
LOG_FILE = "session_log.csv"
FACE_MEMORY_SECONDS = 5  # не показувати двічі одну фразу одній людині протягом цього часу

# Читання фраз і відповідних відео з CSV
def load_phrases(csv_path):
    phrases = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['min_age'] = int(row['min_age'])
            row['max_age'] = int(row['max_age'])
            phrases.append(row)
    return phrases

# Пошук відповідної фрази під атрибути
def select_phrase(phrases, age, gender, emotion):
    for ph in phrases:
        if ph['gender'] == gender and ph['min_age'] <= age <= ph['max_age']:
            # Якщо в CSV задано конкретну емоцію — перевіряємо
            if ph.get('emotion', '') and ph.get('emotion', '') != emotion:
                continue
            return ph
    return None

# Відтворення відео поверх усього (mpv або vlc)
def play_video_on_top(video_path):
    # Блокуючий виклик, відео закриється після програвання
    try:
        # mpv --ontop --no-terminal --force-window=yes video.mp4
        subprocess.Popen([
            "mpv",
            "--ontop",
            "--no-terminal",
            "--force-window=yes",
            "--geometry=50%:50%",
            "--autofit=640x360",
            video_path
        ])
    except FileNotFoundError:
        print("mpv не знайдено. Встановіть mpv або змініть на vlc.")

# Збереження події у лог
def log_event(face_id, age, gender, emotion, phrase, video_path):
    with open(LOG_FILE, "a", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(sep=' ', timespec='seconds'),
            face_id,
            age,
            gender,
            emotion,
            phrase,
            video_path
        ])

# Порівняння ембеддінгів (L2-норма)
def embedding_distance(e1, e2):
    return np.linalg.norm(e1 - e2)

def main():
    # Підготовка
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)
    if not os.path.exists(PHRASES_CSV):
        print(f"Файл {PHRASES_CSV} не знайдено!")
        return
    phrases = load_phrases(PHRASES_CSV)

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "face_id", "age", "gender", "emotion", "phrase", "video_path"])

    # ініціалізація детектора обличчя
    face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    # історія показаних фраз (face_id: (last_time, embedding))
    recent_faces = {}

    # Вебкамера
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не вдалося відкрити вебкамеру!")
        return

    print("Починаємо аналіз у реальному часі. Для виходу натисніть 'q' у вікні відео.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не вдалося зчитати кадр з вебкамери.")
            break

        faces = face_app.get(frame)
        if faces:
            for face in faces:
                embedding = face['embedding']
                age = int(face['age'])
                gender = face['gender']  # 'male'/'female'
                # Для DeepFace потрібне BGR->RGB
                x1, y1, x2, y2 = [int(v) for v in face['bbox']]
                cropped_face = frame[y1:y2, x1:x2]
                if cropped_face.size < 10:
                    continue
                try:
                    deep_result = DeepFace.analyze(cropped_face, actions=['emotion'], enforce_detection=False)
                    emotion = deep_result['dominant_emotion']
                except Exception:
                    emotion = 'neutral'

                gender_str = 'man' if gender == 'male' else 'woman'

                # Face ID (embedding) — пошук схожого
                found = False
                matched_face_id = None
                for face_id, (last_time, emb) in recent_faces.items():
                    if embedding_distance(embedding, emb) < 0.7:  # Поріг можна підібрати
                        found = True
                        matched_face_id = face_id
                        break

                now = time.time()
                if not found:
                    # Нова людина — генеруємо новий face_id
                    face_id = f"{now:.0f}_{np.random.randint(1000)}"
                    recent_faces[face_id] = (now, embedding)
                    phrase_row = select_phrase(phrases, age, gender_str, emotion)
                    if phrase_row:
                        video_path = phrase_row['video_path']
                        full_video_path = os.path.join(VIDEO_DIR, os.path.basename(video_path))
                        phrase = phrase_row['phrase']
                        print(f"НОВА ЛЮДИНА: {face_id}, {age}, {gender_str}, {emotion} => {phrase}")
                        play_video_on_top(full_video_path)
                        log_event(face_id, age, gender_str, emotion, phrase, full_video_path)
                    else:
                        print(f"Не знайдено фрази для: {age}, {gender_str}, {emotion}")
                else:
                    # Вже бачили цю людину — дивимось чи достатньо часу минуло
                    last_time, _ = recent_faces[matched_face_id]
                    if now - last_time > FACE_MEMORY_SECONDS:
                        recent_faces[matched_face_id] = (now, embedding)
                        phrase_row = select_phrase(phrases, age, gender_str, emotion)
                        if phrase_row:
                            video_path = phrase_row['video_path']
                            full_video_path = os.path.join(VIDEO_DIR, os.path.basename(video_path))
                            phrase = phrase_row['phrase']
                            print(f"ПОВТОРНА ЛЮДИНА: {matched_face_id}, {age}, {gender_str}, {emotion} => {phrase}")
                            play_video_on_top(full_video_path)
                            log_event(matched_face_id, age, gender_str, emotion, phrase, full_video_path)
                        else:
                            print(f"Не знайдено фрази для: {age}, {gender_str}, {emotion}")
                    else:
                        # Не відтворюємо — недавно вже було
                        pass

        # Відображення відео для зручності
        cv2.imshow("Webcam FaceAnalysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Вихід...")
            break

        # Очищення старих face_id
        to_del = []
        for face_id, (t, _) in recent_faces.items():
            if time.time() - t > 60:
                to_del.append(face_id)
        for face_id in to_del:
            del recent_faces[face_id]

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()