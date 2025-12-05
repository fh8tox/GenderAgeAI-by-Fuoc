import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# =======================
# CẤU HÌNH MODEL & THAM SỐ
# =======================
AGE_MODEL_PATH = r"C:\Users\phuoc\Downloads\GenderAgeAI\models\best_age_model.keras"
GENDER_MODEL_PATH = r"C:\Users\phuoc\Downloads\GenderAgeAI\models\best_gender_model.keras"
YOLO_MODEL_PATH = r"C:\Users\phuoc\Downloads\GenderAgeAI\models\yolov8n-face-lindevs.pt"

IMG_SIZE = 128
GENDER_LABELS = ["Nam", "Nu"]
CONF_THRESHOLD = 0.3
RESIZE_FACTOR = 0.5
FRAME_SKIP = 2

# =======================
# LOAD MODEL
# =======================
print("⏳ Loading age model...")
age_model = load_model(AGE_MODEL_PATH, compile=False)
print("✅ Age model loaded!")

print("⏳ Loading gender model...")
gender_model = load_model(GENDER_MODEL_PATH, compile=False)
print("✅ Gender model loaded!")

print("⏳ Loading YOLO face detector...")
yolo = YOLO(YOLO_MODEL_PATH)
print("✅ YOLO model loaded!")


# =======================
# HÀM DỰ ĐOÁN TUỔI & GIỚI TÍNH
# =======================
def predict_age_gender(face_img):
    img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # ---- Dự đoán tuổi (Regression) ----
    age_pred = age_model.predict(img, verbose=0)[0][0]
    age_pred = max(0, min(age_pred, 100))  # tránh giá trị âm

    # ---- Dự đoán giới tính ----
    gender_prob = gender_model.predict(img, verbose=0)[0][0]
    gender = GENDER_LABELS[int(gender_prob > 0.5)]

    return int(age_pred), gender


# =======================
# FACE CACHE
# =======================
frame_count = 0
faces_cache = []


def detect_faces(frame):
    global frame_count, faces_cache
    frame_count += 1

    small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

    if frame_count % FRAME_SKIP == 0:
        results = yolo(small_frame, verbose=False)
        new_faces = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                if conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box)

                # Chuyển ngược về kích thước thật
                x1 = int(x1 / RESIZE_FACTOR)
                y1 = int(y1 / RESIZE_FACTOR)
                x2 = int(x2 / RESIZE_FACTOR)
                y2 = int(y2 / RESIZE_FACTOR)

                if x2 - x1 < 40 or y2 - y1 < 40:
                    continue

                new_faces.append((x1, y1, x2, y2))

        faces_cache = new_faces.copy()

    return faces_cache


# =======================
# CHẠY REALTIME
# =======================
def run_webcam_realtime():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Không mở được webcam!")
        return

    print("🎥 Camera đang chạy... Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)

        for (x1, y1, x2, y2) in faces:
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            age, gender = predict_age_gender(face)
            label = f"{gender}, {age} tuoi"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Age & Gender Detection Realtime", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# =======================
# MAIN
# =======================
if __name__ == "__main__":
    run_webcam_realtime()
