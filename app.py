import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, render_template, Response, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import os

# Import đúng từ module realtime_detect
from src.realtime_detect import (
    predict_age_gender,
    detect_faces,
    yolo,
    CONF_THRESHOLD
)

# ==== Cấu hình ==== #
UPLOAD_FOLDER = "./static/uploads"
RESULT_FOLDER = "./static/results"
ALLOWED_IMG_EXT = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ==== Flask app ==== #
app = Flask(__name__)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ==== Helper ==== #
def allowed_file(filename, allowed_ext):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext


# ==== Stream webcam ==== #
def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
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

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ==== Routes ==== #
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename, ALLOWED_IMG_EXT):
        return redirect(request.url)

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    img = cv2.imread(save_path)
    if img is None:
        return "❌ Không đọc được ảnh!", 400

    # Phát hiện khuôn mặt bằng YOLO (chính xác)
    results = yolo(img, verbose=False)

    faces = []
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):
            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box)
            if x2 - x1 < 40 or y2 - y1 < 40:
                continue

            faces.append((x1, y1, x2, y2))

    # Không có mặt
    if len(faces) == 0:
        return render_template(
            'result_image.html',
            message="⚠️ Không phát hiện khuôn mặt!",
            filename=filename,
            result_image=None,
            predictions=[]
        )

    predictions = []

    # Dự đoán tuổi + giới tính cho từng khuôn mặt
    for (x1, y1, x2, y2) in faces:
        face = img[y1:y2, x1:x2]

        age, gender = predict_age_gender(face)
        predictions.append({'age': age, 'gender': gender})

        label = f"{gender}, {age} tuoi"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    result_path = os.path.join(RESULT_FOLDER, filename)
    cv2.imwrite(result_path, img)

    return render_template(
        'result_image.html',
        filename=filename,
        result_image=filename,
        predictions=predictions
    )


# ==== Run ==== #
if __name__ == "__main__":
    app.run(debug=True)
