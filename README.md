🧠 Gender & Age Prediction AI
Xác định Giới tính và Dự đoán Tuổi từ Khuôn mặt qua Ảnh hoặc Webcam
📌 Giới thiệu

Dự án xây dựng một hệ thống AI có khả năng:

Phát hiện khuôn mặt từ ảnh hoặc webcam (real-time)

Xác định giới tính (Male / Female)

Dự đoán tuổi (Regression)

Hiển thị bounding box + thông tin lên ảnh/video

Hệ thống sử dụng YOLOv8 để phát hiện khuôn mặt và MobileNetV2 để phân loại giới tính và dự đoán tuổi.
Giao diện demo được xây dựng bằng Flask và OpenCV.

🚀 Công nghệ sử dụng
Thành phần	Công nghệ
Phát hiện khuôn mặt	YOLOv8 (Ultralytics)
Mô hình dự đoán giới tính	MobileNetV2 (Binary Classification)
Mô hình dự đoán tuổi	MobileNetV2 (Regression + MAE Loss)
Web demo	Flask
Xử lý ảnh	OpenCV
Tăng cường dữ liệu	ImageDataGenerator
🏗 Kiến trúc hệ thống (Pipeline dự đoán)

Nhận ảnh đầu vào (webcam hoặc upload ảnh)

YOLOv8 phát hiện khuôn mặt

Crop khuôn mặt bằng OpenCV

Tiền xử lý (resize, normalize)

Mô hình Gender Model dự đoán giới tính

Mô hình Age Model dự đoán tuổi

Ghép kết quả vào ảnh và trả về giao diện hoặc webcam

📊 Huấn luyện mô hình
Data Augmentation

Đã sử dụng các kỹ thuật tăng cường:

rotation_range=10,
width_shift_range=0.05,
height_shift_range=0.05,
zoom_range=0.1,
horizontal_flip=True

Huấn luyện & Lưu mô hình

Train từng mô hình (gender/age) trên tập train + validation

Lưu mô hình tốt nhất qua ModelCheckpoint (.keras hoặc .h5)

Tự động sinh biểu đồ Loss / Accuracy

Hỗ trợ resume training (train tiếp từ checkpoint trước đó)


▶️ Chạy demo Flask
Cài thư viện
pip install -r requirements.txt

Chạy server
python run.py


Giao diện demo sẽ chạy tại:

http://127.0.0.1:5000/

🎥 Chạy mode webcam
python src/realtime_detect.py


Nhấn Q để thoát webcam.

📌 Kết quả đạt được

Gender Model Accuracy: 90–92%

Age Model MAE: ~6.5 tuổi

Hoạt động tốt trong môi trường thực, ánh sáng không ổn định, khuôn mặt nghiêng
