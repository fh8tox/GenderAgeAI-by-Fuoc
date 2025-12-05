import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==== Cấu hình đường dẫn ====
RAW_DIR = r"C:\Users\phuoc\Downloads\GenderAgeAI\data\raw\UTKFace"
OUTPUT_DIR = r"C:\Users\phuoc\Downloads\GenderAgeAI\data\processed"
LABEL_DIR = os.path.join(OUTPUT_DIR, "labels")
os.makedirs(LABEL_DIR, exist_ok=True)

# ==== Kích thước ảnh đầu vào ====
IMG_SIZE = 128

def parse_filename(filename):
    """
    Tách thông tin age, gender từ tên file.
    Ví dụ: 23_1_0_20170109150557335.jpg.chip.jpg
    Trả về: (age, gender)
    """
    try:
        parts = filename.split("_")
        age = int(parts[0])
        gender = int(parts[1])
        return age, gender
    except Exception:
        return None, None

def load_utkface_dataset():
    X, ages, genders = [], [], []

    print(f"Đang đọc ảnh từ thư mục: {RAW_DIR}")

    for file in tqdm(os.listdir(RAW_DIR)):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            filepath = os.path.join(RAW_DIR, file)
            age, gender = parse_filename(file)
            if age is None:
                continue

            img = cv2.imread(filepath)
            if img is None:
                continue

            # Resize về kích thước chuẩn
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            ages.append(age)
            genders.append(gender)

    X = np.array(X, dtype="float32") / 255.0  # chuẩn hóa 0–1
    ages = np.array(ages)
    genders = np.array(genders)

    print(f"\n✅ Đã đọc {len(X)} ảnh thành công.")
    return X, ages, genders

def save_npz(X, ages, genders):
    npz_path = os.path.join(OUTPUT_DIR, "utkface_preprocessed.npz")
    np.savez(npz_path, X=X, age=ages, gender=genders)
    print(f"💾 Đã lưu dữ liệu tiền xử lý tại: {npz_path}")

def save_csv(ages, genders):
    df = pd.DataFrame({
        "age": ages,
        "gender": genders
    })
    csv_age = os.path.join(LABEL_DIR, "age_labels.csv")
    csv_gender = os.path.join(LABEL_DIR, "gender_labels.csv")
    df["age"].to_csv(csv_age, index=False)
    df["gender"].to_csv(csv_gender, index=False)
    print(f"📑 Đã lưu nhãn tại: {LABEL_DIR}")

if __name__ == "__main__":
    X, ages, genders = load_utkface_dataset()
    save_npz(X, ages, genders)
    save_csv(ages, genders)
