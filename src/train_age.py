# ============================
# train_age_resume.py – Train + Resume Age Model (FULL VERSION)
# ============================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# =======================
# Tạo thư mục
# =======================
os.makedirs("plots", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

CHECKPOINT_PATH = "checkpoints/best_age_model.keras"
HISTORY_PATH = "checkpoints/history.json"


# =======================
# Load dữ liệu NPZ
# =======================
data = np.load(r"C:\Users\phuoc\Downloads\GenderAgeAI\data\processed\utkface_preprocessed.npz")
X = data['X']
ages = data['age']

# =======================
# Train/Val/Test Split
# =======================
X_train, X_testval, y_train, y_testval = train_test_split(X, ages, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_testval, y_testval, test_size=0.5, random_state=42)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)


# =======================
# Data augmentation
# =======================
datagen_train = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)

datagen_val = ImageDataGenerator()

batch_size = 64
train_gen = datagen_train.flow(X_train, y_train, batch_size=batch_size)
val_gen = datagen_val.flow(X_val, y_val, batch_size=batch_size)



# =======================
# Build Age Model
# =======================
def build_age_model(input_shape):
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1)(x)
    model = Model(base.input, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="mse", metrics=["mae"])
    return model


# =======================
# Callback: Lưu epoch
# =======================
class EpochSaver(tf.keras.callbacks.Callback):
    """Lưu epoch hiện tại để lần sau tiếp tục."""
    def on_epoch_end(self, epoch, logs=None):
        with open("checkpoints/epoch.txt", "w") as f:
            f.write(str(epoch + 1))


# =======================
# Callback: Lưu toàn bộ lịch sử train
# =======================
class HistorySaver(tf.keras.callbacks.Callback):
    """Lưu lịch sử train/val vào history.json để ghép biểu đồ."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Nếu file chưa có thì tạo mới
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r") as f:
                history = json.load(f)
        else:
            history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}

        # Thêm log mới
        history["loss"].append(float(logs.get("loss", 0)))
        history["val_loss"].append(float(logs.get("val_loss", 0)))
        history["mae"].append(float(logs.get("mae", 0)))
        history["val_mae"].append(float(logs.get("val_mae", 0)))

        # Ghi lại file
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=4)


# =======================
# Load model nếu có checkpoint
# =======================
initial_epoch = 0

if os.path.exists(CHECKPOINT_PATH):
    print("🔄 Checkpoint detected! Loading model để train tiếp...")
    model = load_model(CHECKPOINT_PATH)

    # Lấy epoch hiện tại từ file
    if os.path.exists("checkpoints/epoch.txt"):
        with open("checkpoints/epoch.txt", "r") as f:
            initial_epoch = int(f.read())

    print(f"▶️ Tiếp tục train từ epoch {initial_epoch}")

else:
    print("🆕 Không có checkpoint → Train mới từ đầu")
    model = build_age_model(X_train.shape[1:])


# =======================
# Callbacks
# =======================
ckpt = ModelCheckpoint(
    CHECKPOINT_PATH,
    monitor="val_loss",
    save_best_only=False,
    save_weights_only=False
)

early = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=False)
reduce = ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.2, min_lr=1e-6)


# =======================
# Train (tiếp tục hoặc mới)
# =======================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    initial_epoch=initial_epoch,
    epochs=50,
    callbacks=[ckpt, early, reduce, EpochSaver(), HistorySaver()]
)


# =======================
# VẼ BIỂU ĐỒ GHÉP NHIỀU LẦN TRAIN
# =======================
if os.path.exists(HISTORY_PATH):
    print("📊 Đang load toàn bộ history từ các lần train trước...")
    with open(HISTORY_PATH, "r") as f:
        full_history = json.load(f)

    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(full_history["loss"], label="Train Loss")
    plt.plot(full_history["val_loss"], label="Val Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()

    # MAE
    plt.subplot(1, 2, 2)
    plt.plot(full_history["mae"], label="Train MAE")
    plt.plot(full_history["val_mae"], label="Val MAE")
    plt.title("Training MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/age_training_plot_full.png")
    plt.show()
else:
    print("⚠ Chưa có history.json để vẽ biểu đồ ghép.")


# =======================
# Evaluate Test Set
# =======================
pred = model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, pred)
print("MAE tuổi:", mae)

