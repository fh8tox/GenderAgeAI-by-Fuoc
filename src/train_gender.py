# ============================
# train_gender_resume.py – Train + Resume Gender Model (FULL VERSION)
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
from sklearn.metrics import classification_report, confusion_matrix

# =======================
# Tạo thư mục
# =======================
os.makedirs("plots", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

CHECKPOINT_PATH = "checkpoints/best_gender_model.keras"
EPOCH_FILE = "checkpoints/epoch_gender.txt"
HISTORY_PATH = "checkpoints/history_gender.json"

# =======================
# Load dữ liệu NPZ
# =======================
data = np.load(r"C:\Users\phuoc\Downloads\GenderAgeAI\data\processed\utkface_preprocessed.npz")
X = data['X']
genders = data['gender']

# =======================
# Train/Val/Test Split
# =======================
X_train, X_testval, y_train, y_testval = train_test_split(X, genders, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_testval, y_testval, test_size=0.5, random_state=42)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# =======================
# Data Augmentation
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
# Build Gender Model
# =======================
def build_gender_model(input_shape):
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(base.input, output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# =======================
# Load model nếu có checkpoint
# =======================
initial_epoch = 0

if os.path.exists(CHECKPOINT_PATH):
    print("🔄 Checkpoint detected → Loading model để train tiếp...")
    model = load_model(CHECKPOINT_PATH)

    if os.path.exists(EPOCH_FILE):
        with open(EPOCH_FILE, "r") as f:
            initial_epoch = int(f.read())

    print(f"▶️ Tiếp tục train từ epoch {initial_epoch}")

else:
    print("🆕 Không có checkpoint → Train từ đầu")
    model = build_gender_model(X_train.shape[1:])


# =======================
# Callback: Lưu epoch
# =======================
class EpochSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open(EPOCH_FILE, "w") as f:
            f.write(str(epoch + 1))


# =======================
# Callback: Lưu toàn bộ lịch sử train
# =======================
class HistorySaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Load hoặc tạo mới
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r") as f:
                history = json.load(f)
        else:
            history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

        history["loss"].append(float(logs.get("loss", 0)))
        history["val_loss"].append(float(logs.get("val_loss", 0)))
        history["accuracy"].append(float(logs.get("accuracy", 0)))
        history["val_accuracy"].append(float(logs.get("val_accuracy", 0)))

        # Lưu file
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=4)


# =======================
# Callbacks
# =======================
ckpt = ModelCheckpoint(
    CHECKPOINT_PATH,
    monitor="val_loss",
    save_best_only=False,   # Lưu tất cả epoch (để resume chuẩn)
    save_weights_only=False
)

early = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=False)
reduce = ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.2, min_lr=1e-6)


# =======================
# Train
# =======================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    initial_epoch=initial_epoch,
    epochs=50,
    callbacks=[ckpt, early, reduce, EpochSaver(), HistorySaver()]
)


# =======================
# Plot Full Training History (ghép nhiều lần train)
# =======================
if os.path.exists(HISTORY_PATH):
    print("📊 Loading full training history...")
    with open(HISTORY_PATH, "r") as f:
        full_history = json.load(f)

    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(full_history["loss"], label="Train Loss")
    plt.plot(full_history["val_loss"], label="Val Loss")
    plt.title("Gender Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Crossentropy")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(full_history["accuracy"], label="Train Acc")
    plt.plot(full_history["val_accuracy"], label="Val Acc")
    plt.title("Gender Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/gender_training_plot_full.png")
    plt.show()
else:
    print("⚠ Chưa có file history_gender.json để vẽ biểu đồ full.")


# =======================
# Evaluate Test
# =======================
pred_prob = model.predict(X_test)
pred = (pred_prob > 0.5).astype(int)

print(classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

