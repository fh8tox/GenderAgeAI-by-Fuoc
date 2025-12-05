# ============================
# train_age_group.py – Train Age Group Model (with plots)
# ============================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical

# Tạo thư mục lưu biểu đồ
os.makedirs("plots", exist_ok=True)

# =======================
# Load dữ liệu NPZ
# =======================
data = np.load(r"C:\Users\phuoc\Downloads\GenderAgeAI\data\processed\utkface_preprocessed.npz")
X = data['X']
ages = data['age']

# =======================
# Chuyển tuổi -> nhóm tuổi
# =======================
def age_to_group(age):
    if age <= 12: return 0
    elif age <= 19: return 1
    elif age <= 29: return 2
    elif age <= 39: return 3
    elif age <= 49: return 4
    elif age <= 59: return 5
    else: return 6

age_groups = np.array([age_to_group(a) for a in ages])
num_classes = 7

# Chuyển sang one-hot
age_groups_oh = to_categorical(age_groups, num_classes)

# =======================
# Train/Val/Test Split
# =======================
X_train, X_testval, y_train, y_testval = train_test_split(X, age_groups_oh, test_size=0.4, random_state=42)
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
# Build Age Group Model
# =======================
def build_age_group_model(input_shape):
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(base.input, output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = build_age_group_model(X_train.shape[1:])
model.summary()

# =======================
# Callback
# =======================
ckpt = ModelCheckpoint("best_model_age_group.keras", monitor="val_loss", save_best_only=True)
early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce = ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.2, min_lr=1e-5)

# =======================
# Train
# =======================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[ckpt, early, reduce]
)

# =======================
# Vẽ biểu đồ Loss & Accuracy
# =======================
plt.figure(figsize=(10, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curve (Age Group Model)")
plt.xlabel("Epoch")
plt.ylabel("Crossentropy Loss")
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy Curve (Age Group Model)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("plots/age_group_training_plot.png")
plt.show()

# =======================
# Evaluate Test Set
# =======================
pred_prob = model.predict(X_test)
pred = np.argmax(pred_prob, axis=1)
true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(true, pred))

print("Confusion Matrix:")
cm = confusion_matrix(true, pred)
print(cm)

# =======================
# Vẽ Confusion Matrix (Heatmap)
# =======================
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[f"Nhóm {i}" for i in range(num_classes)],
            yticklabels=[f"Nhóm {i}" for i in range(num_classes)])

plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.title("Confusion Matrix – Age Group Model")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")
plt.show()