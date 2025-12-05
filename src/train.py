from xml.parsers.expat import model
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix

# =====================
# Load dữ liệu NPZ
# =====================
data = np.load(r"C:\Users\phuoc\Downloads\GenderAgeAI\data\processed\utkface_preprocessed.npz")
X = data['X']
ages = data['age']
genders = data['gender']

# =====================
# Chia train/val/test
# =====================
X_train, X_testval, y_age_train, y_age_testval, y_gender_train, y_gender_testval = train_test_split(
    X, ages, genders, test_size=0.4, random_state=42
)

X_val, X_test, y_age_val, y_age_test, y_gender_val, y_gender_test = train_test_split(
    X_testval, y_age_testval, y_gender_testval, test_size=0.5, random_state=42
)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# =====================
# Data augmentation
# =====================
datagen_train = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)

datagen_val = ImageDataGenerator()  # Không augment cho validation

batch_size = 64

train_gen_age = datagen_train.flow(X_train, y_age_train, batch_size=batch_size)
val_gen_age = datagen_val.flow(X_val, y_age_val, batch_size=batch_size)

train_gen_gender = datagen_train.flow(X_train, y_gender_train, batch_size=batch_size)
val_gen_gender = datagen_val.flow(X_val, y_gender_val, batch_size=batch_size)

# =====================
# Model Age Prediction
# =====================
def build_age_model(input_shape):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation=None)(x)  # Regression
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

age_model = build_age_model(X_train.shape[1:])
age_model.summary()

checkpoint_age = ModelCheckpoint('best_model_age.keras', monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)

history_age = age_model.fit(
    train_gen_age,
    validation_data=val_gen_age,
    epochs=50,
    callbacks=[checkpoint_age, early_stop, reduce_lr]
)

# =====================
# Model Gender Prediction
# =====================
def build_gender_model(input_shape):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)  # Binary classification
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

gender_model = build_gender_model(X_train.shape[1:])
gender_model.summary()

checkpoint_gender = ModelCheckpoint('best_model_gender.keras', monitor='val_loss', save_best_only=True)

history_gender = gender_model.fit(
    train_gen_gender,
    validation_data=val_gen_gender,
    epochs=50,
    callbacks=[checkpoint_gender, early_stop, reduce_lr]
)

# =====================
# Đánh giá test seô
# =====================
# Age
y_age_pred = age_model.predict(X_test).flatten()
print("MAE tuổi:", mean_absolute_error(y_age_test, y_age_pred))

# Gender
y_gender_pred = (gender_model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_gender_test, y_gender_pred))
cm = confusion_matrix(y_gender_test, y_gender_pred)
print("Confusion matrix:\n", cm)

