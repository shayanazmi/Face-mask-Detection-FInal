"""
train_model.py
Contains:
- Data Augmentation
- Data Generators
- VGG16 Transfer Learning Model
- Training Loop (model.fit)
- Saves best model + training logs
"""

import os
import time
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, Callback

BASE_DIR = r"D:\college\sem4\project\Face mask Detection FInal"

# Dataset split directory
SPLIT_PATH = os.path.join(BASE_DIR, "data", "split")

# Training folders
TRAIN_DIR = os.path.join(SPLIT_PATH, "train")
VALID_DIR = os.path.join(SPLIT_PATH, "valid")
TEST_DIR  = os.path.join(SPLIT_PATH, "test")

# Model save location
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "best_mask_model.h5")

# ===============================================================
# Custom Console Dashboard Callback
# ===============================================================
class ConsoleDashboard(Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print("\n" + "=" * 60)
        print(f"{'EPOCH':<8} | {'LOSS':<8} | {'ACC':<8} | {'VAL_LOSS':<10} | {'VAL_ACC':<8}")
        print("-" * 60)

    def on_epoch_end(self, epoch, logs=None):
        print(f"{epoch+1:<8} | {logs['loss']:<8.4f} | {logs['accuracy']:<8.4f} | "
              f"{logs['val_loss']:<10.4f} | {logs['val_accuracy']:<8.4f}")

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print("=" * 60)
        print(f"Training Complete! Total Time: {total_time/60:.2f} minutes")


# ===============================================================
# Data Augmentation
# ===============================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
    channel_shift_range=20.0,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(SPLIT_PATH, "train"),
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(SPLIT_PATH, "valid"),
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)


# ===============================================================
# Model Architecture (VGG16 Transfer Learning)
# ===============================================================
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = Sequential(vgg.layers)
for layer in model.layers:
    layer.trainable = False

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)


# ===============================================================
# Callbacks
# ===============================================================
checkpoint = ModelCheckpoint(
    "best_mask_model.h5",
    monitor="val_auc",
    save_best_only=True,
    mode="max"
)

csv_logger = CSVLogger("training_log.csv")
dashboard = ConsoleDashboard()


# ===============================================================
# TRAINING
# ===============================================================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[checkpoint, csv_logger, dashboard]
)

print("\nTraining finished. Best model saved as best_mask_model.h5")