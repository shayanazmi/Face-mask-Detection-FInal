"""
evaluate_model.py
Evaluates the trained model on the test dataset.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator




BASE_DIR = r"D:\college\sem4\project\Face mask Detection FInal"

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_mask_model.h5")
DATA_SPLIT_PATH = os.path.join(BASE_DIR, "data", "split")
TEST_PATH = os.path.join(DATA_SPLIT_PATH, "test")
RESULTS_PATH = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_PATH, exist_ok=True)


# ======================================================
# LOAD MODEL
# ======================================================

model = load_model(MODEL_PATH)
print("Loaded model from:", MODEL_PATH)


# ======================================================
# LOAD TEST DATA
# ======================================================

datagen = ImageDataGenerator(rescale=1./255)

test_generator = datagen.flow_from_directory(
    TEST_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)


# ======================================================
# PREDICT
# ======================================================

predictions = model.predict(test_generator)
y_pred = (predictions > 0.5).astype(int).flatten()
y_true = test_generator.classes


# ======================================================
# CLASSIFICATION REPORT
# ======================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=["with_mask", "without_mask"]
)

with open(os.path.join(RESULTS_PATH, "classification_report.txt"), "w") as f:
    f.write(report)


# ======================================================
# CONFUSION MATRIX
# ======================================================

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["with_mask", "without_mask"])

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(RESULTS_PATH, "confusion_matrix.png"))
plt.close()


# ======================================================
# ROC CURVE
# ======================================================

fpr, tpr, _ = roc_curve(y_true, predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(RESULTS_PATH, "roc_curve.png"))
plt.close()


print("Evaluation complete. Results saved to:", RESULTS_PATH)


RESULTS_PATH = r"D:\college\sem4\project\Face mask Detection FInal"

# Load training history from CSV
history_csv = os.path.join(RESULTS_PATH, "training_log.csv")
history = pd.read_csv(history_csv)

# Plot Accuracy
plt.figure()
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_PATH, "training_accuracy.png"))
plt.close()

# Plot Loss
plt.figure()
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_PATH, "training_loss.png"))
plt.close()

print("Training accuracy and loss graphs saved in:", RESULTS_PATH)