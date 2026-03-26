# Face Mask Detection using Transfer Learning using VGG16 😷

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

A professional real-time Face Mask Detection system featuring an interactive AI Dashboard. This project utilizes **VGG16 Transfer Learning** for classification and **Haar Cascades** for face localization, achieving a high **93% classification accuracy**.

## 📊 Project Highlights
- **93% Test Accuracy:** Rigorously evaluated on a 1,326-image unseen test set.
- **AI Dashboard:** A Streamlit-based interface providing real-time compliance metrics, confidence trends, and latency monitoring.
- **Advanced Preprocessing:** Custom logic to maintain image aspect ratio through resizing and black-padding, ensuring zero distortion for the VGG16 model.
- **Active Learning:** Built-in feedback system allows users to mark detections as "Correct" or "Incorrect," automatically saving data for future retraining.

## 📁 Project Structure & Scripts
| File | Description |
| :--- | :--- |
| **`data_code_ver2.py`** | **Training Core:** Handles 70/15/15 data splitting, aspect ratio normalization, and VGG16 Transfer Learning training. |
| **`new_inference_code.py`** | **AI Dashboard:** The primary Streamlit application for real-time monitoring and advanced analytics. |
| **`ver2_evaluation.py`** | **Advanced Metrics:** Suite to calculate ROC-AUC, Mean IoU, Dice Coefficient, and mAP. |
| **`live_webcam.py`** | **Direct Inference:** Standard script focused on core webcam detection logic. |
| **`my_mask_detector_v2_2.h5`** | **The Brain:** The final trained VGG16-based model weights. |
| **`haarcascade_frontalface_default.xml`** | **Face Localization:** Pre-trained XML used for high-speed face detection. |

## 📈 Model Performance (V2.2 Evaluation)

The model delivers robust performance across both classes:

| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **With Mask** | 0.95 | 0.90 | 0.93 |
| **Without Mask** | 0.91 | 0.96 | 0.93 |

**Object Detection Metrics:**
- **Mean IoU (Overlap):** 0.3125
- **Dice Coefficient:** 0.4762
- **Mean Absolute Error:** 10.62 pixels

## 🧠 Technical Workflow
1. **Preprocessing:** Images are normalized to $224 \times 224$ pixels while preserving aspect ratio.
2. **Architecture:** Base VGG16 (ImageNet weights) with a custom head:
   - `Flatten` → `Dense(128, ReLU)` → `Dropout(0.5)` → `Dense(1, Sigmoid)`
3. **Inference:** Haar Cascade detects face ROIs $\rightarrow$ VGG16 classifies the ROI $\rightarrow$ Result displayed via Streamlit with a **0.65 classification threshold**.

## 📦 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shayanazmi/Face_Mask_Detection_using_Transfer_Learning_CNN_HaarCascade.git
   cd Face_Mask_Detection_using_Transfer_Learning_CNN_HaarCascade
