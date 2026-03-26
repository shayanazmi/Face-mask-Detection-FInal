# Face Mask Detection System

### Real-Time Face Mask Detection using VGG16 Transfer Learning and Haar-Cascade

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![Accuracy](https://img.shields.io/badge/Accuracy-99%25-green.svg)]()

---

## 🎯 Overview

A hybrid face mask detection system combining **Haar-Cascade** for fast face detection and **VGG16** for accurate classification. Achieves **99% accuracy** with real-time performance via Streamlit dashboard.

**Key Highlights:**
- Two-stage detection: Haar-Cascade (face localization) → VGG16 (mask classification)
- Transfer learning with frozen VGG16 base + custom classification head
- SMOTE-balanced dataset from PyImageSearch & Kaggle sources
- Real-time Streamlit dashboard with live metrics

---



## 🔧 Installation

```bash
# Clone repository
git clone https://github.com/shayanazmi/Face-mask-Detection-FInal.git
cd Face-mask-Detection-FInal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Real-Time Detection Dashboard

```bash
streamlit run streamlit_app/app.py
```

### Train Model

```bash
python src/train.py --epochs 20 --batch-size 32
```

### Evaluate Model

```bash
python src/evaluate.py --model models/best_mask_model.h5
```

---

## 🧠 Model Architecture

```
Input (224×224×3)
    ↓
VGG16 Base (Frozen) - 14.7M params
    ↓
Flatten (25,088)
    ↓
Dense (128, ReLU) - 3.2M params
    ↓
Dropout (0.5)
    ↓
Output (1, Sigmoid)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 32 |
| Epochs | 20 |
| Dropout | 0.5 |
| Decision Threshold | 0.65 |

---

## 📊 Dataset

| Split | With Mask | Without Mask | Total |
|-------|-----------|--------------|-------|
| Train | 3,058 | 3,123 | 6,181 |
| Validation | 655 | 669 | 1,324 |
| Test | 656 | 670 | 1,326 |

**Sources:** PyImageSearch COVID-19 Dataset + Kaggle Face Mask Dataset

**Preprocessing:** SMOTE balancing, 224×224 resize, normalization, augmentation

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Accuracy | 99% |
| Precision | 99% |
| Recall | 99% |
| F1-Score | 99% |
| AUC-ROC | 0.9987 |

### Confusion Matrix

```
              Predicted
            Mask  No Mask
Actual Mask  649      7
      No Mask   8    662
```

---

## 🛠️ Technologies

- **TensorFlow/Keras** - Deep learning
- **OpenCV** - Haar-Cascade face detection
- **Streamlit** - Real-time dashboard
- **Scikit-learn** - SMOTE, metrics
- **NumPy/Pandas** - Data processing


---

## 📧 Contact

**Shayan Azmi** - [@shayanazmi](https://github.com/shayanazmi)

**Repository:** https://github.com/shayanazmi/Face-mask-Detection-FInal

---

<div align="center">

⭐ **Star this repo if you found it helpful!**

</div>
```
