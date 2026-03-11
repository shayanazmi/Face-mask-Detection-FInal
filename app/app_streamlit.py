"""
app_streamlit.py
Real-time Face Mask Detection Dashboard
--------------------------------------------------
Uses:
- VGG16 Transfer Learning Mask Classifier
- Haarcascade Face Detector
- Streamlit UI Dashboard

This file handles real-time inference, statistics tracking,
and user feedback capture. 
"""

import streamlit as st
import cv2
import numpy as np
import os
import time
from tensorflow.keras.models import load_model

# ======================================================================
# 1. PAGE SETUP & FIXED PATHS
# ======================================================================
st.set_page_config(page_title="Face Mask Detection using transfer learning", layout="wide")

# NEW OFFICIAL PROJECT BASE PATH
BASE_DIR = r"D:\college\sem4\project\Face mask Detection FInal"

# UPDATED PATHS ACCORDING TO NEW FOLDER STRUCTURE
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_mask_model.h5")
HAAR_PATH  = os.path.join(BASE_DIR, "models", "haarcascade_frontalface_default.xml")

# Streamlit feedback capture folder lives inside app/
SAVE_DIR = os.path.join(BASE_DIR, "app", "session_captures")

# Create capture folders if missing
for sub in ["correct/mask", "correct/no_mask", "incorrect/mask", "incorrect/no_mask"]:
    os.makedirs(os.path.join(SAVE_DIR, sub), exist_ok=True)

THRESHOLD = 0.65  # Fixed as per original script


# ======================================================================
# 2. LOAD MODEL & HAARCASCADE (Cached for Performance)
# ======================================================================
@st.cache_resource
def load_components():
    model = load_model(MODEL_PATH)
    haar = cv2.CascadeClassifier(HAAR_PATH)
    return model, haar


model, haar = load_components()


# ======================================================================
# 3. SESSION STATE INITIALIZATION
# ======================================================================
if "total" not in st.session_state:
    st.session_state.total = 0
    st.session_state.masks = 0
    st.session_state.no_masks = 0
    st.session_state.conf_history = []
    st.session_state.lat_history = []


# ======================================================================
# 4. UI LAYOUT
# ======================================================================
st.title("Face Mask Detection Dashboard")

# Top statistic cards
c1, c2, c3, c4 = st.columns(4)
total_card = c1.empty()
mask_card = c2.empty()
no_mask_card = c3.empty()
comp_card = c4.empty()

st.markdown("---")

# Prediction cards
m1, m2, m3 = st.columns(3)
status_card = m1.empty()
conf_card = m2.empty()
lat_card = m3.empty()

# Main layout: video + charts
col_feed, col_charts = st.columns([2, 1])

with col_feed:
    video_display = st.empty()
    b1, b2 = st.columns(2)
    save_ok = b1.button("✅ Mark Correct")
    save_bad = b2.button("❌ Mark Incorrect")

with col_charts:
    st.subheader("Confidence Trend")
    conf_plot = st.empty()

    st.subheader("Inference Latency Histogram")
    lat_plot = st.empty()

# Sidebar
st.sidebar.header("System Controls")
run = st.sidebar.checkbox("Start Webcam", value=False)
st.sidebar.info(f"Threshold: {THRESHOLD}")
st.sidebar.write(f"Model: {os.path.basename(MODEL_PATH)}")


# ======================================================================
# 5. REAL-TIME INFERENCE LOOP
# ======================================================================
if run:
    cap = cv2.VideoCapture(0)

    while run:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            t0 = time.time()

            face_crop = frame[y:y+h, x:x+w]

            # Preprocessing
            img = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img / 255.0, axis=0)

            # Prediction
            pred = model.predict(img, verbose=0)[0][0]
            latency = (time.time() - t0) * 1000  # ms

            label_id = 0 if pred < THRESHOLD else 1
            if label_id == 0:
                label = "MASK ON"
                color = (0, 255, 0)
                confidence = (1 - pred) * 100
                st.session_state.masks += 1
            else:
                label = "NO MASK"
                color = (0, 0, 255)
                confidence = pred * 100
                st.session_state.no_masks += 1

            st.session_state.total += 1

            # Append to history
            st.session_state.conf_history.append(confidence)
            st.session_state.lat_history.append(latency)

            if len(st.session_state.conf_history) > 30:
                st.session_state.conf_history.pop(0)
            if len(st.session_state.lat_history) > 100:
                st.session_state.lat_history.pop(0)

            # Update UI counters
            total_card.metric("Total Scanned", st.session_state.total)
            mask_card.metric("With Mask", st.session_state.masks)
            no_mask_card.metric("No Mask", st.session_state.no_masks)

            comp = (st.session_state.masks / st.session_state.total) * 100
            comp_card.metric("Compliance %", f"{comp:.1f}%")

            # Prediction cards
            status_card.markdown(f"### {'✅ SAFE' if label_id == 0 else '⚠️ WARNING'}")
            conf_card.metric("Confidence", f"{confidence:.1f}%")
            lat_card.metric("Latency", f"{latency:.1f} ms")

            # Charts
            conf_plot.line_chart(st.session_state.conf_history)
            lat_plot.bar_chart(np.histogram(st.session_state.lat_history, bins=10)[0])

            # Drawing on frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
            cv2.putText(frame, label, (x+5, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

            # Saving feedback frames
            if save_ok or save_bad:
                status = "correct" if save_ok else "incorrect"
                cls = "mask" if label_id == 0 else "no_mask"
                fname = os.path.join(SAVE_DIR, status, cls, f"{int(time.time())}.jpg")
                cv2.imwrite(fname, face_crop)
                st.toast(f"Saved to {status}/{cls}", icon="💾")

            break  # Process one face for speed

        video_display.image(frame, channels="BGR", use_container_width=True)

    cap.release()

else:
    st.info("System in Standby. Enable 'Start Webcam' to begin.")

