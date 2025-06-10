# AI-Based Gesture Recognition SOS Alert System 🚨🖐️✊

## 📌 Project Overview

This project is an AI-powered real-time gesture recognition system designed to trigger SOS alerts using simple hand gestures captured through a webcam. It enhances emergency communication, especially in environments like vehicles, hospitals, or for individuals with physical disabilities.

- 👋 **Wave Gesture** – Triggers an emergency SOS with location sharing.
- ✊ **Fist Gesture** – Cancels the alert.

Built using **Python**, **OpenCV**, **MediaPipe**, and an **SVM classifier**, the system also integrates **GPS tracking** to provide real-time location data when an SOS is raised.

---

## 💡 Key Features

- 🔍 Real-time gesture detection via webcam.
- 🧠 AI model using **SVM** (Support Vector Machine).
- 📍 Auto-fetches GPS location using IP-based geolocation.
- 📤 Sends SOS alert (can be integrated via SMS/email APIs).
- 💻 Simple UI feedback on gesture recognition and alert status.
- 💡 Designed for low-latency, offline-first response.

---

## 🛠️ Tech Stack

### 👨‍💻 Software:
- Python 3.x
- OpenCV
- MediaPipe
- scikit-learn
- pandas, numpy
- geocoder
- joblib
- requests

### 💻 Hardware:
- Laptop/Desktop
- Integrated/External Webcam
- Internet Connection (for IP-based geolocation)

---

## 🧠 How It Works

1. The webcam captures live video feed.
2. MediaPipe detects and tracks hand landmarks.
3. Extracted features (e.g., keypoint distances) are fed to a pre-trained **SVM classifier**.
4. If a "wave" is detected → system triggers SOS and fetches location.
5. If a "fist" is detected → it cancels any active alert.

---

## 🧪 Model Training

- Dataset: Custom-labeled gesture image set (Wave, Fist).
- Preprocessing: Landmarks extraction from hand gestures.
- Classifier: **SVM (Support Vector Machine)** using scikit-learn.
- Saved model: `gesture_model.pkl` (via joblib)

---

## 🚀 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/gesture-sos-alert.git
   cd gesture-sos-alert
2.Create virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
