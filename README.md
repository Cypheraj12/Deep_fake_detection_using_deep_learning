# 🕵️ DeepFake Image Detection & Digital Forensics Lab

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deepfakedetectionusingdeeplearning-gyrnacaq2wanhxbpxvuc6r.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://deepfakedetectionusingdeeplearning-gyrnacaq2wanhxbpxvuc6r.streamlit.app/)

An interactive, cloud-ready deep learning application that inspects face images to detect AI-generated and manipulated facial imagery using **MobileNetV2 Transfer Learning** combined with **Error Level Analysis (ELA)**.

---

## 🚀 Live Demo

Experience the live interactive application hosted on Streamlit Cloud:  
👉 **[DeepFake Detection & Digital Forensics Lab · Streamlit](https://deepfakedetectionusingdeeplearning-gyrnacaq2wanhxbpxvuc6r.streamlit.app/)**

---

## ✨ Features

- **Interactive Streamlit Web Dashboard**: Upload any JPG, JPEG, or PNG portrait image for instant analysis.
- **MobileNetV2 Deep Feature Extraction**: Evaluates facial representation layers using ImageNet weights and custom classification heads.
- **Error Level Analysis (ELA)**: Real-time digital forensic heatmap visualizing compression rate differentials and digital tampering edges.
- **Confidence Meters & Probability Scoring**: Displays real vs. fake certainty scores with visual progress meters.
- **Web Audio Alert Engine**: Cross-platform synthetic alarm sound triggers automatically when a manipulated deepfake is detected.
- **Configurable Sensitivity**: Real-time slider to adjust forensic detection thresholds.

---

## 🧠 Model Architecture & Forensic Pipeline

- **Backbone**: MobileNetV2 (ImageNet Pretrained)
- **Top Layers**: Global Average Pooling 2D $\rightarrow$ Dense (128, ReLU) $\rightarrow$ Dropout (0.5) $\rightarrow$ Dense (1, Sigmoid)
- **Input Resolution**: 128 × 128 px
- **Forensic Blending**: Dual verification merging neural confidence scores with ELA pixel variance metrics.

---

## 💻 Local Setup & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/Cypheraj12/Deep_fake_detection_using_deep_learning.git
cd Deep_fake_detection_using_deep_learning
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit Application
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```bash
Deep_fake_detection_using_deep_learning/
├── app.py                     # Streamlit application with ELA & neural inference
├── requirements.txt           # Cloud-optimized dependencies (tensorflow-cpu, streamlit)
├── runtime.txt                # Python 3.11 runtime specification
├── deepfake.ipynb             # Research & training notebook
├── index.html                 # Optional static web interface
├── style.css                  # Optional static styles
├── script.js                  # Optional static scripts
├── vercel.json                # Optional Vercel configuration
└── README.md                  # Documentation
```

---

## ⚠️ Disclaimer

This system is designed for educational and research purposes. Digital forensic methods should be used as part of a multi-factor verification pipeline.

---

## 👤 Author

**Anant Joshi**  
- GitHub: [@Cypheraj12](https://github.com/Cypheraj12)  
- Portfolio: [my-portfolio](https://my-portfolio-psi-liart-71.vercel.app/)
