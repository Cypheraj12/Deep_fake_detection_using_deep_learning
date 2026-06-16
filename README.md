# 🕵️ DeepFake Image Detection using MobileNetV2

## Overview

This project detects whether an image is **Real** or **DeepFake** using a deep learning model based on **MobileNetV2** and **TensorFlow**.

The application provides a simple web interface built with Streamlit where users can upload an image and instantly receive a prediction along with confidence scores.

---

## Features

* Upload JPG, JPEG, and PNG images
* DeepFake detection using MobileNetV2
* Real/Fake confidence scores
* Human-review warning for uncertain predictions
* Streamlit web interface
* Lightweight and fast inference

---

## Dataset Structure

Dataset/

├── Train/

│ ├── Fake/

│ └── Real/

├── Validation/

│ ├── Fake/

│ └── Real/

└── Test/

├── Fake/

└── Real/

---

## Model Architecture

* MobileNetV2 (ImageNet Weights)
* Global Average Pooling
* Dense Layer (128 Neurons)
* Dropout (0.5)
* Sigmoid Output Layer

Loss Function:

* Binary Crossentropy

Optimizer:

* Adam

Metric:

* Accuracy

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## Prediction Logic

Output Interpretation:

| Real Probability      | Result     |
| --------------------- | ---------- |
| ≥ 0.75                | Real Image |
| ≤ 0.25                | Fake Image |
| Between 0.25 and 0.75 | Uncertain  |

Final Decision Threshold:

* Real > 0.65 → Real Image
* Real < 0.35 → Fake Image
* Otherwise → Human Review Required

---

## Technologies Used

* Python
* TensorFlow
* Keras
* MobileNetV2
* NumPy
* Streamlit
* PIL

---

## Future Improvements

* Video DeepFake Detection
* Face Extraction Pipeline
* Explainable AI Visualizations
* Model Fine-Tuning
* Real-Time Webcam Detection

---

## Author

Pranshu Verma, Snehil Verma, Anant Joshi

B.Tech Student | Data Science and Artificial Intelligence & Machine Learning

---

## License

This project is intended for educational and research purposes.
