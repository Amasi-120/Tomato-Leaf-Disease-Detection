# 🍅 Tomato Leaf Disease Detection

## 📌 Project Overview

Tomato Leaf Disease Detection is an AI-powered image classification system that detects whether a tomato leaf is **Healthy** or **Diseased** using Deep Learning.

The project uses a transfer learning approach with **MobileNetV2** to classify tomato leaf images and provide prediction results through an interactive **Streamlit web application**.

---

## 🎯 Objective

The main goal of this project is to develop an automated system that helps identify tomato leaf diseases quickly and accurately using Artificial Intelligence and Computer Vision.

---

## 🧠 Model Description

The model is built using:

- **MobileNetV2** pretrained on ImageNet
- Transfer Learning
- Fine-tuning of the last layers
- Global Average Pooling
- Dense classification layer
- Sigmoid activation for binary classification

### Classes:
- 🍃 Healthy
- 🦠 Diseased

---

## 📊 Dataset

The project uses the **PlantVillage Dataset**.

Only tomato leaf images were selected and converted into a binary classification problem:

- Tomato Healthy
- Tomato Late Blight (Diseased)

The dataset was divided into:

- Training set: 70%
- Validation set: 15%
- Testing set: 15%

---

## ⚙️ Technologies Used

### Programming Language:
- Python

### Deep Learning:
- TensorFlow
- Keras
- MobileNetV2

### Data Processing:
- NumPy
- Pillow
- Pandas

### Deployment:
- Streamlit

### Development Environment:
- Google Colab
- VS Code

---

## 📈 Model Performance

The trained model achieved high performance:

- Validation Accuracy: ~98%
- Test Accuracy: ~97%

The model was evaluated using:

- Accuracy
- Classification Report
- Confusion Matrix

---

