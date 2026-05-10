# Home-Security-System-using-VGG16-model
# 🎥 Keyframe Extraction & Object Detection Using VGG16 and YOLOv5

## 📌 Project Overview

This project presents an intelligent video analysis system that performs **keyframe extraction** and **object detection** using deep learning techniques. The system identifies important frames from a video by analyzing visual differences between frames using the **VGG16 Convolutional Neural Network** and cosine similarity. Extracted keyframes are further processed using **YOLOv5** for real-time object detection.

A user-friendly web application is developed using **Streamlit**, allowing users to upload videos, extract keyframes, detect objects, and visualize evaluation metrics interactively.

---

# 🚀 Features

- 📤 Upload video files
- 🖼️ Automatic keyframe extraction
- 🧠 Deep feature extraction using VGG16
- 🔍 Object detection using YOLOv5
- 📊 Cosine similarity analysis
- 📈 Similarity graph visualization
- 📋 Precision, Recall, and F1-score evaluation
- 🌐 Interactive Streamlit web interface

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application framework |
| OpenCV | Video processing |
| TensorFlow / Keras | VGG16 implementation |
| PyTorch | YOLOv5 model loading |
| NumPy | Numerical operations |
| Matplotlib | Graph plotting |
| Scikit-learn | Cosine similarity calculation |

---

# 🧠 Deep Learning Models Used

## 🔹 VGG16

Used for:
- Feature extraction
- Scene understanding
- Visual representation of frames

## 🔹 YOLOv5

Used for:
- Real-time object detection
- Detecting objects in extracted keyframes

---

# 📂 Project Structure

```bash
project-folder/
│
├── app.py
├── keyframe_extractor.py
├── requirements.txt
├── README.md
└── sample_videos/
