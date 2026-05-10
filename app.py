import streamlit as st 
import tempfile 
import cv2 
import matplotlib.pyplot as plt 
import torch 
from keyframe_extractor import extract_keyframes, get_timestamp, evaluate_keyframes, 
extract_vgg_features, load_vgg_model 
from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np 
 
st.set_page_config(page_title="Keyframe Extractor", layout="wide") 
st.title("🎥 Keyframe Extraction & Object Detection Using VGG16 & YOLOv5") 
 
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"]) 
 
threshold = st.slider("Similarity Threshold", 0.1, 1.0, 0.5) 
frame_skip = st.slider("Frame Skip Interval", 1, 30, 10) 
detect_objects = st.checkbox("Enable Object Detection (YOLOv5)", value=True) 
 
ground_truth_keyframes = [10, 60, 140, 220, 300, 400] 
 
def plot_similarity_graph(similarities): 
fig, ax = plt.subplots() 
ax.plot(similarities, marker='o', linestyle='-') 
ax.set_title("Cosine Similarity Between Keyframes") 
ax.set_xlabel("Keyframe Index") 
ax.set_ylabel("Cosine Similarity with Previous Frame") 
ax.grid(True) 
return fig 
if uploaded_video is not None: 
tfile = tempfile.NamedTemporaryFile(delete=False) 
tfile.write(uploaded_video.read()) 
video_path = tfile.name 
st.video(video_path) 
if st.button("Extract Keyframes"): 
model = load_vgg_model() 
keyframes, fps = extract_keyframes(video_path, model, threshold, frame_skip) 
st.subheader("🖼️ Extracted Keyframes") 
if detect_objects: 
st.write("Using YOLOv5 for object detection...") 
try: 
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True, 
verbose=False) 
except Exception as e: 
st.error(f"Failed to load YOLOv5 model: {e}") 
detect_objects = False 
for i, (frame_num, frame) in enumerate(keyframes): 
timestamp = get_timestamp(frame_num, fps) 
if detect_objects: 
results = yolo_model(frame) 
frame_with_boxes = np.squeeze(results.render()) 
st.image(frame_with_boxes, caption=f"Keyframe {i} - Frame {frame_num} - 
Time {timestamp}", width=300) 
labels = results.pandas().xyxy[0]['name'].tolist() 
if labels: 
st.write(f"Detected Objects: {', '.join(labels)}") 
else: 
st.write("No objects detected.") 
else: 
st.image(frame, caption=f"Keyframe {i} - Frame {frame_num} - Time 
{timestamp}", width=300) 
st.write(f"**Index:** {i}, **Frame Number:** {frame_num}, **Timestamp:** 
{timestamp}") 
cap = cv2.VideoCapture(video_path) 
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
cap.release() 
st.subheader("📊 Evaluation Metrics") 
precision, recall, f1_score = evaluate_keyframes(keyframes, total_frames, 
ground_truth_keyframes) 
st.write(f"**Model:** VGG16") 
st.write(f"**Total Frames:** {total_frames}") 
st.write(f"**Keyframes Extracted:** {len(keyframes)}") 
st.write(f"**Precision:**{precision:.2f}") 
st.write(f"**Recall:** {recall:.2f} ") 
st.write(f"**F1 Score:** {f1 score:.2f}  ") 
st.subheader("📈 Keyframe Similarity Graph") 
features = [extract_vgg_features(frame, model) for _, frame in keyframes] 
similarities = [cosine_similarity([features[i-1]], [features[i]])[0][0] for i in range(1, 
len(features))] 
fig = plot_similarity_graph(similarities) 
st.pyplot(fig) 
