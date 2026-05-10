import cv2 
import numpy as np 
import os 
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input 
from tensorflow.keras.models import Model 
from sklearn.metrics.pairwise import cosine_similarity 
from datetime import timedelta 
def load_vgg_model(): 
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 
3)) 
model = Model(inputs=base_model.input, outputs=base_model.output) 
return model 
def extract_vgg_features(frame, model): 
resized = cv2.resize(frame, (224, 224)) 
img = np.expand_dims(resized, axis=0) 
img = preprocess_input(img) 
features = model.predict(img, verbose=0) 
return features.flatten() 
def extract_keyframes(video_path, model, threshold=0.5, frame_skip=10): 
cap = cv2.VideoCapture(video_path) 
if not cap.isOpened(): 
print("Error: Cannot open video file.") 
return [], 0 
fps = cap.get(cv2.CAP_PROP_FPS) 
keyframes = [] 
prev_feature = None 
frame_count = 0 
while True: 
ret, frame = cap.read() 
if not ret: 
break 
if frame_count % frame_skip == 0: 
feature = extract_vgg_features(frame, model) 
if prev_feature is None: 
keyframes.append((frame_count, frame)) 
prev_feature = feature 
else: 
similarity = cosine_similarity([prev_feature], [feature])[0][0] 
if similarity < threshold: 
keyframes.append((frame_count, frame)) 
prev_feature = feature 
frame_count += 1 
cap.release() 
return keyframes, fps 
def get_timestamp(frame_number, fps): 
seconds = frame_number / fps 
return str(timedelta(seconds=int(seconds))) 
def evaluate_keyframes(keyframes, total_frames, ground_truth_frames): 
pred_frames = [frame_num for frame_num, _ in keyframes] 
pred_set = set(pred_frames) 
true_set = set(ground_truth_frames) 
tp = len(pred_set & true_set) 
fp = len(pred_set - true_set) 
fn = len(true_set - pred_set) 
precision = tp / (tp + fp) if (tp + fp) > 0 else 0 
recall = tp / (tp + fn) if (tp + fn) > 0 else 0 
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0 
return precision, recall, f1_score 
