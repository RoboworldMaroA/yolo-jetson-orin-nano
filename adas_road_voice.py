
# Author: Marek Augustyn
# 15 August 2026
#####  Voice commands  ########
### Implemented , when user press button voice control  n ####
### Installed certificate that allows me to run website on https 
# docker exec 9e80ed866c88 pkill -f adas_road.py
# sudo systemctl restart adas.service
# Track logs: tail -f /home/maro/yolo_app/flask_output.log
# It is ADAS app v1 - base ion web_lazy_v6_new_model.py
# From this version I will be changing a program and the name for new version of the ADAS ,
# To run this code It will be automaticaly started ehn you boot up a Jetson Orin Nano.
# It will start a program naed: web_lazy_v6_new_model.py and inside will be reference to run external python script
# Program inside the web_lazy_v6_new_model.py will have an updated name of the new revision
#Description:
# Program is ready fro collection licence plate, but I am waint for camera holder that I will install in the car and
# then do the training on my laptop
######
# Implemented new way of licence plate recognition that allows me do the faster object detection
# Add paddle ONNX model for licence plate recognition, which is faster than EasyOCR and more accurate for Irish plates.
# I can do the recoding, photo, Zoom , Rotate a screen View if I mount a camera opposite side.
# This program allows to pick from the website between different models for inference on video stream. 
# It includes a new model for Irish licence plate recognition, 
# which detects cars, detects plates within those cars, and applies OCR to recognize the plate text.
# Recognized plates are saved as cropped images and logged in a CSV file. The web interface provides buttons to switch between
# detection, pose estimation, segmentation, custom model, and plate recognition modes. 
# The producer thread captures video frames, runs inference with the active model, 
# annotates the frames, and saves detected cups as images. 
# The MJPEG generator streams the annotated video to the web interface.
# ### It is what you can use: ### 
# ## 1. EasyOCR custom model recognition Irish Licence Plates with YOLOv8n trained on Irish plates. 
# The program processes video frames, detects cars, detects plates within those cars, 
# and then applies OCR to recognize the plate text.
# Recognized plates are saved as cropped images and logged in a CSV file.
# Program allow recognize object using yolo pretrained model and stream video with detections over web server
# ## 2. Object Detection ##
# Use a Yolo model for object detection. It save detected cups as images
# ## 3. Pose Estimation ##
# Use a Yolo model for pose estimation. It can be used to detect human poses in the video stream.
# ## 4. Segmentation ##
# Use a Yolo model for segmentation. It can be used to segment objects in the video stream.
# ## 5. Custom Model ##
# Use a custom Yolo model for specific object detection. It recognze signs Start Stop, Turn Left, Turn Right
# ## 6. Licence Plate Recognition ##
# Use a Yolo model traned on cutom datafor licence plate recognition. It can be used to detect licence plates in the video stream.
# ### Interface ###
# You can swith from the web browser between models like Pose Estimation and Object Detection, Segmentation, 
# Custom MOdel Start Stop and Plate Recognition
# If cub is reconize with 60% confidence or higher, it save the frame with detected cup in the /app folder with
# name detected_cup_TIMESTAMP.jpg

# It uses a yolo docker container, USB camera is connected to the Jetson Orin Nano
#.      
#.      docker exec 9e80ed866c88 pkill -f web_lazy_v6_new_model.py
#.      docker exec 9e80ed866c88 pkill -f adas_road.py
#.       sudo systemctl restart adas.service
#.       sudo systemctl status adas.service
#.       tail -f /home/maro/yolo_app/flask_output.log
# Check instruction.txt for how to run the program with docker
# Run:
# Use new docker Image with easyOCR and licence plate recognition model, which is based on YOLOv8n trained on Irish plates
# https://192.168.0.12:5010/
#can be use also with roboworld.react.marekaugustyn.whshost.com:5010 
#25 July 2026 Dodanie kolejki dla nagrywania wideo, bedzie robic nagrywanie poza glowna petlo dzieki temu nie spowolni nagrywania. 

from ultralytics import YOLO
from flask import Flask, Response, render_template, jsonify, request
import threading
import time
import cv2
import os

#Import for easy OCR and licence plate recognition
import torch
import easyocr
reader = easyocr.Reader(["en"], gpu=True) # Initialize ONCE here

import re
import csv
# import os

import numpy as np
import onnxruntime as ort

import subprocess

#Imports for recoding option
from datetime import datetime

# import for shouting down Jetson Orin Nano from the web interface
import requests
import subprocess

import json

#Imports for separate thred used for recording video
import queue
import threading

class LPRNetRecognizer:
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.available = os.path.exists(onnx_path)

        self.alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.blank_index = len(self.alphabet)

        if not self.available:
            print(f"PaddleOCR ONNX not found yet: {onnx_path}")
            self.session = None
            self.input_name = None
            return

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        print(f"PaddleOCR ONNX loaded: {onnx_path}")
        print("ONNX providers:", self.session.get_providers())
        print("Input:", self.session.get_inputs()[0].shape)
        print("Output:", self.session.get_outputs()[0].shape)

    def preprocess(self, plate_crop):
        img_h = 32
        img_w = 160

        img = cv2.resize(plate_crop, (img_w, img_h))
        img = img.astype(np.float32)

        # PaddleOCR style normalization
        img = img / 255.0
        img = (img - 0.5) / 0.5

        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img.astype(np.float32)

    def decode_ctc(self, output):
        output = np.squeeze(output)

        # PaddleOCR ONNX output: [time, classes]
        pred = np.argmax(output, axis=1)

        result = []
        prev = None

        for idx in pred:
            idx = int(idx)

            if idx == self.blank_index:
                prev = idx
                continue

            if idx != prev and idx < len(self.alphabet):
                result.append(self.alphabet[idx])

            prev = idx

        return "".join(result)

    def format_irish_plate(self, text):
        text = text.upper()
        text = re.sub(r"[^A-Z0-9]", "", text)

        match = re.search(r"(\d{2,3})([A-Z]{1,2})(\d{1,6})", text)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

        return text

    def predict(self, plate_crop):
        if not self.available or self.session is None:
            return "PADDLE-NOT-READY"

        try:
            input_tensor = self.preprocess(plate_crop)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            raw_text = self.decode_ctc(outputs[0])
            print("PADDLE RAW:", raw_text)
            return self.format_irish_plate(raw_text)
        except Exception as e:
            print(f"PaddleOCR ONNX error: {e}")
            return "PADDLE-ERROR"

    
    def format_irish_plate(self, text):
        text = text.upper()
        text = re.sub(r"[^A-Z0-9]", "", text)

        # Szukaj irlandzkiego wzoru w środku wyniku, np. 251D21332
        match = re.search(r"(\d{2,3})([A-Z]{1,2})(\d{1,6})", text)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

        # Awaryjnie pokaż tylko pierwsze 12 znaków zamiast całego długiego ciągu
        return text[:12]

    def predict(self, plate_crop):
        if not self.available or self.session is None:
            return "LPRNET-NOT-READY"

        try:
            input_tensor = self.preprocess(plate_crop)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            raw_text = self.decode_ctc(outputs[0])
            return self.format_irish_plate(raw_text)
        except Exception as e:
            print(f"LPRNet ONNX error: {e}")
            return "LPRNET-ERROR"



# Kolejka na klatki do nagrywania (max 5 klatek, żeby nie zapchać RAMu)
record_queue = queue.Queue(maxsize=5)

# Adres Helpera widziany z wnętrza Dockera:
HELPER_URL = "https://172.17.0.1:5055"  # lub "http://192.168.0.12:5055"
# HELPER_URL = "http://172.17.0.1:5055"  # lub "http://192.168.0.12:5055"
# HELPER_URL = "http://127.0.0.1:5055"  # lub "http://192.168.0.12:5055"
is_recording = False

# Focus variables for the Irish plate recognition pipeline
camera_focus = 1
# camera_exposure = -5
camera_exposure_abs = 140
camera_gain = 0
cap_global = None
V4L2_CTL = "/usr/bin/v4l2-ctl"
camera_digital_zoom = 1.0

camera_zoom_abs = 140

#variable flip frame for camera
camera_flip_180 = False

# variable for recodning 
recording_enabled = False
recording_writer = None
recording_fps = 30
recording_width = 1920
recording_height = 1080
RECORDING_DIR = "/app/road_recordings"

#make an images
# images_button_activated = False
# IMAGES_DIR = "/app/images_from_camera_stream_jetson_nano"
IMAGES_DIR = "/app/images_from_camera_stream_jetson_nano"
latest_raw_frame = None

#save to .csv files registration
last_saved_plate = ""
last_saved_time = 0

#Presets
CAMERA_PRESETS_FILE = "/app/camera_presets.json"



# Configuration

#Create a folder that store cropped images licence from the LPRN function:
PLATE_DATASET_DIR = "/app/dataset_irish_plates/images"

os.makedirs(PLATE_DATASET_DIR, exist_ok=True)

print(f"Plate dataset directory ready: {PLATE_DATASET_DIR}")


# MOdel for testing teached regonize first 20 images of Irish plates
MODEL_PATH_LPRNET_ONNX = os.environ.get(
    "MODEL_PATH_LPRNET_ONNX",
    "/app/irish_plate_overfit.onnx"
)


# add a new wproffesional way instead of EasyOCR: lprnet 
MODEL_PATH_LPRNET = os.environ.get(
    "MODEL_PATH_LPRNET",
    "/app/lprnet.engine"
)

# MODEL_PATH_LPRNET_ONNX = os.environ.get(
#     "MODEL_PATH_LPRNET_ONNX",
#     "/app/lprnet.onnx"
# )


MODEL_PATH_DETECT = os.environ.get("MODEL_PATH_DETECT", "/app/yolov8n.engine")
MODEL_PATH_POSE = os.environ.get("MODEL_PATH_POSE", "/app/yolo11n-pose.engine")
MODEL_PATH_SEGMENTATION = os.environ.get("MODEL_PATH_SEGMENTATION", "/app/yolo11n-seg.engine")
MODEL_PATH_CUSTOM_MODEL = os.environ.get("MODEL_PATH_CUSTOM_MODEL", "/app/start_stop_yolo8.engine")
# MODEL_PATH_PLATE_RECOGNITION = os.environ.get("MODEL_PATH_PLATE_RECOGNITION", "/app/licence_plate.pt")  # replace with actual plate recognition model path
MODEL_PATH_PLATE_RECOGNITION = os.environ.get("MODEL_PATH_PLATE_RECOGNITION", "/app/licence_plate.engine")  # replace with actual plate recognition model path

CAMERA_SOURCE = int(os.environ.get("CAMERA_SOURCE", "0"))
FPS_LIMIT = float(os.environ.get("FPS_LIMIT", "20.0"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", "640"))

#Varsiable for Easy OCR
recognized_plates_log = []  # Each entry: (frame_number, car_idx, plate_idx, plate_text)

# Check for MPS (Apple Metal Performance Shaders) device availability
#print(f"MPS built: {torch.backends.mps.is_built()}")
#print(f"MPS available: {torch.backends.mps.is_available()}")
#print(f"Has MPS: {torch.backends.mps.is_built()}")
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load the YOLO models
##model_path = "/app/Yolo_Models/yolov8m.pt"
#model_path = "/app/yolov8n.engine"
# model_path = "/app/Yolo_Models/yolo11n.pt"
# model_registation_plate_path = "Model_recognize_licence_plates/runs/detect/train14/weights/best.pt"
##model_registation_plate_path = "/app/licence_plate.engine"

# model = YOLO(model_path)
model = YOLO(MODEL_PATH_DETECT)
##model_registration_plate = YOLO(model_registation_plate_path)
model_registration_plate = YOLO(MODEL_PATH_PLATE_RECOGNITION)
#Check if model for segmentation is downloaded
try:
    model_segmentation = YOLO(MODEL_PATH_SEGMENTATION)
    print("Model for segmentation loaded successfully.")
    print("Model class names for segmentation:", model_segmentation.names)
except:
    print("Model for segmentation not found.")

#Check if model for pose estimation is downloaded
try:
    model_pose = YOLO(MODEL_PATH_POSE)
    print("Model for pose estimation loaded successfully.")
    print("Model class names for pose estimation:", model_pose.names)
except:
    print("Model for pose estimation not found.")

#Check if model for custom model is downloaded
try:
    model_custom = YOLO(MODEL_PATH_CUSTOM_MODEL)
    print("Model for custom model loaded successfully.")
    print("Model class names for custom model:", model_custom.names)
except:
    print("Model for custom model not found.")

print("Model loaded successfully.")
print("Model class names:", model.names)
print("Model for registration plates loaded successfully.")
print("Model class names for registration plates:", model_registration_plate.names)


# lprnet_recognizer = LPRNetRecognizer(MODEL_PATH_LPRNET)
lprnet_recognizer = LPRNetRecognizer(MODEL_PATH_LPRNET_ONNX)


app = Flask(__name__, template_folder="templates", static_folder="static")

# Shared state
latest_frame = None
frame_lock = threading.Lock()
stop_event = threading.Event()

# Simple FPS tracking
fps_smoothed = 0.0
last_frame_time = None

class ModelManager:
    """Lazy-load YOLO models and switch active model safely."""
    def __init__(self, mapping, default="lprnet-anpr"):
        # mapping: name -> path
        self._mapping = mapping
        self._lock = threading.Lock()
        self._models = {name: None for name in mapping}
        self._loading = {name: threading.Event() for name in mapping}
        self._last_error = {name: None for name in mapping}
        self.active = default

    def _load_worker(self, name, path):
        try:
            model = YOLO(path)
            with self._lock:
                self._models[name] = model
                self._last_error[name] = None
        except Exception as e:
            with self._lock:
                self._last_error[name] = str(e)
        finally:
            # signal loading finished
            self._loading[name].set()

    def ensure_loaded(self, name):
        """Start background load if not already loaded. Non-blocking."""
        with self._lock:
            if name not in self._mapping:
                return
            if self._models.get(name) is not None:
                return
            if self._loading[name].is_set() and self._models[name] is None:
                # previous load finished with error; clear event and retry
                self._loading[name].clear()

            if not self._loading[name].is_set():
                # mark as loading and spawn loader thread
                self._loading[name].clear()
                t = threading.Thread(target=self._load_worker, args=(name, self._mapping[name]), daemon=True)
                t.start()

    def switch(self, name):
        with self._lock:
            if name not in self._mapping:
                raise KeyError(name)
            self.active = name
        # trigger lazy load
        self.ensure_loaded(name)

    def get_active(self):
        with self._lock:
            return self._models.get(self.active)

    def is_loaded(self, name):
        with self._lock:
            return self._models.get(name) is not None

    def is_loading(self, name):
        return (not self._loading[name].is_set()) and (self._models.get(name) is None)

    def last_error(self, name):
        with self._lock:
            return self._last_error.get(name)

    def status(self):
        with self._lock:
            return {
                "active": self.active,
                "loaded": {n: (self._models[n] is not None) for n in self._models},
                "errors": dict(self._last_error)
            }

# create manager with two models
#manager = ModelManager({"detection": MODEL_PATH_DETECT, "pose": MODEL_PATH_POSE}, default="detection")
# create manager with two models, segmentation model and custom model
manager = ModelManager({
    "detection": MODEL_PATH_DETECT,
    "pose": MODEL_PATH_POSE,
    "segmentation": MODEL_PATH_SEGMENTATION,
    "custom-model": MODEL_PATH_CUSTOM_MODEL,
    "plate-recognition": MODEL_PATH_PLATE_RECOGNITION,
    "irish-plate-recognition": MODEL_PATH_PLATE_RECOGNITION,
    "improved-irish-plate-recognition": MODEL_PATH_PLATE_RECOGNITION,
    "lprnet-anpr": MODEL_PATH_PLATE_RECOGNITION,
    "road-detection": "",
    "road-detection-segmentation": ""
}, default="lprnet-anpr")
# kick off load for default model in background
manager.ensure_loaded(manager.active)


#function that help rptate a frame by 180 degrees if the camera is mounted upside down
def apply_camera_orientation(frame):
    global camera_flip_180

    if camera_flip_180:
        return cv2.rotate(frame, cv2.ROTATE_180)

    return frame


# Function from recognize_Irish_Plates_Video_v6_flask.py to clean OCR text for common misrecognitions in Irish plates.
# It can be used to improve the accuracy of plate recognition by correcting common OCR errors.
def clean_plate_text(text):
    """Clean OCR text for common misrecognitions in Irish plates."""
    text = text.upper().strip()
    text = text.replace(" ", "-")
    text = text.replace("LT", "WW")

    no_hyphens = text.replace("-", "")
    no_hyphens = (
        no_hyphens.replace("O", "0")
        .replace("I", "1")
        .replace("S", "5")
        .replace("L", "1")
        .replace("T", "7")
    )

    for first_len in range(3, 0, -1):
        for mid_len in range(1, 3):
            if first_len + mid_len <= len(no_hyphens):
                first = no_hyphens[:first_len]
                middle = no_hyphens[first_len:first_len + mid_len]
                last = no_hyphens[first_len + mid_len :]

                if first.isdigit() and 1 <= len(last) <= 5 and last.isdigit():
                    middle_fixed = (
                        middle.replace("0", "D")
                        .replace("1", "I")
                        .replace("5", "S")
                        .replace("7", "T")
                        .replace("O", "D")
                    )
                    if middle_fixed.isalpha():
                        if len(middle_fixed) == 1 and middle_fixed == "P":
                            middle_fixed = "D"
                        return f"{first}-{middle_fixed}-{last}"

    match = re.match(r"^(\d{1,3})([A-Z]{1,2})(\d{1,5})$", no_hyphens)
    if match:
        middle_fixed = match.group(2)
        if len(middle_fixed) == 1 and middle_fixed == "P":
            middle_fixed = "D"
        return f"{match.group(1)}-{middle_fixed}-{match.group(3)}"

    return no_hyphens


def debug_plate_recognition(plate_text):
    cleaned = clean_plate_text(plate_text)
    is_valid = is_irish_plate(plate_text)
    print(f"  Raw: {plate_text}")
    print(f"  Cleaned: {cleaned}")
    print(f"  Valid: {is_valid}")


def is_irish_plate(text):
    cleaned = clean_plate_text(text)
    pattern = r"^\d{1,3}-[A-Z]{1,2}-\d{1,5}$"
    return bool(re.match(pattern, cleaned))


def preprocess_image(image, frame_number=None, car_idx=None, plate_idx=None, plate_text=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # SAVED_ONLY_RECOGNIZED_PLATES = os.path.join(".", "images_only_reconized_plates")
    SAVED_ONLY_RECOGNIZED_PLATES = os.path.join(".", "/app/images_only_recognized_plates")
    os.makedirs(SAVED_ONLY_RECOGNIZED_PLATES, exist_ok=True)

    if plate_text:
        plate_text_clean = re.sub(r"\W+", "", plate_text)
    else:
        plate_text_clean = "unknown"

    filename = f"license_only_f{frame_number}_c{car_idx}_p{plate_idx}_{plate_text_clean}_{int(time.time()*1000)}.jpg"
    image_path = os.path.join(SAVED_ONLY_RECOGNIZED_PLATES, filename)
    cv2.imwrite(image_path, image)
    print(f"Saved cropped plate image: {image_path}")

    return thresh


def recognize_plate(cropped_img, frame_number=None, car_idx=None, plate_idx=None, plate_text=None):
    # reader = easyocr.Reader(["en"], gpu=True)
    processed_image = preprocess_image(
        cropped_img,
        frame_number=frame_number,
        car_idx=car_idx,
        plate_idx=plate_idx,
        plate_text=plate_text,
    )
    results = reader.readtext(processed_image)
    plates = []
    for (bbox, text, prob) in results:
        if prob > 0.2:
            plates.append((bbox, text))
    return plates


def process_cropped_licence_plate(cropped_img, index):
    if not cropped_img.size:
        print(f"Cropped image {index} is empty, skipping save.")
        return

    os.makedirs("/app/Cropped_licence_plates", exist_ok=True)
    cropped_image_path = f"/app/Cropped_licence_plates/cropped_licence_plate_{index}.png"
    cv2.imwrite(cropped_image_path, cropped_img)
    print(f"Cropped licence plate image saved to {cropped_image_path}")

    # cv2.imshow(f"Cropped Licence Plate {index}", cropped_img)
    # cv2.waitKey(100)
    #cv2.destroyAllWindows()

#It is main function that is trigered when the "irish-plate-recognition" model is active.
#It processes each video frame, detects cars, detects plates within those cars, 
#and applies OCR to recognize the plate text. 
#Recognized plates are saved as cropped images and logged in a CSV file. 
#The function also annotates the video frame with detected cars and recognized plate text.
def process_frame(frame, frame_number):
    results = model(frame, imgsz=IMG_SIZE, verbose=False)
    if not results:
        return frame

    result = results[0]
    car_boxes = []
    allowed_car_classes = [2, 5, 7]

    for car_idx, box in enumerate(result.boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        class_id = int(box.cls[0].item())
        prob = round(box.conf[0].item(), 2)
        if prob < 0.85 or class_id not in allowed_car_classes:
            continue

        car_boxes.append((car_idx, x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 118, 232), 3)
        cv2.putText(
            frame,
            f"{result.names[class_id]} {prob}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.9,
            (26, 9, 156),
            4,
        )

    unique_plates = set()
    for car_idx, x1, y1, x2, y2 in car_boxes:
        car_crop = frame[y1:y2, x1:x2]
        licence_results = model_registration_plate(car_crop, imgsz=IMG_SIZE)

        if not licence_results:
            continue

        licence_result = licence_results[0]
        for plate_idx, plate_box in enumerate(licence_result.boxes):
            px1, py1, px2, py2 = [int(coord) for coord in plate_box.xyxy[0]]
            plate_crop = car_crop[py1:py2, px1:px2]

            plates = recognize_plate(
                plate_crop,
                frame_number=frame_number,
                car_idx=car_idx,
                plate_idx=plate_idx,
                plate_text=None,
            )

            for (bbox, plate_text) in plates:
                cleaned_plate = clean_plate_text(plate_text)
                if not is_irish_plate(plate_text):
                    continue

                if cleaned_plate in unique_plates:
                    continue

                unique_plates.add(cleaned_plate)
                process_cropped_licence_plate(
                    plate_crop, f"car{car_idx}_plate{plate_idx}_{cleaned_plate}"
                )
                recognized_plates_log.append(
                    (frame_number, car_idx, plate_idx, cleaned_plate)
                )

                abs_px1, abs_py1 = x1 + px1, y1 + py1
                (w, h), _ = cv2.getTextSize(
                    cleaned_plate, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 8
                )
                cv2.rectangle(
                    frame,
                    (abs_px1 + 5, abs_py1 - h - 60),
                    (abs_px1 + w + 270, abs_py1),
                    (255, 255, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    cleaned_plate,
                    (abs_px1, abs_py1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.5,
                    (26, 9, 156),
                    10,
                )
    # Development only
    # print(f"All unique plates found in frame: {unique_plates}")
    with open("/app/recognized_plates_video.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame_number", "car_idx", "plate_idx", "plate_text"])
        writer.writerows(recognized_plates_log)
        print("Recognized plates saved to recognized_plates_video.csv")

    # print("Recognized plates saved to recognized_plates_video.csv")

    return frame


def process_frame_optimized(frame, frame_number):
    # 1. Lower the imgsz for the initial vehicle search to gain speed
    results = model(frame, imgsz=640, conf=0.5) 
    
    # 2. Only run OCR on every Nth frame to prevent video lag
    if frame_number % 5 != 0:
        return model_registration_plate(frame)[0].plot() # Just show detection

    # 3. Add logic to sharpen or contrast-enhance the crop before EasyOCR
    # (Existing logic follows...)
    return process_frame(frame, frame_number)

#This function I have to processing on URL http://
# def generate_frames():
#     cap = cv2.VideoCapture(0)  # 0 dla domyślnej kamery; zmień na ścieżkę do kamery jeśli potrzeba
#     frame_number = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         processed_frame = process_frame(frame, frame_number)
#         ret, buffer = cv2.imencode('.jpg', processed_frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#         frame_number += 1



def process_frame_lprnet(frame, frame_number):
    global last_saved_plate, last_saved_time
    """
    New professional ANPR pipeline:
    Vehicle detection -> plate detection -> LPRNet recognition.
    This does NOT use EasyOCR.
    """
    cv2.putText(
        frame,
        "LPRNET MODE",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    results = model(frame, imgsz=IMG_SIZE, conf=0.4, verbose=False)

    if not results:
        return frame

    result = results[0]
    allowed_car_classes = [2, 5, 7]  # car, bus, truck

    for car_idx, box in enumerate(result.boxes):
        class_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        if conf < 0.4 or class_id not in allowed_car_classes:
            continue

        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        car_crop = frame[y1:y2, x1:x2]

        if car_crop.size == 0:
            continue

        licence_results = model_registration_plate(car_crop, imgsz=IMG_SIZE, conf=0.4)

        if not licence_results:
            continue

        licence_result = licence_results[0]

        for plate_idx, plate_box in enumerate(licence_result.boxes):
            px1, py1, px2, py2 = [int(coord) for coord in plate_box.xyxy[0]]
            plate_crop = car_crop[py1:py2, px1:px2]

            if plate_crop.size == 0:
                continue

            # cv2.imwrite(
            #     f"/app/debug_plate_{frame_number}.jpg",
            #     plate_crop
            # )
            #Save cropped plate images for debugging and training purposes
            print(f"Saving cropped plate image: frame {frame_number}, car {car_idx}, plate {plate_idx}")
            # cv2.imwrite(f"/app/dataset_irish_plates/images/plate_{frame_number}_{plate_idx}.jpg",plate_crop)
            # filename = f"/app/dataset_irish_plates/images/plate_{frame_number}_{plate_idx}.jpg"
            # ok = cv2.imwrite(filename, plate_crop)
            # print("SAVE:", filename, ok)

            # filename = (f"{PLATE_DATASET_DIR}/"f"plate_{frame_number}_{car_idx}_{plate_idx}.jpg")
            # ok = cv2.imwrite(filename, plate_crop)
            # print(f"SAVE: {filename} -> {ok}")

            if frame_number % 1 == 0:
                os.makedirs("/app/dataset_irish_plates/images", exist_ok=True)
                filename = (f"/app/dataset_irish_plates/images/"f"plate_{frame_number}_{car_idx}_{plate_idx}.jpg")
                ok = cv2.imwrite(filename, plate_crop)
                print(f"SAVE: {filename} -> {ok}")
            
            plate_text = lprnet_recognizer.predict(plate_crop)

            abs_px1 = x1 + px1
            abs_py1 = y1 + py1
            abs_px2 = x1 + px2
            abs_py2 = y1 + py2

            cv2.rectangle(
                frame,
                (abs_px1, abs_py1),
                (abs_px2, abs_py2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                plate_text,
                (abs_px1, max(abs_py1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # saving to .csv only if the plate is new or 10 seconds have passed since the last save
            now = time.time()

            if (
                plate_text
                and "ERROR" not in plate_text
                and "NOT-READY" not in plate_text
                and (
                    plate_text != last_saved_plate
                    or now - last_saved_time > 10
                )
            ):
                last_saved_plate = plate_text
                last_saved_time = now

                timestamp = time.strftime("%Y%m%d_%H%M%S")

                os.makedirs("/app/anpr_results/crops", exist_ok=True)
                os.makedirs("/app/anpr_results/frames", exist_ok=True)

                crop_filename = f"/app/anpr_results/crops/{timestamp}_{frame_number}_{car_idx}_{plate_idx}.jpg"
                frame_filename = f"/app/anpr_results/frames/{timestamp}_{frame_number}_{car_idx}_{plate_idx}.jpg"

                cv2.imwrite(crop_filename, plate_crop)
                cv2.imwrite(frame_filename, frame)

                csv_path = "/app/anpr_results/plates.csv"
                write_header = not os.path.exists(csv_path)

                with open(csv_path, "a") as f:
                    if write_header:
                        f.write("timestamp,plate_text,crop_file,frame_file\n")

                    f.write(
                        f"{timestamp},{plate_text},{crop_filename},{frame_filename}\n"
                    )
            # if plate_text and "ERROR" not in plate_text and "NOT-READY" not in plate_text:
            #     timestamp = time.strftime("%Y%m%d_%H%M%S")
            #     os.makedirs("/app/anpr_results/crops", exist_ok=True)
            #     os.makedirs("/app/anpr_results/frames", exist_ok=True)
            #     crop_filename = f"/app/anpr_results/crops/{timestamp}_{frame_number}.jpg"
            #     frame_filename = f"/app/anpr_results/frames/{timestamp}_{frame_number}.jpg"
            #     cv2.imwrite(crop_filename, plate_crop)
            #     cv2.imwrite(frame_filename, frame)

            #     with open("/app/anpr_results/plates.csv", "a") as f:
            #         f.write(
            #             f"{timestamp},{plate_text},{crop_filename},{frame_filename}\n"
            #         )

    return frame

# digital zoom function to crop and resize the frame based on the zoom factor
def apply_digital_zoom(frame, zoom):
    if zoom <= 1.0:
        return frame

    h, w = frame.shape[:2]

    new_w = int(w / zoom)
    new_h = int(h / zoom)

    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2

    cropped = frame[y1:y1 + new_h, x1:x1 + new_w]

    return cv2.resize(
        cropped,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )


# Lane detetion - OPEN CV
def process_frame_road_detection(frame, frame_number):
    h, w = frame.shape[:2]

    roi = frame[int(h * 0.55):h, 0:w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=80
    )

    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            if abs(slope) < 0.4:
                continue

            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))

            cv2.line(
                frame,
                (x1, y1 + int(h * 0.55)),
                (x2, y2 + int(h * 0.55)),
                (0, 255, 255),
                2
            )

    direction = "UNKNOWN"

    if len(left_lines) > 0 and len(right_lines) > 0:
        direction = "STRAIGHT"
    elif len(left_lines) > len(right_lines):
        direction = "RIGHT CURVE"
    elif len(right_lines) > len(left_lines):
        direction = "LEFT CURVE"

    cv2.putText(
        frame,
        f"ROAD: {direction}",
        (50, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    return frame

#Road Detection - Segmentation
def process_frame_road_detection_segmentation(frame, frame_number):
    cv2.putText(
        frame,
        "ROAD SEGMENTATION MODE",
        (50, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    return frame


# Get the Docker host gateway IP address for use in the Flask app
def get_docker_host_gateway():
    result = subprocess.run(
        "ip route | awk '/default/ {print $3}'",
        shell=True,
        capture_output=True,
        text=True
    )
    return result.stdout.strip()


# Load camera presets
def load_camera_presets():
    if not os.path.exists(CAMERA_PRESETS_FILE):
        return {}

    with open(CAMERA_PRESETS_FILE, "r") as f:
        return json.load(f)

# Separate worked used for recording video
def recording_worker():
    """Wątek działający w tle, który kompresuje i wysyła klatki, nie blokując kamery."""
    while True:
        try:
            frame_to_send = record_queue.get(timeout=1)
            
            # 1. Kompresja w osobnym wątku
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90] # 90 daje super jakość, a ulży CPU
            _, img_encoded = cv2.imencode('.jpg', frame_to_send, encode_param)
            
            # 2. Wysyłka HTTP w osobnym wątku
            requests.post(
                f"{HELPER_URL}/recording/push_frame",
                data=img_encoded.tobytes(),
                headers={'Content-Type': 'image/jpeg'},
                timeout=0.2
            )
            record_queue.task_done()
        except queue.Empty:
            continue
        except Exception:
            pass

# Producer
def producer():
    # global latest_frame, fps_smoothed, last_frame_time
    # cap = cv2.VideoCapture(CAMERA_SOURCE)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # cap.set(cv2.CAP_PROP_FPS, 30)
    global latest_frame, fps_smoothed, last_frame_time, cap_global
    global latest_raw_frame
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap_global = cap
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 17)
    #focus setup
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, camera_focus)
    #exposure setup
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    # cap.set(cv2.CAP_PROP_EXPOSURE, camera_exposure)
    subprocess.run([V4L2_CTL, "-c", "auto_exposure=1"])
    subprocess.run([V4L2_CTL, "-c", "exposure_dynamic_framerate=0"])
    subprocess.run([V4L2_CTL, "-c", f"exposure_time_absolute={camera_exposure_abs}"])
    subprocess.run([V4L2_CTL, "-c", f"gain={camera_gain}"])
    #zoom setup
    subprocess.run([V4L2_CTL, "-d", "/dev/video0", "--set-ctrl", f"zoom_absolute={camera_zoom_abs}"])
    #gain setup
    # cap.set(cv2.CAP_PROP_GAIN, camera_gain)

    print("Camera resolution:",cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        "FPS:",
        cap.get(cv2.CAP_PROP_FPS)
    )

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    print(
        "FOURCC:",
        "".join([
            chr((fourcc >> 8*i) & 0xFF)
            for i in range(4)
        ])
    )

    last_time = 0.0
    last_frame_time = time.time()
    frame_number = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        
        #get rotated frame if the camera is mounted upside down
        frame = apply_camera_orientation(frame)
        frame = apply_digital_zoom(frame, camera_digital_zoom)
        # frame = apply_digital_zoom(frame,camera_digital_zoom)
        # Last frame used to save when user press save_image_button
        latest_raw_frame = frame.copy()

        #recordning using external flask app inside shuddown_server.py
        # Recording -> Przesyłanie klatki do adas-helper (poza Dockerem)
        # Old version
        # if recording_enabled:
        #     try:
        #         # Zmień rozdzielczość jeśli trzeba (lub po prostu wyślij klatkę)
        #         rec_frame = cv2.resize(frame, (recording_width, recording_height))
                
        #         # Kodowanie do wysokiej jakości JPEG (95% jakości, żeby nie stracić ostrości)
        #         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        #         _, img_encoded = cv2.imencode('.jpg', rec_frame, encode_param)
                
        #         # Wysyłamy prosto do Helpera (port 5055)
        #         requests.post(
        #             f"{HELPER_URL}/recording/push_frame",
        #             data=img_encoded.tobytes(),
        #             headers={'Content-Type': 'image/jpeg'},
        #             timeout=0.05  # bardzo krótki timeout, aby nie zwalniać pętli detekcji
        #         )
        #     except Exception:
        #         pass # Ignorujemy ewentualne chwilowe zgubienie klatki
        # Wysyłanie do nagrywania BEZ BLOKOWANIA pętli:
        if recording_enabled:
            if not record_queue.full():
                record_queue.put_nowait(frame.copy()) # Wrzucenie do RAMu zajmuje 0.0001 sekundy!

        # frame = cv2.resize(frame, (1280, 720))
        frame = cv2.resize(frame, (1920, 1080))
        # get active model (may be None if still loading)
        model = manager.get_active()
        annotated = frame
        detections = []
        
        try:
            if model is None:
                # show loading overlay
                txt = f"Loading model: {manager.active}..."
                cv2.putText(annotated, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            elif manager.active == "irish-plate-recognition":
                # full car -> plate detection -> OCR pipeline
                annotated = process_frame(frame, frame_number)
            elif manager.active == "improved-irish-plate-recognition":
                # placeholder for future improved pipeline (e.g. with better pre/post-processing)
                annotated = process_frame_optimized(frame, frame_number)
            elif manager.active == "lprnet-anpr":
                annotated = process_frame_lprnet(frame, frame_number)
            elif manager.active == "road-detection":
                frame = process_frame_road_detection(frame, frame_number)
            elif manager.active == "road-detection-segmentation":
                annotated = process_frame_road_detection_segmentation(frame, frame_number)
            else:
                # run inference on single-frame
                
                results = model(frame, imgsz=IMG_SIZE, verbose=False)
                r = results[0]
                annotated = r.plot() if hasattr(r, "plot") else frame

                if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        try:
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            cls_name = r.names[cls_id]
                            if conf > 0.55:
                                detections.append({
                                    "class": cls_name,
                                    "confidence": conf,
                                    "box": box.xyxy[0].tolist()
                                })
                        except Exception:
                            continue

                    # save cup frames (rate-limited can be added)
                    for d in detections:
                        if d["class"] == "cup":
                            save_path = f"/app/detected_cup_{int(time.time())}.jpg"
                            cv2.imwrite(save_path, annotated)

        except Exception as e:
            # keep running producer even if model inference fails
            errtxt = f"Model error: {e}"
            cv2.putText(annotated, errtxt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # compute FPS
        now = time.time()
        dt = now - last_frame_time if last_frame_time else 0.0
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        alpha = 0.2
        fps_smoothed = fps_smoothed * (1.0 - alpha) + inst_fps * alpha if fps_smoothed > 0 else inst_fps
        last_frame_time = now

        fps_text = f"FPS: {fps_smoothed:.1f} | Active: {manager.active}"
        (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (8, 8+55), (12 + tw, 14 + th +55), (0, 0, 0), -1)
        cv2.putText(annotated, fps_text, (10, 12 + th + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        ret2, buf = cv2.imencode('.jpg', annotated)
        if not ret2:
            continue
        with frame_lock:
            latest_frame = buf.tobytes()

        frame_number += 1

        # throttle
        if FPS_LIMIT > 0:
            wait = max(0.0, (1.0 / FPS_LIMIT) - (time.time() - last_time))
            if wait > 0:
                time.sleep(wait)
            last_time = time.time()

    cap.release()


def mjpeg_generator():
    boundary = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
    try:
        while not stop_event.is_set():
            with frame_lock:
                frame = latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
            yield boundary + frame + b'\r\n'
            time.sleep(0.01)
    except GeneratorExit:
        return

# Use index_lazy_v3.html for the web interface, which includes buttons for switching between detection, pose estimation, segmentation and custom model. The producer thread captures video frames, runs inference with the active model, annotates the frames, and saves detected cups as images. The MJPEG generator streams the annotated video to the web interface.
@app.route('/')
def index():
    return render_template('index_lazy_v6.html')


@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/switch', methods=['POST'])
def switch_model():
    data = request.json or request.form
    name = data.get('model') if isinstance(data, dict) else None
    if name not in ("detection", "pose", "segmentation", "custom-model", "plate-recognition", "irish-plate-recognition","improved-irish-plate-recognition","lprnet-anpr","road-detection","road-detection-segmentation"):
        return jsonify({"error": "invalid model"}), 400
    try:
        manager.switch(name)
    except KeyError:
        return jsonify({"error": "unknown model"}), 400
    return jsonify({"switching": True, "active": manager.active, "loaded": manager.is_loaded(name)})


@app.route('/status')
def status():
    s = manager.status()
    s.update({"fps": fps_smoothed})
    return jsonify(s)

# @app.route("/camera/focus", methods=["POST"])
# def set_camera_focus():
#     global camera_focus, cap_global

#     data = request.get_json() or {}
#     value = int(data.get("focus", camera_focus))

#     value = max(0, min(255, value))
#     camera_focus = value

#     ok = False
#     if cap_global is not None:
#         cap_global.set(cv2.CAP_PROP_AUTOFOCUS, 0)
#         ok = cap_global.set(cv2.CAP_PROP_FOCUS, camera_focus)

#     return jsonify({
#         "focus": camera_focus,
#         "ok": bool(ok)
#     })
@app.route("/camera/focus", methods=["POST"])
def set_camera_focus():
    global camera_focus, cap_global

    data = request.get_json() or {}
    
    # 1. Sprawdzamy czy użytkownik chce włączyć Auto Focus
    is_auto = data.get("auto", False)

    if is_auto:
        # Włączamy Auto Focus przez V4L2 oraz OpenCV
        subprocess.run([V4L2_CTL, "-c", "focus_automatic_continuous=1"], check=False)
        if cap_global is not None:
            cap_global.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
        print("[Camera] Auto Focus: WŁĄCZONY")
        return jsonify({"ok": True, "mode": "auto"})

    # 2. Jeśli nie auto -> wykonuje się Twój dotychczasowy kod dla Manual Focus
    value = int(data.get("focus", camera_focus))
    value = max(0, min(255, value))
    camera_focus = value

    ok = False
    # Najpierw wymuszamy wyłączenie autofocusa w systemie, żeby nie nadpisywał ręcznej wartości
    subprocess.run([V4L2_CTL, "-c", "focus_automatic_continuous=0"], check=False)
    
    if cap_global is not None:
        cap_global.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        ok = cap_global.set(cv2.CAP_PROP_FOCUS, camera_focus)

    print(f"[Camera] Manual Focus ustawiony na: {camera_focus}")
    return jsonify({
        "focus": camera_focus,
        "mode": "manual",
        "ok": bool(ok)
    })

@app.route("/camera/exposure_abs", methods=["POST"])
def set_camera_exposure_abs():
    global camera_exposure_abs

    data = request.get_json() or {}
    value = int(data.get("exposure", camera_exposure_abs))
    value = max(3, min(2047, value))

    subprocess.run([
        "v4l2-ctl",
        "-c",
        "auto_exposure=1"
    ])

    subprocess.run([
        "v4l2-ctl",
        "-c",
        "exposure_dynamic_framerate=0"
    ])

    result = subprocess.run([
        "v4l2-ctl",
        "-c",
        f"exposure_time_absolute={value}"
    ])

    camera_exposure_abs = value

    return jsonify({
        "exposure": camera_exposure_abs,
        "ok": result.returncode == 0
    })

@app.route("/camera/gain", methods=["POST"])
def set_camera_gain():
    global camera_gain, cap_global

    data = request.get_json() or {}
    value = int(data.get("gain", camera_gain))

    value = max(0, min(255, value))
    camera_gain = value

    ok = False
    if cap_global is not None:
        ok = cap_global.set(cv2.CAP_PROP_GAIN, camera_gain)

    return jsonify({"gain": camera_gain, "ok": bool(ok)})

# camera_zoom_abs = 100

# @app.route("/camera/zoom_abs", methods=["POST"])
# def set_camera_zoom_abs():
#     global camera_zoom_abs

#     data = request.get_json() or {}
#     value = int(data.get("zoom", camera_zoom_abs))

#     # u Ciebie realnie działa 100-177
#     value = max(100, min(500, value))
#     camera_zoom_abs = value

#     result = subprocess.run([
#         V4L2_CTL,
#         "-d",
#         "/dev/video0",
#         "--set-ctrl",
#         f"zoom_absolute={value}"
#     ])

#     return jsonify({
#         "zoom": camera_zoom_abs,
#         "ok": result.returncode == 0
#     })
#Digital zoom is a software-based zoom that crops and resizes the image, while zoom_absolute is a hardware-based zoom that adjusts the camera lens. The digital zoom can be set to a value between 1.0 (no zoom) and 3.0 (3x zoom), and it is applied in the producer function before processing the frame.
@app.route("/camera/digital_zoom", methods=["POST"])
def set_camera_digital_zoom():
    global camera_digital_zoom

    data = request.get_json() or {}

    value = float(
        data.get("zoom", camera_digital_zoom)
    )

    value = max(1.0, min(3.0, value))

    camera_digital_zoom = value

    return jsonify({
        "zoom": camera_digital_zoom,
        "ok": True
    })
#Orientation
@app.route("/camera/orientation", methods=["POST"])
def set_camera_orientation():
    global camera_flip_180

    data = request.get_json() or {}
    mode = data.get("mode", "normal")

    if mode == "car":
        camera_flip_180 = True
    else:
        camera_flip_180 = False

    return jsonify({
        "mode": "car" if camera_flip_180 else "normal",
        "flip_180": camera_flip_180,
        "ok": True
    })

#Recording end point
@app.route("/recording/toggle", methods=["POST"])
def toggle_recording():
    global recording_enabled, recording_writer, recording_fps, recording_width, recording_height

    data = request.get_json() or {}

    recording_fps = int(data.get("fps", recording_fps))
    recording_width = int(data.get("width", recording_width))
    recording_height = int(data.get("height", recording_height))

    os.makedirs(RECORDING_DIR, exist_ok=True)

    if not recording_enabled:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RECORDING_DIR, f"road_recording_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        recording_writer = cv2.VideoWriter(
            filename,
            fourcc,
            recording_fps,
            (recording_width, recording_height)
        )

        if not recording_writer.isOpened():
            recording_writer = None
            return jsonify({"ok": False, "error": "VideoWriter failed"}), 500

        recording_enabled = True

        return jsonify({
            "ok": True,
            "recording": True,
            "file": filename,
            "fps": recording_fps,
            "width": recording_width,
            "height": recording_height
        })

    else:
        recording_enabled = False

        if recording_writer is not None:
            recording_writer.release()
            recording_writer = None

        return jsonify({
            "ok": True,
            "recording": False
        })

@app.route("/recording/save_image", methods=["POST"])
def save_current_image():
    global latest_raw_frame

    if latest_raw_frame is None:
        return jsonify({
            "ok": False,
            "error": "No frame available"
        }), 500

    os.makedirs(IMAGES_DIR, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(IMAGES_DIR, f"frame_{timestamp}.jpg")

    ok = cv2.imwrite(filename, latest_raw_frame)

    return jsonify({
        "ok": bool(ok),
        "file": filename
    })


# Implement a shutdown endpoint that sends a POST request to the Jetson helper service to initiate a system shutdown. The endpoint retrieves the Docker host gateway IP address, constructs the URL for the shutdown request, and sends the request with a timeout of 3 seconds. The response from the helper service is returned in JSON format, indicating whether the shutdown request was successful and providing details about the helper URL and response.
@app.route("/system/shutdown", methods=["POST"])
def shutdown_jetson():
    host_ip = get_docker_host_gateway()
    url = f"https://{host_ip}:5055/shutdown"

    r = requests.post(url, timeout=3)

    return jsonify({
        "ok": True,
        "helper_url": url,
        "helper_response": r.json()
    })


#end point presents for camera settings. It loads the presets from a JSON file and returns them as a JSON response. This allows the web interface to retrieve and display the available camera presets for user selection.
@app.route("/camera/presets", methods=["GET"])
def get_camera_presets():
    presets = load_camera_presets()
    return jsonify(presets)

@app.route("/camera/preset/<preset_name>", methods=["POST"])
def apply_camera_preset(preset_name):
    global camera_focus, camera_exposure_abs, camera_gain, camera_digital_zoom

    presets = load_camera_presets()

    if preset_name not in presets:
        return jsonify({
            "ok": False,
            "error": f"Preset not found: {preset_name}"
        }), 404

    preset = presets[preset_name]

    camera_focus = int(preset.get("focus", camera_focus))
    camera_exposure_abs = int(preset.get("exposure_abs", camera_exposure_abs))
    camera_gain = int(preset.get("gain", camera_gain))
    camera_digital_zoom = float(preset.get("digital_zoom", camera_digital_zoom))

    if cap_global is not None:
        cap_global.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap_global.set(cv2.CAP_PROP_FOCUS, camera_focus)

    subprocess.run([V4L2_CTL, "-c", "auto_exposure=1"])
    subprocess.run([V4L2_CTL, "-c", "exposure_dynamic_framerate=0"])
    subprocess.run([V4L2_CTL, "-c", f"exposure_time_absolute={camera_exposure_abs}"])
    subprocess.run([V4L2_CTL, "-c", f"gain={camera_gain}"])

    return jsonify({
        "ok": True,
        "preset": preset_name,
        "focus": camera_focus,
        "exposure_abs": camera_exposure_abs,
        "gain": camera_gain,
        "digital_zoom": camera_digital_zoom
    })

# end point for setting recording status
@app.route('/set_recording_status', methods=['POST'])
def set_recording_status():
    global recording_enabled, recording_width, recording_height, recording_fps
    data = request.json or {}
    
    recording_enabled = data.get('enabled', False)
    
    if 'width' in data and 'height' in data:
        recording_width = int(data.get('width'))
        recording_height = int(data.get('height'))
        
    if 'fps' in data:
        recording_fps = int(data.get('fps'))
        
    print(f"[ADAS] Nagrywanie: {recording_enabled} ({recording_width}x{recording_height} @ {recording_fps} FPS)")
    return jsonify({"ok": True})


# Uruchamiamy wątek tła raz przy starcie aplikacji
threading.Thread(target=recording_worker, daemon=True).start()


# if __name__ == '__main__':
#     prod = threading.Thread(target=producer, daemon=True)
#     prod.start()
#     try:
#         app.run(host='0.0.0.0', port=5010, threaded=True)
#     finally:
#         stop_event.set()
#         prod.join(timeout=2.0)



