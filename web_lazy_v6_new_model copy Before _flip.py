
# Author: Marek Augustyn
# 21 June 2026
# I am workin on Add a new way of licence plate recognition that allows me do the faster object detection
# Add paddle ONNX model for licence plate recognition, which is faster than EasyOCR and more accurate for Irish plates.
# Test only: 

# Base on web_lazy_v5_easyOCR_v2.py
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

# It uses a yolo docker container, Uspb camera is connected to the Jetson Orin Nano
# Check instruction.txt for how to run the program with docker
# Run:
# Use new docker Image with easyOCR and licence plate recognition model, which is based on YOLOv8n trained on Irish plates
# http://192.168.0.12:5010/
#can be use also with roboworld.react.marekaugustyn.whshost.com:5010 

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
import os

import numpy as np
import onnxruntime as ort

import subprocess


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




# Focus variables for the Irish plate recognition pipeline
camera_focus = 30
# camera_exposure = -5
camera_exposure_abs = 210
camera_gain = 0
cap_global = None
V4L2_CTL = "/usr/bin/v4l2-ctl"
camera_digital_zoom = 1.0

camera_zoom_abs = 150

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
    "lprnet-anpr": MODEL_PATH_PLATE_RECOGNITION
}, default="lprnet-anpr")
# kick off load for default model in background
manager.ensure_loaded(manager.active)


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



# Producer
def producer():
    # global latest_frame, fps_smoothed, last_frame_time
    # cap = cv2.VideoCapture(CAMERA_SOURCE)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # cap.set(cv2.CAP_PROP_FPS, 30)
    global latest_frame, fps_smoothed, last_frame_time, cap_global
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap_global = cap
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
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
        
        frame = apply_digital_zoom(frame,camera_digital_zoom)

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
    if name not in ("detection", "pose", "segmentation", "custom-model", "plate-recognition", "irish-plate-recognition","improved-irish-plate-recognition","lprnet-anpr"):
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

@app.route("/camera/focus", methods=["POST"])
def set_camera_focus():
    global camera_focus, cap_global

    data = request.get_json() or {}
    value = int(data.get("focus", camera_focus))

    value = max(0, min(255, value))
    camera_focus = value

    ok = False
    if cap_global is not None:
        cap_global.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        ok = cap_global.set(cv2.CAP_PROP_FOCUS, camera_focus)

    return jsonify({
        "focus": camera_focus,
        "ok": bool(ok)
    })

# @app.route("/camera/exposure", methods=["POST"])
# def set_camera_exposure():
#     global camera_exposure, cap_global

#     data = request.get_json() or {}
#     value = int(data.get("exposure", camera_exposure))

#     value = max(-13, min(0, value))
#     camera_exposure = value

#     ok = False
#     if cap_global is not None:
#         cap_global.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
#         ok = cap_global.set(cv2.CAP_PROP_EXPOSURE, camera_exposure)

#     return jsonify({"exposure": camera_exposure, "ok": bool(ok)})

# @app.route("/camera/exposure", methods=["POST"])
# def set_camera_exposure():
#     global camera_exposure

#     value = int(request.json["exposure"])

#     value = max(3, min(2047, value))

#     subprocess.run([
#         "v4l2-ctl",
#         "-c",
#         f"exposure_time_absolute={value}"
#     ])

#     camera_exposure = value

#     return jsonify({
#         "exposure": value,
#         "ok": True
#     })

# camera_exposure_abs = 210

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

if __name__ == '__main__':
    prod = threading.Thread(target=producer, daemon=True)
    prod.start()
    try:
        app.run(host='0.0.0.0', port=5010, threaded=True)
    finally:
        stop_event.set()
        prod.join(timeout=2.0)



