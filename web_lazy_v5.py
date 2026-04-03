
# Author: Marek Augustyn
# 03 April 2026
# Working on add a OCR custom model Plate Recognition
# Program allow recognize object using yolo pretrained model and stream video with detections over web server
# It save detected cups as images
# You can swith from the web browser between models like Pose Estimation and Object Detection, Segmentation, Custom MOdel Start Stop and Plate Recognition
# If cub is reconize with 60% confidence or higher, it save the frame with detected cup in the /app folder with name detected_cup_TIMESTAMP.jpg
# It use a yolo docker container with mounted webcam and output folder for detected cups
# Check instruction.txt for how to run the program with docker
# Run:
# http://192.168.0.12:5004/

from ultralytics import YOLO
from flask import Flask, Response, render_template, jsonify, request
import threading
import time
import cv2
import os

# Configuration
MODEL_PATH_DETECT = os.environ.get("MODEL_PATH_DETECT", "/app/yolov8n.engine")
MODEL_PATH_POSE = os.environ.get("MODEL_PATH_POSE", "/app/yolo11n-pose.engine")
MODEL_PATH_SEGMENTATION = os.environ.get("MODEL_PATH_SEGMENTATION", "/app/yolo11n-seg.engine")
MODEL_PATH_CUSTOM_MODEL = os.environ.get("MODEL_PATH_CUSTOM_MODEL", "/app/start_stop_yolo8.engine")
MODEL_PATH_PLATE_RECOGNITION = os.environ.get("MODEL_PATH_PLATE_RECOGNITION", "/app/licence_plate.pt")  # replace with actual plate recognition model path
CAMERA_SOURCE = int(os.environ.get("CAMERA_SOURCE", "0"))
FPS_LIMIT = float(os.environ.get("FPS_LIMIT", "20.0"))

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
    def __init__(self, mapping, default="detection"):
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
manager = ModelManager({"detection": MODEL_PATH_DETECT, "pose": MODEL_PATH_POSE, "segmentation": MODEL_PATH_SEGMENTATION, "custom-model": MODEL_PATH_CUSTOM_MODEL, "plate-recognition": MODEL_PATH_PLATE_RECOGNITION}, default="detection")
# kick off load for default model in background
manager.ensure_loaded(manager.active)


def producer():
    global latest_frame, fps_smoothed, last_frame_time
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    last_time = 0.0
    last_frame_time = time.time()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # get active model (may be None if still loading)
        model = manager.get_active()
        annotated = frame
        detections = []

        try:
            if model is None:
                # show loading overlay
                txt = f"Loading model: {manager.active}..."
                cv2.putText(annotated, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            else:
                # run inference on single-frame
                results = model(frame)
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
        cv2.rectangle(annotated, (8, 8+35), (12 + tw, 14 + th +35), (0, 0, 0), -1)
        cv2.putText(annotated, fps_text, (10, 12 + th + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        ret2, buf = cv2.imencode('.jpg', annotated)
        if not ret2:
            continue
        with frame_lock:
            latest_frame = buf.tobytes()

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
    return render_template('index_lazy_v4.html')


@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/switch', methods=['POST'])
def switch_model():
    data = request.json or request.form
    name = data.get('model') if isinstance(data, dict) else None
    if name not in ("detection", "pose", "segmentation", "custom-model", "plate-recognition"):
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


if __name__ == '__main__':
    prod = threading.Thread(target=producer, daemon=True)
    prod.start()
    try:
        app.run(host='0.0.0.0', port=5004, threaded=True)
    finally:
        stop_event.set()
        prod.join(timeout=2.0)
