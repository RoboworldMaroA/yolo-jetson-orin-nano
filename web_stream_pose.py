from ultralytics import YOLO
from flask import Flask, Response, render_template_string
import os
import cv2
import time
import threading

# Config (can override via environment)
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/yolo11n-pose.engine")  # adjust name if needed
CAMERA_SOURCE = int(os.environ.get("CAMERA_SOURCE", "0"))
FPS_LIMIT = float(os.environ.get("FPS_LIMIT", "10.0"))  # lower FPS for pose

app = Flask(__name__)

# load model once; explicitly set task='pose' if using .pt
# If using a TensorRT engine, YOLO(MODEL_PATH) should work; for .pt use YOLO(MODEL_PATH, task='pose')
if MODEL_PATH.endswith('.pt'):
    model = YOLO(MODEL_PATH, task='pose')
else:
    model = YOLO(MODEL_PATH)

INDEX_HTML = """
<!doctype html>
<title>YOLO11 Pose Stream</title>
<h1>YOLO11 Pose Stream</h1>
<img src="{{ url_for('video_feed') }}" width="720" />
<p>Press Ctrl+C in container to stop server.</p>
"""

latest_frame = None
frame_lock = threading.Lock()
stop_event = threading.Event()

def producer():
    """Single background producer: runs model.predict(stream=True) and updates latest_frame."""
    global latest_frame
    try:
        results = model.predict(source=CAMERA_SOURCE, stream=True, save=False, verbose=False)
        last_time = 0.0
        for r in results:
            if stop_event.is_set():
                break
            # r.plot() draws keypoints + skeleton for pose models
            frame = r.plot()
            # throttle
            if FPS_LIMIT > 0:
                wait = max(0.0, (1.0 / FPS_LIMIT) - (time.time() - last_time))
                if wait > 0:
                    time.sleep(wait)
                last_time = time.time()
            ret, buf = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            with frame_lock:
                latest_frame = buf.tobytes()
    except Exception as e:
        print("Producer error:", e)
    finally:
        stop_event.set()

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

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    prod_thread = threading.Thread(target=producer, daemon=True)
    prod_thread.start()
    try:
        app.run(host='0.0.0.0', port=5001, threaded=True)
    finally:
        stop_event.set()
        prod_thread.join(timeout=2.0)