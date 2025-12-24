from ultralytics import YOLO
from flask import Flask, Response, render_template_string
import os
import cv2
import time
import threading

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/yolov8n.engine")
CAMERA_SOURCE = int(os.environ.get("CAMERA_SOURCE", "0"))
FPS_LIMIT = float(os.environ.get("FPS_LIMIT", "30.0"))  # optional throttle

app = Flask(__name__)

# load model once
model = YOLO(MODEL_PATH)

INDEX_HTML = """
<!doctype html>
<title>YOLO Camera Stream</title>
<h1>YOLO Camera Stream</h1>
<img src="{{ url_for('video_feed') }}" width="1024" />
<p>Press Ctrl+C in container to stop server.</p>
"""

# Shared state between producer and clients
latest_frame = None
frame_lock = threading.Lock()
stop_event = threading.Event()

# ...existing code...
# add FPS state variables
fps_smoothed = 0.0
last_frame_time = None

def producer():
    """Single background producer: runs model.predict(stream=True) and updates latest_frame."""
    global latest_frame, fps_smoothed, last_frame_time
    try:
        # ask model to run inference at a larger input size (e.g. 1920)
        # NOTE: larger imgsz => more GPU memory and slower FPS
        results = model.predict(
            source=CAMERA_SOURCE,
            stream=True,
            save=False,
            verbose=False,
            imgsz=1920  # increase model input size
        )
        last_time = 0.0
        last_frame_time = time.time()
        for r in results:
            if stop_event.is_set():
                break

            # get annotated frame from result
            frame = r.plot()
            print("Predictions:", r.names)  # number of boxes

            # compute instantaneous FPS
            now = time.time()
            dt = now - last_frame_time if last_frame_time is not None else 0.0
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            # exponential moving average for smoother display
            alpha = 0.2
            fps_smoothed = fps_smoothed * (1.0 - alpha) + inst_fps * alpha if fps_smoothed > 0 else inst_fps
            last_frame_time = now

            # draw FPS on frame (top-left)
            fps_text = f"FPS: {fps_smoothed:.1f}"
            # draw semi-transparent rectangle background for readability
            (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (8, 8), (12 + tw, 14 + th), (0, 0, 0), -1)  # black box
            cv2.putText(frame, fps_text, (10, 12 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

            # throttle (keep low to avoid OOM)
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
# ...existing code...

def mjpeg_generator():
    """Yield the latest_frame repeatedly for a connected client."""
    global latest_frame
    boundary = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
    try:
        while not stop_event.is_set():
            with frame_lock:
                frame = latest_frame
            if frame is None:
                # no frame yet
                time.sleep(0.05)
                continue
            yield boundary + frame + b'\r\n'
            # small sleep to avoid busy loop; adjust to control client FPS
            time.sleep(0.01)
    except GeneratorExit:
        # client disconnected; just return and keep producer running
        return

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # start background producer thread
    prod_thread = threading.Thread(target=producer, daemon=True)
    prod_thread.start()
    try:
        app.run(host='0.0.0.0', port=5001, threaded=True)
    finally:
        # on shutdown signal, request producer stop and wait
        stop_event.set()
        prod_thread.join(timeout=2.0)