from ultralytics import YOLO
from flask import Flask, Response, render_template_string
import os
import cv2
import time

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/yolov8n.engine")
CAMERA_SOURCE = int(os.environ.get("CAMERA_SOURCE", "0"))
FPS_LIMIT = float(os.environ.get("FPS_LIMIT", "15.0"))  # optional throttle

app = Flask(__name__)

# load model once
model = YOLO(MODEL_PATH)

INDEX_HTML = """
<!doctype html>
<title>YOLO Camera Stream</title>
<h1>YOLO Camera Stream</h1>
<img src="{{ url_for('video_feed') }}" width="720" />
<p>Press Ctrl+C in container to stop server.</p>
"""

def mjpeg_generator():
    # model.predict returns a generator when stream=True
    results = model.predict(source=CAMERA_SOURCE, stream=True, save=False, verbose=False)
    last_time = 0.0
    for r in results:
        # annotated BGR frame (numpy)
        frame = r.plot()
        # optional throttle
        if FPS_LIMIT > 0:
            wait = max(0.0, (1.0 / FPS_LIMIT) - (time.time() - last_time))
            if wait > 0:
                time.sleep(wait)
            last_time = time.time()
        # encode to JPEG
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        jpg = buf.tobytes()
        # multipart yield
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
    # end generator

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # listen on all interfaces so host can access via mapped port
    app.run(host='0.0.0.0', port=5001, threaded=True)