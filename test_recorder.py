import os
import time
import threading
import subprocess
import cv2
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

RECORDING_DIR = "/home/maro/yolo_app/road_recordings"
os.makedirs(RECORDING_DIR, exist_ok=True)

recording = False
cap = None
writer = None
record_thread = None

def start_autofocus():
    """Włącza Auto Focus przez v4l2-ctl na starcie."""
    try:
        subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "-c", "focus_automatic_continuous=1"], check=False)
        print("[TEST] Auto Focus włączony.")
    except Exception as e:
        print(f"[TEST] Błąd włączania Auto Focus: {e}")

def recording_worker(width=1920, height=1080, fps=20):
    global recording, cap, writer
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(RECORDING_DIR, f"test_recording_{timestamp}.mp4")

    print(f"[TEST] Otwieranie kamery /dev/video0 ({width}x{height} @ {fps} FPS)...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        print("[TEST ERROR] Nie można otworzyć kamery!")
        recording = False
        return

    start_autofocus()

    # Zapis bezpośrednio przez kodek MP4V (bez GStreamera)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

    if not writer.isOpened():
        print("[TEST ERROR] VideoWriter NIE wystartował!")
        cap.release()
        recording = False
        return

    print(f"[TEST SUCCESS] Zapis uruchomiony! Plik: {filepath}")
    frames_written = 0

    while recording:
        ret, frame = cap.read()
        if not ret:
            print("[TEST WARN] Błąd odczytu klatki z kamery!")
            time.sleep(0.01)
            continue

        writer.write(frame)
        frames_written += 1
        time.sleep(1 / fps)

    print(f"[TEST] Zapisano {frames_written} klatek. Zamykanie pliku...")
    writer.release()
    cap.release()
    writer = None
    cap = None
    print(f"[TEST COMPLETE] Zapisano plik: {filepath}")

# --- HTML INTERFACE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Test Nagrywania NVENC</title></head>
<body style="font-family: sans-serif; text-align: center; padding-top: 50px; background: #1a1a1a; color: white;">
    <h1>Test GStreamer / NVENC Recorder</h1>
    <h2 id="status" style="color: yellow;">Stan: Bezczynny</h2>
    <button onclick="startRec()" style="padding: 15px 30px; font-size: 18px; background: green; color: white; border: none; cursor: pointer;">START NAGRYWANIA</button>
    <button onclick="stopRec()" style="padding: 15px 30px; font-size: 18px; background: red; color: white; border: none; cursor: pointer; margin-left: 10px;">STOP NAGRYWANIA</button>

    <script>
        async function startRec() {
            let res = await fetch('/start', {method: 'POST'});
            let data = await res.json();
            document.getElementById('status').innerText = data.message;
        }
        async function stopRec() {
            let res = await fetch('/stop', {method: 'POST'});
            let data = await res.json();
            document.getElementById('status').innerText = data.message;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/start', methods=['POST'])
def start():
    global recording, record_thread
    if recording:
        return jsonify({"message": "Nagrywanie już trwa!"})
    
    recording = True
    record_thread = threading.Thread(target=recording_worker, daemon=True)
    record_thread.start()
    return jsonify({"message": "Nagrywanie WŁĄCZONE (sprawdzaj konsolę)"})

@app.route('/stop', methods=['POST'])
def stop():
    global recording
    if not recording:
        return jsonify({"message": "Nagrywanie nie było aktywne!"})
    
    recording = False
    return jsonify({"message": "Zatrzymywanie nagrywania... Czekaj na komunikat w konsoli."})

if __name__ == '__main__':
    print("[TEST] Aplikacja testowa gotowa na porcie 8080...")
    app.run(host='0.0.0.0', port=5001)