#.    sudo systemctl restart adas-helper.service
#.    sudo systemctl status adas-helper.service
#.    journalctl -u adas-helper.service -f
# Before new version sending data directly to the hellper app, and in new version shut_down_server.py I will be using Gstreamer
from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import os
import time
import cv2
import threading
import queue

app = Flask(__name__)
CORS(app)

# Globalne zmienne dla potoku wideo na hoście
recording_active = False
video_queue = queue.Queue(maxsize=300)
video_writer = None
recorder_thread = None

RECORDING_DIR = "/home/maro/yolo_app/road_recordings"
IMAGES_DIR = "/home/maro/yolo_app/images_from_camera_stream_jetson_nano"

def nvenc_write_worker(output_path, width, height, fps):
    """Wątek procesowy obsługujący sprzętowe kodowanie NVENC na Jetsonie."""
    global recording_active, video_writer, video_queue
    
    # Optymalny potok GStreamer dla Jetson Orin Nano Super
    gst_pipeline = (
        f"appsrc ! videoconvert ! video/x-raw, format=BGRx ! "
        f"nvvidconv ! video/x-raw(memory:NVMM), format=NV12 ! "
        f"nvv4l2h264enc bitrate=4000000 control-rate=1 preset-level=1 ! h264parse ! "
        f"qtmux ! filesink location={output_path}"
    )
    
    video_writer = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, float(fps), (width, height))
    
    if not video_writer.isOpened():
        print("[Helper] Blad: Nie udalo sie otworzyc potoku GStreamer NVENC.")
        recording_active = False
        return

    print(f"[Helper] Sprzetowe nagrywanie uruchomione: {output_path}")
    
    while recording_active or not video_queue.empty():
        try:
            # Timeout pozwala wątkowi cyklicznie sprawdzać stan flagi recording_active
            frame = video_queue.get(timeout=0.5)
            video_writer.write(frame)
            video_queue.task_done()
        except queue.Empty:
            continue
            
    video_writer.release()
    video_writer = None
    print("[Helper] Potok NVENC bezpiecznie zamkniety, plik zapisany.")

# --- NOWE ENDPOINTY DLA WIDEO ---

@app.route("/video_save", methods=["POST"])
def video_save():
    """Zapisuje pojedynczą klatkę wysłaną z aplikacji ADAS jako plik JPG."""
    if "frame" not in request.files:
        return jsonify({"ok": False, "message": "Brak pliku frame w zadaniu"}), 400
        
    file = request.files["frame"]
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(IMAGES_DIR, f"frame_{timestamp}.jpg")
    file.save(filename)
    
    return jsonify({"ok": True, "message": "Klatka zapisana pomyslnie na hostie", "file": filename})

@app.route("/recording/start", methods=["POST"])
def start_recording():
    """Inicjalizuje proces nagrywania sprzętowego na hoście."""
    global recording_active, recorder_thread, video_queue
    
    if recording_active:
        return jsonify({"ok": False, "message": "Nagrywanie juz trwa"}), 400
        
    data = request.get_json() or {}
    width = int(data.get("width", 1280))
    height = int(data.get("height", 720))
    fps = int(data.get("fps", 60))
    
    os.makedirs(RECORDING_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RECORDING_DIR, f"road_recording_{timestamp}.mp4")
    
    # Czyszczenie starej kolejki przed nowym nagrywaniem
    with video_queue.mutex:
        video_queue.queue.clear()
        
    recording_active = True
    recorder_thread = threading.Thread(
        target=nvenc_write_worker, 
        args=(filename, width, height, fps),
        daemon=True
    )
    recorder_thread.start()
    
    return jsonify({"ok": True, "message": "Nagrywanie NVENC wystartowalo", "file": filename})

@app.route("/recording/stop", methods=["POST"])
def stop_recording():
    """Zamyka potok i kończy zapis pliku wideo."""
    global recording_active, recorder_thread
    
    if not recording_active:
        return jsonify({"ok": False, "message": "Nagrywanie nie jest aktywne"}), 400
        
    recording_active = False
    if recorder_thread:
        recorder_thread.join(timeout=5.0)
        
    return jsonify({"ok": True, "message": "Zadanie zakonczenia nagrywania wyslane do NVENC"})

@app.route("/recording/push_frame", methods=["POST"])
def push_frame():
    """Obiera skompresowaną klatkę z Dockera i wrzuca do kolejki zapisu NVENC."""
    global recording_active, video_queue
    
    if not recording_active:
        return jsonify({"ok": False, "message": "Nagrywanie wyłączone"}), 200
        
    if "frame" not in request.files:
        return jsonify({"ok": False, "message": "Brak klatki"}), 400
        
    # Szybka konwersja binarnego pliku z powrotem na macierz OpenCV
    import numpy as np
    file = request.files["frame"]
    filestr = file.read()
    np_img = np.frombuffer(filestr, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    try:
        # block=False zapobiega przytykaniu Dockera, jeśli dysk wolno zapisuje
        video_queue.put(frame, block=False)
        return jsonify({"ok": True})
    except queue.Full:
        return jsonify({"ok": False, "message": "Kolejka wideo helpera przepelniona"}), 503

# --- TWOJE ISTNIEJĄCE ENDPOINTY SYSTEMOWE ---

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"ok": True, "message": "helper alive"})

@app.route("/shutdown", methods=["POST"])
def shutdown():
    subprocess.Popen(["/home/maro/shutdown_jetson.sh"])
    return jsonify({"ok": True, "message": "Shutdown started"})

@app.route("/reboot", methods=["POST"])
def reboot():
    subprocess.Popen(["/home/maro/reboot_jetson.sh"])
    return jsonify({"ok": True, "message": "Jetson rebooting..."})

@app.route("/restart_adas", methods=["POST"])
def restart_adas():
    subprocess.Popen(["/home/maro/restart_adas.sh"])
    return jsonify({"ok": True, "message": "ADAS restarting..."})

@app.route("/restart_helper", methods=["POST"])
def restart_helper():
    subprocess.Popen(["/home/maro/restart_helper.sh"])
    return jsonify({"ok": True, "message": "Helper restarting..."})

@app.route("/system_status")
def system_status():
    with open("/sys/devices/virtual/thermal/thermal_zone0/temp") as f:
        cpu_temp = int(f.read()) / 1000.0
    mem = subprocess.check_output(["free", "-m"]).decode().splitlines()
    mem_values = mem[1].split()
    mem_total = int(mem_values[1])
    mem_used = int(mem_values[2])
    disk_cmd = subprocess.check_output(["df", "-h", "/"]).decode().splitlines()
    disk_values = disk_cmd[1].split()
    load = os.getloadavg()
    
    try:
        subprocess.check_output(["systemctl","is-active","adas.service"])
        adas_running = True
    except subprocess.CalledProcessError:
        adas_running = False

    try:
        subprocess.check_output(["systemctl","is-active","adas-helper.service"])
        helper_running = True
    except subprocess.CalledProcessError:
        helper_running = False

    return jsonify({
        "current_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_temperature": cpu_temp,
        "memory_used_mb": mem_used,
        "memory_total_mb": mem_total,
        "disk_used": disk_values[2],
        "disk_total": disk_values[1],
        "disk_percent": disk_values[4],
        "uptime": subprocess.check_output(["uptime", "-p"]).decode().strip(),
        "cpu_load_1min": round(load[0],2),
        "cpu_load_5min": round(load[1],2),
        "cpu_load_15min": round(load[2],2),
        "adas_running": adas_running,
        "helper_running": helper_running
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055)