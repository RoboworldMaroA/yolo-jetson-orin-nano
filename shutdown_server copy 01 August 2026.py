# Author: Marek Augustyn
# Date: 02 August 2026
# Description: adar.helper is runing this flask app so you can manage from the website camera parametes and monitorin Jetson operations
#.    sudo systemctl stop adas-helper.service
#.    sudo systemctl restart adas-helper.service
#.    sudo systemctl status adas-helper.service
#.    journalctl -u adas-helper.service -f

# 24 July Add a Gstreamer instead standart Open CV librarries
#.    sudo systemctl restart adas-helper.service
#.    sudo systemctl status adas-helper.service
#.    journalctl -u adas-helper.service -f

# 24 July Add a Gstreamer instead standard OpenCV libraries
from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import os
import time
import cv2
import threading
import queue
import re
import numpy as np

import requests  # Import for shouting down Jetson Orin Nano from the web interface

app = Flask(__name__)
CORS(app)  # Pełna obsługa Cross-Origin z kontenera Dockera

# Globalne zmienne dla potoku wideo na hoście
recording_active = False
video_queue = queue.Queue(maxsize=300)
video_writer = None
recorder_thread = None

RECORDING_DIR = "/home/maro/yolo_app/road_recordings"
IMAGES_DIR = "/home/maro/yolo_app/images_from_camera_stream_jetson_nano"

os.makedirs(RECORDING_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


# def nvenc_write_worker(output_path, width, height, fps):
#     """Wątek procesowy obsługujący sprzętowe kodowanie NVENC na Jetsonie."""
#     global recording_active, video_writer, video_queue
    
#     # Optymalny potok GStreamer dla Jetson Orin Nano / Orin Nano Super
#     gst_pipeline = (
#         f"appsrc ! videoconvert ! video/x-raw, format=BGRx ! "
#         f"nvvidconv ! video/x-raw(memory:NVMM), format=NV12 ! "
#         f"nvv4l2h264enc bitrate=8000000 control-rate=1 preset-level=1 ! h264parse ! "
#         f"qtmux ! filesink location={output_path}"
#     )
    
#     video_writer = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, float(fps), (width, height))
    
#     if not video_writer.isOpened():
#         print("[Helper] Błąd: Nie udało się otworzyć potoku GStreamer NVENC. Sprawdź wsparcie GStreamer w OpenCV.")
#         recording_active = False
#         return

#     print(f"[Helper] Sprzętowe nagrywanie NVENC uruchomione: {output_path} ({width}x{height} @ {fps}fps)")
    
#     while recording_active or not video_queue.empty():
#         try:
#             # Timeout pozwala wątkowi cyklicznie sprawdzać stan flagi recording_active
#             frame = video_queue.get(timeout=0.5)
#             video_writer.write(frame)
#             video_queue.task_done()
#         except queue.Empty:
#             continue
            
#     video_writer.release()
#     video_writer = None
#     print("[Helper] Potok NVENC bezpiecznie zamknięty, plik MP4 zapisany.")
# def nvenc_write_worker(output_path, width, height, fps):
#     """Wątek procesowy obsługujący sprzętowe kodowanie NVENC na Jetsonie z kontrolą PTS."""
#     global recording_active, video_writer, video_queue
    
#     # Optymalny potok GStreamer
#     gst_pipeline = (
#         f"appsrc ! videoconvert ! video/x-raw, format=BGRx ! "
#         f"nvvidconv ! video/x-raw(memory:NVMM), format=NV12 ! "
#         f"nvv4l2h264enc bitrate=6000000 control-rate=1 preset-level=1 ! h264parse ! "
#         f"qtmux ! filesink location={output_path}"
#     )
    
#     video_writer = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, float(fps), (width, height))
    
#     if not video_writer.isOpened():
#         print("[Helper] Blad: Nie udalo sie otworzyc potoku GStreamer NVENC.")
#         recording_active = False
#         return

#     print(f"[Helper] Sprzetowe nagrywanie uruchomione: {output_path} ({fps} FPS)")
    
#     # Wylliczenie optymalnego czasu trwania jednej klatki (w sekundach)
#     frame_delay = 1.0 / float(fps)
#     last_write_time = time.time()

#     while recording_active or not video_queue.empty():
#         try:
#             frame = video_queue.get(timeout=0.5)
            
#             # KONTROLA PTS: Upewniamy się, że nie zapisujemy klatek zbyt szybko po sobie
#             now = time.time()
#             elapsed = now - last_write_time
#             if elapsed < frame_delay:
#                 time.sleep(frame_delay - elapsed)
            
#             video_writer.write(frame)
#             last_write_time = time.time()
#             video_queue.task_done()
            
#         except queue.Empty:
#             continue
            
#     video_writer.release()
#     video_writer = None
#     print("[Helper] Potok NVENC bezpiecznie zamkniety, plik MP4 zapisany.")


import time

def nvenc_write_worker(filename, width=1920, height=1080, target_fps=30):
    """Zapisuje klatki z dynamicznym dopasowaniem FPS do realnego tempa napływu."""
    global recording_active, video_queue
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, target_fps, (width, height))
    
    if not writer.isOpened():
        print(f"[Helper ERROR] Nie udało się otworzyć pliku do zapisu: {filename}")
        recording_active = False
        return

    print(f"[Helper] Start zapisu Full HD -> {filename}")
    
    frames_written = 0
    start_time = time.time()
    
    while recording_active or not video_queue.empty():
        try:
            frame = video_queue.get(timeout=0.2)
            
            # Dopasowanie rozmiaru do Full HD jeśli trzeba
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
                
            writer.write(frame)
            frames_written += 1
            video_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Helper ERROR] Błąd zapisu klatki: {e}")

    writer.release()
    
    elapsed_time = time.time() - start_time
    if elapsed_time > 0 and frames_written > 0:
        real_fps = frames_written / elapsed_time
        print(f"[Helper] Zakończono! Zapisano {frames_written} klatek w {elapsed_time:.1f}s (Średni realny FPS: {real_fps:.1f})")


# --- ENDPOINTY DLA ZAPISU WIDEO I ZDJĘĆ ---

@app.route("/video_save", methods=["POST"])
def video_save():
    """Zapisuje pojedynczą klatkę wysłaną z aplikacji ADAS jako plik JPG w pełnej jakości."""
    if "frame" not in request.files:
        return jsonify({"ok": False, "message": "Brak pliku frame w żądaniu"}), 400
        
    file = request.files["frame"]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(IMAGES_DIR, f"frame_{timestamp}.jpg")
    file.save(filename)
    
    return jsonify({"ok": True, "message": "Klatka zapisana pomyślnie na hoście", "file": filename})


# @app.route("/recording/start", methods=["POST"])
# def start_recording():
#     """Inicjalizuje proces nagrywania sprzętowego na hoście."""
#     global recording_active, recorder_thread, video_queue
    
#     if recording_active:
#         return jsonify({"ok": False, "message": "Nagrywanie już trwa"}), 400
        
#     data = request.get_json() or {}
#     width = int(data.get("width", 1280))
#     height = int(data.get("height", 720))
#     fps = int(data.get("fps", 30))
    
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
#     filename = os.path.join(RECORDING_DIR, f"road_recording_{timestamp}.mp4")
    
#     # Czyszczenie starej kolejki przed nowym nagrywaniem
#     with video_queue.mutex:
#         video_queue.queue.clear()
        
#     recording_active = True
#     recorder_thread = threading.Thread(
#         target=nvenc_write_worker, 
#         args=(filename, width, height, fps),
#         daemon=True
#     )
#     recorder_thread.start()
    
#     return jsonify({"ok": True, "message": "Nagrywanie NVENC wystartowało", "file": filename})



# @app.route("/recording/start", methods=["POST"])
# def start_recording():
#     """Inicjalizuje proces nagrywania sprzętowego na hoście."""
#     global recording_active, recorder_thread, video_queue
    
#     if recording_active:
#         return jsonify({"ok": False, "message": "Nagrywanie już trwa"}), 400
        
#     data = request.get_json() or {}
#     width = int(data.get("width", 1280))
#     height = int(data.get("height", 720))
#     fps = int(data.get("fps", 30))
    
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
#     filename = os.path.join(RECORDING_DIR, f"road_recording_{timestamp}.mp4")
    
#     # Czyszczenie starej kolejki przed nowym nagrywaniem
#     with video_queue.mutex:
#         video_queue.queue.clear()
        
#     recording_active = True
#     recorder_thread = threading.Thread(
#         target=nvenc_write_worker, 
#         args=(filename, width, height, fps),
#         daemon=True
#     )
#     recorder_thread.start()
    
#     # Powiadomienie adas.py w Dockerze o starcie wysyłania klatek
#     try:
#         requests.post(
#             'http://127.0.0.1:5010/set_recording_status', 
#             json={'enabled': True, 'width': width, 'height': height, 'fps': fps}, 
#             timeout=0.5
#         )
#         print(f"[Helper] Start nagrywania: {width}x{height} @ {fps} FPS")
#     except Exception as e:
#         print(f"[Helper] Ostrzeżenie: Nie udało się powiadomić adas.py: {e}")

#     return jsonify({"ok": True, "message": "Nagrywanie NVENC wystartowało", "file": filename})

@app.route("/recording/start", methods=["POST"])
def start_recording():
    global recording_active, recorder_thread, video_queue
    
    if recording_active:
        return jsonify({"ok": False, "message": "Nagrywanie już trwa"}), 400
        
    data = request.get_json() or {}
    # Ustawiamy domyślnie Full HD (1920x1080)
    width = int(data.get("width", 1920))
    height = int(data.get("height", 1080))
    fps = int(data.get("fps", 30))
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RECORDING_DIR, f"road_recording_{timestamp}.mp4")
    
    with video_queue.mutex:
        video_queue.queue.clear()
        
    recording_active = True
    recorder_thread = threading.Thread(
        target=nvenc_write_worker, 
        args=(filename, width, height, fps),
        daemon=True
    )
    recorder_thread.start()
    
    # Informujemy adas.py w Dockerze o transmisji Full HD
    try:
        requests.post(
            'http://127.0.0.1:5010/set_recording_status', 
            json={'enabled': True, 'width': width, 'height': height, 'fps': fps}, 
            timeout=0.5
        )
    except Exception as e:
        print(f"[Helper WARN] Nie udało się powiadomić adas.py: {e}")

    return jsonify({"ok": True, "message": "Nagrywanie Full HD wystartowało", "file": filename})


@app.route('/recording/stop', methods=['POST'])
def stop_recording():
    global recording_active, video_writer
    print("[Helper] Otrzymano zadanie zatrzymania nagrywania...")
    
    recording_active = False
    
    if video_writer is not None:
        try:
            video_writer.release()
            video_writer = None
            print("[Helper] Plik wideo zostal bezpiecznie zamkniety.")
        except Exception as e:
            print(f"[Helper] Blad podczas zamykania wideo: {e}")

    # Wewnątrz stop_recording():
    recording_active = False
    
    try:
        requests.post(
            'http://127.0.0.1:5010/set_recording_status', 
            json={'enabled': False}, 
            timeout=0.5
        )
        print("[Helper] Wyslano sygnal STOP do adas.py")
    except Exception as e:
        print(f"[Helper] Ostrzezenie: Nie udalo sie powiadomic adas.py: {e}")

    return jsonify({"ok": True, "message": "Nagrywanie zostalo zatrzymane"})


@app.route("/recording/status", methods=["GET"])
def recording_status():
    """Zwraca obecny stan rejestratora."""
    global recording_active, video_queue
    return jsonify({
        "recording": recording_active,
        "queue_size": video_queue.qsize()
    })


# @app.route("/recording/push_frame", methods=["POST"])
# def push_frame():
#     """Odbiera skompresowaną klatkę z Dockera i wrzuca do kolejki zapisu NVENC."""
#     global recording_active, video_queue
    
#     if not recording_active:
#         return jsonify({"ok": False, "message": "Nagrywanie wyłączone"}), 200
        
#     if "frame" not in request.files:
#         return jsonify({"ok": False, "message": "Brak klatki"}), 400
        
#     file = request.files["frame"]
#     filestr = file.read()
#     np_img = np.frombuffer(filestr, np.uint8)
#     frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
#     try:
#         # block=False zapobiega blokowaniu kontenera ADAS
#         video_queue.put(frame, block=False)
#         return jsonify({"ok": True})
#     except queue.Full:
#         return jsonify({"ok": False, "message": "Kolejka wideo helpera przepełniona"}), 503
@app.route("/recording/push_frame", methods=["POST"])
def push_frame():
    global recording_active, video_queue

    if not recording_active:
        print("[DEBUG] Odrzucono: recording_active jest FALSE!")
        return jsonify({"ok": False, "reason": "Not active"}), 400

    file_bytes = np.frombuffer(request.data, np.uint8)
    if len(file_bytes) == 0:
        print("[DEBUG] Odrzucono: Odebrano 0 bajtów data!")
        return jsonify({"ok": False, "reason": "Empty data"}), 400

    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        print("[DEBUG] Odrzucono: cv2.imdecode zwrocil None!")
        return jsonify({"ok": False, "reason": "Decode failed"}), 400

    # Jeśli doszło tutaj - klatka trafia do kolejki
    if video_queue.full():
        try:
            video_queue.get_nowait()
        except queue.Empty:
            pass
    video_queue.put(frame)

    return jsonify({"ok": True}), 200

# --- ENDPOINTY SYSTEMOWE I DIAGNOSTYCZNE ---

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
    # Odczyt temperatury CPU
    try:
        with open("/sys/devices/virtual/thermal/thermal_zone0/temp") as f:
            cpu_temp = round(int(f.read()) / 1000.0, 1)
    except Exception:
        cpu_temp = 0.0

    # Odczyt pamięci RAM
    try:
        mem = subprocess.check_output(["free", "-m"]).decode().splitlines()
        mem_values = re.split(r'\s+', mem[1])
        mem_total = int(mem_values[1])
        mem_used = int(mem_values[2])
    except Exception:
        mem_total, mem_used = 0, 0

    # Odczyt Dyskowe
    try:
        disk_cmd = subprocess.check_output(["df", "-h", "/"]).decode().splitlines()
        disk_values = re.split(r'\s+', disk_cmd[1])
        disk_total = disk_values[1]
        disk_used = disk_values[2]
        disk_percent = disk_values[4]
    except Exception:
        disk_total, disk_used, disk_percent = "0G", "0G", "0%"

    load = os.getloadavg()
    
    # Status Usług systemd
    try:
        subprocess.check_output(["systemctl", "is-active", "adas.service"])
        adas_running = True
    except subprocess.CalledProcessError:
        adas_running = False

    try:
        subprocess.check_output(["systemctl", "is-active", "adas-helper.service"])
        helper_running = True
    except subprocess.CalledProcessError:
        helper_running = False

    return jsonify({
        "current_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_temperature": cpu_temp,
        "memory_used_mb": mem_used,
        "memory_total_mb": mem_total,
        "disk_used": disk_used,
        "disk_total": disk_total,
        "disk_percent": disk_percent,
        "uptime": subprocess.check_output(["uptime", "-p"]).decode().strip(),
        "cpu_load_1min": round(load[0], 2),
        "cpu_load_5min": round(load[1], 2),
        "cpu_load_15min": round(load[2], 2),
        "adas_running": adas_running,
        "helper_running": helper_running,
        "recording_active": recording_active
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055, threaded=True)