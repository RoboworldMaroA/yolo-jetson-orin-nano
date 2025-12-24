import os
import re
import time
import subprocess
import threading
from flask import Flask, Response, render_template, jsonify, request
import cv2

CAM_DEVICE = os.environ.get("CAM_DEVICE", "/dev/video0")
PORT = int(os.environ.get("STREAM_PORT", "5002"))  # choose different port if needed

app = Flask(__name__, template_folder="templates", static_folder="static")

# Shared frame storage
latest_frame = None
frame_lock = threading.Lock()
stop_event = threading.Event()

# Capture object created in producer thread
cap = None

def run_cmd(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return p.stdout.strip()
    except Exception as e:
        return None

def parse_list_ctrls(output):
    """Parse `v4l2-ctl --list-ctrls` output into dict."""
    controls = {}
    if not output:
        return controls
    for line in output.splitlines():
        # Example lines:
        # brightness (int)    : min=0 max=255 step=1 default=128 value=1
        # white_balance_automatic (bool)   : default=1 value=1
        # power_line_frequency (menu)   : min=0 max=2 default=2 value=2 (60 Hz)
        # m = re.match(r'^\s*([a-zA-Z0-9_]+)\s+\([a-zA-Z]+\)\s*:\s*(.*)$', line)
        # *** CHANGE THIS LINE ***
        # Original: m = re.match(r'^\s*([a-zA-Z0-9_]+)\s+\([a-zA-Z]+\)\s*:\s*(.*)$', line)
        # Fix for output with hex IDs:
        m = re.match(r'^\s*([a-zA-Z0-9_]+)\s+0x[0-9a-fA-F]+\s+\(([a-zA-Z]+)\)\s*:\s*(.*)$', line)

        if not m:
            continue

        # After this change, m.group(1) is the control name, m.group(4) is the rest of the data
        # Your original code needs slight adjustment to group indexing for the rest of the parsing to work if the old groups were (1, 2)
        
        # Based on your original parsing logic structure, the simplest fix is to ensure the control name is group 1 and the data is group 2.
        # Let's adjust the regex to only capture the needed parts: name and data.
        
        # Simplified FIX regex for your output:
        m = re.match(r'^\s*([a-zA-Z0-9_]+)\s+0x[0-9a-fA-F]+\s+\([a-zA-Z]+\)\s*:\s*(.*)$', line)
        name = m.group(1)
        rest = m.group(2)
        ctrl = {"name": name, "raw": rest}
        # try parse ints
        num_match = re.search(r'min=(\S+)\s+max=(\S+)\s+step=(\S+)\s+default=(\S+)\s+value=(\S+)', rest)
        if num_match:
            ctrl.update({
                "type": "int",
                "min": int(num_match.group(1)),
                "max": int(num_match.group(2)),
                "step": int(num_match.group(3)),
                "default": int(num_match.group(4)),
                "value": int(num_match.group(5))
            })
        else:
            # bool or menu or simple default/value
            dv_match = re.search(r'default=(\S+)\s+value=(\S+)', rest)
            if dv_match:
                default = dv_match.group(1)
                value = dv_match.group(2)
                if default in ("0","1") and value in ("0","1"):
                    ctrl.update({"type":"bool", "default": int(default), "value": int(value)})
                else:
                    ctrl.update({"type":"other", "default": default, "value": value})
            else:
                ctrl.update({"type":"other", "value": rest})
        controls[name] = ctrl
    return controls

def list_controls():
    """Return parsed available controls for CAM_DEVICE using v4l2-ctl (if present)."""
    out = run_cmd(["v4l2-ctl", "-d", CAM_DEVICE, "--list-ctrls"])
    if out is None:
        return {}
    return parse_list_ctrls(out)

def get_control(name):
    """Get single control value via v4l2-ctl (--get-ctrl)."""
    out = run_cmd(["v4l2-ctl", "-d", CAM_DEVICE, "--get-ctrl", name])
    if not out:
        return None
    # out like: "brightness: 128"
    parts = out.split(":")
    if len(parts) >= 2:
        return parts[1].strip()
    return out.strip()

def set_control(name, value):
    """Set control via v4l2-ctl; return True on success."""
    # try v4l2-ctl first
    res = run_cmd(["v4l2-ctl", "-d", CAM_DEVICE, "--set-ctrl", f"{name}={value}"])
    # run_cmd returns stdout or None on error; v4l2-ctl usually prints nothing on success
    if res is None:
        # fallback to OpenCV if cap exists and property mapping known
        prop_map = {
            "brightness": cv2.CAP_PROP_BRIGHTNESS,
            "contrast": cv2.CAP_PROP_CONTRAST,
            "saturation": cv2.CAP_PROP_SATURATION,
            "gain": cv2.CAP_PROP_GAIN,
            "focus": cv2.CAP_PROP_FOCUS,
            "exposure": cv2.CAP_PROP_EXPOSURE
        }
        try:
            if cap is not None and name in prop_map:
                cap.set(prop_map[name], float(value))
                return True
        except Exception:
            return False
        return False
    return True

def producer():
    global latest_frame, cap
    cap = cv2.VideoCapture(0)
    # set a reasonable resolution (can be adjusted by client)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    last_time = time.time()
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        # overlay current control values for user feedback
        ctrls = list_controls()
        y = 20
        for k, v in ctrls.items():
            txt = f"{k}: {v.get('value') if isinstance(v, dict) else v}"
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            y += 22
            if y > frame.shape[0] - 30:
                break
        # encode jpeg
        ret2, buf = cv2.imencode('.jpg', frame)
        if not ret2:
            continue
        with frame_lock:
            latest_frame = buf.tobytes()
        # limit CPU use
        time.sleep(0.01)
    if cap:
        cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/controls')
def controls():
    ctrls = list_controls()
    # convert to simple JSON
    return jsonify(ctrls)

@app.route('/get_control')
def http_get_control():
    name = request.args.get('name')
    if not name:
        return jsonify({"error":"name required"}), 400
    val = get_control(name)
    if val is None:
        return jsonify({"error":"not supported or v4l2-ctl missing"}), 404
    return jsonify({"name": name, "value": val})

@app.route('/set_control', methods=['POST'])
def http_set_control():
    data = request.json or request.form
    name = data.get('name')
    value = data.get('value')
    if name is None or value is None:
        return jsonify({"error":"name and value required"}), 400
    ok = set_control(name, value)
    if not ok:
        return jsonify({"error":"failed to set control"}), 500
    return jsonify({"name":name, "value": value})

def mjpeg_generator():
    boundary = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
    while not stop_event.is_set():
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield boundary + frame + b'\r\n'
        time.sleep(0.01)

@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # start producer thread
    t = threading.Thread(target=producer, daemon=True)
    t.start()
    try:
        app.run(host='0.0.0.0', port=PORT, threaded=True)
    finally:
        stop_event.set()
        t.join(timeout=2.0)