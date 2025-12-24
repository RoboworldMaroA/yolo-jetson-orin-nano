# ...existing code...
from ultralytics import YOLO
import cv2
import os
import sys

# --- Configuration (can be overridden by environment variables) ---
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/yolov8n.engine")
CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "0")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/results/output_video.mp4")

# Convert numeric index if possible
try:
    cam_source = int(CAMERA_SOURCE)
except Exception:
    cam_source = CAMERA_SOURCE  # keep as string (device path)

# Quick camera accessibility check before loading model
# ...existing code...
cap = cv2.VideoCapture(cam_source)

if not cap.isOpened():
    print(f"Error: cannot open camera '{CAMERA_SOURCE}'.")
    sys.exit(1)

cap.release()

# --- Inference ---
model = YOLO(MODEL_PATH)

# --- NEW: display annotated frames in a window ---
try:
    # Use the model's streaming API (yields Results)
    results_gen = model.predict(source=cam_source, stream=True, save=False, verbose=False)

    print("Press 'q' to quit.")
    for result in results_gen:
        # result.plot() returns an OpenCV BGR numpy array with boxes/labels drawn
        annotated = result.plot()

        # Show annotated frame
        cv2.imshow("YOLO Camera", annotated)

        # Optional: save frames to a video file (uncomment to enable)
        # You can initialize a VideoWriter on first frame if needed.

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    cv2.destroyAllWindows()

print("Done.")
# ...existing code...