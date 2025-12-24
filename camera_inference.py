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
cap = cv2.VideoCapture(cam_source)

if not cap.isOpened():
    print(f"Error: cannot open camera '{CAMERA_SOURCE}'.")
    sys.exit(1)

# --- Inference ---
model = YOLO(MODEL_PATH)

# Start capturing from the camera
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to a lower resolution
    frame_resized = cv2.resize(frame, (320, 240))

    # Pass the resized frame to the model and get results
    results_generator = model.predict(
        source=frame_resized,
        save=False,  # Set to False to avoid saving each frame
        project='/results',
        name='video_output',
        exist_ok=True,
        verbose=False,
        stream=True  # Enable streaming mode
    )

    # Process results
    for result in results_generator:
        # Here you can access result.boxes, result.masks, etc.
        print(result)  # Print or process the result as needed

cap.release()
print(f"Video inference completed. Output saved to /results/")