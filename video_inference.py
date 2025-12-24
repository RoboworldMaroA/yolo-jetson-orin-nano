
from ultralytics import YOLO
import cv2

# --- Configuration ---
# NOTE: This path is *inside* the Docker container's file system
MODEL_PATH = '/app/yolov8n.engine' 

# --- Inference ---
# 1. Load the exported TensorRT engine model
model = YOLO(MODEL_PATH)

# ... (Imports and model loading are the same)
INPUT_PATH = '/data/dogs.mp4'  # <<< Change this to your video file path
OUTPUT_PATH = '/results/output_video.mp4' # <<< Change the output to a video file

# Run prediction with the 'save=True' flag to save the output video
# The prediction returns a generator stream for videos/webcams
results_generator = model.predict(
    source=INPUT_PATH, 
    save=True,
    # CRITICAL CHANGE: Set the 'project' argument to the mounted folder
    project='/results',
    # Optional: Set a specific sub-folder name (e.g., 'video_run_1')
    name='video_output', 
    exist_ok=True,
    verbose=False
)
# Note: You don't need a loop here; 'save=True' handles the file writing automatically.

print(f"Video inference completed. Output saved to /results/")