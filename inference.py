# This program performs object detection inference using a YOLOv8 model
# It has been exported to TensorRT format and is run inside a Docker container.
# The input image is read from a mounted directory, and the output image
# with detected bounding boxes is saved to another mounted directory. 
# Ultalitics and OpenCV libraries are used for model loading, inference, and image processing.
from ultralytics import YOLO
import cv2

# --- Configuration ---
# NOTE: This path is *inside* the Docker container's file system
MODEL_PATH = '/app/yolov8n.engine' 
INPUT_PATH = '/data/test_image.jpg'
INPUT_PATH_2 = '/data/irish_licence_distance.png'
OUTPUT_PATH = '/results/output_image_3.jpg' 

# --- Inference ---
# 1. Load the exported TensorRT engine model
model = YOLO(MODEL_PATH)

# 2. Run prediction, stream=False returns a list of Results
# The model will internally load the image from the INPUT_PATH
results = model.predict(source=INPUT_PATH_2, save=False, verbose=False)

# 3. Process the results (drawing the bounding boxes)
for result in results:
    # Get the image with bounding boxes already drawn by Ultralytics
    # This uses OpenCV-compatible BGR format (NumPy array)
    annotated_img = result.plot() 
    
    # 4. Save the annotated image to the mounted output folder
    cv2.imwrite(OUTPUT_PATH, annotated_img)
    
    # Optional: Print detection details
    print(f"Detections found: {len(result.boxes)}")
    print(f"Saved annotated image to: {OUTPUT_PATH}")

print("Inference completed successfully.")