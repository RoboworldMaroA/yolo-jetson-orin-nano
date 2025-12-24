
No good reason to not use this code
from ultralytics import YOLO 

from   jetson_utils import videoSource, videoOutput, Log
from   jetson_utils import cudaToNumpy
from   jetson_utils import cudaFromNumpy

# Load YOLO TRT model
model = YOLO("/home/jet/robotics/yolo/networks/yolo11n.engine")

# Jetson_Utils initialize
input  = videoSource()
output = videoOutput()
 
# Run Inference on Video Frames
while True:

    # capture the next image
    frame = input.Capture()

    if frame is None: # timeout
        continue  
        
    # Exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break

    # Convert Jetson Cuda image to Numpy array  
    frame_numpy = cudaToNumpy(frame)

    # Run Yolo Inference
    results = model(frame_numpy)  #, show=True)
 
    for resx in results:
        boxes     = resx.boxes      # Boxes object for bounding box outputs
        masks     = resx.masks      # Masks object for segmentation masks outputs
        keypoints = resx.keypoints  # Keypoints object for pose outputs
        probs     = resx.probs      # Probs object for classification outputs
        obb       = resx.obb        # Oriented boxes object for OBB outputs

        # Display image and bounding box in Jetson_Utils output window
        output.Render(cudaFromNumpy(resx.plot()))

        print(boxes)   # Stream object detection results