# /app/test_display.py
import cv2
import numpy as np
import os

# Force software rendering for robustness (already tried, but good to keep)
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

print("OpenCV version:", cv2.__version__)

# Create a simple red square image
img = np.zeros((400, 400, 3), dtype=np.uint8)
img[:, :] = (0, 0, 255) # Red color (BGR format)

cv2.putText(img, "Success!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

# Display the image
cv2.imshow("X11 Test Window", img)

# Wait indefinitely for a key press (required for imshow to work)
cv2.waitKey(0) 

cv2.destroyAllWindows()