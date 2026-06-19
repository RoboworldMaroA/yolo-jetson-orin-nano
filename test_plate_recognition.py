import cv2
from web_lazy_v6_new_model import LPRNetRecognizer

MODEL_PATH_LPRNET_ONNX = "/app/lprnet.onnx"

lprnet_recognizer = LPRNetRecognizer(
    MODEL_PATH_LPRNET_ONNX
)

plate = cv2.imread("/app/debug_plate_973.jpg")

result = lprnet_recognizer.predict(plate)

print("RESULT:", result)