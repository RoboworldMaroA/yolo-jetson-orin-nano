#Author: Marek Augustyn
#Date: 05 April 2026 Updated for video, unified Irish plate cleaning/validation
# It is recognize Irish licence from the video and save to .csv file
#
from ultralytics import YOLO
import torch
import cv2
import easyocr
import re
import csv
import os
import time

from flask import Flask, Response, render_template_string
import threading

recognized_plates_log = []  # Each entry: (frame_number, car_idx, plate_idx, plate_text)

# Check for MPS (Apple Metal Performance Shaders) device availability
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Has MPS: {torch.backends.mps.is_built()}")
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load the YOLO models
model_path = "Yolo_Models/yolov8m.pt"
# model_path = "Yolo_Models/yolo11n.pt"
# model_registation_plate_path = "Model_recognize_licence_plates/runs/detect/train14/weights/best.pt"
model_registation_plate_path = "/app/licence_plate.pt"

model = YOLO(model_path)
model_registration_plate = YOLO(model_registation_plate_path)
print("Model loaded successfully.")
print("Model class names:", model.names)
print("Model for registration plates loaded successfully.")
print("Model class names for registration plates:", model_registration_plate.names)


app = Flask(__name__)


def clean_plate_text(text):
    """Clean OCR text for common misrecognitions in Irish plates."""
    text = text.upper().strip()
    text = text.replace(" ", "-")
    text = text.replace("LT", "WW")

    no_hyphens = text.replace("-", "")
    no_hyphens = (
        no_hyphens.replace("O", "0")
        .replace("I", "1")
        .replace("S", "5")
        .replace("L", "1")
        .replace("T", "7")
    )

    for first_len in range(3, 0, -1):
        for mid_len in range(1, 3):
            if first_len + mid_len <= len(no_hyphens):
                first = no_hyphens[:first_len]
                middle = no_hyphens[first_len:first_len + mid_len]
                last = no_hyphens[first_len + mid_len :]

                if first.isdigit() and 1 <= len(last) <= 5 and last.isdigit():
                    middle_fixed = (
                        middle.replace("0", "D")
                        .replace("1", "I")
                        .replace("5", "S")
                        .replace("7", "T")
                        .replace("O", "D")
                    )
                    if middle_fixed.isalpha():
                        if len(middle_fixed) == 1 and middle_fixed == "P":
                            middle_fixed = "D"
                        return f"{first}-{middle_fixed}-{last}"

    match = re.match(r"^(\d{1,3})([A-Z]{1,2})(\d{1,5})$", no_hyphens)
    if match:
        middle_fixed = match.group(2)
        if len(middle_fixed) == 1 and middle_fixed == "P":
            middle_fixed = "D"
        return f"{match.group(1)}-{middle_fixed}-{match.group(3)}"

    return no_hyphens


def debug_plate_recognition(plate_text):
    cleaned = clean_plate_text(plate_text)
    is_valid = is_irish_plate(plate_text)
    print(f"  Raw: {plate_text}")
    print(f"  Cleaned: {cleaned}")
    print(f"  Valid: {is_valid}")


def is_irish_plate(text):
    cleaned = clean_plate_text(text)
    pattern = r"^\d{1,3}-[A-Z]{1,2}-\d{1,5}$"
    return bool(re.match(pattern, cleaned))


def preprocess_image(image, frame_number=None, car_idx=None, plate_idx=None, plate_text=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    SAVED_ONLY_RECOGNIZED_PLATES = os.path.join(".", "images_only_reconized_plates")
    os.makedirs(SAVED_ONLY_RECOGNIZED_PLATES, exist_ok=True)

    if plate_text:
        plate_text_clean = re.sub(r"\W+", "", plate_text)
    else:
        plate_text_clean = "unknown"

    filename = f"license_only_f{frame_number}_c{car_idx}_p{plate_idx}_{plate_text_clean}_{int(time.time()*1000)}.jpg"
    image_path = os.path.join(SAVED_ONLY_RECOGNIZED_PLATES, filename)
    cv2.imwrite(image_path, image)
    print(f"Saved cropped plate image: {image_path}")

    return thresh


def recognize_plate(cropped_img, frame_number=None, car_idx=None, plate_idx=None, plate_text=None):
    reader = easyocr.Reader(["en"], gpu=True)
    processed_image = preprocess_image(
        cropped_img,
        frame_number=frame_number,
        car_idx=car_idx,
        plate_idx=plate_idx,
        plate_text=plate_text,
    )
    results = reader.readtext(processed_image)
    plates = []
    for (bbox, text, prob) in results:
        if prob > 0.2:
            plates.append((bbox, text))
    return plates


def process_cropped_licence_plate(cropped_img, index):
    if not cropped_img.size:
        print(f"Cropped image {index} is empty, skipping save.")
        return

    os.makedirs("Croped_licence_plates", exist_ok=True)
    cropped_image_path = f"Croped_licence_plates/cropped_licence_plate_{index}.png"
    cv2.imwrite(cropped_image_path, cropped_img)
    print(f"Cropped licence plate image saved to {cropped_image_path}")

    # cv2.imshow(f"Cropped Licence Plate {index}", cropped_img)
    # cv2.waitKey(100)
    cv2.destroyAllWindows()


def process_frame(frame, frame_number):
    results = model(frame)
    if not results:
        return frame

    result = results[0]
    car_boxes = []
    allowed_car_classes = [2, 5, 7]

    for car_idx, box in enumerate(result.boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        class_id = int(box.cls[0].item())
        prob = round(box.conf[0].item(), 2)
        if prob < 0.85 or class_id not in allowed_car_classes:
            continue

        car_boxes.append((car_idx, x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 118, 232), 3)
        cv2.putText(
            frame,
            f"{result.names[class_id]} {prob}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.9,
            (26, 9, 156),
            4,
        )

    unique_plates = set()
    for car_idx, x1, y1, x2, y2 in car_boxes:
        car_crop = frame[y1:y2, x1:x2]
        licence_results = model_registration_plate(car_crop)

        if not licence_results:
            continue

        licence_result = licence_results[0]
        for plate_idx, plate_box in enumerate(licence_result.boxes):
            px1, py1, px2, py2 = [int(coord) for coord in plate_box.xyxy[0]]
            plate_crop = car_crop[py1:py2, px1:px2]

            plates = recognize_plate(
                plate_crop,
                frame_number=frame_number,
                car_idx=car_idx,
                plate_idx=plate_idx,
                plate_text=None,
            )

            for (bbox, plate_text) in plates:
                cleaned_plate = clean_plate_text(plate_text)
                if not is_irish_plate(plate_text):
                    continue

                if cleaned_plate in unique_plates:
                    continue

                unique_plates.add(cleaned_plate)
                process_cropped_licence_plate(
                    plate_crop, f"car{car_idx}_plate{plate_idx}_{cleaned_plate}"
                )
                recognized_plates_log.append(
                    (frame_number, car_idx, plate_idx, cleaned_plate)
                )

                abs_px1, abs_py1 = x1 + px1, y1 + py1
                (w, h), _ = cv2.getTextSize(
                    cleaned_plate, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 8
                )
                cv2.rectangle(
                    frame,
                    (abs_px1 + 5, abs_py1 - h - 60),
                    (abs_px1 + w + 270, abs_py1),
                    (255, 255, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    cleaned_plate,
                    (abs_px1, abs_py1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.5,
                    (26, 9, 156),
                    10,
                )

    print(f"All unique plates found in frame: {unique_plates}")
    with open("recognized_plates_video.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame_number", "car_idx", "plate_idx", "plate_text"])
        writer.writerows(recognized_plates_log)

    print("Recognized plates saved to recognized_plates_video.csv")

    return frame


def generate_frames():
    cap = cv2.VideoCapture(0)  # 0 dla domyślnej kamery; zmień na ścieżkę do kamery jeśli potrzeba
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame, frame_number)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        frame_number += 1

@app.route('/')
def index():
    return render_template_string('''
    <html>
    <head>
        <title>Camera Stream</title>
    </head>
    <body>
        <h1>Live Camera Stream with Plate Recognition</h1>
        <img src="/video_feed" style="width:100%; height:auto;">
    </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def main():
    app.run(host='0.0.0.0', port=5010, debug=True)

if __name__ == "__main__":
    main()