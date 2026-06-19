# Proggrm take the images and generate a draft training file for easyocr. 
#The output file will contain the relative path to the image and the corresponding plate text, separated by a tab.
#Runnig process:
#1. Make sure you have the images in the "dataset_renamed/images" directory
#2. Activate conatiner with easyOCR:
#  sudo docker exec -it 9e80ed866c88 /bin/bash
#2. Run the script to generate the draft training file: 
#  python /app/make_train_draft_easyocr.py
import os
import re
from pathlib import Path

import cv2
import easyocr

IMAGE_DIR = Path("/app/dataset_renamed/images")
OUTPUT_FILE = Path("/app/dataset_renamed/train_draft.txt")

reader = easyocr.Reader(["en"], gpu=True)

def clean_plate(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text

images = sorted(
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.png"))
)

with open(OUTPUT_FILE, "w") as f:
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = reader.readtext(img, detail=0)
        raw = "".join(results)
        plate = clean_plate(raw)

        relative_path = f"images/{img_path.name}"
        f.write(f"{relative_path}\t{plate}\n")

        print(relative_path, "->", plate)

print(f"Saved: {OUTPUT_FILE}")
