from pathlib import Path
import shutil

src = Path("dataset_irish_plates/images")
dst = Path("dataset_renamed/images")
dst.mkdir(parents=True, exist_ok=True)

files = sorted(src.glob("*.jpg"))

for i, file in enumerate(files, start=1):
    new_name = f"plate_{i:07d}.jpg"
    shutil.copy2(file, dst / new_name)

print(f"Copied {len(files)} files to {dst}")
