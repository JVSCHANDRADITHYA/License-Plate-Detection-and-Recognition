import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import easyocr
import re

# --- Paths ---
model_path = "best.pt"
image_dir = r"F:\License-Plate-Detection-and-Recognition\DATA SCIENTIST_ASSIGNMENT\test"
results_dir = "annotated_results"
os.makedirs(results_dir, exist_ok=True)

# --- Load YOLOv8 model ---
model = YOLO(model_path)

# --- Initialize EasyOCR ---
reader = easyocr.Reader(['en'], gpu=True)

# --- Get all image paths ---
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
image_paths = [os.path.join(image_dir, f) for f in image_files]

# --- Process images ---
display_count = 0
max_display = 5

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        continue

    results = model(img_path)[0]

    if results.boxes:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = img[y1:y2, x1:x2]

            detections = reader.readtext(cropped, detail=0, min_size=6)
            raw_text = ''.join(detections)
            raw_text = re.sub(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', '', raw_text)
            digits_only = ''.join(filter(str.isdigit, raw_text))

            # Limit to 7 digits: max 3 from start and 4 from end
            if len(digits_only) > 7:
                digits_only = digits_only[:3] + digits_only[-4:]

            # Draw bounding box and text
            annotated_img = img.copy()
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_img, digits_only, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Save annotated image
            out_path = os.path.join(results_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, annotated_img)

            # Show 4-5 samples only
            if display_count < max_display:
                plt.figure(figsize=(8, 6))
                plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title(f"{os.path.basename(img_path)} | Plate: {digits_only}")
                plt.show()
                display_count += 1

            break  # Only the first box is processed per image

print(f"\n✅ All annotated results saved to: {results_dir}")

