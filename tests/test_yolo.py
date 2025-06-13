import os
import cv2
import math
import matplotlib.pyplot as plt
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = "best.pt"
IMAGE_DIR = r"F:\License-Plate-Detection-and-Recognition\DATA SCIENTIST_ASSIGNMENT\test"
GRID_ROWS = 5
GRID_COLS = 5

# === Load Model ===
model = YOLO(MODEL_PATH)

# === Get all image paths ===
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
image_paths = [os.path.join(IMAGE_DIR, f) for f in image_files]

# === Prepare results ===
processed_images = []

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_path)[0]

    # Draw boxes if any
    if results.boxes:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

    processed_images.append((img_rgb, os.path.basename(img_path)))

# === Plot in grid ===
def plot_images(images, rows, cols):
    total = len(images)
    pages = math.ceil(total / (rows * cols))

    for p in range(pages):
        start = p * rows * cols
        end = min(start + rows * cols, total)
        subset = images[start:end]

        fig, axs = plt.subplots(rows, cols, figsize=(20, 20))
        axs = axs.flatten()

        for i in range(rows * cols):
            if i < len(subset):
                axs[i].imshow(subset[i][0])
                axs[i].set_title(subset[i][1], fontsize=10)
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()

plot_images(processed_images, GRID_ROWS, GRID_COLS)
