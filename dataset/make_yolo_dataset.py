import pandas as pd
import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split

# --- Paths ---
csv_path = r'F:\License-Plate-Detection-and-Recognition\DATA SCIENTIST_ASSIGNMENT\Licplatesdetection_train.csv'
img_dir = r'F:\License-Plate-Detection-and-Recognition\DATA SCIENTIST_ASSIGNMENT\license_plates_detection_train'
base_dir = r'F:\YOLO_LicensePlate_Dataset'  # output dir

# --- YOLO folder structure ---
for split in ['train', 'val']:
    os.makedirs(os.path.join(base_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'labels', split), exist_ok=True)

# --- Load CSV ---
df = pd.read_csv(csv_path)

# --- Split data ---
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# --- Function to process a split ---
def process_split(split_df, split_name):
    for _, row in split_df.iterrows():
        img_name = row['img_id']
        img_path = os.path.join(img_dir, img_name)

        # Open image and get dimensions
        try:
            img = Image.open(img_path)
            w, h = img.size
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            continue

        # YOLO format: class x_center y_center width height (all normalized)
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        x_center = (xmin + xmax) / 2 / w
        y_center = (ymin + ymax) / 2 / h
        width = (xmax - xmin) / w
        height = (ymax - ymin) / h

        # Save image
        dest_img_path = os.path.join(base_dir, 'images', split_name, img_name)
        shutil.copy(img_path, dest_img_path)

        # Save label
        label_name = img_name.replace('.jpg', '.txt')
        label_path = os.path.join(base_dir, 'labels', split_name, label_name)

        with open(label_path, 'w') as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# --- Process both splits ---
process_split(train_df, 'train')
process_split(val_df, 'val')

# --- Create YAML config ---
yaml_path = os.path.join(base_dir, 'license_plate.yaml')
with open(yaml_path, 'w') as f:
    f.write(f"""path: {base_dir}
train: images/train
val: images/val
nc: 1
names: ['plate']
""")

print("✅ Done! You can now run YOLOv8 training:")
print(f"yolo detect train data={yaml_path} model=yolov8n.pt epochs=50 imgsz=640 batch=16")
