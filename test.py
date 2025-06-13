import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 model for license plate detection
yolo_model = YOLO("best.pt")

# Load CRNN model (from earlier)
class CRNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, 1, 1), torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(64, 128, 3, 1, 1), torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(128, 256, 3, 1, 1), torch.nn.ReLU(),
        )
        self.rnn = torch.nn.LSTM(256 * 8, 256, num_layers=2, bidirectional=True)
        self.fc = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(3, 0, 2, 1)  # (W, B, H, C)
        T, B, H, C = x.size()
        x = x.reshape(T, B, H * C)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# Char maps
CHARS = "0123456789T"
CHAR2IDX = {ch: i+1 for i, ch in enumerate(CHARS)}
CHAR2IDX['<blank>'] = 0
IDX2CHAR = {i: ch for ch, i in CHAR2IDX.items()}

# Decode function
def decode(output):
    output = output.argmax(2).detach().cpu().numpy()
    results = []
    for seq in output.transpose(1, 0):
        prev = -1
        decoded = []
        for ch in seq:
            if ch != prev and ch != 0:
                decoded.append(IDX2CHAR[ch])
            prev = ch
        results.append(''.join(decoded))
    return results

# Load CRNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
crnn_model = CRNN(num_classes=len(CHAR2IDX)).to(device)
crnn_model.load_state_dict(torch.load("crnn_license_plate.pth", map_location=device))
crnn_model.eval()

# CRNN image transform
crnn_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Inference pipeline
def recognize_plate_from_car(image_path):
    image = Image.open(image_path).convert("RGB")
    results = yolo_model.predict(image_path, conf=0.4)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = image.crop((x1, y1, x2, y2))

        # CRNN inference
        input_tensor = crnn_transform(plate_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            output = crnn_model(input_tensor)
        pred_text = decode(output)[0]

        # Draw bbox & text on original image
        draw = ImageDraw.Draw(image)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = None  # fallback
        draw.text((x1, y1 - 30), pred_text, fill="yellow", font=font)

    image.show()
    image.save("output_with_text.jpg")
    print("Done. Saved as output_with_text.jpg")

# === CALL PIPELINE ===
for i in range(1,20):
    path = f"F:/License-Plate-Detection-and-Recognition/DATA SCIENTIST_ASSIGNMENT/license_plates_detection_train/{i}.jpg"
    recognize_plate_from_car(path)
