import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CTCLoss
from torchvision import transforms
from PIL import Image
import pandas as pd

# ====== CONFIG ======
CSV_PATH = r"F:\License-Plate-Detection-and-Recognition\DATA SCIENTIST_ASSIGNMENT\Licplatesrecognition_train.csv"  # your CSV: img_id,text
IMG_DIR = r"F:\License-Plate-Detection-and-Recognition\DATA SCIENTIST_ASSIGNMENT\license_plates_recognition_train"  # your images folder
EPOCHS = 20
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== CHAR MAPS ======
CHARS = "0123456789T"  # Add Arabic or other chars if needed
CHAR2IDX = {ch: i+1 for i, ch in enumerate(CHARS)}  # +1 because 0 is blank
CHAR2IDX['<blank>'] = 0
IDX2CHAR = {i: ch for ch, i in CHAR2IDX.items()}

# ====== DATASET ======
class LicensePlateDataset(Dataset):
    def __init__(self, csv_path, img_dir):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((32, 128)),  # H x W
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["img_id"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = [CHAR2IDX[c] for c in str(row["text"]) if c in CHAR2IDX]
        return image, torch.tensor(label, dtype=torch.long)

# ====== COLLATE FUNCTION ======
def collate_fn(batch):
    images, labels = zip(*batch)
    labels = list(labels)
    return torch.stack(images), labels

# ====== MODEL ======
class CRNN(nn.Module):
    def __init__(self, num_classes=len(CHAR2IDX)):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
        )
        self.rnn = nn.LSTM(256 * 8, 256, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)                    # (B, C, H, W)
        x = x.permute(3, 0, 2, 1)          # (W, B, H, C)
        T, B, H, C = x.size()
        x = x.reshape(T, B, H * C)         # (W, B, H*C)
        x, _ = self.rnn(x)                 # (W, B, 512)
        x = self.fc(x)                     # (W, B, num_classes)
        return x

# ====== DECODE FUNCTION ======
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

# ====== TRAIN FUNCTION ======
def train():
    dataset = LicensePlateDataset(CSV_PATH, IMG_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = CRNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = CTCLoss(blank=0)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            targets = torch.cat(labels).to(DEVICE)
            input_lengths = torch.full(size=(imgs.size(0),), fill_value=imgs.shape[-1] // 4, dtype=torch.long)
            target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long).to(DEVICE)

            output = model(imgs)  # (W, B, C)
            loss = criterion(output.log_softmax(2), targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "crnn_license_plate.pth")
    return model

# ====== INFERENCE EXAMPLE ======
def test_single(model, image_path):
    model.eval()
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img)  # (T, B, C)
    result = decode(output)
    print("Predicted:", result[0])

# ====== RUN EVERYTHING ======
if __name__ == "__main__":
    model = train()
    # test_single(model, "license_plate_images/103.jpg")  # Uncomment to test
