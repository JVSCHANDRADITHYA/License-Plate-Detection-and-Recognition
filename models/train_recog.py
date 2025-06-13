import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

# ---- CONFIG ----
IMG_DIR = 'plates/'
CSV_PATH = 'plate_labels.csv'
CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
MAX_LABEL_LEN = 10
IMG_WIDTH, IMG_HEIGHT = 200, 50
BATCH_SIZE = 16
EPOCHS = 30
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- CHAR TO INDEX MAP ----
char_to_idx = {char: idx + 1 for idx, char in enumerate(CHARS)}
idx_to_char = {idx + 1: char for idx, char in enumerate(CHARS)}
char_to_idx['blank'] = 0
idx_to_char[0] = ''

# ---- Dataset ----
class PlateDataset(Dataset):
    def __init__(self, csv_path, img_dir):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def encode_label(self, label):
        return [char_to_idx[c] for c in label]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['img_id'])
        image = Image.open(img_path).convert('L')
        image = self.transform(image)
        label = self.encode_label(row['text'])
        return image, torch.tensor(label), len(label)

# ---- Model (CRNN) ----
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
        )
        self.rnn = nn.LSTM(128 * (IMG_HEIGHT//4), 256, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        x = x.view(b, w, -1)       # (B, W, C*H)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# ---- Collate Fn for CTC ----
def collate_fn(batch):
    images, labels, label_lens = zip(*batch)
    images = torch.stack(images)
    labels_concat = torch.cat(labels)
    label_lens = torch.tensor(label_lens)
    return images, labels_concat, label_lens

# ---- Training ----
def train():
    dataset = PlateDataset(CSV_PATH, IMG_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = CRNN(num_classes=len(char_to_idx)).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, labels, label_lens in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)  # (B, W, C)
            log_probs = nn.functional.log_softmax(logits, dim=2)

            input_len = torch.full(size=(log_probs.size(0),), fill_value=log_probs.size(1), dtype=torch.long).to(DEVICE)
            loss = criterion(log_probs.permute(1, 0, 2), labels, input_len, label_lens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), 'crnn_plate.pth')
    print("✅ Model saved as crnn_plate.pth")

# ---- Run ----
if __name__ == '__main__':
    train()
