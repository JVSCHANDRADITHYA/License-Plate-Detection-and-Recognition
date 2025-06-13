from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class LicensePlateDetection(Dataset):
    def __init__(self, data_folder, csv_file, transform=None):
        self.data_folder = data_folder
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['img_id']
        img_path = os.path.join(self.data_folder, img_id)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.df.iloc[idx]['xmin'], self.df.iloc[idx]['ymin'], self.df.iloc[idx]['xmax'], self.df.iloc[idx]['ymax']
    

# Example usage:
if __name__ == "__main__":
    csv_path = r'F:\License-Plate-Detection-and-Recognition\DATA SCIENTIST_ASSIGNMENT\Licplatesdetection_train.csv'
    data_folder = r'F:\License-Plate-Detection-and-Recognition\DATA SCIENTIST_ASSIGNMENT\license_plates_detection_train'
    
    dataset = LicensePlateDetection(data_folder, csv_path)
    print(f"Number of samples in dataset: {len(dataset)}")
    img, xmin, ymin, xmax, ymax = dataset[0]
    print(f"Image size: {img.size}, Bounding box: ({xmin}, {ymin}, {xmax}, {ymax})")

    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, edgecolor='red', facecolor='none'))
    image = img.crop((xmin, ymin, xmax, ymax))
    plt.imshow(image)
    plt.show()
    