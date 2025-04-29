# 1-4. resnet finetuning
import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm

# === Dataset for crop images ===
class CropBuildingDataset(Dataset):
    def __init__(self, csv_path, transform=None, image_size=224):
        self.df = pd.read_csv(csv_path)
        self.image_paths = self.df["image_path"].tolist()
        self.labels = self.df["label"].tolist()
        self.hazard_levels = self.df["hazard_level"].tolist()
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype("float32") / 255.0
        return img

    def __getitem__(self, idx):
        img = self._load_image(self.image_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        hazard_level = torch.tensor(self.hazard_levels[idx], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img).permute(2, 0, 1)

        return img, hazard_level.unsqueeze(0), label

# === Model: ResNet34 + Hazard Meta ===
class ResNetWithHazard(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.resnet = models.resnet34(weights="IMAGENET1K_V1")
        self.resnet.fc = nn.Identity()
        self.meta_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_meta):
        img_feat = self.resnet(x_img)
        meta_feat = self.meta_fc(x_meta)
        feat = torch.cat([img_feat, meta_feat], dim=1)
        return self.classifier(feat)

# === Fine-tuning function ===
def train_one_type(train_csv, val_csv, disaster_type, save_dir, device, epochs=10):
    print(f"🔵 Training disaster type: {disaster_type}")

    train_dataset = CropBuildingDataset(train_csv)
    val_dataset = CropBuildingDataset(val_csv)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = ResNetWithHazard().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for img, hazard, label in tqdm(train_loader, desc=f"Epoch {epoch+1} - Train"):
            img, hazard, label = img.to(device), hazard.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(img, hazard)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

        train_acc = correct / total
        print(f"✅ Epoch {epoch+1} Train Loss={total_loss:.4f}, Train Acc={train_acc:.4f}")

        # --- Validation ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for img, hazard, label in tqdm(val_loader, desc=f"Epoch {epoch+1} - Val"):
                img, hazard, label = img.to(device), hazard.to(device), label.to(device)
                output = model(img, hazard)
                pred = output.argmax(dim=1)
                val_correct += (pred == label).sum().item()
                val_total += label.size(0)

        val_acc = val_correct / val_total
        print(f"🧪 Epoch {epoch+1} Val Acc={val_acc:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"model_type_{disaster_type}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"💾 Saved best model for type {disaster_type} at {model_path}")

# === 実行 ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_val_dir = "./type_train_val_split"
    save_dir = "./saved_type_models"

    type_list = [f.replace("train_", "").replace(".csv", "") for f in os.listdir(train_val_dir) if f.startswith("train_")]

    for disaster_type in type_list:
        train_csv = os.path.join(train_val_dir, f"train_{disaster_type}.csv")
        val_csv = os.path.join(train_val_dir, f"val_{disaster_type}.csv")
        train_one_type(train_csv, val_csv, disaster_type, save_dir, device, epochs=10)
