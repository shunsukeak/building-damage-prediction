# hybrid_shape_train.py
import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
import torch.nn.functional as F

# === Focal Loss クラス ===
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, weight=self.weight, reduction=self.reduction)
        return loss

# === Dataset ===
class CropBuildingDatasetWithShape(Dataset):
    def __init__(self, csv_path, transform=None, image_size=224):
        self.df = pd.read_csv(csv_path)
        self.image_paths = self.df["image_path"].tolist()
        self.labels = self.df["label"].tolist()
        self.hazard_levels = self.df["hazard_level"].tolist()
        self.shape_features = self.df[["area", "perimeter", "aspect_ratio", "extent_ratio", "convexity"]].fillna(0).values.astype("float32")
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
        try:
            img = self._load_image(self.image_paths[idx])
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        hazard_level = torch.tensor([self.hazard_levels[idx]], dtype=torch.float32)
        shape_feat = torch.tensor(self.shape_features[idx], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img).permute(2, 0, 1)

        return img, hazard_level, shape_feat, label

# === Model ===
class ResNetWithHazardAndShape(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.resnet = models.resnet34(weights="IMAGENET1K_V1")
        self.resnet.fc = nn.Identity()

        self.hazard_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.shape_fc = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 + 32 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_hazard, x_shape):
        img_feat = self.resnet(x_img)
        hazard_feat = self.hazard_fc(x_hazard)
        shape_feat = self.shape_fc(x_shape)
        feat = torch.cat([img_feat, hazard_feat, shape_feat], dim=1)
        return self.classifier(feat)

# === 学習曲線プロット関数 ===
def plot_training_curves(train_losses, val_accuracies, save_dir, disaster_type):
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve ({disaster_type})")
    plt.legend()
    loss_path = os.path.join(save_dir, f"loss_curve_{disaster_type}.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"📈 Saved loss curve: {loss_path}")

    plt.figure()
    plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Validation Accuracy Curve ({disaster_type})")
    plt.legend()
    acc_path = os.path.join(save_dir, f"accuracy_curve_{disaster_type}.png")
    plt.savefig(acc_path)
    plt.close()
    print(f"📈 Saved accuracy curve: {acc_path}")

# === Fine-tuning function ===
def train_one_type(train_csv, val_csv, disaster_type, save_dir, device, epochs=30, patience=5):
    print(f"🔵 Training disaster type: {disaster_type}")

    train_dataset = CropBuildingDatasetWithShape(train_csv)
    val_dataset = CropBuildingDatasetWithShape(val_csv)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    train_df = pd.read_csv(train_csv)
    class_counts = train_df["label"].value_counts().sort_index().values
    class_counts = torch.tensor(class_counts, dtype=torch.float)
    class_freq = class_counts / class_counts.sum()
    class_weights = 1.0 / class_freq
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = class_weights.to(device)
    print(f"⚖️ Class weights for {disaster_type}: {class_weights}")

    model = ResNetWithHazardAndShape().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = FocalLoss(gamma=2, weight=class_weights)

    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for img, hazard, shape, label in tqdm(train_loader, desc=f"Epoch {epoch+1} - Train"):
            img, hazard, shape, label = img.to(device), hazard.to(device), shape.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(img, hazard, shape)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

        train_acc = correct / total
        train_losses.append(total_loss)
        print(f"✅ Epoch {epoch+1} Train Loss={total_loss:.4f}, Train Acc={train_acc:.4f}")

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for img, hazard, shape, label in tqdm(val_loader, desc=f"Epoch {epoch+1} - Val"):
                img, hazard, shape, label = img.to(device), hazard.to(device), shape.to(device), label.to(device)
                output = model(img, hazard, shape)
                pred = output.argmax(dim=1)
                val_correct += (pred == label).sum().item()
                val_total += label.size(0)

        val_acc = val_correct / val_total
        val_accuracies.append(val_acc)
        print(f"🧪 Epoch {epoch+1} Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"model_type_{disaster_type}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"💾 Saved best model for type {disaster_type} at {model_path}")
        else:
            no_improve_epochs += 1
            print(f"⏳ No improvement for {no_improve_epochs} epoch(s).")

        if no_improve_epochs >= patience:
            print(f"⛔ Early stopping at epoch {epoch+1}")
            break

    plot_training_curves(train_losses, val_accuracies, save_dir, disaster_type)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_val_dir = "./type_train_val_split_with_shape"
    save_dir = "./hybrid_saved_type_models_with_shape"

    type_list = [f.replace("train_", "").replace(".csv", "") for f in os.listdir(train_val_dir) if f.startswith("train_")]

    for disaster_type in type_list:
        train_csv = os.path.join(train_val_dir, f"train_{disaster_type}.csv")
        val_csv = os.path.join(train_val_dir, f"val_{disaster_type}.csv")
        train_one_type(train_csv, val_csv, disaster_type, save_dir, device, epochs=30)
