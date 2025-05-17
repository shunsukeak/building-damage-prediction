import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from sklearn.metrics import classification_report

graph_dir = "../graphs/graphs_with_image"
save_dir = "../graphs/gat_models_by_type"
os.makedirs(save_dir, exist_ok=True)

# === GAT ===
class GATClassifier(nn.Module):
    def __init__(self, in_dim=518, hidden_dim=128, out_dim=4, heads=4, dropout=0.6):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        return self.mlp(x)

def train_gat(data, model, device, epochs=100, patience=10):
    data = data.to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        model.eval()
        pred = out.argmax(dim=1)
        acc = (pred == data.y).float().mean().item()
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f} Acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"⛔ Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model

def evaluate_gat(data, model, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu().numpy()
        true = data.y.cpu().numpy()

    print("📊 Classification Report:")
    print(classification_report(true, pred, target_names=["no-damage", "minor", "major", "destroyed"]))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fname in os.listdir(graph_dir):
        if not fname.endswith(".pt"): continue
        path = os.path.join(graph_dir, fname)
        dtype = fname.replace("graph_", "").replace(".pt", "")

        print(f"\n🚀 Training GAT for disaster type: {dtype}")
        data = torch.load(path)

        model = GATClassifier()
        model = train_gat(data, model, device, epochs=100, patience=10)

        model_path = os.path.join(save_dir, f"gat_{dtype}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"💾 Saved model to {model_path}")

        evaluate_gat(data, model, device)

if __name__ == "__main__":
    main()
