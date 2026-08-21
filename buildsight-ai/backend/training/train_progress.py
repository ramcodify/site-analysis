"""BuildSight AI — 9-Stage Construction Progress Classification Model Training & Test Evaluation

Trains a real convolutional neural network classifier on the 9 construction stages:
  1. Site Preparation
  2. Excavation
  3. Foundation
  4. Structural Work
  5. Brickwork
  6. Roofing
  7. Plastering
  8. Electrical and Plumbing
  9. Finishing

Evaluates on untouched test split and saves:
  - backend/data/models/progress_model.pth
  - backend/data/models/progress_evaluation_report.json
"""

import os
import json
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
PROGRESS_DATA_DIR = BASE_DIR / "dataset" / "progress"
MODELS_DIR = BASE_DIR / "data" / "models"
EXPERIMENTS_DIR = BASE_DIR / "experiments" / "progress_stages_v1"

STAGES = [
    "Site Preparation",
    "Excavation",
    "Foundation",
    "Structural Work",
    "Brickwork",
    "Roofing",
    "Plastering",
    "Electrical and Plumbing",
    "Finishing",
]


class ConstructionStageDataset(Dataset):
    """PyTorch Dataset for construction progress stage classification."""

    def __init__(self, split_dir: Path):
        self.samples = []
        for idx, stage in enumerate(STAGES):
            folder = split_dir / stage.replace(" ", "_")
            if folder.exists():
                for img_p in folder.glob("*.jpg"):
                    self.samples.append((str(img_p), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB").resize((128, 128))
        arr = np.array(img, dtype=np.float32) / 255.0
        # (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, label


class ConstructionStageClassifier(nn.Module):
    """Convolutional neural network for construction stage classification."""

    def __init__(self, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32x32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feat = self.features(x)
        return self.classifier(feat)


def train_and_evaluate(epochs=15, batch_size=16, lr=0.001):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = ConstructionStageDataset(PROGRESS_DATA_DIR / "train")
    val_ds = ConstructionStageDataset(PROGRESS_DATA_DIR / "val")
    test_ds = ConstructionStageDataset(PROGRESS_DATA_DIR / "test")

    logger.info(f"Loaded datasets: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConstructionStageClassifier(num_classes=len(STAGES)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger.info(f"Training 9-stage classifier for {epochs} epochs on {device}...")
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = out.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / max(1, val_total)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    # Save model weights
    save_path = MODELS_DIR / "progress_model.pth"
    torch.save(model.state_dict(), str(save_path))
    logger.info(f"✓ Saved trained progress stage model to {save_path}")

    # Evaluate on untouched test set
    logger.info("Evaluating progress classifier on untouched TEST set...")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Confusion matrix
    cm = np.zeros((len(STAGES), len(STAGES)), dtype=int)
    for p, t in zip(all_preds, all_targets):
        cm[t, p] += 1

    # Per-stage precision, recall, F1
    per_stage_metrics = {}
    for i, stage in enumerate(STAGES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_stage_metrics[stage] = {
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1_score": round(f1, 3),
            "support": int(cm[i, :].sum()),
        }

    overall_acc = float((all_preds == all_targets).mean())

    report = {
        "model_architecture": "ConstructionStageClassifier (Conv3-BatchNorm-ReLU-Linear)",
        "num_classes": 9,
        "stages": STAGES,
        "test_samples_total": len(all_targets),
        "overall_accuracy": round(overall_acc, 4),
        "per_stage_metrics": per_stage_metrics,
        "confusion_matrix": cm.tolist(),
    }

    report_path = MODELS_DIR / "progress_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(EXPERIMENTS_DIR / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"✓ Progress Stage Model Test Accuracy: {overall_acc * 100:.1f}%")
    logger.info(f"✓ Evaluation report saved to {report_path}")
    return report


if __name__ == "__main__":
    train_and_evaluate()
