import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from pathlib import Path
import gc, random, numpy as np
from tqdm import tqdm

print(f"CPU specs: {torch.backends.cpu.get_cpu_capability()}")
print(f"cuDNN: {torch.backends.cudnn.is_available()}")
print(f"CUDA: {torch.backends.cuda.is_built()}")
torch.backends.cudnn.benchmark = True

class StreetViewDataset(Dataset):
    def __init__(self, hf_dataset, label2id, longitude, latitude, transform=None):
        self.examples = hf_dataset
        self.label2id = label2id
        self.longitude = longitude
        self.latitude = latitude
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        row = self.examples[idx]
        img = row['image'].crop((1017, 0, 2033, 561)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.label2id[row['country_iso_alpha2']]
        long = self.label2id[row['longitude']]
        lat = self.label2id[row['latitude']]
        return img, label, long, lat

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # ---------------- CONFIG ----------------
    DATASET_NAME = "stochastic/random_streetview_images_pano_v0.0.2"
    OUTPUT_DIR = Path("/resnet50-finetuned_raw")
    BATCH_SIZE = 256       # bigger batch for better GPU utilization
    NUM_EPOCHS = 16      # for 48 hours
    LR = 1e-4
    IMG_CROP = (1017, 0, 2033, 561)  # (left, top, right, bottom)
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4       # CPU parallelism for DataLoader
    FP16 = True           # mixed precision
    # ---------------------------------------

    # ---------------- SEED ----------------
    set_seed(42)

    # --- Load dataset
    full_dataset = load_dataset(DATASET_NAME)["train"]

    # --- Split
    train_idx, val_idx = train_test_split(list(range(len(full_dataset))),
                                        test_size=0.1,
                                        shuffle=True,
                                        random_state=SEED)

    train_hf = full_dataset.select(train_idx)
    val_hf   = full_dataset.select(val_idx)

    # --- Classes
    labels = sorted(list(set(full_dataset["country_iso_alpha2"])))
    longs = sorted(list(set(full_dataset["longitude"])))
    lats = sorted(list(set(full_dataset["latitude"])))

    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    long = {l: i for i, l in enumerate(longs)}
    rev_long = {i: l for l, i in long.items()}
    lat = {l: i for i, l in enumerate(lats)}
    rev_lat = {i: l for l, i in lat.items()}

    num_labels = len(labels)
    print(f"Number of iso2alphas: {num_labels}")

    # --- Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # --- Datasets + Loaders
    train_dataset = StreetViewDataset(train_hf, label2id, rev_long, rev_lat, transform)
    val_dataset   = StreetViewDataset(val_hf, label2id, rev_long, rev_lat, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    # ---------------- MODEL ----------------
    model = models.resnet50(weights=None)  # random init
    model.fc = nn.Linear(model.fc.in_features, num_labels)
    model = model.to(DEVICE)

    # ---------------- OPTIMIZER & LOSS ----------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = torch.GradScaler('cuda', enabled=FP16)

    # ---------------- TRAINING ----------------
    print('started training loop')
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, labels, longs, lats in tqdm(train_loader):
            imgs, labels, longs, lats = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True), longs.to(DEVICE, non_blocking=True), lats.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=FP16):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {epoch_loss:.4f}")

        # --- Optional: validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Validation Accuracy: {acc:.4f}")

    # ---------------- SAVE ----------------
    torch.save(model.state_dict(), "resnet50_streetview_feature.pth")
    print(f"Model saved")
