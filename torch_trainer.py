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
    def __init__(self, hf_dataset, label2id, transform=None):
        self.examples = hf_dataset
        self.label2id = label2id
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        row = self.examples[idx]
        img = row['image'].crop((1017, 0, 2033, 561)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        # Country label (as class index)
        label = self.label2id[row['country_iso_alpha2']]
        
        # Coordinates (as continuous values)
        # Convert to float and normalize to [-1, 1]
        longitude = float(row['longitude'])
        latitude = float(row['latitude'])
        
        # Normalize: longitude [-180, 180] -> [-1, 1], latitude [-90, 90] -> [-1, 1]
        long_norm = longitude / 180.0
        lat_norm = latitude / 90.0
        
        return img, label, torch.tensor([long_norm, lat_norm], dtype=torch.float32)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    gc.collect()
    torch.cuda.empty_cache()

class ResNet50MultiTask(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet50(weights=None)  # Or weights='IMAGENET1K_V1'
        in_features = self.resnet.fc.in_features
        
        # Keep the backbone for feature extraction
        self.resnet.fc = nn.Identity()
        
        # Two heads for multi-task learning
        self.country_head = nn.Linear(in_features, num_classes)
        self.coordinate_head = nn.Linear(in_features, 2)  # For (longitude, latitude)
    
    def forward(self, x):
        features = self.resnet(x)
        country_logits = self.country_head(features)
        coordinates = torch.tanh(self.coordinate_head(features))  # Bound to [-1, 1]
        return country_logits, coordinates

if __name__ == "__main__":
    # ---------------- CONFIG ----------------
    DATASET_NAME = "stochastic/random_streetview_images_pano_v0.0.2"
    OUTPUT_DIR = Path("/resnet50-finetuned_raw")
    BATCH_SIZE = 2       # Increased for better GPU utilization
    NUM_EPOCHS = 1      # More epochs for proper training
    LR = 1e-4
    IMG_CROP = (1017, 0, 2033, 561)
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    FP16 = True
    COUNTRY_LOSS_WEIGHT = 1.0
    COORD_LOSS_WEIGHT = 0.5  # Adjust based on importance
    # ---------------------------------------
    
    set_seed(42)
    
    # Load dataset
    full_dataset = load_dataset(DATASET_NAME)["train"]
    
    # Split
    train_idx, val_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.1,
        shuffle=True,
        random_state=SEED
    )
    
    train_hf = full_dataset.select(train_idx)
    val_hf = full_dataset.select(val_idx)
    
    # Classes
    labels = sorted(list(set(full_dataset["country_iso_alpha2"])))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(labels)
    print(f"Number of countries: {num_labels}")
    print(f"Total samples: {len(full_dataset)}")
    print(f"Training samples: {len(train_hf)}")
    print(f"Validation samples: {len(val_hf)}")
    
    # Transforms
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets + Loaders
    train_dataset = StreetViewDataset(train_hf, label2id, transform)
    val_dataset = StreetViewDataset(val_hf, label2id, val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    # ---------------- MODEL ----------------
    model = ResNet50MultiTask(num_classes=num_labels)
    model = model.to(DEVICE)
    
    # Load pretrained weights (optional)
    # pretrained = models.resnet50(weights='IMAGENET1K_V1')
    # model.resnet.load_state_dict(pretrained.state_dict(), strict=False)
    
    # ---------------- LOSSES & OPTIMIZER ----------------
    criterion_country = nn.CrossEntropyLoss()
    criterion_coord = nn.MSELoss()  # Mean Squared Error for regression
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.GradScaler(enabled=FP16)
    
    # ---------------- TRAINING LOOP ----------------
    best_val_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_country_loss = 0.0
        train_coord_loss = 0.0
        train_total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        for imgs, country_labels, coords in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            country_labels = country_labels.to(DEVICE, non_blocking=True)
            coords = coords.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type="cuda", enabled=FP16):
                # Forward pass
                country_logits, pred_coords = model(imgs)
                
                # Compute losses
                loss_country = criterion_country(country_logits, country_labels)
                loss_coord = criterion_coord(pred_coords, coords)
                
                # Combined loss
                loss = COUNTRY_LOSS_WEIGHT * loss_country + COORD_LOSS_WEIGHT * loss_coord
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Track losses
            train_country_loss += loss_country.item()
            train_coord_loss += loss_coord.item()
            train_total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'country_loss': loss_country.item(),
                'coord_loss': loss_coord.item(),
                'total_loss': loss.item()
            })
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_country_correct = 0
        val_total = 0
        val_coord_loss = 0.0
        
        with torch.no_grad():
            for imgs, country_labels, coords in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                country_labels = country_labels.to(DEVICE, non_blocking=True)
                coords = coords.to(DEVICE, non_blocking=True)
                
                country_logits, pred_coords = model(imgs)
                
                # Country accuracy
                _, predicted = torch.max(country_logits, 1)
                val_country_correct += (predicted == country_labels).sum().item()
                val_total += country_labels.size(0)
                
                # Coordinate loss
                val_coord_loss += criterion_coord(pred_coords, coords).item()
        
        val_acc = val_country_correct / val_total
        avg_val_coord_loss = val_coord_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train - Country Loss: {train_country_loss/len(train_loader):.4f}, "
              f"Coord Loss: {train_coord_loss/len(train_loader):.4f}, "
              f"Total Loss: {train_total_loss/len(train_loader):.4f}")
        print(f"  Val - Accuracy: {val_acc:.4f}, Coord Loss: {avg_val_coord_loss:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_coord_loss': avg_val_coord_loss,
                'label_mapping': id2label,
            }, "resnet50_streetview_multi_task.pth")
            print(f"  💾 Saved best model with val_acc: {val_acc:.4f}")
    
    print(f"\n✅ Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")