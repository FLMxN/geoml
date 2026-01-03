import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from pathlib import Path
import gc, random, numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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
        # Load ImageNet pretrained ResNet50
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')
        in_features = self.resnet.fc.in_features
        
        # Replace fc layer with Identity for feature extraction
        self.resnet.fc = nn.Identity()
        
        # Country classification head
        self.country_head = nn.Linear(in_features, num_classes)
        
        # Coordinate regression head with better initialization
        self.coordinate_head = nn.Linear(in_features, 2)
        
        # Initialize coordinate head to predict reasonable starting location
        self._initialize_heads()
    
    def _initialize_heads(self):
        """Initialize heads with smart starting values"""
        nn.init.normal_(self.country_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.country_head.bias)
        
        nn.init.normal_(self.coordinate_head.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.coordinate_head.bias)

    
    def forward(self, x):
        features = self.resnet(x)
        country_logits = self.country_head(features)

        coordinates = torch.tanh(self.coordinate_head(features))
        
        return country_logits, coordinates

if __name__ == "__main__":
    # ---------------- CONFIG ----------------
    DATASET_NAME = "stochastic/random_streetview_images_pano_v0.0.2"
    BATCH_SIZE = 32                    
    NUM_EPOCHS = 64                    
    LR = 3e-4                           
    GRADIENT_ACCUMULATION_STEPS = 1     
    IMG_CROP = (1017, 0, 2033, 561)
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    
    COUNTRY_LOSS_WEIGHT = 2
    COORD_LOSS_WEIGHT = 0.5
    
    FP16 = True
    
    PATIENCE = 16
    MIN_DELTA = 0.001
    # ------------------------------------------------
    
    set_seed(SEED)
    
    print(f"🚀 Starting training with:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"   Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Learning rate: {LR}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
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
    
    print(f"\n📊 Dataset statistics:")
    print(f"   Total countries: {num_labels}")
    print(f"   Total samples: {len(full_dataset):,}")
    print(f"   Training samples: {len(train_hf):,}")
    print(f"   Validation samples: {len(val_hf):,}")
    
    # # Calculate coordinate statistics for normalization
    # print("\n🗺️ Calculating coordinate statistics...")
    # longitudes = [float(row['longitude']) for row in full_dataset]
    # latitudes = [float(row['latitude']) for row in full_dataset]
    
    # long_mean, long_std = np.mean(longitudes), np.std(longitudes)
    # lat_mean, lat_std = np.mean(latitudes), np.std(latitudes)
    
    # print(f"   Longitude: mean={long_mean:.2f}°, std={long_std:.2f}°")
    # print(f"   Latitude: mean={lat_mean:.2f}°, std={lat_std:.2f}°")
    # print(f"   Normalized means: long={long_mean/180:.3f}, lat={lat_mean/90:.3f}")
    
    # Enhanced transforms with more augmentations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.33, contrast=0.33, saturation=0.33, hue=0.1),
        transforms.RandomRotation(degrees=5),
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
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    
    # ---------------- MODEL ----------------
    print("\n🧠 Creating model with ImageNet pretrained weights...")
    model = ResNet50MultiTask(num_classes=num_labels)
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # ---------------- LOSSES & OPTIMIZER ----------------
    criterion_country = nn.CrossEntropyLoss()
    criterion_coord = nn.MSELoss()
    
    # Use AdamW with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LR, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=64,
        T_mult=1,
        eta_min=1e-7
    )
    
    scaler = torch.GradScaler(enabled=FP16)
    
    # ---------------- TRAINING LOOP ----------------
    print("\n🚂 Starting training...")
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_country_loss = 0.0
        train_coord_loss = 0.0
        train_total_loss = 0.0
        
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        for batch_idx, (imgs, country_labels, coords) in enumerate(pbar):
            imgs = imgs.to(DEVICE, non_blocking=True)
            country_labels = country_labels.to(DEVICE, non_blocking=True)
            coords = coords.to(DEVICE, non_blocking=True)
            
            with torch.autocast(device_type="cuda", enabled=FP16):
                # Forward pass
                country_logits, pred_coords = model(imgs)
                
                # Compute losses
                loss_country = criterion_country(country_logits, country_labels)
                loss_coord = criterion_coord(pred_coords, coords)
                
                # Combined loss
                loss = (COUNTRY_LOSS_WEIGHT * loss_country + 
                       COORD_LOSS_WEIGHT * loss_coord) / GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass with gradient accumulation
            scaler.scale(loss).backward()
            
            # Update weights every GRADIENT_ACCUMULATION_STEPS
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Track losses
            train_country_loss += loss_country.item() * GRADIENT_ACCUMULATION_STEPS
            train_coord_loss += loss_coord.item() * GRADIENT_ACCUMULATION_STEPS
            train_total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # Update progress bar
            pbar.set_postfix({
                'country_loss': loss_country.item(),
                'coord_loss': loss_coord.item(),
                'lr': optimizer.param_groups[0]['lr']
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
        
        # Calculate metrics
        train_country_loss_avg = train_country_loss / len(train_loader)
        train_coord_loss_avg = train_coord_loss / len(train_loader)
        train_total_loss_avg = train_total_loss / len(train_loader)
        
        val_acc = val_country_correct / val_total
        avg_val_coord_loss = val_coord_loss / len(val_loader)
        
        print(f"\n📊 Epoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"   Train - Country Loss: {train_country_loss_avg:.4f}, "
              f"Coord Loss: {train_coord_loss_avg:.4f}, "
              f"Total Loss: {train_total_loss_avg:.4f}")
        print(f"   Val - Accuracy: {val_acc:.4f} ({val_country_correct}/{val_total}), "
              f"Coord Loss: {avg_val_coord_loss:.4f}")
        print(f"   Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping check
        if val_acc > best_val_acc + MIN_DELTA:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_coord_loss': avg_val_coord_loss,
                'label_mapping': id2label,
                'config': {
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LR,
                    'country_loss_weight': COUNTRY_LOSS_WEIGHT,
                    'coord_loss_weight': COORD_LOSS_WEIGHT,
                }
            }, "resnet50_streetview_imagenet1k.pth")
            print(f"   💾 Saved BEST model with val_acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"   ⏹️ Early stopping triggered after {epoch+1} epochs")
                break
    
    print(f"\n✅ Training completed!")
    print(f"   Best validation accuracy: {best_val_acc:.4f}")