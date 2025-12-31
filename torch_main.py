import pandas as pd
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from tqdm import tqdm
from predictor import predict_image
from transformers import AutoImageProcessor

imgs = ["pics/zahodryazan.jpg", "pics/t1.png", "pics/t2.png", "pics/t3.png", "pics/t4.png", "pics/ryazan21080-371224838.jpg", "pics/ryazan-russia-city-view-3628679470.jpg", "pics/Ryazan-03.jpg", "pics/5df12e8f9e3d0-5140-sobornaja-ploschad.jpeg"]
HEIGHT = 561              # desired target height in pixels
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

id2label_map = {
    7: "AD", 16: "AE", 15: "AR", 43: "AU", 30: "BD",
    26: "BE", 4: "BG", 46: "BR", 48: "BT", 31: "BW",
    3: "CA", 49: "CH", 34: "CL", 17: "CO", 45: "CZ",
    25: "DE", 36: "DK", 12: "EE", 41: "ES", 23: "FI",
    28: "FR", 0: "GB", 53: "GR", 22: "HK", 24: "HR",
    14: "HU", 42: "ID", 13: "IE", 51: "IL", 6: "IS",
    27: "IT", 35: "JP", 10: "KH", 1: "KR", 32: "LT",
    50: "LV", 29: "MX", 9: "MY", 2: "NL", 5: "NO",
    37: "NZ", 44: "PE", 47: "PL", 21: "PT", 38: "RO",
    52: "RU", 40: "SE", 19: "SG", 55: "SI", 8: "SK",
    11: "SZ", 18: "TH", 33: "TW", 39: "UA", 54: "US",
    20: "ZA"
}

iso_alpha2_to_country = {
    "AD": "Andorra", "AE": "United Arab Emirates", "AR": "Argentina", "AU": "Australia",
    "BD": "Bangladesh", "BE": "Belgium", "BG": "Bulgaria", "BR": "Brazil", "BT": "Bhutan",
    "BW": "Botswana", "CA": "Canada", "CH": "Switzerland", "CL": "Chile", "CO": "Colombia",
    "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark", "EE": "Estonia", "ES": "Spain",
    "FI": "Finland", "FR": "France", "GB": "United Kingdom", "GR": "Greece", "HK": "Hong Kong",
    "HR": "Croatia", "HU": "Hungary", "ID": "Indonesia", "IE": "Ireland", "IL": "Israel",
    "IS": "Iceland", "IT": "Italy", "JP": "Japan", "KH": "Cambodia", "KR": "Republic of Korea",
    "LT": "Lithuania", "LV": "Latvia", "MX": "Mexico", "MY": "Malaysia", "NL": "Netherlands",
    "NO": "Norway", "NZ": "New Zealand", "PE": "Peru", "PL": "Poland", "PT": "Portugal",
    "RO": "Romania", "RU": "Russian Federation", "SE": "Sweden", "SG": "Singapore",
    "SI": "Slovenia", "SK": "Slovakia", "SZ": "Eswatini", "TH": "Thailand", "TW": "Taiwan",
    "UA": "Ukraine", "US": "United States", "ZA": "South Africa"
}

# --------------------------
# Utilities
# --------------------------
def lowres(image: Image.Image, new_height: int = HEIGHT) -> Image.Image:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    new_width = int(round(new_height * aspect_ratio))
    resized_img = image.crop((int((orig_width-new_width)/2), 0, int((new_width/2)+new_width), 561))
    return resized_img

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def clean_state_dict_keys(state_dict: dict) -> dict:
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith("module."):
            new_key = k[len("module."):]
        new_sd[new_key] = v
    return new_sd

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.id2label = id2label_map
        self.country_head = nn.Linear(in_features, num_classes)
        self.coordinate_head = nn.Linear(in_features, 2)  # (longitude, latitude)
    
    def forward(self, x):
        features = self.resnet(x)
        country_logits = self.country_head(features)
        coordinates = self.coordinate_head(features)
        return country_logits, coordinates

def load_model_checkpoint(path: str, device: torch.device, num_classes=56):
    model = ResNet50FeatureExtractor(num_classes=num_classes)
    
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load checkpoint
    checkpoint = torch.load(path, map_location="cpu")
    
    print("📊 Checkpoint structure:")
    if isinstance(checkpoint, dict):
        for k, v in checkpoint.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape {v.shape}")
            elif isinstance(v, dict):
                print(f"  {k}: dict with {len(v)} keys")
            else:
                print(f"  {k}: {type(v).__name__} = {v}")
    
    # Extract the actual model weights
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("\n✅ Extracted model_state_dict from checkpoint")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("\n✅ Extracted state_dict from checkpoint")
        else:
            # If the checkpoint itself is the state_dict
            state_dict = checkpoint
            print("\n✅ Checkpoint is already the state_dict")
    else:
        state_dict = checkpoint
    
    # Clean keys (remove 'module.' prefix if present)
    state_dict = clean_state_dict_keys(state_dict)
    
    # Debug: Show first few keys
    print("\n📋 State dict keys (first 10):")
    for i, (k, v) in enumerate(list(state_dict.items())[:10]):
        if hasattr(v, 'shape'):
            print(f"  {k}: shape {v.shape}")
        else:
            print(f"  {k}: {type(v)}")
    
    # Check if we have the new multi-task structure
    has_country_head = any('country_head' in k for k in state_dict.keys())
    has_coordinate_head = any('coordinate_head' in k for k in state_dict.keys())
    
    if has_country_head and has_coordinate_head:
        print("\n✅ Multi-task checkpoint detected (both heads present)")
    elif has_country_head:
        print("\n⚠️ Only country_head found, coordinate_head missing")
    else:
        print("\n⚠️ Old checkpoint detected (no custom heads)")
    
    # Load the weights
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ Checkpoint loaded successfully (strict mode)")
    except Exception as e:
        print(f"⚠️ Strict load failed: {e}")
        print("Trying non-strict load...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        print(f"\n📊 Load summary:")
        print(f"  Missing keys: {len(missing)}")
        if missing:
            for m in missing[:5]:  # Show first 5
                print(f"    - {m}")
            if len(missing) > 5:
                print(f"    ... and {len(missing)-5} more")
        
        print(f"\n  Unexpected keys: {len(unexpected)}")
        if unexpected:
            for u in unexpected[:5]:  # Show first 5
                print(f"    - {u}")
            if len(unexpected) > 5:
                print(f"    ... and {len(unexpected)-5} more")
    
    model.to(device)
    model.eval()
    return model, checkpoint

def project_and_plot(embs: np.ndarray, sample_emb: np.ndarray,
                     id2label_map, labels,
                     show_text=True):
    
    embeddings = embs.squeeze()
    df = pd.DataFrame(embeddings)
    df['label'] = labels
    centroids = df.groupby('label').mean().to_numpy()
    classes = df['label'].unique()
       
    all_points = np.concatenate([centroids, sample_emb], axis=0)

    scaled = StandardScaler().fit_transform(all_points)
    umap_2d = UMAP(n_components=2, n_neighbors=4, random_state=42).fit_transform(scaled)

    plane_centroids = umap_2d[:-1]
    plane_sample = umap_2d[-1]

    plt.figure(figsize=(12, 8))
    if plane_centroids.shape[0] > 0:
        # color by class if provided, else single color
        if classes is not None:
            plt.scatter(plane_centroids[:, 0], plane_centroids[:, 1], c=classes, cmap='tab20', s=20)
        else:
            plt.scatter(plane_centroids[:, 0], plane_centroids[:, 1], s=20)
    plt.scatter(plane_sample[0], plane_sample[1], c='red', s=120, edgecolor='black', marker='X', label='sample')

    if show_text and plane_centroids.shape[0] > 0 and classes is not None and id2label_map is not None:
        for i, lbl in enumerate(classes):
            alpha2 = id2label_map.get(int(lbl), str(lbl))
            plt.text(plane_centroids[i, 0], plane_centroids[i, 1], alpha2, fontsize=12,
                     ha='center', va='center')

    plt.legend()
    plt.title("UMAP projection (centroids + sample)")
    plt.show()

def diagnose_model(model, checkpoint):
    print("\n🔍 MODEL DIAGNOSTICS 🔍")
    
    # 1. Check model structure
    print(f"Country head shape: {model.country_head.weight.shape}")
    print(f"Coordinate head shape: {model.coordinate_head.weight.shape}")
    
    # 2. Check coordinate head outputs
    test_input = torch.randn(1, 2048).to(DEVICE)  # Random features
    coords = model.coordinate_head(test_input)
    print(f"\nRandom feature -> Coordinates: {coords}")
    
    # 3. Check if tanh is applied correctly
    print(f"After tanh: {torch.tanh(coords)}")
    
    # 4. What happens with all-zero features?
    zero_input = torch.zeros(1, 2048).to(DEVICE)
    zero_coords = model.coordinate_head(zero_input)
    print(f"\nZero features -> Coordinates: {zero_coords}")
    print(f"Zero features -> After tanh: {torch.tanh(zero_coords)}")
    print(f"Denormalized: Longitude={torch.tanh(zero_coords)[0,0].item()*180:.1f}°, "
          f"Latitude={torch.tanh(zero_coords)[0,1].item()*90:.1f}°")
    
    # 5. Check checkpoint info
    if 'val_acc' in checkpoint:
        print(f"\nCheckpoint validation accuracy: {checkpoint['val_acc']*100:.2f}%")
    if 'val_coord_loss' in checkpoint:
        print(f"Checkpoint coordinate loss: {checkpoint['val_coord_loss']:.4f}")

# Call it in your main

if __name__ == "__main__":
    if os.path.exists(str(Path(__file__).absolute().parent) + "/np_cache/embeddings.npy"):
        embeddings = np.load("np_cache/embeddings.npy")
        labels = np.load("np_cache/labels.npy")
        print("loaded data via save at /np_cache")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ckpt_path = "E://resnet50_streetview_multi_task.pth"
    model, ckpt = load_model_checkpoint(ckpt_path, device=device, num_classes=56)
    sample_imgs = []

    for i in enumerate(imgs, 0):
        img = lowres(Image.open(i[1]).convert("RGB"))
        sample_imgs.append(img)

    predict_image(samples=sample_imgs, model=model, checkpoint=ckpt)

    diagnose_model(model, ckpt)


    # project_and_plot(embs=embeddings, sample_emb=sample_emb, id2label_map=id2label_map, labels=labels)

    