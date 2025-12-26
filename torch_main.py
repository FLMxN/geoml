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

imgs = ["pics/t1.png"]
IMG = "pics/image.png"   # single image path
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

    def __init__(self, num_classes=56):
        super().__init__()
        # Using the standard torchvision ResNet50 backbone
        backbone = models.resnet50(weights=None)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Linear(num_ftrs, num_classes)
        self.backbone = backbone
        self.id2label = id2label_map

    def forward(self, x, return_features=False):
        # replicate torchvision forward until avgpool to extract features
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        pooled = self.backbone.avgpool(x)
        feats = torch.flatten(pooled, 1)  # shape (B, num_ftrs)
        logits = self.backbone.fc(feats)

        return feats if return_features else logits

def load_model_checkpoint(path: str, device: torch.device, num_classes=56):
    model = ResNet50FeatureExtractor(num_classes=num_classes)
    model.to(device)

    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    state_dict = clean_state_dict_keys(state_dict)

    # 🔧 Fix key prefix mismatch: add "backbone." if not present
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("backbone."):
            new_state_dict["backbone." + k] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict

    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ Checkpoint loaded successfully (strict mode).")
    except Exception as e:
        print("⚠️ Strict load_state_dict failed:", e)
        res = model.load_state_dict(state_dict, strict=False)
        print("Non-strict load_state_dict result:", res)

    model.to(DEVICE)
    model.eval()
    return model

def extract_sample_embedding(model: nn.Module, image_path: str, device: torch.device):
    img = Image.open(image_path).convert("RGB")
    img_resized = lowres(img)
    tensor = preprocess(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = model(tensor, return_features=True)  # Tensor (1, D)
    return feats.cpu().numpy(), img_resized


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


if __name__ == "__main__":
    if os.path.exists(str(Path(__file__).absolute().parent) + "/np_cache/embeddings.npy"):
        embeddings = np.load("np_cache/embeddings.npy")
        labels = np.load("np_cache/labels.npy")
        print("loaded data via save at /np_cache")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ckpt_path = "D:/resnet50-finetuned_raw/resnet50_streetview.pth"
    model = load_model_checkpoint(ckpt_path, device=device, num_classes=56)
    sample_imgs = []

    for i in enumerate(imgs, 0):
        emb, img = extract_sample_embedding(model, image_path=imgs[int(i[0])], device=device)
        sample_imgs.append(img)
    predict_image(samples=sample_imgs, model=model)

    # project_and_plot(embs=embeddings, sample_emb=sample_emb, id2label_map=id2label_map, labels=labels)