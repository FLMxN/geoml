from safetensors.torch import load_file
from transformers import AutoModel, AutoFeatureExtractor
import torch
import datasets
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from umap import UMAP
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys
from pathlib import Path
import os

sys.setrecursionlimit(10000)

dataset = load_dataset("stochastic/random_streetview_images_pano_v0.0.2")
dataset = dataset.cast_column("image", datasets.Image())

print(dataset)

model = AutoModel.from_pretrained("D:/resnet50-finetuned")
model.eval()

method_list = [method for method in dir(model) if 'features' in method.lower()]
print("feature methods:", method_list)

try:
    feature_extractor = AutoFeatureExtractor.from_pretrained("D:/resnet50-finetuned")
    print("Feature extractor found")
except:
    feature_extractor = None
    print("No feature extractor found, using manual preprocessing")

embeddings = []
labels = []

preprocess = transforms.Compose([
    transforms.Resize((531, 531)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

label2id_map = {
    "AD": 7, "AE": 16, "AR": 15, "AU": 43, "BD": 30,
    "BE": 26, "BG": 4, "BR": 46, "BT": 48, "BW": 31,
    "CA": 3, "CH": 49, "CL": 34, "CO": 17, "CZ": 45,
    "DE": 25, "DK": 36, "EE": 12, "ES": 41, "FI": 23,
    "FR": 28, "GB": 0, "GR": 53, "HK": 22, "HR": 24,
    "HU": 14, "ID": 42, "IE": 13, "IL": 51, "IS": 6,
    "IT": 27, "JP": 35, "KH": 10, "KR": 1, "LT": 32,
    "LV": 50, "MX": 29, "MY": 9, "NL": 2, "NO": 5,
    "NZ": 37, "PE": 44, "PL": 47, "PT": 21, "RO": 38,
    "RU": 52, "SE": 40, "SG": 19, "SI": 55, "SK": 8,
    "SZ": 11, "TH": 18, "TW": 33, "UA": 39, "US": 54,
    "ZA": 20
}

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


def collate_fn(batch):
    images = []
    country_labels = []
    for row in batch:
        print(row, type(row))
        img = row['image']
        img_tensor = preprocess(img)
        images.append(img_tensor)
        country_labels.append(row['country_iso_alpha2'])
    
    numeric_labels = [label2id_map[label] for label in country_labels]
    return {"pixel_values": torch.stack(images), "labels": torch.tensor(numeric_labels)}

if os.path.exists(str(Path(__file__).absolute().parent) + "/np_cache/embeddings.npy"):
    embeddings = np.load("embeddings.npy")
    labels = np.load("labels.npy")
else:
    dataloader = DataLoader(dataset['train'], batch_size=16, shuffle=False, collate_fn=collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        for x, batch in enumerate(dataloader, 1):
            print(f'started iteration {x}')            
            pixel_values = batch['pixel_values'].to(device)
            region_labels = batch['labels']
                
            try:
                outputs = model(pixel_values)
                if hasattr(outputs, 'last_hidden_state'):
                    feats = outputs.last_hidden_state.mean(dim=1)
                else:
                    feats = outputs.pooler_output
            except Exception as e:
                print(f"error extracting features: {e}")
                continue
                
            embeddings.append(feats.cpu())
            labels.append(region_labels)
    
    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()

    np.save("embeddings.npy", embeddings)
    np.save("labels.npy", labels)

if embeddings:
    embeddings_2d = embeddings.reshape(embeddings.shape[0], -1)

    umap = UMAP(n_neighbors=30, min_dist=0.25, metric='cosine', random_state=42, init='spectral')
    emb_umap = umap.fit_transform(embeddings_2d)

    unique_labels = np.unique(labels)
    centroids = np.zeros((len(unique_labels), 2))

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        centroids[i] = emb_umap[mask].mean(axis=0)

    plt.figure(figsize=(16, 8))
    scatter = plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c=unique_labels,
        cmap='tab20',
        s=5,
        alpha=0.7
    )

    for i, lbl in enumerate(unique_labels):
        country = id2label_map.get(int(lbl), str(lbl))
        plt.text(
            centroids[i, 0],
            centroids[i, 1],
            country,
            fontsize=12,
            weight='bold',
            ha='center',
            va='center'
        )

    plt.title("UMAP projection of Street View embeddings")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()

else:
    print("No embeddings were generated. Check feature extraction method.")