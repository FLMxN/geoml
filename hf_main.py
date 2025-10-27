from safetensors.torch import load_file
from transformers import AutoModel, AutoFeatureExtractor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import torch
import datasets
from datasets import Dataset, load_dataset
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
from torchvision import transforms, models
from umap import UMAP
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys
from pathlib import Path
import os
from launcher import predict
import pandas as pd
import plotly.express as px
from tqdm import tqdm

IMG = "pics/image.png"

FOV = 75
HORIZONTAL = 0
VERTICAL = 0
HEIGHT = 561
WIDTH = 997

sys.setrecursionlimit(10000)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

dataset = load_dataset("stochastic/random_streetview_images_pano_v0.0.2")
dataset = dataset.cast_column("image", datasets.Image())

model = AutoModel.from_pretrained("D:/resnet50-finetuned")
model.eval()

try:
    feature_extractor = AutoFeatureExtractor.from_pretrained("D:/resnet50-finetuned")
    print("feature extractor found")
except:
    feature_extractor = None
    print("no feature extractor found, using manual preprocessing")

embeddings = []
labels = []

preprocess = transforms.Compose([
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

iso_alpha2_to_country = {
    "AD": "Andorra",
    "AE": "United Arab Emirates",
    "AF": "Afghanistan",
    "AG": "Antigua and Barbuda",
    "AI": "Anguilla",
    "AL": "Albania",
    "AM": "Armenia",
    "AO": "Angola",
    "AQ": "Antarctica",
    "AR": "Argentina",
    "AS": "American Samoa",
    "AT": "Austria",
    "AU": "Australia",
    "AW": "Aruba",
    "AX": "Åland Islands",
    "AZ": "Azerbaijan",
    "BA": "Bosnia and Herzegovina",
    "BB": "Barbados",
    "BD": "Bangladesh",
    "BE": "Belgium",
    "BF": "Burkina Faso",
    "BG": "Bulgaria",
    "BH": "Bahrain",
    "BI": "Burundi",
    "BJ": "Benin",
    "BL": "Saint Barthélemy",
    "BM": "Bermuda",
    "BN": "Brunei Darussalam",
    "BO": "Bolivia (Plurinational State of)",
    "BQ": "Caribbean Netherlands",
    "BR": "Brazil",
    "BS": "The Bahamas",
    "BT": "Bhutan",
    "BV": "Bouvet Island",
    "BW": "Botswana",
    "BY": "Belarus",
    "BZ": "Belize",
    "CA": "Canada",
    "CC": "Cocos (Keeling) Islands",
    "CD": "Democratic Republic of the Congo",
    "CF": "Central African Republic",
    "CG": "Republic of the Congo",
    "CH": "Switzerland",
    "CI": "Ivory Coast",
    "CK": "Cook Islands",
    "CL": "Chile",
    "CM": "Cameroon",
    "CN": "People's Republic of China",
    "CO": "Colombia",
    "CR": "Costa Rica",
    "CU": "Cuba",
    "CV": "Cape Verde",
    "CW": "Curaçao",
    "CX": "Christmas Island",
    "CY": "Cyprus",
    "CZ": "Czech Republic",
    "DE": "Germany",
    "DJ": "Djibouti",
    "DK": "Denmark",
    "DM": "Dominica",
    "DO": "Dominican Republic",
    "DZ": "Algeria",
    "EC": "Ecuador",
    "EE": "Estonia",
    "EG": "Egypt",
    "EH": "Western Sahara",
    "ER": "Eritrea",
    "ES": "Spain",
    "ET": "Ethiopia",
    "FI": "Finland",
    "FJ": "Fiji",
    "FM": "Federated States of Micronesia",
    "FO": "Faroe Islands",
    "FR": "France",
    "GA": "Gabon",
    "GB": "United Kingdom",
    "GD": "Grenada",
    "GE": "Georgia",
    "GF": "French Guiana",
    "GG": "Guernsey",
    "GH": "Ghana",
    "GI": "Gibraltar",
    "GL": "Greenland",
    "GM": "The Gambia",
    "GN": "Guinea",
    "GP": "Guadeloupe",
    "GQ": "Equatorial Guinea",
    "GR": "Greece",
    "GS": "South Georgia and the South Sandwich Islands",
    "GT": "Guatemala",
    "GU": "Guam",
    "GW": "Guinea-Bissau",
    "GY": "Guyana",
    "HK": "Hong Kong",
    "HM": "Heard Island and McDonald Islands",
    "HN": "Honduras",
    "HR": "Croatia",
    "HT": "Haiti",
    "HU": "Hungary",
    "ID": "Indonesia",
    "IE": "Ireland",
    "IL": "Israel",
    "IM": "Isle of Man",
    "IN": "India",
    "IO": "British Indian Ocean Territory",
    "IQ": "Iraq",
    "IR": "Islamic Republic of Iran",
    "IS": "Iceland",
    "IT": "Italy",
    "JE": "Jersey",
    "JM": "Jamaica",
    "JO": "Jordan",
    "JP": "Japan",
    "KE": "Kenya",
    "KG": "Kyrgyzstan",
    "KH": "Cambodia",
    "KI": "Kiribati",
    "KM": "Comoros",
    "KN": "Saint Kitts and Nevis",
    "KP": "Democratic People's Republic of Korea",
    "KR": "Republic of Korea",
    "KW": "Kuwait",
    "KY": "Cayman Islands",
    "KZ": "Kazakhstan",
    "LA": "Lao People's Democratic Republic",
    "LB": "Lebanon",
    "LC": "Saint Lucia",
    "LI": "Liechtenstein",
    "LK": "Sri Lanka",
    "LR": "Liberia",
    "LS": "Lesotho",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "LV": "Latvia",
    "LY": "Libya",
    "MA": "Morocco",
    "MC": "Monaco",
    "MD": "Moldova (Republic of)",
    "ME": "Montenegro",
    "MF": "Saint Martin (French part)",
    "MG": "Madagascar",
    "MH": "Marshall Islands",
    "MK": "North Macedonia",
    "ML": "Mali",
    "MM": "Myanmar",
    "MN": "Mongolia",
    "MO": "Macao",
    "MP": "Northern Mariana Islands",
    "MQ": "Martinique",
    "MR": "Mauritania",
    "MS": "Montserrat",
    "MT": "Malta",
    "MU": "Mauritius",
    "MV": "Maldives",
    "MW": "Malawi",
    "MX": "Mexico",
    "MY": "Malaysia",
    "MZ": "Mozambique",
    "NA": "Namibia",
    "NC": "New Caledonia",
    "NE": "Niger",
    "NF": "Norfolk Island",
    "NG": "Nigeria",
    "NI": "Nicaragua",
    "NL": "Netherlands",
    "NO": "Norway",
    "NP": "Nepal",
    "NR": "Nauru",
    "NU": "Niue",
    "NZ": "New Zealand",
    "OM": "Oman",
    "PA": "Panama",
    "PE": "Peru",
    "PF": "French Polynesia",
    "PG": "Papua New Guinea",
    "PH": "Philippines",
    "PK": "Pakistan",
    "PL": "Poland",
    "PM": "Saint Pierre and Miquelon",
    "PN": "Pitcairn",
    "PR": "Puerto Rico",
    "PT": "Portugal",
    "PW": "Palau",
    "PY": "Paraguay",
    "QA": "Qatar",
    "RE": "Réunion",
    "RO": "Romania",
    "RS": "Serbia",
    "RU": "Russian Federation",
    "RW": "Rwanda",
    "SA": "Saudi Arabia",
    "SB": "Solomon Islands",
    "SC": "Seychelles",
    "SD": "Sudan",
    "SE": "Sweden",
    "SG": "Singapore",
    "SH": "Saint Helena, Ascension and Tristan da Cunha",
    "SI": "Slovenia",
    "SJ": "Svalbard and Jan Mayen",
    "SK": "Slovakia",
    "SL": "Sierra Leone",
    "SM": "San Marino",
    "SN": "Senegal",
    "SO": "Somalia",
    "SR": "Suriname",
    "SS": "South Sudan",
    "ST": "Sao Tome and Principe",
    "SV": "El Salvador",
    "SX": "Sint Maarten (Dutch part)",
    "SY": "Syrian Arab Republic",
    "SZ": "Eswatini",
    "TC": "Turks and Caicos Islands",
    "TD": "Chad",
    "TF": "French Southern and Antarctic Lands",
    "TG": "Togo",
    "TH": "Thailand",
    "TJ": "Tajikistan",
    "TK": "Tokelau",
    "TL": "Timor-Leste",
    "TM": "Turkmenistan",
    "TN": "Tunisia",
    "TO": "Tonga",
    "TR": "Turkey",
    "TT": "Trinidad and Tobago",
    "TV": "Tuvalu",
    "TZ": "Tanzania (United Republic of)",
    "TW": "Taiwan",
    "UA": "Ukraine",
    "UG": "Uganda",
    "UM": "United States Minor Outlying Islands",
    "US": "United States",
    "UY": "Uruguay",
    "UZ": "Uzbekistan",
    "VA": "Holy See",
    "VC": "Saint Vincent and the Grenadines",
    "VE": "Venezuela (Bolivarian Republic of)",
    "VG": "British Virgin Islands",
    "VI": "United States Virgin Islands",
    "VN": "Viet Nam",
    "VU": "Vanuatu",
    "WF": "Wallis and Futuna",
    "WS": "Samoa",
    "YE": "Yemen",
    "YT": "Mayotte",
    "ZA": "South Africa",
    "ZM": "Zambia",
    "ZW": "Zimbabwe",
}

def lowres(img, target_height=561):
    orig_width, orig_height = img.size
    aspect_ratio = orig_width / orig_height
    new_width = int(target_height * aspect_ratio)
    resized_img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
    return resized_img
    
def collate_fn(batch):
    images = []
    country_labels = []
    for row in batch:
        img = row['image'].crop((1017, 0, 2033, 561)).convert('RGB')
        img_tensor = preprocess(img)
        images.append(img_tensor)
        country_labels.append(row['country_iso_alpha2'])
    
    numeric_labels = [label2id_map[label] for label in country_labels]
    return {"pixel_values": torch.stack(images), "labels": torch.tensor(numeric_labels)}

if os.path.exists(str(Path(__file__).absolute().parent) + "/np_cache/embeddings.npy"):
    embeddings = np.load("np_cache/embeddings.npy")
    labels = np.load("np_cache/labels.npy")
    print("loaded data via save at /np_cache")
else:
    dataloader = DataLoader(dataset['train'], batch_size=16, shuffle=False, collate_fn=collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):           
            pixel_values = batch['pixel_values'].to(device)
            region_labels = batch['labels']           
            try:
                outputs = model(pixel_values)
                feats = outputs.pooler_output
                embeddings.append(feats.cpu())
                labels.append(region_labels)
            except:
                print('lulz')
    
        embeddings = torch.cat(embeddings).numpy()
        labels = torch.cat(labels).numpy()

    np.save("np_cache/embeddings.npy", embeddings)
    np.save("np_cache/labels.npy", labels)

embeddings = embeddings.squeeze()

df = pd.DataFrame(embeddings)
df['label'] = labels

centroids = df.groupby('label').mean().to_numpy()
classes = df['label'].unique()

if IMG != None:
    print('preprocessing sample...')
    sample_raw = lowres(Image.open(IMG).convert('RGB'))
    sample = preprocess(sample_raw).unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        pixel_value = sample.to(device)
        try:
            outputs = model(pixel_value)
            feats = outputs.pooler_output
            sample_emb = feats.cpu().numpy().reshape(1, -1)
        except:
            print('lulz2')

all_points = np.concatenate([centroids, sample_emb], axis=0)
scaled = StandardScaler().fit_transform(all_points)

umap_2d = UMAP(n_components=2, n_neighbors=5, metric='euclidean', random_state=42).fit_transform(scaled)
umap_3d = UMAP(n_components=3, n_neighbors=5, metric='euclidean', random_state=42).fit_transform(scaled)

plane_centroids = umap_2d[:-1]
plane_sample = umap_2d[-1]

proj_centroids = umap_3d[:-1]
proj_sample = umap_3d[-1]

def volume_plot():
    coord = pd.DataFrame(proj_centroids, columns=['x','y','z'])
    coord['label'] = list(map(lambda x: iso_alpha2_to_country[id2label_map[x]], classes))
    coord.loc[len(coord)] = [proj_sample[0], proj_sample[1], proj_sample[2], 'X']
    fig = px.scatter_3d(coord, x='x', y='y', z='z', color='label', title="3D embedding projection", text=coord['label'])
    fig.show()

def plane_plot():
    plt.figure(figsize=(16, 8))
    scatter = plt.scatter(
        plane_centroids[:, 0],
        plane_centroids[:, 1],
        c=classes,
        cmap='tab20',
        s=5)
    
    for i, lbl in enumerate(classes):
        country = id2label_map.get(int(lbl), str(lbl))
        plt.text(
            plane_centroids[i, 0],
            plane_centroids[i, 1],
            country,
            fontsize=12,
            weight='bold',
            ha='center',
            va='center')
        
    plt.scatter(
        plane_sample[0],
        plane_sample[1],
        c='red',
        s=60,
        edgecolor='black',
        marker='X')
    plt.tight_layout()
    plt.show()

def distance():
    closest_euc_idx = np.argmin(euclidean_distances(sample_emb, centroids))
    print("closest match (via euclidean):", id2label_map[int(classes[closest_euc_idx])] + " // " + iso_alpha2_to_country[id2label_map[int(classes[closest_euc_idx])]] + " <-- #1 METRIC")
    closest_cos_idx = np.argmin(cosine_distances(sample_emb, centroids))
    print("closest match (via cosine):", id2label_map[int(classes[closest_cos_idx])] + " // " + iso_alpha2_to_country[id2label_map[int(classes[closest_cos_idx])]] + " <-- #2 METRIC")

predict(IMG=sample_raw)
distance()
plane_plot()
input("Press any key to make 3D plot... <-- #3 METRIC")
volume_plot()
