import torch
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoConfig
from PIL import Image
import PIL
import os
import torch.nn.functional as F
from torchvision import transforms

HEIGHT = 561
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

regions = {
    "Europe": ["AD", "BE", "BG", "CH", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GB", "GR", "HR", "HU", "IE", "IS", "IT", "LT", "LV", "NL", "NO", "PL", "PT", "RO", "RU", "SE", "SI", "SK", "UA"],
    "Asia": ["AE", "BD", "BT", "HK", "ID", "IL", "JP", "KH", "KR", "MY", "SG", "TH", "TW"],
    "Oceania": ["AU", "NZ"],
    "North America": ["CA", "MX", "US"],
    "South America": ["AR", "BR", "CL", "CO", "PE"],
    "Africa": ["BW", "SZ", "ZA"]
}

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(model, resized_img: Image.Image, processor=None, top_k=5, device=DEVICE):
        try:
            inputs = processor(resized_img, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            print('HuggingFace AutoModel found')
        except:
            inputs = preprocess(resized_img).unsqueeze(0) #FIX THIS!!!!!!!!!!!!
            inputs = inputs.to(device)
            print('PyTorch model from checkpoint found')

        with torch.no_grad():
            try:
                logits = model(**inputs)
            except:
                logits = model(inputs)
        
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]

        reg_top_probs, reg_top_indices = torch.topk(probabilities, 56)
        reg_top_probs = reg_top_probs.cpu().numpy()[0]
        reg_top_indices = reg_top_indices.cpu().numpy()[0]
        
        if hasattr(model, 'config'):
            print("Using HuggingFace AutoConfig labels")
        else:
            print("Using PyTorch raw labels")

        print(f"\nParticular predictions:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            try:
                label = model.config.id2label[idx]
            except:
                label = model.id2label[idx]
            print(f"    {label}: {prob*100:.2f}")

        eu_score = 0
        asia_score = 0
        ocean_score = 0
        na_score = 0
        sa_score = 0
        africa_score = 0

        print(f"\nRegional predictions:")
        for i, (prob, idx) in enumerate(zip(reg_top_probs, reg_top_indices)):
            try:
                label = model.config.id2label[idx]
            except:
                label = model.id2label[idx]

            match label:
                case country if country in regions["Europe"]:
                    eu_score = eu_score + prob
                case country if country in regions["Asia"]:
                    asia_score = asia_score + prob
                case country if country in regions["North America"]:
                    na_score = na_score + prob
                case country if country in regions["South America"]:
                    sa_score = sa_score + prob
                case country if country in regions["Oceania"]:
                    ocean_score = ocean_score + prob
                case country if country in regions["Africa"]:
                    africa_score = africa_score + prob

        print(f"    Europe: {eu_score*100:.2f}")
        print(f"    Asia: {asia_score*100:.2f}")
        print(f"    North America: {na_score*100:.2f}")
        print(f"    South America: {sa_score*100:.2f}")
        print(f"    Oceania: {ocean_score*100:.2f}")
        print(f"    Africa: {africa_score*100:.2f}")
        
        