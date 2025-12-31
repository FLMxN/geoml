import torch
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoConfig
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from numpy import mean

HEIGHT = 561
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def predict_image(model, samples, checkpoint, processor=None, top_k=3, device=DEVICE):

    state_score = {
    "AD":0, "AE":0, "AR":0, "AU":0,
    "BD":0, "BE":0, "BG":0, "BR":0, "BT":0,
    "BW":0, "CA":0, "CH":0, "CL":0, "CO":0,
    "CZ":0, "DE":0, "DK":0, "EE":0, "ES":0,
    "FI":0, "FR":0, "GB":0, "GR":0, "HK":0,
    "HR":0, "HU":0, "ID":0, "IE":0, "IL":0,
    "IS":0, "IT":0, "JP": 0, "KH":0, "KR":0,
    "LT":0, "LV":0, "MX":0, "MY":0, "NL":0,
    "NO":0, "NZ":0, "PE":0, "PL":0, "PT":0,
    "RO":0, "RU":0, "SE":0, "SG":0,
    "SI":0, "SK":0, "SZ":0, "TH":0, "TW":0,
    "UA":0, "US":0, "ZA":0
}
    eu_score = 0
    asia_score = 0
    ocean_score = 0
    na_score = 0
    sa_score = 0
    africa_score = 0

    longitudes = []
    latitudes = []

    for x in enumerate(samples, 0):
        resized_img = samples[x[0]]
        try:
            inputs = processor(resized_img, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except:
            inputs = preprocess(resized_img).unsqueeze(0)
            inputs = inputs.to(device)

        with torch.no_grad():
            try:
                outputs = model(**inputs)
            except:
                outputs = model(inputs)
        
        # Extract outputs based on model structure
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs.get('classification_logits'))
            longitude = outputs['longitude'].cpu().numpy()[0]
            latitude = outputs['latitude'].cpu().numpy()[0]
        elif isinstance(outputs, tuple):
            logits = outputs[0]
            if len(outputs) > 1:
                # Assuming outputs[1] contains coordinates
                coords = outputs[1].cpu().numpy()[0]
                if len(coords) == 2:
                    longitude, latitude = coords[0], coords[1]
        else:
            logits = outputs
        
        # Store coordinates if found
        if 'longitude' in locals() and 'latitude' in locals():
            print(f"\nCoordinates of {x}: {longitude}, {latitude}")
            longitudes.append(longitude)
            latitudes.append(latitude)
        
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]

        reg_top_probs, reg_top_indices = torch.topk(probabilities, 56)
        reg_top_probs = reg_top_probs.cpu().numpy()[0]
        reg_top_indices = reg_top_indices.cpu().numpy()[0]

        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            try:
                label = model.config.id2label[idx]
            except:
                label = model.id2label[idx]
            state_score[label] = state_score[label] + prob*100/len(samples)

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

    preds = dict(sorted(
    ((k, float(v)) for k, v in state_score.items() if v != 0),
    key=lambda x: x[1],
    reverse=True))

    print(f"\nCoordinates: {mean(longitudes)}, {mean(latitudes)}")

    print(f"\nRegional predictions:")
    print(f"    Europe: {eu_score*100/len(samples):.2f}")
    print(f"    Asia: {asia_score*100/len(samples):.2f}")
    print(f"    North America: {na_score*100/len(samples):.2f}")
    print(f"    South America: {sa_score*100/len(samples):.2f}")
    print(f"    Oceania: {ocean_score*100/len(samples):.2f}")
    print(f"    Africa: {africa_score*100/len(samples):.2f}")

    print(f"\nParticular predictions:")
    for y in preds:
        print(f"    {y}: {preds[y]:.2f}")

    
