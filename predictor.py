from PIL import Image
import torch
import torch.nn.functional as F

iso_alpha2_to_country = {
    "US": "United States",
    "CA": "Canada",
    "GB": "United Kingdom",
    "FR": "France",
    "DE": "Germany",
    "JP": "Japan",
    "CN": "China",
    "IN": "India",
    "BR": "Brazil",
    "RU": "Russia",
    "AU": "Australia",
    "IT": "Italy",
    "ES": "Spain",
    "MX": "Mexico",
    "ZA": "South Africa",
    "KR": "South Korea",
    "NG": "Nigeria",
    "AR": "Argentina",
    "SE": "Sweden",
    "CH": "Switzerland",
    "NL": "Netherlands",
    "BE": "Belgium",
    "NO": "Norway",
    "DK": "Denmark",
    "FI": "Finland",
    "PL": "Poland",
    "PT": "Portugal",
    "GR": "Greece",
    "TR": "Turkey",
    "EG": "Egypt",
    "SA": "Saudi Arabia",
    "AE": "United Arab Emirates",
    "IL": "Israel",
    "TH": "Thailand",
    "MY": "Malaysia",
    "SG": "Singapore",
    "NZ": "New Zealand",
    "IE": "Ireland",
    "AT": "Austria",
    "HU": "Hungary",
    "CZ": "Czech Republic",
    "RO": "Romania",
    "BG": "Bulgaria",
    "HR": "Croatia",
    "SI": "Slovenia",
    "SK": "Slovakia",
    "PH": "Philippines",
    "VN": "Vietnam",
    "ID": "Indonesia",
    "PK": "Pakistan",
    "BD": "Bangladesh",
}


class DeterminedPredictor:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        
    def predict(self, image_path, temperature, use_tta, min_confidence):
        try:
            image = Image.open(image_path).convert('RGB')
            
            if use_tta:
                predictions = []
                inputs = self.processor(image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions.append(outputs.logits)
                
                flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
                inputs = self.processor(flipped, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions.append(outputs.logits)
                
                avg_logits = torch.mean(torch.stack(predictions), dim=0)
                probs = F.softmax(avg_logits / temperature, dim=-1)
            else:
                inputs = self.processor(image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = F.softmax(outputs.logits / temperature, dim=-1)
            
            probs = probs.cpu().numpy()[0]
            
            confident_predictions = []
            for idx, prob in enumerate(probs):
                if prob >= min_confidence:
                    country = self.model.config.id2label[idx]
                    confident_predictions.append((country, prob, idx))
            
            confident_predictions.sort(key=lambda x: x[1], reverse=True)
            
            print(f"prediction:")
            print(f"   config: TTA={use_tta}, Temperature={temperature}, Min Confidence={min_confidence*100:.1f}%")
            print(f"   value:")
            
            if confident_predictions:
                for i, (country, prob, idx) in enumerate(confident_predictions[:3]):
                    confidence = prob * 100
                    print(f"   {i+1}. {country} // {iso_alpha2_to_country[country]}: {prob:.4f} ({confidence:.1f}%)")
            else:
                top_idx = probs.argmax()
                top_country = self.model.config.id2label[top_idx]
                top_prob = probs[top_idx]
                print(f"   last resort: {top_country} ({top_prob*100:.1f}%)")
            
            return confident_predictions if confident_predictions else [(top_country, top_prob, top_idx)]
            
        except Exception as e:
            print(f"error: {e}")
            import traceback
            traceback.print_exc()
            return None
