import torch
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoConfig
from PIL import Image
import PIL
import os
import torch.nn.functional as F
from predictor import DeterminedPredictor
from coordinates import CoordinatePredictor

TARGET_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_correctly(model_path, processor_path_name):
    config = AutoConfig.from_pretrained(model_path)
    # print(f"📋 Model should have: {config.num_labels} classes")
    # print(f"📋 Example labels: {list(config.id2label.values())[:10]}...")
    
    model = ResNetForImageClassification.from_pretrained(
        model_path,
        num_labels=config.num_labels,
        id2label=config.id2label,
        label2id=config.label2id,
        ignore_mismatched_sizes=True,
    )
    
    processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
    
    model = model.to(DEVICE)
    model.eval()
    
    # print(f"✅ Model loaded with: {model.config.num_labels} classes")
    # print(f"✅ Actual labels: {list(model.config.id2label.values())[:10]}...")
    # print(f"✅ Using processor from: {processor_path_name}")
    
    return model, processor, DEVICE

def predict_image(model, processor, device, image_path, top_k=5):
    try:
        image = Image.open(image_path).convert('RGB')

        width, height = image.size
        if width < height:
            new_width = TARGET_SIZE
            new_height = int(height * (new_width / width))
        else:
            new_height = TARGET_SIZE
            new_width = int(width * (new_height / height))


        resized_img = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        inputs = processor(resized_img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        print(f"\n🔍 Predictions for {os.path.basename(image_path)}:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            label = model.config.id2label[idx]
            confidence = prob * 100
            print(f"  {i+1}. {label}: {prob:.4f} ({confidence:.2f}%)")
        
        # Return the top prediction
        top_label = model.config.id2label[top_indices[0]]
        top_confidence = top_probs[0]
        
        return top_label, top_confidence
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None, None

# Main execution
def predict(IMG):
    model_path = "D:/resnet50-finetuned"
    processor_path = "D:/resnet50-finetuned"  # Original model for processor
    
    model, processor, device = load_model_correctly(model_path, processor_path)

    predictor = DeterminedPredictor(model, processor, device)
    # coord_predictor = CoordinatePredictor("D:/resnet50-finetuned")

    settings = [
        {"temperature": 0.1, "use_tta": True, "min_confidence": 0.01},
        {"temperature": 0.5, "use_tta": True, "min_confidence": 0.01},
    ]

    for i, setting in enumerate(settings, 1):
        print(f"{'='*60}")
        print(f"{'Volume' if i > 1 else 'Precision'}")
        predictor.predict(IMG, **setting)