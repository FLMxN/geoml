import torch
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoConfig
from PIL import Image
import PIL
import os
import torch.nn.functional as F

HEIGHT = 561
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_image(model, processor, image, top_k=5, device=DEVICE):
    # try:
        new_height = HEIGHT
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        new_width = int(round(new_height * aspect_ratio))
        resized_img = image.crop((int((orig_width-new_width)/2), 0, int((new_width/2)+new_width), 561))

        inputs = processor(resized_img, return_tensors="pt") #FIX THIS!!!!!!!!!!!!
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = model(**inputs)
        
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        print(f"\n🔍 Predictions:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            label = model.config.id2label[idx]
            confidence = prob * 100
            print(f"  {i+1}. {label}: {prob:.4f} ({confidence:.2f}%)")
        
        top_label = model.config.id2label[top_indices[0]]
        top_confidence = top_probs[0]
        
        return top_label, top_confidence
        
    # except Exception as e:
    #     print(f"❌ Error processing image: {e}")
    #     return None, None