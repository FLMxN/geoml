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
from torch_main import load_model_checkpoint
import torchvision.transforms.functional as TF

warnings.filterwarnings('ignore')

#-------------------------------- CONFIG -----------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
do_loop = 0
target1_idx = 33
target2_idx = 35
#---------------------------------------------------------------------------

ckpt_path = "E://resnet50_streetview_imagenet1k.pth"
model, ckpt = load_model_checkpoint(ckpt_path, device=DEVICE, num_classes=56)

model.eval()
for p in model.parameters():
    p.requires_grad = False

img_raw = torch.randn(1, 3, 224, 224, device=DEVICE, requires_grad=True)
# target_idx = label2id["JP"]

def tv_loss(x):
    return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))


mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1,3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1,3,1,1)
transform = transforms.Compose([transforms.PILToTensor()])

optimizer = torch.optim.Adam([img_raw], lr=0.0001)

def exact_match_percentage(tensor1, tensor2):
    # Check if tensors have the same shape
    if tensor1.shape != tensor2.shape:
        return 0.0
    
    # Calculate percentage of exactly equal elements
    equal_elements = torch.sum(tensor1 == tensor2).item()
    total_elements = tensor1.numel()
    percentage = (equal_elements / total_elements) * 100
    return percentage

def get_visual(targets):
    for target_idx in targets:
        for step in range(15000):
            optimizer.zero_grad()
            img_norm = (img_raw - mean) / std

            logits, _ = model(img_norm)
            target_logit = logits[0, target_idx]

            color_loss = torch.mean((img_raw.mean(dim=(2,3)) - mean.squeeze())**2)
            loss = -target_logit + 1e-2 * tv_loss(img_raw) + 1e-1 * color_loss

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                img_raw.clamp_(0, 1)

            if step % 10 == 0:
                print(f"step {step}, logit {target_logit.item():.2f}")
                with torch.no_grad():
                    img_raw.data = TF.gaussian_blur(
                            img_raw.data,
                            kernel_size=[5, 5],
                            sigma=[0.5, 0.5]
                        )
            
        result = img_raw.detach().cpu().squeeze().permute(1,2,0).numpy()
        result = np.clip(result, 0.0, 1.0)
        vis = (result - result.min()) / (result.max() - result.min() + 1e-8)
        vis = (vis * 255).astype(np.uint8)
        Image.fromarray(vis).save(f"visualizer/features_{target_idx}.png")

try:
    pic1 = transform(Image.open(f'visualizer/features_{target1_idx}.png'))
except:
    do_loop += 1
try:
    pic2 = transform(Image.open(f'visualizer/features_{target2_idx}.png'))
except:
    do_loop += 2

match do_loop:
    case 1:
        get_visual(target1_idx)
    case 2:
        get_visual(target2_idx)
    case 3:
        get_visual([target1_idx, target2_idx])
    case _: 
        pass

pic1 = transform(Image.open(f'visualizer/features_{target1_idx}.png'))
pic2 = transform(Image.open(f'visualizer/features_{target2_idx}.png'))
print(f"Exact match: {exact_match_percentage(pic1, pic2):.2f}%")
