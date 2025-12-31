import torch
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

pretrained = models.resnet50(weights='IMAGENET1K_V1')
pretrained.eval()
first_layer_weights = pretrained.conv1.weight.data.cpu()
print(f"First layer shape: {first_layer_weights.shape}")  # [64, 3, 7, 7]

fig, axes = plt.subplots(8, 8, figsize=(12, 12))
for i in range(64):
    ax = axes[i//8, i%8]
    # Average across RGB channels for visualization
    filter_img = first_layer_weights[i].mean(0)
    ax.imshow(filter_img, cmap='gray')
    ax.axis('off')
plt.suptitle("ImageNet Learned Filters (Edge Detectors)")
plt.show()