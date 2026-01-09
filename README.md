# Kopernik - geography-adjacent ResNet-based CNN by Google Street View data
## Introduction
**Kopernik** is an open-source machine learning project, focused on predicting geographic data based on landscape pictures inside or outside of urbanity.

### License
*Kopernik*
Copyright (C) *2026* *FLMxN*

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

## Installation
### Dependencies
Kopernik mostly requires a standart package of machine learning and image processing libraries for Python >= 3.12 (including CUDA-supporting version of PyTorch and collateral) as well as CUDA-supporting GPU for training and/or inference.
```
pip install torch torchvision scikit-learn datasets numpy tqdm pathlib
```
### Model setup
In order to use pretrained fine-tuned model of the latest developer version, download it via [Hugging Face](https://huggingface.co/flmxn/resnet50-streetview/blob/main/resnet50_streetview_imagenet1k.pth)

Otherwise, to train Kopernik with your own configuration, look into [torch_trainer.py](torch_trainer.py)

Before start, make sure to insert the path to your model and sample pictures in the inference config at [torch_main.py](torch_main.py)

## Inference
### Startup
If you completed [model setup](#model-setup) properly, inference requires no any additional preparations --> launch [torch_main.py](torch_main.py) and voila!
### Output
***Disclaimer***: <ins>Please be aware that any coordinate-related output contains no meaningful information by the day of Jan 9, 2026</ins>

**Example ([pics/t2.png](pics/t2.png))**
```
Using device: cuda
Checkpoint structure:
  epoch: int = 56
  model_state_dict: dict with 322 keys
  optimizer_state_dict: dict with 2 keys
  val_acc: float = 0.47106690777576854
  val_coord_loss: float = 0.06438215609107699
  label_mapping: dict with 56 keys
  config: dict with 4 keys

✅ Extracted model_state_dict from checkpoint

📋 State dict keys (first 10):
  resnet.conv1.weight: shape torch.Size([64, 3, 7, 7])
  resnet.bn1.weight: shape torch.Size([64])
  resnet.bn1.bias: shape torch.Size([64])
  resnet.bn1.running_mean: shape torch.Size([64])
  resnet.bn1.running_var: shape torch.Size([64])
  resnet.bn1.num_batches_tracked: shape torch.Size([])
  resnet.layer1.0.conv1.weight: shape torch.Size([64, 64, 1, 1])
  resnet.layer1.0.bn1.weight: shape torch.Size([64])
  resnet.layer1.0.bn1.bias: shape torch.Size([64])
  resnet.layer1.0.bn1.running_mean: shape torch.Size([64])

✅ Multi-task checkpoint detected (both heads present)
✅ Checkpoint loaded successfully (strict mode)

Coordinates of (0, <PIL.Image.Image image mode=RGB size=988x561 at 0x167719A1CD0>): -0.09785331040620804, 0.3668176233768463

Coordinates of (1, <PIL.Image.Image image mode=RGB size=997x561 at 0x16768D72900>): -0.06961575150489807, 0.41150110960006714

Coordinates: -0.08373452723026276, 0.3891593813896179

Regional predictions:
    Europe: 98.67
    Asia: 0.87
    North America: 0.39
    South America: 0.07
    Oceania: 0.00
    Africa: 0.00

Particular predictions:
    DE: 49.78
    CZ: 31.73
    SI: 9.20
    PT: 5.97
    AD: 1.93
```
### Output interpretation
```
Using device: cuda
Checkpoint structure:
  epoch: int = 56
  model_state_dict: dict with 322 keys
  optimizer_state_dict: dict with 2 keys
  val_acc: float = 0.47106690777576854
  val_coord_loss: float = 0.06438215609107699
  label_mapping: dict with 56 keys
  config: dict with 4 keys
```
[Developer model](#model-setup) was trained through 56 epochs using CUDA and reached validation accuracy of 47% among 56 countries.

```
✅ Extracted model_state_dict from checkpoint

📋 State dict keys (first 10):
  resnet.conv1.weight: shape torch.Size([64, 3, 7, 7])
  resnet.bn1.weight: shape torch.Size([64])
  resnet.bn1.bias: shape torch.Size([64])
  resnet.bn1.running_mean: shape torch.Size([64])
  resnet.bn1.running_var: shape torch.Size([64])
  resnet.bn1.num_batches_tracked: shape torch.Size([])
  resnet.layer1.0.conv1.weight: shape torch.Size([64, 64, 1, 1])
  resnet.layer1.0.bn1.weight: shape torch.Size([64])
  resnet.layer1.0.bn1.bias: shape torch.Size([64])
  resnet.layer1.0.bn1.running_mean: shape torch.Size([64])

✅ Multi-task checkpoint detected (both heads present)
✅ Checkpoint loaded successfully (strict mode)
```
Debug data, optional. [Developer model](#model-setup) successfully loaded with no compromise, displaying first 10 keys of own state dictionary.

```
Coordinates of (0, <PIL.Image.Image image mode=RGB size=988x561 at 0x167719A1CD0>): -0.09785331040620804, 0.3668176233768463

Coordinates of (1, <PIL.Image.Image image mode=RGB size=997x561 at 0x16768D72900>): -0.06961575150489807, 0.41150110960006714

Coordinates: -0.08373452723026276, 0.3891593813896179
```
Effectively useless data by the date of Jan 09, 2026. [Developer model](#model-setup) has loaded and processed [t2.png](pics/t2.png) with 2 different scaling strategies (stretch and crop). Coordinate-related data serves no meaning (yet).

```
Regional predictions:
    Europe: 98.67
    Asia: 0.87
    North America: 0.39
    South America: 0.07
    Oceania: 0.00
    Africa: 0.00
```
Features of [t2.png](pics/t2.png) seem to mostly represent features of the European countries.

```
Particular predictions:
    DE: 49.78
    CZ: 31.73
    SI: 9.20
    PT: 5.97
    AD: 1.93
```
Features of [t2.png](pics/t2.png) seem to mostly represent features of Germany (DE) and Czech Republic (CZ), followed by Slovenia (SI), Portugal (PT) and Andorra (AD). All labels respect the naming standart of **ISO 3166-1 alpha-2**.


