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
Kopernik mostly requires a standart package of machine learning and image processing libraries for Python >= 3.12 (including CUDA-supporting version of PyTorch and collateral)
```
pip install torch torchvision scikit-learn datasets numpy tqdm pathlib
```
### Model setup
In order to use pretrained fine-tuned model of the latest developer version, download it via [Hugging Face](https://huggingface.co/flmxn/resnet50-streetview/blob/main/resnet50_streetview_imagenet1k.pth)

Otherwise, to train Kopernik with your own configuration, look into [torch_trainer.py](torch_trainer.py)

Before start, make sure to insert the path to your model and sample pictures in the inference config at [torch_main.py](torch_main.py)
