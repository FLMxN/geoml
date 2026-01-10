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

### Dataset
Dataset used for training the [developer model](#model-setup) consists of <ins>**approximately 11.5k panorama images of 56 countries**</ins> (around 175 panoramas for each class). Dataset itself and additional information can be found on [Hugging Face](https://huggingface.co/datasets/stochastic/random_streetview_images_pano_v0.0.2)

**Credits**: [stochastic](https://huggingface.co/stochastic) (Winson Truong)

## Installation
### Dependencies
Kopernik mostly requires a standart package of machine learning and image processing libraries for Python >= 3.12 (including CUDA-supporting version of PyTorch and collateral) as well as CUDA-supporting GPU for training and/or inference.
```
pip install torch torchvision scikit-learn datasets numpy tqdm pathlib dotenv
```
### Model setup
In order to use pretrained fine-tuned model of the latest <ins>**developer version**</ins>, download it via [Hugging Face](https://huggingface.co/flmxn/resnet50-streetview/blob/main/resnet50_streetview_imagenet1k.pth)

Otherwise, to train Kopernik with your own configuration, look into [torch_trainer.py](torch_trainer.py)

Before start, make sure to confugire the path to your model in [.env](.env) file and sample pictures in the inference config at [torch_main.py](torch_main.py)

## Understanding predictions and labels
> [!NOTE]
> This section is constantly improving and getting new updates, based on tests and feedback.

> [!IMPORTANT]
> When working with a mathematical model, it is a common mistake to take it's predictions as-is. It is crucial to understand the meaning behind each label in a particular model architecture. [Developer model](#model-setup) is far from flawless, just as any other mathematical predictor of such nature.

To fairly estimate and interpret predictions of [this particular](#model-setup) model, let's categorize labels inside it's computing space.
+ <ins>**Null point attractors:**</ins> **Poland** (PL), **New Zealand** (NZ), **Bhutan** (BT), **Eswatini** (SZ), **Cambodia** (KH) and **Argentina** (AR). These labels usually represent particular ambiguity of image features. Results, containing these labels as *top_k* are <ins>*unreliable*</ins> and serve no straightforward meaning due to lack of images' informative features.

> [!TIP]
> If you want to group <ins>**null point attractors**</ins> as **UNDEFINED** when listing predictions, check out [this](#startup)
  
+ <ins>**Common attractors:**</ins>
  - **Taiwan** (TW) and **South Korea** (KR) labels represent features of developed mostly Asian countries, although certain European or American landscapes can fit into this space as well. More often than not represents large urban centres of China, India, Japan, Australia, USA and Russia.
  - **Czech Republic** (CZ) label represents features of mostly post-soviet countries, although certain Eastern European landscapes can fit into this space as well. More often than not represents rural or small urban centres of Russia, Ukraine, Belarus, Poland, Estonia, Latvia and Lithuania.
  - **Finland** (FI) label represents features of mostly northern countries, although certain European landscapes can fit into this space as well. More often than not represents northern medium urban centres of Norway, Sweden, Finland, Denmark, Iceland and Russia.
  - **Spain** (ES) label represents features of mostly European countries, although certain American landscapes can fit into this space as well. More often than not represents urban centres of France, Spain, Portugal and Italy.

+ <ins>**Particular attractors:**</ins>
  - **Botswana** (BW) label may represent features of rural or small urban centres of Japan.
  - **Mexico** (MX) label may represent features of rural or small urban centres of South America.
  - **Romania** (RO) label may represent features of highways or other kind of purely road images.
  - **Canada** (CA) label may represent features of neighbourhoods in USA.

## Inference
### Startup
If you completed [model setup](#model-setup) properly, inference requires no any additional preparations.
```
python torch_main.py
```
> [!TIP]
> Default output mode is verbose. In order to hide debug logs and display more user-experience information, set **IS_PRETTY** in the inference config to <ins>*True*</ins> or pass additional argument when starting inference from shell:
> ```
> python torch_main.py pretty
> ``` 

### Verbose output digest
> [!TIP]
> Make sure to read [this](#understanding-predictions-and-labels) before getting into predictions provided below.
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
Debug data, optional. [Developer model](#model-setup) was trained through 56 epochs using CUDA and reached validation accuracy of 47% among 56 countries.

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
(0, <PIL.Image.Image image mode=RGB size=988x561 at 0x2A2733D63C0>) is loaded

(1, <PIL.Image.Image image mode=RGB size=997x561 at 0x2A27CA29FA0>) is loaded
```
[Developer model](#model-setup) has loaded and processed [t2.png](pics/t2.png) with 2 different scaling strategies (stretch and crop).

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
Features of [t2.png](pics/t2.png) seem to mostly represent features of **Germany** (DE) and **Czech Republic** (CZ), followed up by **Slovenia** (SI), **Portugal** (PT) and **Andorra** (AD). All labels respect the naming standart of [**ISO 3166-1 alpha-2**](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).

## Feature visualising
> [!NOTE]
> This section is not complete and requires extension.

In **debug** or **research purposes**, Kopernik is able to get a mean image of class features among all it's examples in the dataset. In order to do this, configure the [torch_visualizer.py](torch_visualizer.py) and run to get 2 images of chosen classes and **compare** them in % of similarity.
