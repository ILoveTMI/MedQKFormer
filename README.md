<div align="center">
<h1> MedQKFormer: Spiking Transformer with Q-K Decomposed Attention for Medical Image Segmentation </h1>
</div>

## 🎈 News

[2025.7.31] **Training and testing code released <-- We are here !**



## ⭐ Abstract

Spiking Self-Attention (SSA) has shown potential in medical image segmentation due to its event-driven and energy-efficient nature. However, segmentation performance degrades in the presence of misleading co-occurrence between salient and non-salient objects, primarily because (1) existing spike attention mechanisms rely only on activated neurons, ignoring contextual cues from inactivated or low-spike-value neurons; and (2) conventional spiking neurons struggle to evaluate spatial feature importance. To overcome these limitations, we propose MedQKFormer, a spiking transformer featuring two core components: Spike-Decomposing Q-K Attention (SDQK-A) and Normalized Integer Spike-Fire Neurons (NISF). SDQK-A models three types of neuronal interactions—activated-activated, activated-inactivated, and inactivated-inactivated—enabling richer contextual representation. NISF quantizes spike outputs into normalized integers, enhancing spatial discriminability while naturally improving training stability and preserving SNN energy efficiency. MedQKFormer achieves state-of-the-art segmentation performance with computational efficiency suitable for practical deployment.

## 🚀 Introduction

<div align="center">
    <img width="400" alt="image" src="figures/challenge.png?raw=true">
</div>

<div align="center">
The challenges: The misleading co-occurrence of salient and non-salient objects.
</div>

## 📻 Overview

<div align="center">
<img width="800" alt="image" src="figures/network.png?raw=true">
</div>

<div align="center">
Illustration of the overall architecture.
</div>


## 📆 TODO

- [x] Release code

## 🎮 Getting Started

### 1. Install Environment

```
conda create -n Net python=3.8
conda activate Net
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs PyWavelets
```

### 2. Prepare Datasets

#### 2.1 Datasets  
Our method is evaluated on 5 public datasets of different modalities, including ISIC2018, Kvasir, BUSI, Monu-Seg, and COVID-19.  

- **ISIC2018**: A relatively large dataset for skin cancer detection, containing 2,594 skin lesion images. It is split into 2,076 training images and 518 testing images.  ([link](https://challenge.isic-archive.com/data/#2018))
- **Kvasir**: Focuses on pixel-level segmentation of colorectal polyps, with 1,000 endoscopic images. The split is 800 training images and 200 testing images.  ([link](https://challenge.isic-archive.com/data/#2018))
- **BUSI**: A breast ultrasound imaging dataset categorized into three classes (normal, benign, malignant), comprising 780 images in total (624 for training, 156 for testing).  ([link](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset))
- **Monu-Seg**: A medical imaging dataset for cell nucleus segmentation, containing 74 images (59 for training, 15 for testing).  ([link](https://www.kaggle.com/datasets/tuanledinh/monuseg2018))
- **COVID-19**: Includes 894 images for segmenting lung infection regions in CT scans, split into 716 training images and 178 testing images. ([link](https://drive.usercontent.google.com/download?id=1FHx0Cqkq9iYjEMN3Ldm9FnZ4Vr1u3p-j&export=download&authuser=0))

- Folder organization: put datasets into ./data folder.

#### 2.2 Implementation Details  
We implement the method using PyTorch, with experiments conducted on an NVIDIA TITAN RTX GPU. Key parameters are as follows:  
- **Optimizer**: AdamW  
- **Learning Rate Scheduler**: CosineAnnealingLR  
- **Input Size**: Resized to 256 × 256  
- **Data Augmentation**: Horizontal flipping, vertical flipping, and rotation (to enhance model robustness)  
- **Training Epochs**: 200  
- **Initial Learning Rate**: 1e-4  
- **Batch Size**: 12  
- **Random Seed**: 41

### 3. Train the Net

```
python train.py --datasets ISIC2018
concrete information see train.py, please
```

### 3. Test the Net

```
python test.py --datasets ISIC2018
concrete information see test.py, please
```

### 3. Code example

```
python Test/example.py
```

## ⭐ Visualization

<div align="center">
<img width="800" alt="image" src="figures/com_pic.png?raw=true">
</div>

<div align="center">
We compare our method against 14 state-of-the-art methods. The red box indicates the area of incorrect predictions.
</div>

## ✨ Quantitative comparison

<div align="center">
<img width="800" alt="image" src="figures/com_tab.png?raw=true">
</div>

<div align="center">
Performance comparison with 14 SOTA methods on ISIC2018, Kvasir, BUSI, COVID-19 and Monu-Seg datasets.
</div>


## 🖼️ Visualization of Ablation Results

<div align="center">
<img width="800" alt="image" src="figures/aba.png?raw=true">
</div>



## 🖼️ Convergence Analysis

<div align="center">
<img width="800" alt="image" src="figures/curve.png?raw=true">
</div>



## 🎫 License

The content of this project itself is licensed under [LICENSE](https://github.com/ILoveAAAI/MedQKFormer?tab=Apache-2.0-1-ov-file).
