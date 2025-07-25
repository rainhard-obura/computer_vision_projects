#  Beyond Visible Spectrum: AI for Sustainable Agriculture

This repository contains the official code for the **"Beyond Visible Spectrum"** challenge submission, which explores the application of deep learning for **hyperspectral crop disease classification** and **synthetic hyperspectral data generation**. The solution is focused on early disease detection using hyperspectral UAV imagery spanning wavelengths from 450nm to 950nm.

---

##  Challenge Overview

The goal of the challenge is to harness hyperspectral imaging and artificial intelligence to tackle real-world agricultural problems, primarily:

### 1. **Crop Disease Classification**
- Classify hyperspectral image patches as *diseased* or *healthy*.
- Enable **early detection** by capturing spectral cues invisible to the naked eye.
- Improve precision agriculture by minimizing yield loss through timely interventions.

### 2. **Synthetic Hyperspectral Data Generation**
- Explore generative models (GANs, VAEs, diffusion models) to simulate realistic hyperspectral datasets.
- Address **data scarcity** by augmenting datasets with high-fidelity synthetic samples.
- Facilitate robust and generalizable model training across diverse crop conditions.

---

##  Dataset Details

- **Data Format**: `.npy` hyperspectral image patches (e.g. `sample1024.npy`)
- **Spectral Bands**: 100–125 bands per image (trimmed or padded to 100 in preprocessing)
- **Image Dimensions**: Usually `128x128` or `64x64`
- **Label**: Categorical integer label for disease class (0 = healthy)

The dataset is split into:
- `train.csv`: Contains `id` and `label` columns.
- Image files in: `/path/to/data/` directory, loaded as numpy arrays.

---

##  Model Architecture

We implement a **Spectral-Spatial Attention CNN** designed to extract both spectral and spatial features from hyperspectral imagery.

###  Core Components:
- **3 Convolutional Blocks** with BatchNorm and LeakyReLU
- **Channel Attention (CA)**: Highlights important spectral channels.
- **Spatial Attention (SA)**: Emphasizes critical regions in the image.
- **Adaptive Average Pooling**: Global feature aggregation.
- **Fully Connected Classifier**: Classifies into the target crop disease class.

###  Custom Attention Modules:
- **ChannelAttention** – Uses max and average pooled features.
- **SpatialAttention** – Learns spatial weights using convolutional operations.

---

##  Data Augmentation

We apply Kornia-based differentiable data augmentations:
- Random horizontal and vertical flips
- Random affine transformations
- Random cropping with padding

---

##  Training Configuration

| Parameter        | Value         |
|------------------|---------------|
| Epochs           | 50            |
| Batch Size       | 32            |
| Learning Rate    | 0.001         |
| Optimizer        | Adam          |
| Loss Function    | CrossEntropy  |
| Device           | CUDA/CPU      |

---

##  Training & Evaluation

The training pipeline includes:
- Train/Validation split (80/20)
- Loss and accuracy tracking across epochs
- Best model checkpointing based on validation loss
- Evaluation via softmax prediction and accuracy visualization

```bash
