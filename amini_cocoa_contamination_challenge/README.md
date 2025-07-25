#  Cocoa Disease Detection Challenge

##  Objective

The goal of this challenge was to develop an accurate and lightweight object detection model capable of identifying and classifying visible diseases in cocoa plants using images. The solution had to meet strict resource constraints to ensure deployment feasibility on low-resource smartphones.

>  Constraints included:
> - Training must be completed within 9 hours (T4 GPU)
> - Inference must complete within 3 hours
> - Model must be suitable for edge deployment (ONNX or TensorFlow Lite compatible)

---

##  Methodology

###  Data Handling

- **Dataset**: RGB images annotated with YOLO-style bounding boxes for three classes: `healthy`, `cssvd`, and `anthracnose`.
- **Class Balancing**: Addressed class imbalance via bounding-box-preserving augmentations using Albumentations.
- **Folds**: Used `StratifiedGroupKFold` to split the data into 10 folds while preserving class distribution and avoiding data leakage across the same cocoa plant image.

###  Training Pipeline

- **Model Used**: `YOLOv8s` (referred to as `yolo11s` in the original plan).
- **Input Size**: `800x800` pixels.
- **Augmentation**: Applied bounding-box-aware augmentations.
- **Loss Balancing**: Implemented custom `YOLOWeightedDataset` class to compute class-wise sampling weights.
- **Training Strategy**:
  - Trained models on **folds 6, 7, and 8** as a 3-model ensemble.
  - Used `AdamW` optimizer, `lr0=3e-4`, `batch=10`, `weight_decay=1e-2`.
  - Each model trained for **50 epochs**, with early stopping optionally enabled.
  - Mosaic augmentation disabled after 20 epochs (`close_mosaic=20`).

###  Training Environment

- **Platform**: Kaggle notebooks
- **Total Training Time**: 8 hours 33 minutes (well within the 9-hour limit)

---

##  Deployment & Inference

- **Model Export**: Trained models are compatible with ONNX and TensorFlow Lite for edge deployment.
- **Ensemble Strategy**: Combined predictions from three folds (6, 7, 8) using ensemble techniques.
- **Inference Time**: Total time for ensemble inference was **40 minutes**, well within the 3-hour constraint.

---

##  Results

- **Improved robustness** through fold ensembling and class rebalancing.
- **Lightweight architecture** allowed deployment without sacrifici
