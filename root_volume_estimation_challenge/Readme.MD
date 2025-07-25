# Cassava Root Volume Estimation Challenge

## 🥥 Objective

The goal of this project is to accurately estimate the volume of cassava roots from Ground Penetrating Radar (GPR) images using a combination of computer vision and machine learning techniques. The challenge involves segmenting root structures from radar images and building predictive models to estimate their total volume, measured in cubic centimeters.

---

## 📁 Dataset Overview

The dataset includes:

- **Images**:  
  GPR scans of cassava roots at various depths and from different orientations (left/right).  
  Structure:  

<pre lang="markdown"> <code> data/
├── train/
│ └── [FolderName]/
│ ├── Left_0.png
│ ├── Left_1.png
│ ├── ...
│ └── Right_19.png
├── test/
└── ... </code> </pre>


- **CSV Files**:
- `Train.csv`: Contains `FolderName`, `Start`, `End`, and the target `RootVolume`.
- `Test.csv`: Contains `FolderName`, `Start`, and `End` only — no `RootVolume`.

- **Segmentation Models Provided**:
- `Models/best_early.pt`
- `Models/best_full.pt`
- `Models/best_late.pt`

---

##  Methodology

### 1. **Image Segmentation using Pretrained YOLOv11 Models**

- Three YOLOv11 segmentation models were provided, each trained on cassava GPR slices:
- **`best_early.pt`**: Trained on shallow layers (0–6)
- **`best_full.pt`**: Trained on all layers (0–19)
- **`best_late.pt`**: Trained on deeper layers (12–19)

- For each image layer:
- The appropriate YOLO model is used to segment the root.
- Bounding boxes are extracted and used to calculate the segmented area.
- These area features are aggregated for each scan.

### 2. **Feature Extraction and Engineering**

- For each folder:
- Loop through layers between `Start` and `End` values.
- Use bounding boxes from YOLO detections to estimate the 2D root area per layer.
- Sum/average the estimated areas across layers to approximate root volume geometry.
- Additional features: number of detections, left/right view comparison, depth range.

### 3. **Root Volume Prediction**

- **Modeling Strategy**:
- Train a regression model (e.g., Random Forest, Gradient Boosting Regressor) on extracted features.
- Optimize using cross-validation and RMSE (Root Mean Squared Error) as the evaluation metric.

- **Final Model**:
- A tuned `RandomForestRegressor` yielded the best results on the validation set.

### 4. **Evaluation Metric**

- Root Mean Squared Error (RMSE) is used to evaluate performance.
- RMSE =  
\[
\sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2}
\]

---

##  Results

| Model | RMSE (Validation) | RMSE (Leaderboard) |
|-------|-------------------|--------------------|
| Baseline (Area sum only) | ~1.32              | 1.29               |
| Random Forest Regressor  | ~1.08              | **1.0677**         |

---

##  Key Learnings

- Effective use of pretrained YOLO segmentation models can significantly reduce preprocessing time and enhance root localization accuracy.
- Estimating volume from 2D slices is feasible with area approximations, provided that depth indexing is handled properly.
- Combining left and right views improves robustness of volume estimation.

---

##  Tech Stack

- **Languages**: Python  
- **Libraries**: PyTorch, OpenCV, Scikit-learn, NumPy, Pandas  
- **Models**: YOLOv11 for segmentation, RandomForestRegressor for volume prediction  
- **Environment**: Local training on GPU (YOLO), CPU-inference ready

---

##  Folder Structure

<pre lang="markdown"> <code> root_volume_estimation_challenge /
├── data/
│ ├── train/
│ └── test/
├── Models/
│ ├── best_early.pt
│ ├── best_full.pt
│ └── best_late.pt
├── root_volume_estimation.ipynb
├── working_notebook.ipynb
├── final_submission.csv
└── README.md </code> </pre>