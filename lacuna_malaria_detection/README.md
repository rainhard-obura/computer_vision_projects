#  Malaria Parasite Detection Using YOLOv8

##  Objective

The goal of this project is to develop a **multiclass object detection and classification model** to detect **malaria parasites**—specifically the **trophozoite stage**—and distinguish between infected and uninfected blood cells in microscopic slide images. This work has direct diagnostic relevance in **sub-Saharan Africa**, where malaria continues to be a major public health concern.

A scalable, automated diagnostic tool can:
- Enable **large-scale screenings** and **early warning systems**,
- **Relieve overburdened healthcare workers** by automating routine diagnostics,
- Improve **accuracy**, **speed**, and **accessibility** of malaria diagnosis in **low-resource settings**.

---

##  Dataset

The dataset comprises high-resolution **microscopy images** and corresponding bounding box annotations identifying:
- `Trophozoite` (malaria-infected cell),
- `WBC` (white blood cell),
- `NEG` (images with no relevant features).

### Files:
- `Train.csv` – Bounding box annotations for training images.
- `Test.csv` – Test set with image filenames.
- `SampleSubmission.csv` – Sample format for submission.
- `images.zip` – Raw microscopy slide images.

---

##  Methodology

### 1. **Environment Setup**
- Utilized libraries: `pandas`, `OpenCV`, `scikit-learn`, `matplotlib`, `ultralytics`, `tqdm`, `multiprocessing`.
- Target model: **YOLOv8 (You Only Look Once)** for real-time object detection.

### 2. **Data Preprocessing**
- **Unzipped and loaded images** and metadata.
- Mapped image paths and label-encoded classes (`Trophozoite`: 0, `WBC`: 1, `NEG`: 2).
- **Filtered out `NEG` class** during training (as per YOLOv5 guidance).
- **Split data** into train and validation sets (75%-25%) using stratified sampling.

### 3. **Annotation Conversion**
- Converted bounding boxes from (xmin, ymin, xmax, ymax) format to **YOLO format**:
  - `(class_id, x_center, y_center, width, height)` — all normalized by image dimensions.
- Saved labels to `.txt` files corresponding to each image.

### 4. **Data Organization**
Structured dataset into the YOLO-compliant format:
```
datasets/
└── dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

### 5. **Model Training**
- Used pretrained `yolov8m.pt` model.
- Training parameters:
  - `imgsz=2048`, `batch=8`, `epochs=30`, `patience=5`.
- Trained on GPU (`device=0`).
- Training controlled using a `data.yaml` file defining class names and paths.

### 6. **Model Evaluation**
- Loaded best model weights from:
  ```
  /content/runs/detect/train/weights/best.pt
  ```
- Evaluated performance on validation set.

### 7. **Inference on Test Set**
- Loaded test images from `/content/datasets/dataset/images/test`.
- Ran inference on each image using the trained model.
- Captured:
  - Bounding box coordinates,
  - Detected class,
  - Confidence score.
- For images with no detections, assigned `NEG` class with default bbox.

---

##  Results

- The trained YOLOv8 model successfully detects and classifies:
  - **Trophozoites**, indicating malaria infection.
  - **WBCs**, which are distractors in medical diagnostics.
- Achieves robust detection and localization performance with **fast inference time**.
- Produces a submission-ready `.csv` with:
  ```
  Image_ID, class, confidence, ymin, xmin, ymax, xmax
  ```

>  *Sample Predictions:*
```csv
Image_ID, class, confidence, ymin, xmin, ymax, xmax
img_001.jpg, Trophozoite, 0.94, 0.18, 0.32, 0.44, 0.60
img_002.jpg, NEG, 1.0, 0, 0, 0, 0
```

---

##  Project Structure

```
malaria_detection_project/
├── data/
│   ├── images/
│   └── labels/
├── Train.csv
├── Test.csv
├── SampleSubmission.csv
├── yolov8_inference_and_training.ipynb
├── data.yaml
└── README.md
```

---

##  Future Work

- Introduce **hard-negative mining** using `NEG` images to reduce false positives.
- Explore **model distillation** or **quantization** for deployment on **mobile/edge devices**.
- Integrate with mobile applications or diagnostic APIs for **field deployment**.

---

##  Key Takeaways

- YOLOv8 is highly effective for real-time object detection in medical imaging.
- Class imbalance must be handled carefully (e.g., `NEG` samples).
- Object detection can be used beyond traditional image classification in digital pathology.

