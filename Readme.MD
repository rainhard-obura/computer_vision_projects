# Computer Vision Projects
This repository collects  applied computer vision challenges tackled using Python and Jupyter notebooks. Each contains a self‑contained dataset, code, model results, and analysis.



---

##  Projects Overview

### 1. Amini Canopy Crop Challenge
- **Goal**: Segment canopy vs. non-canopy regions in aerial images.
- **Approach**: Image preprocessing → segmentation model (U-Net or similar) → mask visualization & coverage analysis.
- Check the notebook in its folder for detailed methodology and results.

### 2. Amini Cocoa Contamination Challenge
- **Goal**: Detect and classify contamination in cocoa bean images.
- **Approach**: Feature engineering, binary/multi-class classification, performance metrics, and visual validation of model output.

### 3. Root Volume Estimation Challenge
- **Goal**: Estimate the volume of tree roots or soil volume underneath from image data.
- **Approach**: Image-to-depth regression or segmentation; conversion from pixel measurements to volume estimates followed by accuracy assessment.

---

##  Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Libraries (install via `requirements.txt` if added):
  - `numpy`, `pandas`, `scikit-learn`
  - `opencv-python`, `Pillow`
  - `tensorflow` or `torch` (depending on model)
  - `matplotlib`, `seaborn`

---

##  Usage

1. **Clone** the repo:
   ```bash
   git clone https://github.com/rainhard-obura/computer_vision_projects.git
   cd computer_vision_projects
