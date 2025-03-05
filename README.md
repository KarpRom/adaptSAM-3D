# **Adapt SAM2 3D: Enhancing Segment Anything 2.1 for 3D Object Extraction**

Segment Anything Model 2 (SAM2) is a foundation model designed for **promptable visual segmentation** in images and videos.  
Version 2 introduces **object tracking through time** using a streaming memory mechanism.

This repository enables **fast 3D object labeling** using **point prompts**, treating 2D slices of volumetric images as video frames.

Unlike other solutions, this repository provides a **simple and efficient** way to perform **3D volume estimation** using SAM2.

---

## ⚡ Quickstart

Want to try **AdaptSAM 3D** immediately? Here's a minimal example:

```python
import sam2
import tifffile as tif
from adapt_sam_3d import AdaptSAMPredictor

# Load the SAM2 predictor
predictor = AdaptSAMPredictor(
    model_cfg="configs/sam2.1/sam2.1_hiera_b+.yaml",
    sam2_checkpoint="checkpoints/sam2.1_hiera_b+.pt"
)

# Load a 3D image
mat = tif.imread("example/data/img_0000_0576_0768.tif")

# Define a point prompt (x, y, z)
point_prompt = [10, 20, 30]

# Generate the prediction
prediction = predictor.predict(mat, point_prompt)

# Save the result
tif.imwrite("predicted.tif", prediction)
```

## 🚀 Installation

### 1️⃣ Create a Virtual Environment (Optional)

```sh
conda create -n adaptsam python=3.10
```

### 2️⃣ Install SAM2

```sh
git clone https://github.com/facebookresearch/sam2
cd sam2
pip install .
```

### 3️⃣ Install AdaptSAM-3D

```sh
git clone https://github.com/KarpRom/adaptSAM-3D
cd adaptSAM-3D
pip install .
```

---

## Finetuning Sam

---

## ❗ Troubleshooting

- **No space left when installing SAM2?**  
  Set the `TMPDIR` environment variable to another location before installing:
  ```sh
  TMPDIR=<path> pip install .e
  ```
