# Adapt SAM2 3D: Adapting Segment Anything 2.1 to extract 3D objects from volumetric data.

Segment Anything Model 2 is a foundation model that aims to solve promptable visual segmentation in images and videos.
The version 2 allows to perform object tracking through time using a streaming memory.
This repository provides a way to quickly label 3D objects through point prompt by treating 2D slices of 3D images as video frames.

Unlike other repositories, this one aims to provide a simple and quick way to perform 3D volume estimation using SAM2.

<image process>

## Installation

0. Create a new virtual env

```sh
conda create -n adaptsam python=3.10
```

1. Follow sam2 installation

```sh
git clone https://github.com/facebookresearch/sam2
cd sam2
pip install .
```

### Troubleshoot

- No space left when installing sam2: set the variable `TMPDIR` to another location. Example: `TMPDIR=<path> pip install .e`

## Getting Started

```python
import sam2
import tifffile as tif

# Prepare the predictor by loading SAM2
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
sam2_checkpoint = "checkpoints/sam2.1_hiera_b+.pt"
predictor = AdaptSAMPredictor(model_cfg, sam2_checkpoint)

# Load data
data_path = "example/3d_cell.tif"
mat = tif.memmap(data_path)

# Create a point prompt
# Same axis order as mat
point_prompt = [10,20,30]

# Predict on the full matrix with the point prompt
prediction = predictor.predict(mat, point_prompt)

tif.imwrite("predicted.tif", prediction)
```

## Finetuning Sam
