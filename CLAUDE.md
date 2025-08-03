# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Soccer field detection and segmentation project demonstrating the limitations of classical computer vision and the effectiveness of deep learning (YOLOv8) for field element detection.

## Key Commands

### Environment Setup
```bash
# Install all dependencies at once
pip install -r requirements.txt

# Or install core packages manually
pip install ultralytics opencv-python matplotlib numpy pyyaml psutil ipywidgets roboflow
```

### Running Inference

**Quick Inference (Jupyter Notebook):**
```python
# In DL_ellipse_detection.ipynb
results = inference_jupyter("sample1.png")  # Auto-detects latest model
```

**Classical CV Demo (Shows Poor Performance):**
```bash
python cv_ellipse_detection.py  # Outputs to output.png
```

### Training Models

**Automatic Adaptive Training:**
```python
# In DL_learning_model.ipynb
trainer = YOLOv8MPSTrainerAuto(dataset_path="./field-6")
model_path, results = trainer.train_adaptive(
    model_size='auto',  # 'n', 's', 'm', 'l', 'x', or 'auto'
    epochs=100,
    imgsz=640
)
```

**Download Dataset from Roboflow:**
```python
from roboflow import Roboflow
rf = Roboflow(api_key="your_api_key")
project = rf.workspace("yssp").project("field-xzc0o")
dataset = project.version(6).download("yolov8")
```

## Architecture Details

### Key Functions and Classes

**`inference_jupyter(image_path, model_path=None)`** - Main inference function
- Auto-detects latest model in `runs/segment/*/weights/best.pt`
- Returns YOLO results object
- Saves annotated image to `result.jpg`
- Displays side-by-side comparison in Jupyter

**`YOLOv8MPSTrainerAuto`** - Adaptive training class
- Automatically profiles system (memory, CPU, GPU)
- Selects optimal model size based on available memory:
  - 64GB+: Large (l)
  - 32-64GB: Medium (m)
  - 16-32GB: Small (s)
  - <16GB: Nano (n)
- Configures batch size, workers, and caching strategy
- Handles MPS (Apple Silicon), CUDA, and CPU

### Dataset Organization

```
field-6/
├── data.yaml           # Class definitions and paths
├── train/
│   ├── images/        # Training images (excluded from git)
│   └── labels/        # YOLO format annotations
├── valid/
│   ├── images/        # Validation images (excluded from git)
│   └── labels/        # YOLO format annotations
└── test/
    ├── images/        # Test images (excluded from git)
    └── labels/        # YOLO format annotations
```

### Model Output Structure

```
runs/segment/
├── field_seg_auto_l_20250803_040736/
│   ├── weights/
│   │   ├── best.pt    # Best model weights
│   │   └── last.pt    # Last epoch weights
│   ├── args.yaml      # Training configuration
│   ├── results.csv    # Training metrics
│   └── train_batch*.jpg  # Training visualizations
```

## Important Implementation Details

### Memory Optimization (MPS/Apple Silicon)
```python
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
```

### Default Training Parameters
- Optimizer: AdamW
- Learning rate: 0.01
- Image augmentation: Mosaic, flip, scale
- NMS IOU threshold: 0.7
- Confidence threshold: 0.25 (inference)

### Class IDs
```
0: 18Yard
1: 18Yard Circle
2: 5Yard
3: First Half Central Circle
4: First Half Field
5: Second Half Central Circle
6: Second Half Field
```

### Known Issues
- `cv_ellipse_detection.py` fails with partial occlusions and lighting variations
- Large models (x) may require 128GB+ unified memory on Apple Silicon
- Dataset images are large (~1.9GB for training set)

## Roboflow Integration

Project uses Roboflow dataset `field-xzc0o` version 6. Search Roboflow Universe for similar datasets using keywords: `soccer field`, `football pitch`, `soccer segmentation`.