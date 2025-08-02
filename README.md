# Text2Field v0.1

A computer vision project for detecting and segmenting soccer field elements using both classical CV and deep learning approaches.

## Overview

Text2Field is designed to identify and segment various components of a soccer field from images and videos. The project implements two approaches:

1. **Classical Computer Vision**: HSV-based detection with ellipse fitting (demonstrates limitations)
2. **Deep Learning**: YOLOv8 segmentation model with adaptive training (recommended approach)

## Why Deep Learning?

The traditional computer vision approach implemented in `cv_ellipse_detection.py` demonstrates poor accuracy due to:

- Sensitivity to lighting variations
- Inability to handle field color variations
- Susceptibility to noise and shadows
- Difficulty detecting partially occluded circles

This highlights why a more robust deep learning approach (YOLOv8) is necessary for reliable field element detection.

## Finding Soccer Datasets on Roboflow

[Roboflow Universe](https://universe.roboflow.com/) hosts numerous soccer-related datasets and models:

- Soccer field segmentation
- Player detection
- Ball tracking
- Goal detection

Search keywords: `soccer field`, `football pitch`, `soccer segmentation`

## How to Use This Project

### 1. Running Inference (DL_ellipse_detection.ipynb)

To run inference with the pre-trained model:

```python
# Open and run DL_ellipse_detection.ipynb
from ultralytics import YOLO

# Auto-detect and load the model
results = inference_jupyter("sample1.png")
```

### 2. Building Your Own Model (DL_learning_model.ipynb)

To train a model on your own dataset:

```python
# Open and run DL_learning_model.ipynb
trainer = YOLOv8MPSTrainerAuto(dataset_path="./field-6")

# Train with automatically optimized settings
model_path, results = trainer.train_adaptive(
    model_size='auto',  # Auto-selects based on system memory
    epochs=100,
    imgsz=640
)
```

## Project Structure

```
Text2Field_v0.1/
├── cv_ellipse_detection.py     # Classical CV approach (low accuracy example)
├── DL_ellipse_detection.ipynb  # Inference notebook
├── DL_learning_model.ipynb     # Model training notebook
├── field-6/                    # Dataset directory
│   ├── train/                  # Training data
│   ├── valid/                  # Validation data
│   └── test/                   # Test data
├── sample1.png                 # Sample images
├── sample2.png
└── runs/                       # Trained models
```

## Detectable Classes

- 18-Yard Box
- 18-Yard Circle
- 5-Yard Box
- First Half Central Circle
- First Half Field
- Second Half Central Circle
- Second Half Field

## Installation

```bash
# Clone the repository
git clone https://github.com/RNMUDS/Text2Field_v0.1.git
cd Text2Field_v0.1

# Install dependencies
pip install -r requirements.txt
```

## Key Features

- 🎯 Multi-class segmentation of 7 field elements
- 🚀 Automatic hardware adaptation (Apple Silicon/CUDA/CPU)
- 💾 Memory-aware batch size and model selection
- 📊 Interactive Jupyter interface for inference
- 🔧 Pre-trained models included

## System Requirements

Minimum:
- 8GB RAM
- Python 3.8+

Recommended:
- 16GB+ RAM
- GPU with CUDA support or Apple Silicon

## License

MIT License

## Acknowledgments

- Dataset provided by [Roboflow](https://roboflow.com)
- Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)