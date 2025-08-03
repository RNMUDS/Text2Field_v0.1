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
├── cv_ellipse_detection.py     # Classical CV approach (demonstrates limitations)
├── DL_ellipse_detection.ipynb  # YOLOv8 inference notebook
├── DL_learning_model.ipynb     # YOLOv8 model training notebook
├── detection_persons.py        # Player detection and team classification
├── manual_click_transform.py   # Interactive perspective transform tool
├── interactive_transform_notebook.py  # Jupyter-friendly transform tool
├── correct_homography.py       # Homography calculation utilities
├── correct_field_transform.py  # Field perspective correction
├── calibrate_homography.py     # Camera calibration for field mapping
├── analyze_image.py            # Image analysis utilities
├── field-6/                    # YOLOv8 dataset directory
│   ├── train/                  # Training data
│   ├── valid/                  # Validation data
│   └── test/                   # Test data
└── runs/                       # Trained models
```

## File Descriptions

### Core Detection Files

#### `cv_ellipse_detection.py`
Traditional computer vision approach using HSV color space and ellipse fitting. Demonstrates why deep learning is necessary:
- Uses HSV thresholding to detect field
- Attempts to fit ellipse to center circle
- Fails with partial occlusions and lighting variations

#### `DL_ellipse_detection.ipynb`
YOLOv8-based inference notebook for field element segmentation:
- `inference_jupyter()` - Auto-detects latest model and runs inference
- Displays side-by-side comparison of original and annotated images
- Supports interactive file upload widget

#### `DL_learning_model.ipynb`
Adaptive YOLOv8 training system:
- `YOLOv8MPSTrainerAuto` class - Automatically adapts to system resources
- Memory-aware model selection (n/s/m/l/x)
- Support for Apple Silicon MPS, CUDA, and CPU

### Player and Ball Detection

#### `detection_persons.py`
Advanced player detection with team classification:
- Detects players using YOLOv5
- Classifies teams by jersey color (red/blue)
- Identifies referee by black uniform
- Detects ball position
- Maps players to 2D field representation
- Creates tactical field visualization

Key functions:
- `detect_players_and_ball_refined()` - Main detection function
- `transform_to_2d_field_with_perspective()` - Maps image coords to field coords
- `create_accurate_soccer_field()` - Generates tactical view

### Perspective Transform Tools

#### `manual_click_transform.py`
Interactive tool for manual perspective correction:
- Click 4 corners of the field in order
- Applies perspective transform to bird's-eye view
- Saves transformation matrix for reuse
- Usage: `python manual_click_transform.py <image_file>`

#### `interactive_transform_notebook.py`
Jupyter notebook version of perspective transform:
- Uses matplotlib's `ginput()` for point selection
- Designed for Google Colab compatibility

### Calibration and Analysis Tools

#### `correct_homography.py`
Calculates homography matrix from known field points:
- Uses multiple feature points for accuracy
- Handles camera distortion from side angles

#### `correct_field_transform.py`
Specialized transform for stadium camera angles:
- Corrects for perspective distortion
- Maps visible field area to metric coordinates

#### `calibrate_homography.py`
Camera calibration utilities:
- `get_homography_matrix()` - Returns calibrated homography
- `test_homography()` - Validates transformation accuracy

#### `analyze_image.py`
Image analysis and debugging tools:
- Edge detection visualization
- Line detection using Hough transform
- Helps identify field features

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

## Usage Examples

### 1. Detect Players and Teams
```python
# Run player detection with team classification
python detection_persons.py

# This will:
# - Detect all players in the image
# - Classify them into red/blue teams
# - Identify the referee
# - Find the ball
# - Create a tactical field visualization
```

### 2. Manual Perspective Transform Practice
```python
# Interactive perspective correction
python manual_click_transform.py sample1.png

# Click 4 corners in order:
# 1. Top-left (red)
# 2. Top-right (green)
# 3. Bottom-right (blue)
# 4. Bottom-left (yellow)
```

### 3. Field Element Detection (YOLOv8)
```python
# In Jupyter notebook
from DL_ellipse_detection import inference_jupyter
results = inference_jupyter("sample1.png")
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