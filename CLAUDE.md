# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a soccer field detection and segmentation project using computer vision and deep learning approaches. The project aims to detect and segment various elements of a soccer field from images and videos.

## Key Commands

### Environment Setup
```bash
# Install ultralytics for YOLOv8
pip install ultralytics

# Install OpenCV for computer vision tasks
pip install opencv-python

# Install other dependencies
pip install matplotlib numpy pyyaml psutil ipywidgets
```

### Running Detection/Segmentation

**Computer Vision Approach (Classical):**
```bash
python cv_ellipse_detection.py
```

**Deep Learning Approach (YOLOv8):**
```python
# Use the inference function in DL_ellipse_detection.ipynb
# The model automatically detects the latest trained model in runs/segment/
```

### Training Models
```python
# Use the YOLOv8MPSTrainerAuto class in DL_learning_model.ipynb
# It automatically adapts to system memory and hardware capabilities
```

## Project Architecture

### Core Components

1. **cv_ellipse_detection.py**: Classical computer vision approach for detecting center circles using HSV color space and ellipse fitting. Currently has issues with accurate ellipse detection.

2. **DL_ellipse_detection.ipynb**: YOLOv8-based inference notebook for field segmentation. Contains:
   - Simple inference function
   - Jupyter-friendly inference with automatic model detection
   - Interactive UI for file uploads

3. **DL_learning_model.ipynb**: YOLOv8 training notebook with memory-adaptive configuration. Features:
   - Automatic system profiling (memory, CPU, GPU)
   - Dynamic batch size and model selection based on available resources
   - Support for Apple Silicon MPS, CUDA, and CPU

### Dataset Structure

The project uses a Roboflow dataset (`field-6/`) with 7 classes:
- 18Yard
- 18Yard Circle  
- 5Yard
- First Half Central Circle
- First Half Field
- Second Half Central Circle
- Second Half Field

Dataset splits:
- Train: 2523 images
- Valid: 161 images
- Test: 101 images

### Model Storage

Trained models are stored in `runs/segment/` with the following naming convention:
- `field_seg_auto_{model_size}_{timestamp}/`
- Best model weights: `weights/best.pt`

## Key Technical Considerations

1. **Memory Management**: The training script automatically adapts to available system memory (especially important for Apple Silicon unified memory).

2. **MPS Support**: Full support for Apple Silicon GPU acceleration with appropriate environment variables set.

3. **Adaptive Training**: The `YOLOv8MPSTrainerAuto` class automatically selects optimal:
   - Model size (n/s/m/l/x based on memory)
   - Batch size
   - Worker threads
   - Cache strategy

4. **Data Augmentation**: Configured for soccer field imagery with appropriate augmentation parameters.