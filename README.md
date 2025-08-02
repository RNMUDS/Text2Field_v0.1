# Text2Field v0.1

A computer vision project for detecting and segmenting soccer field elements using both classical CV and deep learning approaches.

## Overview

Text2Field is designed to identify and segment various components of a soccer field from images and videos, including:
- Field boundaries (First/Second half)
- Central circles
- 18-yard boxes and circles
- 5-yard boxes

The project implements two approaches:
1. **Classical Computer Vision**: HSV-based detection with ellipse fitting
2. **Deep Learning**: YOLOv8 segmentation model with adaptive training

## Features

- 🎯 Multi-class segmentation of soccer field elements
- 🚀 Automatic hardware adaptation for training (CPU/MPS/CUDA)
- 💾 Memory-aware batch size and model selection
- 📊 Interactive Jupyter interface for inference
- 🔧 Pre-trained models available

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Text2Field_v0.1.git
cd Text2Field_v0.1

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Using Pre-trained Model

```python
# In Jupyter notebook
from ultralytics import YOLO

# Load the model
model = YOLO('runs/segment/field_seg_64gb_n_20250802_214746/weights/best.pt')

# Run inference
results = model('sample1.png')
results[0].show()
```

### Training Your Own Model

Open `DL_learning_model.ipynb` and run the adaptive training:

```python
trainer = YOLOv8MPSTrainerAuto(dataset_path="./field-6")
model_path, results = trainer.train_adaptive(
    model_size='auto',  # Automatically selects based on system memory
    epochs=100,
    imgsz=640
)
```

## Project Structure

```
Text2Field_v0.1/
├── cv_ellipse_detection.py    # Classical CV approach
├── DL_ellipse_detection.ipynb # Inference notebook
├── DL_learning_model.ipynb    # Training notebook
├── field-6/                   # Dataset directory
│   ├── train/
│   ├── valid/
│   └── test/
├── sample1.png               # Sample images
├── sample2.png
└── runs/                     # Trained models
```

## Dataset

The project uses a Roboflow dataset with 7 classes:
- 18Yard
- 18Yard Circle
- 5Yard
- First Half Central Circle
- First Half Field
- Second Half Central Circle
- Second Half Field

Dataset statistics:
- Training: 2,523 images
- Validation: 161 images
- Test: 101 images

## Hardware Requirements

Minimum:
- 8GB RAM
- Python 3.8+

Recommended:
- 16GB+ RAM
- GPU with CUDA support or Apple Silicon

The training script automatically adapts to available hardware.

## Results

The YOLOv8 model achieves high accuracy in detecting and segmenting field elements. Example results can be found in the `runs/segment/` directory after training.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset provided by [Roboflow](https://roboflow.com)
- Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{text2field2024,
  title={Text2Field: Soccer Field Element Detection and Segmentation},
  author={Your Name},
  year={2024},
  version={0.1}
}
```