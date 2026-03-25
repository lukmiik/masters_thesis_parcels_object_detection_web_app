# Object Detection Pipeline

Streamlit application for object detection model training, inference, and evaluation.
The app provides a multi-page interface for comparing models, running inference, editing
regions of interest (ROI), and training custom detectors.

This project compares object detection models (YOLOv11, RF-DETR, Faster R-CNN) trained
on
the [Roboflow Mailers and Boxes dataset](https://app.roboflow.com/lukmik-e4qay/mailers-and-boxes-v10),
which contains images of small and big packages for mail sorting applications.

## Features

- **Model Comparison**: Compare detection results across multiple models side-by-side
- **Inference Runner**: Run inference on videos with customizable ROI and detection
  parameters
- **ROI Editor**: Define and edit regions of interest for focused detection
- **Training**: Train custom object detection models using YOLOv11, RF-DETR, or Faster
  R-CNN
- **Metrics Extraction**: Evaluate model performance with precision, recall, and mAP
  metrics

## Installation

### Recommended: Using uv

[uv](https://docs.astral.sh/uv/) is the recommended package manager for this project.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository and navigate to the project
cd masters_app_demo

# Install dependencies
uv sync

# Run the application
uv run streamlit run app/main.py
```

### Alternative: Using pip

```bash
# Clone the repository and navigate to the project
cd masters_app_demo

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Run the application
streamlit run app/main.py
```

## Training Notebooks

The training notebooks can be run in two ways:
- **Locally**: Run via the Training page in the Streamlit app
- **Google Colab**: Use the links below

| Model        | Notebook                  | Colab Link                                                                                                                                                                                                           |
|--------------|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| YOLOv11      | `train_yolov11.ipynb`     | [<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">](https://colab.research.google.com/github/lukmiik/masters_thesis_app/blob/main/training_notebooks/train_yolov11.ipynb)     |
| RF-DETR      | `train_rf_detr.ipynb`     | [<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">](https://colab.research.google.com/github/lukmiik/masters_thesis_app/blob/main/training_notebooks/train_rf_detr.ipynb)     |
| Faster R-CNN | `train_faster_rcnn.ipynb` | [<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">](https://colab.research.google.com/github/lukmiik/masters_thesis_app/blob/main/training_notebooks/train_faster_rcnn.ipynb) |

For Colab, set the `ROBOFLOW_API_KEY` environment variable or set
`USE_ROBOFLOW_API = False` in the notebook to use a locally downloaded dataset.

## Project Structure

```
masters_app_demo/
├── app/
│   ├── main.py              # Main entry point
│   └── pages/
│       ├── view.py          # Model comparison page
│       ├── inference.py     # Inference runner page
│       ├── roi_editor.py    # ROI editor page
│       └── training.py       # Training page
├── src/
│   ├── detection.py         # Detection utilities
│   ├── inference.py         # Inference engine
│   ├── metrics_extractor.py # Metrics calculation
│   ├── tracking.py          # Object tracking
│   ├── video_processor.py   # Video processing
│   └── paths.py             # Path configuration
├── training_notebooks/
│   ├── train_yolov11.ipynb
│   ├── train_rf_detr.ipynb
│   └── train_faster_rcnn.ipynb
├── input_videos/            # Input video directory
├── outputs/                 # Output directory
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## Requirements

- Python 3.13+
- Streamlit 1.36+
- PyTorch 2.9.1+
- Ultralytics, Supervision, Roboflow

See `pyproject.toml` for full dependency list.
