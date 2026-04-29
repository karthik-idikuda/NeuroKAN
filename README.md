# NeuroKAN: A Hybrid Convolutional Kolmogorov-Arnold Network for Axiomatic Explainable Alzheimer's Detection from MRI

## Overview

NeuroKAN bridges the gap between the high accuracy of modern Deep Learning and the strict explainability required in healthcare. This research framework focuses on early-stage Alzheimer's disease classification by hybridizing a lightweight Convolutional Neural Network (EfficientNetV2-S) with Kolmogorov-Arnold Networks (KANs).

Furthermore, we utilize state-of-the-art Axiomatic Explainability via **Integrated Gradients with Noise Tunneling**, providing clinicians mathematically robust Heatmaps (3D and 2D) of brain atrophy rather than black-box approximations.

### Why NeuroKAN?
*   **Efficiency**: Achieves comparable performance to heavy ensemble models with 10x fewer parameters.
*   **Axiomatic Explainability**: Replaces basic, noisy Grad-CAM and SHAP with robust, Noise-Tunneled Integrated Gradients that satisfy the "Completeness Axiom."
*   **Biologically Plausible Learning**: By utilizing KAN layers with Fourier expansion, the network learns non-linear progressions on its edges, perfectly modeling brain tissue atrophy trajectories over time.

## Directory Structure

*   `src/model.py`: PyTorch implementation of the `NeuroKAN` model architecture and the custom `KANLinear` layers.
*   `src/dataset.py`: Handles preprocessing and loading of the customized Parquet format MRI datasets.
*   `src/train.py`: Training loop with learning-rate scheduling and curriculum-oriented data loading.
*   `src/visualize.py`: Explainability module with Integrated Gradients and 3D visualizations.
*   `backend/main.py`: FastAPI backend serving three models (CNN, NeuroKAN, Random Forest).
*   `requirements.txt`: Project dependencies.

## Getting Started

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Training
```bash
cd src
python train.py --model_type kan --epochs 15
```

### Visualization
```bash
cd src
python visualize.py --data_dir ../data --model_path ../models/neurokan_final.pth
```

## Model Architecture

NeuroKAN combines:
1. **EfficientNetV2-S Backbone** - Feature extraction from MRI images
2. **KAN Classifier Head** - Kolmogorov-Arnold layers with Fourier-series based spline functions
