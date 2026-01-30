# DeepECH: Global Dynamic Prediction of ECH Waves via Two-Stage Deep Learning

## Overview

This project implements a **two-stage deep learning framework** to predict the occurrence and intensity of **Electron Cyclotron Harmonic (ECH) waves** in the Earth's magnetosphere, based on Van Allen Probes data (2013-2018).

ECH waves are essential for magnetospheric dynamics, scattering energetic electrons and contributing to diffuse auroras. This model utilizes multi-resolution geomagnetic indices (SYM-H, SME) and spatial coordinates to provide reliable global predictions, especially during geomagnetic storms.

## Key Research Highlights (from the Manuscript)

- **Two-Stage Architecture**: A classification network first determines wave presence (91.2% accuracy), followed by a regression network for amplitude prediction ($R^2 = 0.7094$).
- **Multi-Resolution Features**: Extracts temporal features from SYM-H and SME sequences at both 1-hour and 1-minute resolutions using an **Attention-LSTM** network.
- **Spatial Modeling**: Incorporates L-shell, Magnetic Local Time (MLT), and Magnetic Latitude (MLAT) for precise spatial mapping.
- **Scientific Impact**: Offers instantaneous, global wave distribution maps, overcoming the limitations of traditional statistical models.

## Model Components

1. **Classification Model (`transpace_cls.py`)**: Predicts the probability of ECH wave occurrence.
2. **Regression Model (`transpace.py`)**: Estimates the wave amplitude (in dB) when waves are present.
3. **Inference Script (`run_inference.py`)**: Orchestrates the two-stage logic to generate L-MLT distribution grids.

## Getting Started

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running Inference
Load pre-trained weights and generate sample grids:
```bash
python run_inference.py
```
Results are saved in the `results/` folder:
- `regression_grid.mat`: Global ECH wave amplitude distribution.
- `classification_grid.mat`: ECH wave occurrence probability map.

## Project Structure
- `checkpoints/`: Pre-trained model weights.
- `data/`: Sample input feature vectors (Pickle format).
- `results/`: Output `.mat` files for visualization (e.g., in MATLAB).

