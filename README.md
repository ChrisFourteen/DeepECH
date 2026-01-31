# DeepECH: Global Dynamic Prediction of ECH Waves via Two-Stage Deep Learning

## Overview

This project implements a **two-stage deep learning framework** to predict the occurrence and intensity of **Electron Cyclotron Harmonic (ECH) waves** in the Earth's magnetosphere, based on Van Allen Probes data (2013-2018).

ECH waves are essential for magnetospheric dynamics, scattering energetic electrons and contributing to diffuse auroras. This model utilizes multi-resolution geomagnetic indices (SYM-H, SME) and spatial coordinates to provide reliable global predictions, especially during geomagnetic storms.


## Model Components

1. **Classification Model (`models/transpace_cls.py`)**: Predicts the probability of ECH wave occurrence.
2. **Regression Model (`models/transpace.py`)**: Estimates the wave amplitude (in dB) when waves are present.
3. **Classification Inference (`run_cls.py`)**: Runs single-sample classification.
4. **Regression Inference (`run_reg.py`)**: Runs single-sample regression.

## Data Format
The input feature vector structured as follows:
### 1. Position Input (Indices 0-3)
- `[0]`: **L-shell** - Geocentric distance to the magnetic field line's equatorial point (in Earth radii, $R_E$).
- `[1]`: **cos(MLT)** - Cosine of the Magnetic Local Time.
- `[2]`: **sin(MLT)** - Sine of the Magnetic Local Time.
- `[3]`: **cos(MLAT)^6** - Sixth power of the cosine of the Magnetic Latitude.

### 2. Sequence Input (Indices 4-224)
Based on the manuscript and model architecture, the sequence input consists of multi-resolution geomagnetic indices (SYM-H and SME) used to capture temporal activity across four sequences:

- `[4-52]`: **Sequence 1** (Length 49)
- `[53-101]`: **Sequence 2** (Length 49)
- `[102-162]`: **Sequence 3** (Length 61)
- `[163-223]`: **Sequence 4** (Length 61)

These sequences are processed by the **Attention-based LSTM** modules to extract temporal features representing the geomagnetic disturbance levels.

## Getting Started

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running Inference

#### ECH Classification
```bash
python run_cls.py --index 60
```

#### ECH Regression
```bash
python run_reg.py --index 60
```

## Project Structure
- `models/`: Model architecture definitions.
- `checkpoints/`: Pre-trained model weights.
- `data/`: Sample input feature vectors (Pickle format).
- `run_cls.py`: Script for classification inference.
- `run_reg.py`: Script for regression inference.
- `README_zh.md`: Chinese documentation.

