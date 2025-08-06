# 🛰️ Landslide Image Classification from Multi-Modal Satellite Imagery
A deep learning approach for detecting landslides using multi-temporal Sentinel-1 (SAR) and Sentinel-2 (optical) satellite imagery. This project implements a two-stage detection pipeline combining traditional machine learning for efficient negative filtering with a multi-modal Feature Pyramid Network (FPN) for precise landslide identification.
📊 Overview
This solution addresses the challenge of detecting landslides from satellite imagery by leveraging both optical and radar data to overcome limitations like cloud coverage. The approach combines:

Multi-modal data fusion: Integrating optical (Sentinel-2) and SAR (Sentinel-1) imagery
Two-stage detection: XGBoost pre-filter + Deep CNN for efficiency
Temporal analysis: Using pre- and post-event imagery to detect changes
Advanced architecture: Custom Multi-Modal FPN with separate branches for different data modalities

[INSERT: Example visualization showing input satellite images (optical + SAR) and predicted landslide mask]
🏗️ Architecture
Two-Stage Pipeline

Stage 1 - Negative Filter (XGBoost)

Extracts hand-crafted features (NDVI, SAVI, polarization ratios, etc.)
Filters out obvious non-landslide areas with 100% precision
Reduces computational load by ~30-40%


Stage 2 - Multi-Modal FPN

Separate CNN branches for optical and SAR data
Feature Pyramid Network for multi-scale feature extraction
Ensemble of 10 models from cross-validation



[INSERT: Architecture diagram showing the two-stage pipeline and FPN structure]
Key Features

Input: 64×64 pixel patches with 13 channels:

5 optical bands (RGB, NIR, cloud mask)
4 descending SAR bands (VV/VH pre/post)
4 ascending SAR bands (VV/VH pre/post)


Preprocessing:

BigEarthNet v2 normalization
Temporal difference computation
Cloud coverage detection
Multiple vegetation and water indices



📈 Performance
The model achieves robust performance through:

10-fold cross-validation ensuring generalization
Test-time augmentation with 8 transformation combinations
Ensemble prediction from multiple folds
F1-optimized thresholds for each model

[INSERT: Performance metrics table or confusion matrix visualization]
🚀 Quick Start
Installation
bash# Clone the repository
git clone https://github.com/PhilipJanz/Landslide-Image-Classification.git
cd Landslide-Image-Classification

# Install dependencies
pip install -r requirements.txt
Data Preparation

Place raw satellite images in data/raw/train/images/ and data/raw/test/images/
Run preprocessing:

bashpython src/utils/image_preprocessing.py
python src/utils/feature_preprocessing.py
Training
bash# Train the negative filter model
python src/model/neg_filter_model.py

# Train the main CNN model
python src/model/train.py

# Optional: Hyperparameter optimization
python src/model/hyperparameter_opt.py
Prediction
bashpython src/model/predict.py
📁 Project Structure
├── data/
│   ├── raw/                    # Original satellite imagery
│   ├── processed/               # Preprocessed images and features
│   └── submissions/             # Model predictions
├── models/                      # Saved model checkpoints
├── src/
│   ├── config.py               # Configuration settings
│   ├── model/
│   │   ├── train.py            # Main training script
│   │   ├── predict.py          # Inference pipeline
│   │   ├── fpn_architecture.py # Multi-modal FPN implementation
│   │   └── neg_filter_model.py # XGBoost pre-filter
│   └── utils/
│       ├── image_preprocessing.py
│       ├── feature_preprocessing.py
│       ├── augmentation.py
│       └── dataset_loader.py
└── notebooks/                   # Exploratory analysis
🔧 Technical Details
Model Architecture

Optical Branch: Processes 5 bands (RGB + NIR + cloud mask)
SAR Branches: Separate processing for ascending/descending passes
FPN Backbone: 4-level feature pyramid with lateral connections
Fusion: Multi-scale feature aggregation with global pooling

Training Strategy

Optimizer: Adam with cosine annealing and warmup
Loss: Weighted BCE loss to handle class imbalance
Augmentation: Random flips, rotations, and patch erasing
Regularization: Dropout and weight decay

[INSERT: Training curves showing loss/accuracy/F1 over epochs]
Inference Pipeline

Load ensemble of 10 models from cross-validation
Apply 8-fold test-time augmentation
Average predictions with uncertainty estimation
Apply F1-optimized thresholds
Post-process with negative filter results

🌍 Environmental Impact
This project includes carbon emission tracking using CodeCarbon to monitor the environmental footprint of model training.
