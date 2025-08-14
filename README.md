# 🛰️ Landslide Image Classification from Multi-Modal Satellite Imagery
Tackling Zindi's [Landslide Classification Challenge](https://zindi.africa/competitions/classification-for-landslide-detection) - a computer vision task on bi-temporal Sentinel-1 (SAR) and Sentinel-2 (optical) satellite imagery. This project  implements classic data preprocessing, augmentation and deep learning using a lightweight multi-modal architecture (< 1M parameters). An efficient data filtering methode has proven to be not only shrink training costs by more than half (279 -> 126 Wh), but also significantly boost performance. Another architectural choice, the Feature Pyramid Network (FPN), couldn't increase performance. 
The solution achieved a test F1-score of .877 (winning F1: .907) and hit place 23 (of 303) on the final leaderboard. 

## 📊 Data Overview
The given data unites bi-temporal Sentinel-1 (SAR) and Sentinel-2 (optical) satellite imagery. The SAR data from Sentinel provides VH and VV bands both for pre- and post-desaster as well as descending and ascending, which gives the possibility to train a *change detection network*. 

![](assets/example_image.png)

The training dataset is highly unbalanced (~17% positives) and contains even images of waters and other obvious non-landslide images. This observation motivated the application of a pre-filter model that would catch obvious negatives to exclude them from the deep learning training.   

## 🏗️ Model Pipeline and Architecture
This solution addresses the challenge by leveraging both optical and radar data in the same model to overcome limitations like cloud coverage. The approach combines:

- Multi-modal data fusion: Integrating optical (Sentinel-2) and SAR (Sentinel-1) imagery
- Two-stage detection: XGBoost pre-filter + Deep CNN for efficiency
- Temporal analysis: Using pre- and post-event imagery to detect changes
- Advanced architecture: Custom Multi-Modal FPN with separate branches for different data modalities

### Stage 1 - Negative Filter (XGBoost)

- Extracts hand-crafted features (NDVI, SAVI, polarization ratios, etc.)
- Filters out obvious non-landslide areas with 100% precision (dont loose any positive datapoint)


### Stage 2 - Multi-Modal FPN

- Separate CNN branches for optical and SAR data
- Feature Pyramid Network for multi-scale feature extraction
- Ensemble of 10 models from cross-validation

### Inference Stage
Predicting the test data is supported by the following concepts the boost performance and robustness:

- 10-fold cross-validation ensuring generalization
- Test-time augmentation with 8 transformation combinations (h_flip x v_flip x rotation_90°)
- F1-optimized thresholds for each model


## 📈 Performance

The following image shows the behaivior of the validation F1-score during training for different approaches. The curves represent the mean values over 10 folds. The blue curve represents the final approach that got submitted to the challenge. All other curves are realized by removing a single building block from the final approach - allowing a direct comparisson of the concepts improtance. However, since all approaches use the same hyperparameters, it leaves the possibility that each model might be more effective given hyperparameter tuning.

![](assets/f1_comparison.png)

### Feature Pyramid Network
The FPN architecture enables the model to simultanously detect large- and small-scale patters. In a direct comparison with a conventional CNN there is no difference in performance. This result suggests that the scale of the images is not large enough that a advances architecture such as the FPN, could lead to any advantage. 

### Data Filtering
Cleaning the dataset from *obvious negatives* enabled the model to focus on vavluable examples of landslide and non-landslide images, what boostes performance by a noticable amount. This methode enables a much faaster and more balanced training (~39% positives vs ~17% pre-filtering) 

### Multi Modality
The *only_SAR* and *only_optical* curves fall significantly under the blue one, indicating that multi-modality plays a key roll for landslide detection. While the SAR model trains converges faster the optical counterpart performed better after completion of 100 training epochs.


## 🚀 Quick Start
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
