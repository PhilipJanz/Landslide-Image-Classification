import numpy as np
import sys
from pathlib import Path

# Add src to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import BAND_LABELS



EPSILON = 1e-6  # for safe division

def extract_features(img):
    # Assume img shape is (H, W, C), we want (C, H, W)
    img = img.transpose(2, 0, 1)
    img_dict = {band_label: band for band, band_label in zip(img, BAND_LABELS)}

    # Create new features (bands)
    img_dict["desc_VV_diff"] = img_dict["desc_VV_post"] - img_dict["desc_VV_pre"]
    img_dict["desc_VH_diff"] = img_dict["desc_VH_post"] - img_dict["desc_VH_pre"]
    img_dict["asc_VV_diff"] = img_dict["asc_VV_post"] - img_dict["asc_VV_pre"]
    img_dict["asc_VH_diff"] = img_dict["asc_VH_post"] - img_dict["asc_VH_pre"]

    img_dict["desc_VV_VH_polarization_ratio"] = img_dict["desc_VV_post"] - img_dict["desc_VH_post"]
    img_dict["asc_VV_VH_polarization_ratio"] = img_dict["asc_VV_post"] - img_dict["asc_VH_post"]

    nir = img_dict["nir"]
    red = img_dict["red"]
    green = img_dict["green"]
    blue = img_dict["blue"]

    img_dict["ndvi"] = (nir - red) / np.mean(nir + red)
    img_dict["savi"] = 1.5 * (nir - red) / np.mean(nir + red + 0.5)
    img_dict["gndvi"] = (nir - green) / np.mean(nir + green)
    img_dict["evi"] = 2.5 * (nir - red) / np.mean(nir + 6 * red - 7.5 * blue + 1)
    img_dict["ndwi"] = (green - nir) / np.mean(green + nir)
    img_dict["ngrdi"] = (green - red) / np.mean(green + red)

    # Now compute mean and std for each feature
    features = []
    feature_names = []

    for key, arr in img_dict.items():
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        max_val = np.max(arr)
        min_val = np.min(arr)
        q9_val = np.quantile(arr, .9)
        q1_val = np.quantile(arr, .1)
        features.extend([mean_val, std_val, max_val, min_val, q1_val, q9_val])
        feature_names.extend([f"{key}_mean", f"{key}_std", f"{key}_max", f"{key}_min", f"{key}_q1", f"{key}_q9"])

    return features, feature_names
