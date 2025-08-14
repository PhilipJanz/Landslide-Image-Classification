from pathlib import Path

# ---------------------------------------------------------------------------- #
#                                 Paths                                        #
# ---------------------------------------------------------------------------- #
# Define the absolute path to the project root directory.
# This makes your project portable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define paths to key data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "01_raw"
PROCESSED_DATA_DIR = DATA_DIR / "02_processed"
EMBEDDING_DATA_DIR = DATA_DIR / "03_embeddings"


# Paths to the separate train and test image folders
TRAIN_IMAGE_DIR = RAW_DATA_DIR / "train"
TEST_IMAGE_DIR = RAW_DATA_DIR / "test"


# Paths to the separate train and test image folders
PROCESSED_TRAIN_IMAGE_DIR = PROCESSED_DATA_DIR / "train"
PROCESSED_TEST_IMAGE_DIR = PROCESSED_DATA_DIR / "test"

# Path to preprocessed features for stage-1 modeling
PROCESSED_FEATURE_PATH = PROCESSED_DATA_DIR / "features"

# Paths to the CSV files in the raw data directory
TRAIN_CSV_PATH = RAW_DATA_DIR / "Train.csv"
TEST_CSV_PATH = RAW_DATA_DIR / "Test.csv"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "SampleSubmission.csv"

# Define paths for outputs
MODEL_DIR = PROJECT_ROOT / "models"
FM_MODEL_DIR = MODEL_DIR / "foundation_models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SUBMISSIONS_DIR = OUTPUT_DIR / "submissions"

# band information after preprocessing
BAND_DESCRIPTIONS = [
    "Cloud Map", "Red", "Green", "Blue", "Near Infrared",
    "Descending VV post", "Descending VH post",
    "Descending VV pre", "Descending VH pre",
    "Ascending VV post", "Ascending VH post",
    "Ascending VV pre", "Ascending VH pre"
]
BAND_LABELS = [
    "cloud", "red", "green", "blue", "nir",
    "desc_VV_post", "desc_VH_post",
    "desc_VV_pre", "desc_VH_pre",
    "asc_VV_post", "asc_VH_post",
    "asc_VV_pre", "asc_VH_pre",
]

# ---------------------------------------------------------------------------- #
#                        Model & Training Hyperparameters                      #
# ---------------------------------------------------------------------------- #
# Reproducibility
SEED = 42

# Image processing settings
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_CHANNELS = 13

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 100

DEVICE = "cuda"  # or "cpu"

# Name for the saved model file
MODEL_NAME = "landslide_MMCNN_final42"
