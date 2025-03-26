from pathlib import Path

# Główne ścieżki
BASE_DIR = Path(__file__).parent
TRAIN_PATH = BASE_DIR / "dataset" / "train"
VAL_PATH = BASE_DIR / "dataset" / "val"
TEST_PATH = BASE_DIR / "dataset" / "test"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "animal_model.keras"

# Parametry domyślne
CATEGORIES = ["cat", "wild", "dog"]
IMG_SIZE = 128
DEFAULT_EPOCHS = 50
DEFAULT_SHOW_IMAGES = False
DEFAULT_NUM_IMAGES = 5
