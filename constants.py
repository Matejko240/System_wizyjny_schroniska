from pathlib import Path

# Główne ścieżki
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "animal_model.keras"
TRAIN_PATH = BASE_DIR / "dataset" / "train"
VAL_PATH = BASE_DIR / "dataset" / "val"
TEST_PATH = BASE_DIR / "dataset" / "test"

# Parametry domyślne
CATEGORIES = ["cat", "wild", "dog"]
IMG_SIZE = 128
DEFAULT_EPOCHS = 5
DEFAULT_SHOW_IMAGES = False
DEFAULT_NUM_IMAGES = 5
