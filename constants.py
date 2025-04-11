from pathlib import Path

# Główne ścieżki
BASE_DIR = Path(__file__).parent
TRAIN_PATH = BASE_DIR / "dataset" / "train"
VAL_PATH = BASE_DIR / "dataset" / "val"
TEST_PATH = BASE_DIR / "dataset" / "test"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "animal_model.keras"
DATASET_JSON_PATH = MODELS_DIR / "animal_model.keras.dataset.json"
# Parametry domyślne
CATEGORIES = ["cat", "wild", "dog"]
IMG_SIZE = 128
DEFAULT_EPOCHS = 50
DEFAULT_SHOW_IMAGES = False
DEFAULT_NUM_IMAGES = 5
PATIENCE = 5 #Na podstawie ilu epok wstecz działa early stop
DEFAULT_VAL_IMAGES = 1500
DEFAULT_TRAIN_IMAGES = 13000
MIN_DELTA = 0.01 #zmiana pogorszenia się walidacji aby zadziałał early stop
RUNS = 5 # ilość powtórzeń trenowania modelu