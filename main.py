import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from pathlib import Path
from tensorflow.keras.models import load_model
from dataset_loader import CATEGORIES, DATASET_PATH, VAL_PATH, TEST_PATH
from model import load_or_train_model
from classifier import classify_random_images, classify_all_images_in_folder

# Parametry użytkownika
EPOCHS = 30  # Ustaw tutaj liczbę epok
SHOW_IMAGES = True # Jeśli True -> obrazy będą wyświetlane, jeśli False -> tylko klasyfikacja w terminalu
NUM_IMAGES = 5 # Ilość obrazków z folderu walidacyjnego

# Ścieżka do modelu
MODEL_PATH = Path(__file__).parent / "animal_model.keras"

# Wczytanie lub trening modelu
model = load_or_train_model(MODEL_PATH, DATASET_PATH, CATEGORIES, epochs=EPOCHS)


# Klasyfikacja losowego obrazka z walidacyjnego zbioru danych
classify_random_images(VAL_PATH, model, CATEGORIES, 128, num_images=NUM_IMAGES, show_images=SHOW_IMAGES)

# Klasyfikacja wszystkich obrazów w folderze testowym
classify_all_images_in_folder(TEST_PATH, model, CATEGORIES, 128, show_images=SHOW_IMAGES)

# Sprawdzenie konkretnego obrazka po ścieżce
#CUSTOM_IMAGE_PATH = TEST_PATH / "gandalf.jpg"
#classify_custom_image(CUSTOM_IMAGE_PATH)

