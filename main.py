import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model
from dataset_loader import CATEGORIES, DATASET_PATH, load_images, split_dataset, datagen
from model import create_model
from classifier import classify_animal

# Ścieżka do modelu
MODEL_PATH = Path(__file__).parent / "animal_model.h5"

# Trenowanie modelu, jeśli nie istnieje
if not MODEL_PATH.exists():
    print("🚀 Brak modelu, rozpoczynam trenowanie...")
    
    # Wczytanie danych
    data, labels = load_images(DATASET_PATH, CATEGORIES, 128)
    X_train, X_test, y_train, y_test = split_dataset(data, labels)

    # Stworzenie modelu
    model = create_model(128, len(CATEGORIES))

    # Trenowanie modelu z augmentacją
    model.fit(datagen.flow(X_train, y_train, batch_size=32), 
              epochs=20, 
              validation_data=(X_test, y_test))

    # Zapis modelu
    model.save(MODEL_PATH)
    print("✅ Model zapisany jako", MODEL_PATH)
else:
    print("📂 Wczytuję istniejący model...")
    model = load_model(MODEL_PATH)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # ✅ Dodaj kompilację


# Ścieżka do folderu walidacyjnego
VAL_PATH = Path(__file__).parent / "dataset" / "val"

def get_random_image(val_path, categories):
    """Losuje losowy obrazek z walidacyjnego zbioru danych"""
    category = random.choice(categories)
    category_path = val_path / category
    if not category_path.exists():
        return None, None
    images = os.listdir(category_path)
    if not images:
        return None, None
    image_name = random.choice(images)
    return str(category_path / image_name), category

def show_image(image_path):
    """Wyświetla obrazek za pomocą Matplotlib i nie blokuje kodu"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV używa BGR zamiast RGB
    plt.imshow(img)
    plt.axis("off")  # Ukryj osie
    plt.show(block=False)
    plt.pause(5)
    plt.close()

def classify_custom_image(image_path):
    """Sprawdza konkretny obrazek po podanej ścieżce"""
    if not os.path.exists(image_path):
        print(f"❌ Błąd: Plik {image_path} nie istnieje!")
        return

    print(f"🔍 Sprawdzam obrazek: {image_path}")

    # Wyświetlenie obrazka
    show_image(image_path)

    # Klasyfikacja obrazka
    result = classify_animal(image_path, model, CATEGORIES, 128)
    print(f"🤖 Wynik klasyfikacji: {result}")

# Wczytanie losowego obrazka z folderu walidacyjnego
random_image_path, actual_category = get_random_image(VAL_PATH, CATEGORIES)

if random_image_path:
    print(f"🖼️ Wylosowany obrazek: {random_image_path}")
    print(f"✅ Faktyczna kategoria: {actual_category}")

    show_image(random_image_path)
    predicted_result = classify_animal(random_image_path, model, CATEGORIES, 128)
    print(f"🤖 Wynik klasyfikacji: {predicted_result}")
else:
    print("❌ Nie udało się wylosować obrazka.")

# Sprawdzenie konkretnego obrazka po ścieżce
CUSTOM_IMAGE_PATH = r"C:\Users\rjane\Desktop\projekty\System_wizyjny_schroniska\naklejka-twarz-leoparda.jpg"
classify_custom_image(CUSTOM_IMAGE_PATH)
