from pathlib import Path
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Automatyczna ścieżka do folderu dataset/train
BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "dataset" / "train"

# Kategorie zwierząt
CATEGORIES = ["cat", "wild", "dog"]
IMG_SIZE = 128  # Rozmiar obrazów

def load_images(dataset_path, categories, img_size):
    """Wczytuje i przetwarza obrazy do treningu."""
    data, labels = [], []
    for category in categories:
        path = dataset_path / category
        label = categories.index(category)

        if not path.exists():
            print(f"BŁĄD: Folder {path} nie istnieje!")
            continue
        
        for img_name in os.listdir(path):
            img_path = path / img_name
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.resize(img, (img_size, img_size))
                data.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Błąd przy wczytywaniu obrazu {img_path}: {e}")

    if len(data) == 0:
        raise ValueError("Nie załadowano żadnych obrazów! Sprawdź ścieżki do dataset.")

    return np.array(data) / 255.0, np.array(labels)

def split_dataset(data, labels, test_size=0.2):
    """Dzieli dane na zestawy treningowe i testowe."""
    return train_test_split(data, labels, test_size=test_size, random_state=42)



