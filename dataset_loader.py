import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import random
from constants import IMG_SIZE, CATEGORIES, VAL_PATH

def load_images(dataset_path, categories=CATEGORIES, img_size=IMG_SIZE):
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
        raise ValueError("Nie załadowano żadnych obrazów!")

    return np.array(data) / 255.0, np.array(labels)

def split_dataset(data, labels, test_size=0.2):
    return train_test_split(data, labels, test_size=test_size, random_state=42)

def get_random_images(val_path=VAL_PATH, categories=CATEGORIES, num_images=1):
    image_list = []

    for _ in range(num_images):
        category = random.choice(categories)
        category_path = val_path / category
        if not category_path.exists():
            continue
        images = os.listdir(category_path)
        if not images:
            continue
        image_name = random.choice(images)
        image_list.append((str(category_path / image_name), category))

    return image_list

def show_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Błąd: Plik {image_path} nie istnieje!")
        return

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Nazwa pliku: {os.path.basename(image_path)}", fontsize=12, fontweight="bold")
    plt.show(block=False)
    plt.pause(5)
    plt.close()
