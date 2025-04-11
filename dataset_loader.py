import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import json
from constants import *
from logger_utils import *

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

def get_balanced_subset_and_remainder(base_path, used_images, categories=CATEGORIES, img_size=IMG_SIZE):
    used, unused = [], []

    total_available = 0
    per_category_files = {}

    # Licz całkowitą liczbę zdjęć w folderze (per kategoria)
    for category in categories:
        cat_path = base_path / category
        files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        per_category_files[category] = files
        total_available += len(files)

    reserve_total = total_available - used_images
    reserve_per_cat = reserve_total // len(categories)

    for category in categories:
        files = per_category_files[category]
        random.shuffle(files)

        if len(files) < reserve_per_cat:
            log(f"⚠️ Za mało zdjęć w {category}: {len(files)} < {reserve_per_cat}")
            reserved_for_test = len(files)
        else:
            reserved_for_test = reserve_per_cat

        test_imgs = files[:reserved_for_test]
        train_imgs = files[reserved_for_test:]

        cat_path = base_path / category
        used += [(cat_path / f, category) for f in train_imgs]
        unused += [(cat_path / f, category) for f in test_imgs]

        log(f"📂 {category}: razem {len(files)}, test: {len(test_imgs)}, tren.: {len(train_imgs)}")

    return used, unused




def load_from_paths(image_paths, categories=CATEGORIES, img_size=IMG_SIZE):
    data, labels = [], []
    for path, category in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        data.append(img)
        labels.append(categories.index(category))
    return np.array(data) / 255.0, np.array(labels)


def evaluate_model_on_paths(model, image_paths, categories=CATEGORIES, img_size=IMG_SIZE):
    if not image_paths:
        log("⚠️ Brak danych testowych do oceny.")
        return {}

    data, labels = load_from_paths(image_paths, categories, img_size)
    predictions = model.predict(data)
    predicted_labels = np.argmax(predictions, axis=1)

    correct = np.sum(predicted_labels == labels)
    total = len(labels)
    accuracy = (correct / total) * 100 if total > 0 else 0.0

    # Tworzenie confusion matrix jako słownik
    confusion = defaultdict(lambda: defaultdict(int))
    for true, pred in zip(labels, predicted_labels):
        true_cat = categories[true]
        pred_cat = categories[pred]
        confusion[true_cat][pred_cat] += 1

    log(f"\n📊 Wyniki testu na {total} obrazach:")
    log(f"✔️ Trafne: {correct}")
    log(f"❌ Nietrafione: {total - correct}")
    log(f"🎯 Skuteczność: {accuracy:.2f}%\n")

    for true_cat in categories:
        for pred_cat in categories:
            count = confusion[true_cat][pred_cat]
            if count > 0:
                msg = f"📌 {count}× '{true_cat}' rozpoznano jako '{pred_cat}'"
                if true_cat != pred_cat:
                    msg += " ❌"
                else:
                    msg += " ✅"
                log(msg)


    # Oblicz metryki jakości
    metrics_per_class = {}

    for idx, cls in enumerate(categories):
        TP = confusion[cls].get(cls, 0)
        FN = sum(confusion[cls][c] for c in categories if c != cls)
        FP = sum(confusion[c][cls] for c in categories if c != cls)
        TN = sum(confusion[c1][c2] for c1 in categories for c2 in categories
                if c1 != cls and c2 != cls)

        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        type1_error = FP / (FP + TN) if (FP + TN) > 0 else 0
        type2_error = FN / (FN + TP) if (FN + TP) > 0 else 0

        metrics_per_class[cls] = {
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "precision": round(precision, 4),
            "f1_score": round(f1, 4),
            "type1_error": round(type1_error, 4),
            "type2_error": round(type2_error, 4),
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN
        }
    log("📐 Metryki jakości dla każdej klasy:")
    for cls, m in metrics_per_class.items():
        log(f"  🐾 Klasa '{cls}':")
        for k, v in m.items():
            log(f"     {k}: {v}")
        
    # Zwracamy dane testowe do pliku .history.json
    return {
        "total": total,
        "correct": int(correct),
        "incorrect": int(total - correct),
        "accuracy": accuracy,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "metrics_per_class": metrics_per_class
    }
def save_dataset_paths(train_paths, val_paths, test_paths, output_path):
    def serialize(paths):
        return [(str(p), c) for p, c in paths]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "train": serialize(train_paths),
            "val": serialize(val_paths),
            "test": serialize(test_paths)
        }, f, indent=2)
    log(f"💾 Zapisano ścieżki zestawu danych do {output_path}")

def load_dataset_paths(dataset_json_path):
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def deserialize(pairs):
        return [(Path(p), c) for p, c in pairs]

    log(f"📂 Wczytuję ścieżki zestawu danych z {dataset_json_path}")
    return {
        "train": deserialize(data["train"]),
        "val": deserialize(data["val"]),
        "test": deserialize(data["test"]),
    }
