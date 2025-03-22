import cv2
import numpy as np
from pathlib import Path
import os
from dataset_loader import get_random_images, show_image
from constants import IMG_SIZE, CATEGORIES  # <- dodano

_log_callback = print  # Domyślna funkcja logująca

def set_logger(logger_function):
    global _log_callback
    _log_callback = logger_function

def log(msg):
    _log_callback(str(msg))

def classify_animal(image_path, model, categories=CATEGORIES, img_size=IMG_SIZE):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None, f"Nie udało się wczytać obrazu: {image_path}"

    img = cv2.resize(img, (img_size, img_size))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]
    predicted_class = np.argmax(prediction)

    for i, category in enumerate(categories):
        log(f"{category}: {prediction[i] * 100:.2f}%")

    animal = categories[predicted_class]
    access = "Dostęp przyznany" if animal in ["cat", "dog"] else "Dostęp zabroniony"
    result_msg = f"Rozpoznano: {animal}. {access} \n" + "=" * 64
    log(result_msg)
    return animal, result_msg

def classify_random_images(val_path, model, categories=CATEGORIES, img_size=IMG_SIZE, num_images=1, show_images=False):
    random_images = get_random_images(val_path, categories, num_images)
    results = []

    if not random_images:
        msg = "❌ Nie udało się wylosować obrazków."
        log(msg)
        return []

    for image_path, actual_category in random_images:
        log_msg1 = f"🖼️ Wylosowany obrazek: {Path(image_path).relative_to(val_path.parent)}"
        log_msg2 = f"✅ Faktyczna kategoria: {actual_category}"
        log(log_msg1)
        log(log_msg2)

        if show_images:
            show_image(image_path)

        predicted_category, result_msg = classify_animal(image_path, model, categories, img_size)
        results.append((str(image_path), actual_category, predicted_category, result_msg))

    return results

def classify_all_images_in_folder(folder_path, model, categories=CATEGORIES, img_size=IMG_SIZE, show_images=False):
    folder = Path(folder_path)
    results = []
    if not folder.exists() or not folder.is_dir():
        log(f"❌ Błąd: Folder {folder_path} nie istnieje!")
        return []

    for image_name in os.listdir(folder):
        image_path = folder / image_name
        if image_path.is_file():
            if show_images:
                show_image(image_path)

            log(f"🔍 Sprawdzam obrazek: {image_name}")
            predicted_category, result_msg = classify_animal(str(image_path), model, categories, img_size)
            results.append((str(image_path), None, predicted_category, result_msg))

    return results
