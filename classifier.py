import cv2
import numpy as np
from pathlib import Path
import os
from dataset_loader import get_random_images, show_image

def classify_animal(image_path, model, categories, img_size):
    """Klasyfikuje obrazek zwierzęcia i zwraca decyzję."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return f"Nie udało się wczytać obrazu: {image_path}"
    
    img = cv2.resize(img, (img_size, img_size))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)[0]
    predicted_class = np.argmax(prediction)

    for i, category in enumerate(categories):
        print(f"{category}: {prediction[i] * 100:.2f}%")

    animal = categories[predicted_class]
    access = "Dostęp przyznany" if animal in ["cat", "dog"] else "Dostęp zabroniony"
    return f"Rozpoznano: {animal}. {access} \n" + "=" * 64

def classify_random_images(val_path, model, categories, img_size, num_images=1, show_images=False):
    """Losuje podaną liczbę obrazków z folderu walidacyjnego, wyświetla je i klasyfikuje."""
    random_images = get_random_images(val_path, categories, num_images)

    if not random_images:
        print("❌ Nie udało się wylosować obrazków.")
        return
    
    for image_path, actual_category in random_images:
        print(f"🖼️ Wylosowany obrazek: {image_path}")
        print(f"✅ Faktyczna kategoria: {actual_category}")

        if show_images:
            show_image(image_path)
        
        print(f"🤖 Wynik klasyfikacji: {classify_animal(image_path, model, categories, img_size)}")


        
def classify_all_images_in_folder(folder_path, model, categories, img_size, show_images=False):
    """Przetwarza wszystkie obrazy w podanym folderze testowym. Może wyświetlać obrazy, jeśli show_images=True."""
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"❌ Błąd: Folder {folder_path} nie istnieje!")
        return

    for image_name in os.listdir(folder):
        image_path = folder / image_name
        if image_path.is_file():
            if show_images:
                show_image(image_path)

            print(f"🔍 Sprawdzam obrazek: {image_name}")
            print(classify_animal(str(image_path), model, categories, img_size))
