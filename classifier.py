import cv2
import numpy as np

def classify_animal(image_path, model, categories, img_size):
    """Klasyfikuje obrazek zwierzęcia i zwraca decyzję."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return f"Nie udało się wczytać obrazu: {image_path}"
    
    img = cv2.resize(img, (img_size, img_size))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)[0]  # Pobiera wynik jako tablicę
    predicted_class = np.argmax(prediction)

    # Wyświetlenie dokładnych procentowych wyników
    for i, category in enumerate(categories):
        print(f"{category}: {prediction[i] * 100:.2f}%")

    animal = categories[predicted_class]
    access = "Dostęp przyznany" if animal in ["cat", "dog"] else "Dostęp zabroniony"
    return f"Rozpoznano: {animal}. {access}"

