import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input

def create_model(img_size, num_classes):
    """Tworzy i zwraca model CNN do klasyfikacji zwierząt."""
    model = keras.Sequential([
        Input(shape=(img_size, img_size, 3)),  # ✅ Dodano warstwę wejściową
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # ✅ Poprawna warstwa wyjściowa dla klasyfikacji wieloklasowej
    ])

    # ✅ Optymalizacja dla CPU i GPU
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model
