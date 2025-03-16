import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input

from tensorflow.keras.layers import Dropout  # ✅ Dodaj dropout

def create_model(img_size, num_classes):
    """Tworzy i zwraca ulepszony model CNN do klasyfikacji zwierząt."""
    model = keras.Sequential([
        Input(shape=(img_size, img_size, 3)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        Dropout(0.2),  # ✅ Dodano dropout
        
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        Dropout(0.3),  # ✅ Więcej dropout
        
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        Dropout(0.4),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        Dropout(0.5),  # ✅ Zapobiega przeuczeniu
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model
