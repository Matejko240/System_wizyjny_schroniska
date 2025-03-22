import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.keras.models import load_model
from dataset_loader import load_images, split_dataset

_log_callback = print  # Domyślna funkcja logująca

def set_logger(logger_function):
    """Ustawia funkcję logującą komunikaty (np. do GUI)."""
    global _log_callback
    _log_callback = logger_function

def log(msg):
    """Loguje wiadomość za pomocą aktualnie ustawionej funkcji."""
    _log_callback(str(msg))

def create_model(img_size, num_classes):
    """Tworzy i zwraca ulepszony model CNN do klasyfikacji zwierząt."""
    model = keras.Sequential([
        Input(shape=(img_size, img_size, 3)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def load_or_train_model(model_path, dataset_path, categories, img_size=128, epochs=10):
    """Ładuje model z pliku lub trenuje nowy, jeśli model nie istnieje.
    Zwraca model oraz historię trenowania (jeśli dotyczy)."""

    if model_path.exists():
        log("📂 Wczytuję istniejący model...")
        model = load_model(model_path)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model, None  # Brak historii, bo model już był

    log(f"🚀 Brak modelu, rozpoczynam trenowanie na {epochs} epokach...")
    data, labels = load_images(dataset_path, categories, img_size)
    X_train, X_test, y_train, y_test = split_dataset(data, labels)

    model = create_model(img_size, len(categories))

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_test, y_test)
    )

    model.save(model_path)
    log(f"✅ Model zapisany jako {model_path}")

    return model, history
