import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback

from dataset_loader import load_images, split_dataset
import json
from logger_utils import log, set_logger

class LoggingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get("accuracy", 0)
        val_acc = logs.get("val_accuracy", 0)
        loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)
        log(f"📊 Epoka {epoch + 1} zakończona — acc: {acc:.4f}, val_acc: {val_acc:.4f}, loss: {loss:.4f}, val_loss: {val_loss:.4f}")


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
        validation_data=(X_test, y_test),
        callbacks=[LoggingCallback()]
    )


    model.save(model_path)
    log(f"✅ Model zapisany jako {model_path}")
    history_path = model_path.with_suffix('.history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history.history, f, indent=4)
    log(f"📝 Historia treningu zapisana do {history_path}")

    return model, history
