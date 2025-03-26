import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback

from dataset_loader import load_images, split_dataset
import json
from logger_utils import log, set_logger
from constants import MODEL_PATH, DEFAULT_EPOCHS, DEFAULT_SHOW_IMAGES, DEFAULT_NUM_IMAGES, CATEGORIES, IMG_SIZE, TEST_PATH, VAL_PATH, TRAIN_PATH

class LoggingCallback(Callback):
    def on_epoch_end(self, epoch=DEFAULT_EPOCHS, logs=None):
        acc = logs.get("accuracy", 0)
        val_acc = logs.get("val_accuracy", 0)
        loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)
        log(f"📊 Epoka {epoch + 1} zakończona — acc: {acc:.4f}, val_acc: {val_acc:.4f}, loss: {loss:.4f}, val_loss: {val_loss:.4f}")


def create_model(img_size=IMG_SIZE, num_classes=len(CATEGORIES)):
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

def load_or_train_model(model_path=MODEL_PATH, img_size=IMG_SIZE, epochs=DEFAULT_EPOCHS,
                        num_train_images=None, num_val_images=None):
    if model_path.exists():
        log("📂 Wczytuję istniejący model...")
        model = load_model(model_path)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model, None

    log(f"🚀 Rozpoczynam trenowanie na {epochs} epokach...")

    # Załaduj i przytnij zbiór treningowy
    train_data, train_labels = load_images(TRAIN_PATH, CATEGORIES, img_size)
    if num_train_images:
        train_data = train_data[:num_train_images]
        train_labels = train_labels[:num_train_images]

    # Załaduj i przytnij zbiór walidacyjny
    val_data, val_labels = load_images(VAL_PATH, CATEGORIES, img_size)
    if num_val_images:
        val_data = val_data[:num_val_images]
        val_labels = val_labels[:num_val_images]

    model = create_model(img_size, len(CATEGORIES))

    history = model.fit(
        train_data, train_labels,
        epochs=epochs,
        validation_data=(val_data, val_labels),
        callbacks=[LoggingCallback()]
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    log(f"✅ Model zapisany jako {model_path}")

    history_path = model_path.with_suffix('.history.json')
    history_data = {
        "meta": {
            "epochs": epochs,
            "img_size": img_size,
            "num_train_images": num_train_images or len(train_data),
            "num_val_images": num_val_images or len(val_data)
        },
        "history": history.history
    }
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, indent=4)
    log(f"📝 Historia treningu zapisana do {history_path}")


    return model, history
