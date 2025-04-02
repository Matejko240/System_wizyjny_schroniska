import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback
import time
import json

from dataset_loader import load_images, split_dataset
from logger_utils import log, set_logger
from constants import *
from dataset_loader import get_balanced_subset_and_remainder, load_from_paths, evaluate_model_on_paths

class LoggingCallback(Callback):
    def on_epoch_end(self, epoch=DEFAULT_EPOCHS, logs=None):
        acc = logs.get("accuracy", 0)
        val_acc = logs.get("val_accuracy", 0)
        loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)
        log(f"📊 Epoka {epoch + 1} zakończona — acc: {acc:.4f}, val_acc: {val_acc:.4f}, loss: {loss:.4f}, val_loss: {val_loss:.4f}")

class CustomEarlyStopping(Callback):
    def __init__(self, patience=PATIENCE):
        super().__init__()
        self.patience = patience
        self.val_losses = []
        self.early_stop_info = None

    def on_epoch_end(self, epoch, logs=None,):
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return

        self.val_losses.append(val_loss)

        if len(self.val_losses) >= self.patience * 2:
            recent = self.val_losses[-self.patience:]
            previous = self.val_losses[-2*self.patience:-self.patience]
            avg_recent = sum(recent) / len(recent)
            avg_previous = sum(previous) / len(previous)

            if avg_recent > avg_previous:
                reason = f"avg val_loss {avg_recent:.4f} > wcześniejsza avg {avg_previous:.4f}"
                log(f"🛑 Early stopping aktywowany na epoko {epoch+1} ({reason})")
                self.early_stop_info = {
                    "stopped_at_epoch": epoch + 1,
                    "reason": reason,
                    "patience": self.patience
                }
                self.model.stop_training = True



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
                        num_train_images=DEFAULT_TRAIN_IMAGES, num_val_images=DEFAULT_VAL_IMAGES):

    if model_path.exists():
        log("📂 Wczytuję istniejący model...")
        model = load_model(model_path)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model, None

    log(f"🚀 Rozpoczynam trenowanie na {epochs} epokach...")

    train_paths, train_unused = get_balanced_subset_and_remainder(TRAIN_PATH, num_train_images)
    val_paths, val_unused = get_balanced_subset_and_remainder(VAL_PATH, num_val_images)

    test_paths = train_unused + val_unused


    train_data, train_labels = load_from_paths(train_paths, CATEGORIES, img_size)
    val_data, val_labels = load_from_paths(val_paths, CATEGORIES, img_size)

    model = create_model(img_size, len(CATEGORIES))
    early_stop_cb = CustomEarlyStopping(patience=PATIENCE)
    start_time = time.time()
    history = model.fit(
        train_data, train_labels,
        epochs=epochs,
        validation_data=(val_data, val_labels),
        callbacks=[LoggingCallback(), early_stop_cb]
    )
    end_time = time.time()
    training_duration = end_time - start_time

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    log(f"✅ Model zapisany jako {model_path}")


    history_data = {
        "meta": {
            "epochs": epochs,
            "img_size": img_size,
            "training_time_sec": training_duration,
            "num_train_images": num_train_images or len(train_data),
            "num_val_images": num_val_images or len(val_data)
        },
        "history": history.history
    }
    if early_stop_cb.early_stop_info:
        history_data["early_stopping"] = early_stop_cb.early_stop_info
    # test
    test_results = evaluate_model_on_paths(model, test_paths, CATEGORIES, img_size)
    history_data["test_results"] = test_results
    #zapisanie historii
    history_path = model_path.with_suffix('.history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, indent=4)
    log(f"📝 Historia treningu zapisana do {history_path}")
    
    return model, history

