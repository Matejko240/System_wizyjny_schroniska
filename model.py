import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback
import time
import json
from statistics import mean, median, stdev
from dataset_loader import load_images, split_dataset, load_dataset_paths, evaluate_model_on_paths
from logger_utils import log, set_logger
from constants import *
from dataset_loader import get_balanced_subset_and_remainder, load_from_paths, evaluate_model_on_paths, load_dataset_paths, save_dataset_paths

class LoggingCallback(Callback):
    def on_epoch_end(self, epoch=DEFAULT_EPOCHS, logs=None):
        acc = logs.get("accuracy", 0)
        val_acc = logs.get("val_accuracy", 0)
        loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)
        log(f"📊 Epoka {epoch + 1} zakończona — acc: {acc:.4f}, val_acc: {val_acc:.4f}, loss: {loss:.4f}, val_loss: {val_loss:.4f}")

class CustomEarlyStopping(Callback):
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.val_losses = []
        self.early_stop_info = None

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return

        self.val_losses.append(val_loss)

        if len(self.val_losses) < self.patience + 1:
            return  # za mało danych nawet na przesunięcie o 1

        # Tworzymy dwa okna z przesunięciem o jedną epokę
        recent = self.val_losses[-self.patience:]
        previous = self.val_losses[-(self.patience + 1):-1]

        avg_recent = sum(recent) / len(recent)
        avg_previous = sum(previous) / len(previous)

        if avg_recent - avg_previous  > self.min_delta:
            reason = (
                f"val_loss się pogarsza: {avg_previous:.4f} → {avg_recent:.4f}, "
                f"delta = {avg_recent - avg_previous:.6f} > {self.min_delta:.6f}"
            )
            log(f"🛑 Early stopping aktywowany na epoko {epoch+1} ({reason})")
            self.early_stop_info = {
                "stopped_at_epoch": epoch + 1,
                "reason": reason,
                "patience": self.patience,
                "min_delta": self.min_delta
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
                        num_train_images=DEFAULT_TRAIN_IMAGES, num_val_images=DEFAULT_VAL_IMAGES,  dataset_json_path=DATASET_JSON_PATH):

    if model_path.exists():
        log("📂 Wczytuję istniejący model...")
        model = load_model(model_path)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model, None

    log(f"🚀 Rozpoczynam trenowanie na {epochs} epokach...")

    
    # 🔁 Użycie gotowych ścieżek z pliku (jeśli podano)
    if dataset_json_path and Path(dataset_json_path).exists():
        sets = load_dataset_paths(dataset_json_path)
        train_paths = sets["train"][:num_train_images]
        val_paths = sets["val"][:num_val_images]
        test_paths = sets["test"]
    else:
        train_paths, train_unused = get_balanced_subset_and_remainder(TRAIN_PATH, num_train_images)
        val_paths, val_unused = get_balanced_subset_and_remainder(VAL_PATH, num_val_images)
        test_paths = train_unused + val_unused

        if dataset_json_path:
            save_dataset_paths(train_paths, val_paths, test_paths, dataset_json_path)



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

def repeat_evaluation(model_path, dataset_json_path, img_size=IMG_SIZE, runs=RUNS):


    if not model_path.exists():
        log(f"❌ Model {model_path} nie istnieje.")
        return

    if not Path(dataset_json_path).exists():
        log(f"❌ Dataset {dataset_json_path} nie istnieje.")
        return

    model = load_model(model_path)
    sets = load_dataset_paths(dataset_json_path)
    test_paths = sets["test"]

    acc_list = []

    for i in range(runs):
        log(f"\n🔁 Test {i + 1} z {runs}")
        results = evaluate_model_on_paths(model, test_paths, CATEGORIES, img_size)
        acc = results.get("accuracy", 0)
        acc_list.append(acc)

    log("\n📊 Statystyki stabilności:")
    log(f"Średnia accuracy: {mean(acc_list):.2f}%")
    log(f"Mediana accuracy: {median(acc_list):.2f}%")
    if len(acc_list) > 1:
        log(f"Odchylenie standardowe: {stdev(acc_list):.2f}")
    return acc_list

def repeat_training(model_path_base, dataset_json_path, img_size=IMG_SIZE, runs=RUNS, epochs=DEFAULT_EPOCHS):
    acc_list = []

    for i in range(runs):
        log(f"\n🔁 Trenowanie modelu {i + 1} z {runs}...")

        current_model_path = model_path_base.with_stem(f"{model_path_base.stem}_run{i+1}")

        model, history = load_or_train_model(
            model_path=current_model_path,
            img_size=img_size,
            epochs=epochs,
            num_train_images=DEFAULT_TRAIN_IMAGES,
            num_val_images=DEFAULT_VAL_IMAGES,
            dataset_json_path=dataset_json_path
        )

        # Wczytaj plik historii JSON ręcznie
        history_path = current_model_path.with_suffix(".history.json")
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                hist_data = json.load(f)
                acc = hist_data.get("test_results", {}).get("accuracy", 0)
                acc_list.append(acc)
                log(f"🎯 Accuracy testowe po run {i+1}: {acc:.2f}%")
        else:
            log(f"⚠️ Brak pliku historii dla run {i+1}")


    # Statystyki końcowe
    if acc_list:
        log("\n📊 Statystyki stabilności treningu:")
        log(f"Średnia accuracy: {mean(acc_list):.2f}%")
        log(f"Mediana accuracy: {median(acc_list):.2f}%")
        if len(acc_list) > 1:
            log(f"Odchylenie standardowe: {stdev(acc_list):.2f}")

    return acc_list

