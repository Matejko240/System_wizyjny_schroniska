import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback
# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

import numpy as np
import time
import json
from statistics import mean, median, stdev
from dataset_loader import load_images, split_dataset, load_dataset_paths, evaluate_model_on_paths
from logger_utils import log, set_logger
from constants import *
from dataset_loader import get_balanced_subset_and_remainder, load_from_paths, evaluate_model_on_paths, load_dataset_paths, save_dataset_paths
from experiment_logger import log_experiment
from dataset_loader import add_gaussian_noise
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



def create_model(img_size=IMG_SIZE, num_classes=len(CATEGORIES), num_conv_layers=3, activation_function='relu', optimizer='adam'):
    model = keras.Sequential([Input(shape=(img_size, img_size, 3))])

    filters = 32
    for i in range(num_conv_layers):
        model.add(layers.Conv2D(filters, (3, 3), activation=activation_function))
        model.add(layers.MaxPooling2D(2, 2))
        filters *= 2

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=activation_function))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model




def load_or_train_model(model_path=MODEL_PATH, img_size=IMG_SIZE, epochs=DEFAULT_EPOCHS,
                        num_train_images=DEFAULT_TRAIN_IMAGES, num_val_images=DEFAULT_VAL_IMAGES,  dataset_json_path=DATASET_JSON_PATH, batch_size=BATCH_SIZE):

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
        batch_size=batch_size,
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

def repeat_training(
    model_path_base,
    dataset_json_path,
    img_size=IMG_SIZE,
    runs=RUNS,
    epochs=DEFAULT_EPOCHS,
    batch_size=32,
    optimizer_name="adam",
    activation_function="relu",
    conv_layers=3,
    add_noise=False
):
    from experiment_logger import log_experiment
    acc_list = []
    training_times = []
    confusions = []
    metrics_list = []

    for i in range(runs):
        log(f"\n🔁 Trenowanie modelu {i + 1} z {runs}...")

        current_model_path = model_path_base.with_stem(f"{model_path_base.stem}_run{i+1}")
        start_time = time.time()
        """
        # Przygotowanie danych
        if dataset_json_path and Path(dataset_json_path).exists():
            sets = load_dataset_paths(dataset_json_path)
            train_paths = sets["train"][:DEFAULT_TRAIN_IMAGES]
            val_paths = sets["val"][:DEFAULT_VAL_IMAGES]
            test_paths = sets["test"]
        else:
            train_paths, train_unused = get_balanced_subset_and_remainder(TRAIN_PATH, DEFAULT_TRAIN_IMAGES)
            val_paths, val_unused = get_balanced_subset_and_remainder(VAL_PATH, DEFAULT_VAL_IMAGES)
            test_paths = train_unused + val_unused
            if dataset_json_path:
                save_dataset_paths(train_paths, val_paths, test_paths, dataset_json_path)
        """
        train_paths, train_unused = get_balanced_subset_and_remainder(TRAIN_PATH, DEFAULT_TRAIN_IMAGES)
        val_paths, val_unused = get_balanced_subset_and_remainder(VAL_PATH, DEFAULT_VAL_IMAGES)
        test_paths = train_unused + val_unused

        train_data, train_labels = load_from_paths(train_paths, CATEGORIES, img_size)
        val_data, val_labels = load_from_paths(val_paths, CATEGORIES, img_size)

        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        val_data = np.array(val_data)
        val_labels = np.array(val_labels)

        if add_noise:
            from dataset_loader import add_gaussian_noise
            train_data = np.array([add_gaussian_noise(img) for img in train_data])
            val_data = np.array([add_gaussian_noise(img) for img in val_data])

        # Budowa modelu
        model = keras.Sequential([Input(shape=(img_size, img_size, 3))])
        filters = 32
        for _ in range(conv_layers):
            model.add(layers.Conv2D(filters, (3, 3), activation=activation_function))
            model.add(layers.MaxPooling2D(2, 2))
            filters *= 2

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation=activation_function))
        model.add(layers.Dense(len(CATEGORIES), activation='softmax'))

        optimizer_mapping = {
            "adam": keras.optimizers.Adam(),
            "sgd": keras.optimizers.SGD(),
            "rmsprop": keras.optimizers.RMSprop(),
            "adagrad": keras.optimizers.Adagrad()
        }

        # Używamy odpowiedniego optymalizatora w zależności od przekazanej nazwy
        optimizer = optimizer_mapping.get(optimizer_name.lower(), keras.optimizers.Adam())


        model.compile(optimizer=optimizer, 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])

        early_stop_cb = CustomEarlyStopping(patience=PATIENCE)

        history = model.fit(
            train_data, train_labels,
            epochs=epochs,
            validation_data=(val_data, val_labels),
            batch_size=batch_size,
            callbacks=[LoggingCallback(), early_stop_cb]
        )

        end_time = time.time()
        training_times.append(end_time - start_time)

        # Zapisz model
        current_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(current_model_path)

        # Ocena modelu
        test_results = evaluate_model_on_paths(model, test_paths, CATEGORIES, img_size)
        acc = test_results.get("accuracy", 0)
        acc_list.append(acc)

        log(f"🎯 Accuracy testowe po run {i+1}: {acc:.2f}%")

        confusions.append(test_results.get("confusion_matrix", {}))
        metrics_list.append(test_results.get("metrics_per_class", {}))

        # Zapisz historię
        history_data = {
            "meta": {
                "epochs": epochs,
                "img_size": img_size,
                "training_time_sec": end_time - start_time,
                "num_train_images": len(train_data),
                "num_val_images": len(val_data)
            },
            "history": history.history,
            "test_results": test_results
        }  
        if early_stop_cb.early_stop_info:
            history_data["early_stopping"] = early_stop_cb.early_stop_info
        history_path = current_model_path.with_suffix('.history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=4)

    # Statystyki po wszystkich runach
    if acc_list:
        mean_acc = mean(acc_list)
        median_acc = median(acc_list)
        std_acc = stdev(acc_list) if len(acc_list) > 1 else 0
        avg_time = mean(training_times) if training_times else 0

        # Oblicz średnią confusion matrix
        avg_confusion = {cat: {c: 0 for c in CATEGORIES} for cat in CATEGORIES}
        for confusion in confusions:
            for true_cat in confusion:
                for pred_cat in confusion[true_cat]:
                    avg_confusion[true_cat][pred_cat] += confusion[true_cat][pred_cat]
        for true_cat in avg_confusion:
            for pred_cat in avg_confusion[true_cat]:
                avg_confusion[true_cat][pred_cat] /= runs

        # Oblicz średnie metryki jakości
        avg_metrics = {cat: {k: 0 for k in ["recall", "specificity", "precision", "f1_score", "type1_error", "type2_error", "TP", "FP", "FN", "TN"]} for cat in CATEGORIES}
        for metrics in metrics_list:
            for cat in metrics:
                for key in metrics[cat]:
                    avg_metrics[cat][key] += metrics[cat][key]
        for cat in avg_metrics:
            for key in avg_metrics[cat]:
                avg_metrics[cat][key] /= runs

        # Przygotuj dane do zapisania
        experiment_result = {
            "resolution": f"{img_size}x{img_size}",
            "noise": "yes" if add_noise else "no",
            "conv_layers": conv_layers,
            "activation": activation_function,
            "optimizer": optimizer_name,
            "batch_size": batch_size,
            "mean_accuracy": round(mean_acc, 2),
            "median_accuracy": round(median_acc, 2),
            "std_dev_accuracy": round(std_acc, 2),
            "avg_training_time_sec": round(avg_time, 2)
        }

        # Dodaj confusion matrix do zapisu
        for true_cat in avg_confusion:
            for pred_cat in avg_confusion[true_cat]:
                key = f"{true_cat}_{pred_cat}"
                experiment_result[key] = round(avg_confusion[true_cat][pred_cat], 2)

        # Dodaj średnie metryki do zapisu
        for cat in avg_metrics:
            for metric_name in avg_metrics[cat]:
                key = f"{cat}_{metric_name}"
                experiment_result[key] = round(avg_metrics[cat][metric_name], 4)

        # Zapisz wszystko
        log_experiment(experiment_result)
        log("✅ Wynik eksperymentu zapisany do CSV.")

    return acc_list
