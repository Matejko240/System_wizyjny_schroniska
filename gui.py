from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout,QGridLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QSpinBox, QCheckBox, QLineEdit, QTextEdit
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6 import QtGui
import sys
import cv2
import os
from PIL import Image
from pathlib import Path
from classifier import classify_animal, classify_random_images, classify_all_images_in_folder
from tensorflow.keras.models import load_model
from model import load_or_train_model
import matplotlib
matplotlib.use('QtAgg')  # backend zgodny z PyQt
import matplotlib.pyplot as plt
from constants import *
import json
from logger_utils import log, set_logger

class TrainingThread(QThread):
    training_finished = pyqtSignal(object, object)

    def __init__(self, model_path, img_size, epochs, num_train, num_val):
        super().__init__()
        self.model_path = model_path
        self.img_size = img_size
        self.epochs = epochs
        self.num_train = num_train
        self.num_val = num_val

    def run(self):
        model, history = load_or_train_model(
            self.model_path,
            self.img_size,
            self.epochs,
            self.num_train,
            self.num_val
        )
        self.training_finished.emit(model, history)


class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Klasyfikacja obrazów")
        self.setGeometry(100, 100, 600, 700)
        if MODEL_PATH.exists():
            self.model = load_model(MODEL_PATH)
        else:
            self.model, _ = load_or_train_model(MODEL_PATH)

        self.layout = QVBoxLayout()

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setVisible(False)  # domyślnie ukryte
        self.layout.addWidget(self.image_label)


        self.upload_btn = QPushButton("Wybierz obraz")
        self.upload_btn.clicked.connect(self.load_image)
        self.layout.addWidget(self.upload_btn)

        self.classify_btn = QPushButton("Klasyfikuj")
        self.classify_btn.setEnabled(False)
        self.classify_btn.clicked.connect(self.classify_image)
        self.layout.addWidget(self.classify_btn)

        self.history_label = QLabel("Historia klasyfikacji:")
        self.layout.addWidget(self.history_label)
        self.history_box = QTextEdit()
        self.history_box.setReadOnly(True)
        self.history_box.setMinimumHeight(200)  # Ustawienie większej wysokości przy starcie
        self.layout.addWidget(self.history_box)

        self.train_btn = QPushButton("Trenuj model")
        self.train_btn.clicked.connect(self.train_model)
        self.layout.addWidget(self.train_btn)
        
        self.history = None
        self.plot_btn = QPushButton("📈 Pokaż wykres uczenia")
        self.plot_btn.clicked.connect(self.plot_training_curve)
        self.plot_btn.setEnabled(False)
        self.layout.addWidget(self.plot_btn)

        self.plot_from_file_btn = QPushButton("📄 Pokaż historię z pliku")
        self.plot_from_file_btn.clicked.connect(self.plot_training_curve_from_file)
        self.layout.addWidget(self.plot_from_file_btn)

        
        self.classify_random_btn = QPushButton("Klasyfikuj losowe obrazy")
        self.classify_random_btn.clicked.connect(self.classify_random_images)
        self.layout.addWidget(self.classify_random_btn)

        self.classify_all_btn = QPushButton("Klasyfikuj wszystkie obrazy")
        self.classify_all_btn.clicked.connect(self.classify_all_images)
        self.layout.addWidget(self.classify_all_btn)
        
        self.max_train = self.count_images_in_folder(TRAIN_PATH, CATEGORIES)
        self.max_val = self.count_images_in_folder(VAL_PATH, CATEGORIES)

        
        settings_layout = QGridLayout()
        
        # Liczba epok
        settings_layout.addWidget(QLabel("Liczba epok:"), 0, 0)
        self.epochs_input = QSpinBox()
        self.epochs_input.setValue(DEFAULT_EPOCHS)
        settings_layout.addWidget(self.epochs_input, 0, 1)

        # Liczba zdjęć treningowych
        settings_layout.addWidget(QLabel("Liczba zdjęć treningowych:"), 1, 0)
        train_box = QHBoxLayout()
        self.train_img_input = QSpinBox()
        self.train_img_input.setRange(1, self.max_train)
        self.train_img_input.setValue(min(14630, self.max_train))
        train_box.addWidget(self.train_img_input)
        self.train_img_percent = QLabel()
        train_box.addWidget(self.train_img_percent)
        settings_layout.addLayout(train_box, 1, 1)

        # Liczba zdjęć walidacyjnych
        settings_layout.addWidget(QLabel("Liczba zdjęć walidacyjnych:"), 2, 0)
        val_box = QHBoxLayout()
        self.val_img_input = QSpinBox()
        self.val_img_input.setRange(1, self.max_val)
        self.val_img_input.setValue(min(1500, self.max_val))
        val_box.addWidget(self.val_img_input)
        self.val_img_percent = QLabel()
        val_box.addWidget(self.val_img_percent)
        settings_layout.addLayout(val_box, 2, 1)

        # Nazwa pliku modelu
        settings_layout.addWidget(QLabel("Nazwa pliku modelu:"), 3, 0)
        self.model_path_input = QLineEdit()
        self.model_path_input.setText(MODEL_PATH.name)
        settings_layout.addWidget(self.model_path_input, 3, 1)
        
        # Liczba obrazków do klasyfikacji
        settings_layout.addWidget(QLabel("Liczba obrazków do klasyfikacji:"), 4, 0)
        self.num_images_input = QSpinBox()
        self.num_images_input.setValue(DEFAULT_NUM_IMAGES)
        settings_layout.addWidget(self.num_images_input, 4, 1)

                
        # Policz dostępne dane i ustaw limity
        self.train_img_input.setRange(1, self.max_train)
        self.val_img_input.setRange(1, self.max_val)

        # Powiąż zmiany ze zmianą opisu %
        self.train_img_input.valueChanged.connect(self.update_train_percent)
        self.val_img_input.valueChanged.connect(self.update_val_percent)

        self.update_train_percent()
        self.update_val_percent()
        
        self.layout.addLayout(settings_layout)
        
        
        self.show_images_checkbox = QCheckBox("Pokazuj obrazy")
        self.show_images_checkbox.setChecked(DEFAULT_SHOW_IMAGES)
        self.layout.addWidget(self.show_images_checkbox)
        
        self.setLayout(self.layout)
        self.image_path = None

        # Ustawienie funkcji logującej
        set_logger(self.append_to_history)

        
    def append_to_history(self, msg):
        self.history_box.append(str(msg))
        self.history_box.moveCursor(QtGui.QTextCursor.MoveOperation.End)

        doc_height = self.history_box.document().size().height()
        new_height = int(doc_height * 2)

        # Ustaw maksymalną wysokość, np. 400 px
        self.history_box.setMinimumHeight(min(new_height, 300))



    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Wybierz obraz", "", "Obrazy (*.jpg *.png *.jpeg)")
        if file_path:
            self.image_path = file_path
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img.thumbnail((200, 200))

            qt_img = QImage(img.tobytes(), img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            self.image_label.setVisible(True)
            self.image_label.setPixmap(pixmap)
            self.classify_btn.setEnabled(True)

    def classify_image(self):
        if self.image_path:
            classify_animal(self.image_path, self.model, CATEGORIES, IMG_SIZE)

    def train_model(self):
        epochs = self.epochs_input.value()
        model_path = MODELS_DIR / self.model_path_input.text()
        num_train = self.train_img_input.value()
        num_val = self.val_img_input.value()

        self.training_thread = TrainingThread(model_path, IMG_SIZE, epochs, num_train, num_val)
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.training_thread.start()

        
    def on_training_finished(self, model, history):
        self.model = model
        self.history = history
        self.plot_btn.setEnabled(self.history is not None)

    def classify_random_images(self):
        num_images = self.num_images_input.value()
        show_images = self.show_images_checkbox.isChecked()
        results = classify_random_images(VAL_PATH, self.model, CATEGORIES, IMG_SIZE, num_images, show_images) or []
        self.show_statistics(results)

    def show_statistics(self, results):
        total = len(results)
        correct = sum(1 for _, actual, predicted, _ in results if actual == predicted)
        accuracy = (correct / total * 100) if total > 0 else 0.0
        stats = f"\n📊 Statystyki walidacji:\n✔️ Trafne: {correct}\n❌ Nietrafione: {total - correct}\n🎯 Skuteczność: {accuracy:.2f}%"
        self.history_box.append(stats)

    def classify_all_images(self):
        show_images = self.show_images_checkbox.isChecked()
        classify_all_images_in_folder(TEST_PATH, self.model, CATEGORIES, IMG_SIZE, show_images) or []

    def draw_training_plot(self, history_data):
        acc = history_data.get('accuracy', [])
        val_acc = history_data.get('val_accuracy', [])
        loss = history_data.get('loss', [])
        val_loss = history_data.get('val_loss', [])
        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b', label='Treningowa')
        plt.plot(epochs, val_acc, 'r', label='Walidacyjna')
        plt.title('Dokładność (accuracy) modelu')
        plt.xlabel('Epoka')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'b', label='Treningowa')
        plt.plot(epochs, val_loss, 'r', label='Walidacyjna')
        plt.title('Strata (loss) modelu')
        plt.xlabel('Epoka')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show(block=False)


    def plot_training_curve(self):
        if not self.history:
            return
        self.draw_training_plot(self.history.history)

    def plot_training_curve_from_file(self):
        model_path = MODELS_DIR / self.model_path_input.text()
        history_path = model_path.with_suffix('.history.json')

        if not history_path.exists():
            self.append_to_history(f"❌ Plik historii {history_path} nie istnieje.")
            return

        with open(history_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)

        history_data = full_data["history"]
        self.draw_training_plot(history_data)

        
    def count_images_in_folder(self, folder, categories):
        total = 0
        for category in categories:
            path = folder / category
            if path.exists():
                total += len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        return total
    
    def update_train_percent(self):
        val = self.train_img_input.value()
        percent = (val / self.max_train) * 100 if self.max_train else 0
        self.train_img_percent.setText(f"{percent:.1f}% z dostępnych")

    def update_val_percent(self):
        val = self.val_img_input.value()
        percent = (val / self.max_val) * 100 if self.max_val else 0
        self.val_img_percent.setText(f"{percent:.1f}% z dostępnych")
    
    