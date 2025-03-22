from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QSpinBox, QCheckBox, QLineEdit, QTextEdit
from PyQt6.QtGui import QPixmap, QImage
import sys
import cv2
from PIL import Image
from pathlib import Path
from classifier import classify_animal, classify_random_images, classify_all_images_in_folder, set_logger
from tensorflow.keras.models import load_model
from model import load_or_train_model
import matplotlib.pyplot as plt
from constants import MODEL_PATH, DEFAULT_EPOCHS, DEFAULT_SHOW_IMAGES, DEFAULT_NUM_IMAGES, CATEGORIES, IMG_SIZE, TEST_PATH, VAL_PATH, TRAIN_PATH
# Wczytaj model

model = load_model(MODEL_PATH)


class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Klasyfikacja obrazów")
        self.setGeometry(100, 100, 600, 700)

        self.layout = QVBoxLayout()

        self.image_label = QLabel("Brak obrazu")
        self.image_label.setFixedSize(300, 300)
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

        self.classify_random_btn = QPushButton("Klasyfikuj losowe obrazy")
        self.classify_random_btn.clicked.connect(self.classify_random_images)
        self.layout.addWidget(self.classify_random_btn)

        self.classify_all_btn = QPushButton("Klasyfikuj wszystkie obrazy")
        self.classify_all_btn.clicked.connect(self.classify_all_images)
        self.layout.addWidget(self.classify_all_btn)

        
        # Dodanie ustawień
        self.epochs_label = QLabel("Liczba epok:")
        self.layout.addWidget(self.epochs_label)
        self.epochs_input = QSpinBox()
        self.epochs_input.setValue(DEFAULT_EPOCHS)
        self.layout.addWidget(self.epochs_input)
        
        self.model_path_label = QLabel("Nazwa pliku modelu:")
        self.layout.addWidget(self.model_path_label)
        self.model_path_input = QLineEdit()
        self.model_path_input.setText(MODEL_PATH.name)
        self.layout.addWidget(self.model_path_input)
        
        self.num_images_label = QLabel("Liczba obrazków do klasyfikacji:")
        self.layout.addWidget(self.num_images_label)
        self.num_images_input = QSpinBox()
        self.num_images_input.setValue(DEFAULT_NUM_IMAGES)
        self.layout.addWidget(self.num_images_input)

        
        self.show_images_checkbox = QCheckBox("Pokazuj obrazy")
        self.show_images_checkbox.setChecked(DEFAULT_SHOW_IMAGES)
        self.layout.addWidget(self.show_images_checkbox)
        
        self.setLayout(self.layout)
        self.image_path = None

        # Ustawienie funkcji logującej
        set_logger(self.append_to_history)

    def append_to_history(self, msg):
        self.history_box.append(str(msg))

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Wybierz obraz", "", "Obrazy (*.jpg *.png *.jpeg)")
        if file_path:
            self.image_path = file_path
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img.thumbnail((300, 300))

            qt_img = QImage(img.tobytes(), img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            self.image_label.setPixmap(pixmap)
            self.classify_btn.setEnabled(True)

    def classify_image(self):
        if self.image_path:
            classify_animal(self.image_path, model, CATEGORIES, IMG_SIZE)

    def train_model(self):
        epochs = self.epochs_input.value()
        model_path = Path(self.model_path_input.text())
        global model
        model, self.history = load_or_train_model(model_path, TRAIN_PATH, CATEGORIES, IMG_SIZE, epochs)
        self.plot_btn.setEnabled(self.history is not None)

    def classify_random_images(self):
        num_images = self.num_images_input.value()
        show_images = self.show_images_checkbox.isChecked()
        results = classify_random_images(VAL_PATH, model, CATEGORIES, IMG_SIZE, num_images, DEFAULT_SHOW_IMAGES) or []
        self.show_statistics(results)

    def show_statistics(self, results):
        total = len(results)
        correct = sum(1 for _, actual, predicted, _ in results if actual == predicted)
        accuracy = (correct / total * 100) if total > 0 else 0.0
        stats = f"\n📊 Statystyki walidacji:\n✔️ Trafne: {correct}\n❌ Nietrafione: {total - correct}\n🎯 Skuteczność: {accuracy:.2f}%"
        self.history_box.append(stats)

    def classify_all_images(self):
        show_images = self.show_images_checkbox.isChecked()
        classify_all_images_in_folder(TEST_PATH, model, CATEGORIES, IMG_SIZE, show_images) or []

    def plot_training_curve(self):
        if not self.history:
            return

        acc = self.history.history.get('accuracy', [])
        val_acc = self.history.history.get('val_accuracy', [])
        loss = self.history.history.get('loss', [])
        val_loss = self.history.history.get('val_loss', [])
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
        plt.show()
