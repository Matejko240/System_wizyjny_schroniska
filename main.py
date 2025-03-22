import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from pathlib import Path
from tensorflow.keras.models import load_model
import sys
from PyQt6.QtWidgets import QApplication
from gui import ImageClassifierApp


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec())



