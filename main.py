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

# Czulosc, swoistosc,specyficznosc cos takiego znalezc miary ktore mowia o bledach pierwszego i drugiego rodzaju, poprawic early stop okna z roznica 1 np. RAK
# to bedzie koniec jakosci modelu. param ogolny, reszta parametrow dla konkretnych klas. macierz konfuzji,czas treningu
# seria eksperymentow. zmiana bazy danych teraz mamy te 13000 ale mamy sprawdzic dla mniejszej. zachowac proporcje dla val i train. jak jakosc zdjec wplywa. resize zdjec
# dobranie optymalnych parametrow zostawic ladna baze danych zmieniac architekture modelu zmnijeszyc i zwiekszyc lliczbe warstw
# zapisywac wszystkie eksperymenty np w excel
# dokladnie te same zdjecia dla eksperymentow!!!!
# robic po kilka testow zamiast jednego
# 1. dodac parametry 2. dla tych samych danych przeprowadzic eksperymenty 3. zrobic kilka testow dla jednego modleu
# acurracy średnia mediana i odchylenie standardowe aby okreslic stabilnosc modelu.
# early stop ma byc zmiana a nie nierownoscia np 0.01, 0.001
# nakladac kilka wykresow na siebie. wartosc srednia confusion

# 1. Miary błędów pierwszego i drugiego rodzaju (ZROBIONE)
# 2. Poprawiony Early Stopping (ZROBIONE)
# 3. Seria eksperymentów
# 4. Zachowanie tych samych zdjęć (ZROBIONE(jedynie trzeba to polaczyc z seria eksperymentow i ewentualnie jakies wykorzystanie w gui))
# 5. Wielokrotne testy dla jednego modelu
# 6. Nakładanie wykresów i średnia Confusion Matrix
# 7. Resize i jakość zdjęć
# 8. Eksperymenty z architekturą
# od 5 maja 8 rano