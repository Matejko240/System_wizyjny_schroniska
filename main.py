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


"""
Poprawić opiski w tabeli ze wzorami

10 runów accuracy + 5 z poprzedniego

Teraz należałoby już modyfikować model:
Na pewno resize zdjęć, 
Czy pogorszenie zdjęć np. 4 krotnie (czyli zmniejszenie rozdzielczości 2x w każdym) pogorszy
śledzić czas obliczeń (do early stopa)
3 różne rozdzielczości(2x w x i y; oraz jeszcze raz 2x w x oraz y)
Można też spróbować wrzucając szum(np modyfikacja róznych pixeli)(zazwyczaj robi się szub gaussowski)
Testujemy 3 różne rozdzielczości i też możemy 3 różne zaszumienia
Możemy spróbować pojasnić lub pociemnić zdjęcie
Chcemy wsystkie dobre zastąpić tymi poprawionymi. Nie robimy augmentacji

Można zmieniać głębokość (ilość warstw, zobaczyć co się stanie gdy dodamy jedną lub usuniemy jeszcze jedną),
Można zmienić funkcję aktywacji,
optimizery ( zmienić adama na 2 inne ) - wybrać różne, nie 3 samochody szybkie sportowe
funkcja loss
batch size

na razie tyle

gdybyśmy chcieli znaleźć optymalny punkt:
jaki powinnismy ustwaić batch size? jak mamy 32 to bierzemy 8, 16, 64, 128 dajemy na wykres i sprawdzamy czy jest pattern
potem sprwadzamy w mniejszym przedziale, nie trzeba bardzo głęboko wchodzić

testujemy zmianę tylko jednego parametru naraz, sprawdzamy co na co wpływa
"""