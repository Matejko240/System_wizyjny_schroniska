# 🐾 System wizyjny schroniska

Aplikacja do klasyfikacji zdjęć zwierząt (kot, pies, dzikie) zbudowana w Pythonie przy użyciu Keras, OpenCV oraz PyQt6. Projekt pozwala na trenowanie własnego modelu CNN, analizę jakości, klasyfikację nowych zdjęć i prowadzenie serii eksperymentów na spójnym zbiorze danych.

---

## 🚀 Funkcje:
- Trening modelu CNN z funkcją wczesnego zatrzymania (`early stopping`)
- Klasyfikacja zdjęć przez GUI lub folderowo
- Automatyczna analiza wyników (confusion matrix, F1-score, błędy I i II rodzaju)
- Seria eksperymentów na tych samych danych (dla porównań architektur)
- Eksport historii treningu do `.history.json`

---


### 🔠 Pojęcia podstawowe:
| Symbol | Znaczenie |
|--------|-----------|
| **TP** (True Positive)  | Obraz poprawnie rozpoznany jako danej klasy (np. `cat` → `cat`) |
| **TN** (True Negative)  | Obraz poprawnie odrzucony jako inna klasa (np. `dog` ≠ `cat`) |
| **FP** (False Positive) | Obraz błędnie przypisany do klasy (np. `dog` → `cat`) |
| **FN** (False Negative) | Obraz danej klasy błędnie rozpoznany jako inna (np. `cat` → `wild`) |


---


## 📊 Miary jakości modelu:

| Miara                       | Wzór                                            | Znaczenie                                                   |
|-----------------------------|-------------------------------------------------|-------------------------------------------------------------|
| **Czułość (Recall)**        | TP / (TP + FN)                                  | Jak dobrze model wykrywa daną klasę                         |
| **Swoistość (Specificity)** | TN / (TN + FP)                                  | Jak dobrze model ignoruje inne klasy                        |
| **Precyzja (Precision)**    | TP / (TP + FP)                                  | Ile z przewidzianych przykładów to rzeczywiście ta klasa    |
| **F1-score**                | 2 * (Precision * Recall) / (Precision + Recall) | Harmoniczna średnia precyzji i czułości                     |
| **Błąd I rodzaju**          | FP / (FP + TN) = 1 - specificity                | False Positive: fałszywy alarm                              |
| **Błąd II rodzaju**         | FN / (FN + TP) = 1 - recall                     | False Negative: pominięcie faktycznej klasy                 |

---

## ⚙️ Parametry GUI:

| Pole                        | Opis |
|----------------------------|------|
| **Liczba epok**            | Liczba cykli treningowych |
| **Liczba zdjęć treningowych** | Liczba obrazów do trenowania (z folderu `dataset/train`) |
| **Liczba zdjęć walidacyjnych** | Liczba obrazów do walidacji (z folderu `dataset/val`) |
| **Nazwa pliku modelu**     | Nazwa zapisywanego modelu `.keras` i powiązanych plików (`.history.json`, `.dataset.json`) |
| **Liczba obrazków do klasyfikacji** | Ile losowych zdjęć zostanie wybranych do testu |
| **Pokazuj obrazy**         | Czy wyświetlać zdjęcia podczas klasyfikacji (tak/nie) |

---

## 📂 Pliki generowane po treningu:

| Plik | Opis |
|------|------|
| `animal_model.keras` | Zapisany model sieci neuronowej |
| `animal_model.history.json` | Zawiera historię treningu, metryki, early stopping, skuteczność |
| `animal_model.dataset.json` | Zapisany zbiór danych (ścieżki zdjęć train/val/test) dla powtarzalnych eksperymentów |

---

## 🔁 Seria eksperymentów:

Możesz uruchomić wiele treningów z różnymi architekturami, używając **tego samego zestawu danych** – dzięki `dataset.json`.

Dzięki temu możesz porównać modele **sprawiedliwie**, bez wpływu losowości.

---

## 🧪 Analiza wyników:

Po zakończeniu treningu lub wczytaniu pliku `.history.json`, w GUI wyświetlane są:
- dokładność (`accuracy`)
- confusion matrix
- F1-score, czułość, swoistość dla każdej klasy
- błędy I i II rodzaju
- informacja o wczesnym zatrzymaniu (`early stopping`)

---

## ✅ Wymagania

- Python 3.10+
- biblioteki: `tensorflow`, `opencv-python`, `matplotlib`, `pyqt6`, `pillow`, `scikit-learn`

Instalacja:
```bash
pip install -r requirements.txt
