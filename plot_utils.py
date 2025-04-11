import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
from statistics import mean, median, stdev

def plot_confusion_matrix(confusion_dict, categories):
    """
    Wyświetla confusion matrix jako siatkę z nazwami kategorii i wartościami liczbowymi.

    :param confusion_dict: słownik w formacie {prawdziwa_kategoria: {przewidziana_kategoria: liczba}}
    :param categories: lista kategorii w odpowiedniej kolejności
    """
    n = len(categories)
    matrix = np.zeros((n, n), dtype=int)

    for i, true_cat in enumerate(categories):
        for j, pred_cat in enumerate(categories):
            matrix[i, j] = confusion_dict.get(true_cat, {}).get(pred_cat, 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")

    # Opisy osi
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(categories)
    ax.set_yticklabels(categories)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Expected", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, pad=15)

    # Liczby w komórkach
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, matrix[i, j],
                           ha="center", va="center",
                           color="white" if matrix[i, j] > matrix.max() / 2 else "black")

    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)



def plot_confusion_from_history_file(history_path, categories):
    import json
    from pathlib import Path

    history_path = Path(history_path)
    if not history_path.exists():
        print(f"❌ Plik {history_path} nie istnieje.")
        return

    with open(history_path, 'r', encoding='utf-8') as f:
        history_data = json.load(f)

    confusion = history_data.get("test_results", {}).get("confusion_matrix", {})
    if not confusion:
        print("⚠️ Brak danych do wyświetlenia confusion matrix.")
        return

    plot_confusion_matrix(confusion, categories)

def get_confusion_matrix_image(confusion_dict, categories):
    n = len(categories)
    matrix = np.zeros((n, n), dtype=int)

    for i, true_cat in enumerate(categories):
        for j, pred_cat in enumerate(categories):
            matrix[i, j] = confusion_dict.get(true_cat, {}).get(pred_cat, 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(categories)
    ax.set_yticklabels(categories)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Expected")
    ax.set_title("Confusion Matrix")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, matrix[i, j],
                    ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")

    fig.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def get_training_plot_image(history_data):
    acc = history_data.get('accuracy', [])
    val_acc = history_data.get('val_accuracy', [])
    loss = history_data.get('loss', [])
    val_loss = history_data.get('val_loss', [])
    epochs = range(1, len(acc) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(epochs, acc, 'b', label='Treningowa')
    axs[0].plot(epochs, val_acc, 'r', label='Walidacyjna')
    axs[0].set_title('Dokładność')
    axs[0].set_xlabel('Epoka')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    axs[1].plot(epochs, loss, 'b', label='Treningowa')
    axs[1].plot(epochs, val_loss, 'r', label='Walidacyjna')
    axs[1].set_title('Strata')
    axs[1].set_xlabel('Epoka')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def plot_accuracy_statistics(values):


    runs = list(range(1, len(values) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(runs, values, marker='o', label='Accuracy (%)')
    plt.axhline(mean(values), color='green', linestyle='--', label=f'Średnia: {mean(values):.2f}%')
    plt.axhline(median(values), color='orange', linestyle='-.', label=f'Mediana: {median(values):.2f}%')
    if len(values) > 1:
        std = stdev(values)
        plt.fill_between(runs, [mean(values)-std]*len(runs), [mean(values)+std]*len(runs),
                         color='gray', alpha=0.2, label='Odchylenie std')

    plt.title("Stabilność modelu — Accuracy w kolejnych testach")
    plt.xlabel("Test")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)  # krótka pauza pozwala GUI dorysować wykres


