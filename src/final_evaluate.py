import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# import global seeda i fabryki modeli
from src.config import GLOBAL_SEED, set_seed
from src.models import model_factory
from src.data_loader import load_and_prepare_data

# inicjalizacja środowiska dla pełnej powtarzalności wyników
set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "final_model", "mobilenetv3_final_optimized.pth")
    final_model_dir = os.path.join(base_dir, "results", "final_model") # ostateczna lokalizacja na wykresy i raporty
    os.makedirs(final_model_dir, exist_ok=True)
    
    print("=== START: FINAL_EVALUATE (ZAAWANSOWANE CENTRUM ANALITYCZNO-WYKRESOWE) ===")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Nie znaleziono zapisanego modelu wag pod ścieżką: {model_path}. Uruchom najpierw trening finalny!")

    # 1. ŁADOWANIE ZBIORU TESTOWEGO I ZAMROŻONEGO MODELU
    _, _, test_dataset = load_and_prepare_data()
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    model = model_factory('MobileNetV3', num_classes=9, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = [] # zbieranie surowych prawdopodobieństw Softmax potrzebnych do krzywych ROC
    
    print("[EVAL] Rozpoczynanie oficjalnego testu końcowego na ukrytym zbiorze testowym...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if len(labels.shape) > 1 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
                
            outputs = model(images)
            probs = F.softmax(outputs, dim=1) # konwersja logitów na rozkład prawdopodobieństwa
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
            
    # 2. GENEROWANIE I ZAPIS RAPORTU KLASYFIKACJI (TEKST ORAZ STRUCTURALNY JSON)
    class_names = [f"Klasa {i}" for i in range(9)]
    report_text = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, digits=4, output_dict=True)
    
    print("\n--- KOŃCOWY RAPORT KLASYFIKACJI (ZBIÓR TESTOWY) ---")
    print(report_text)
    
    json_report_path = os.path.join(final_model_dir, "final_classification_report.json")
    with open(json_report_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"[SUKCES] Strukturalny raport JSON zapisany w: {json_report_path}")
        
    # 3. WYKRES 1: ZNORMALIZOWANA MACIERZ POMYŁEK
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8), dpi=150)
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Znormalizowana Macierz Pomyłek (Zoptymalizowany MobileNetV3 - Zbiór Testowy)", fontsize=12, fontweight='bold', pad=12)
    plt.ylabel("Faktyczna klasa (Ground Truth)", fontsize=11)
    plt.xlabel("Przewidziana klasa (Predicted)", fontsize=11)
    plt.tight_layout()
    
    matrix_path = os.path.join(final_model_dir, "final_confusion_matrix.png")
    plt.savefig(matrix_path)
    plt.close()
    print(f"[SUKCES] Wykres 1 (Macierz pomyłek) zapisany w: {matrix_path}")

    # 4. WYKRES 2: PROFILE METRYK PER-KLASA (WYKRES SŁUPKOWY PORÓWNAWCZY)
    precisions = [report_dict[f"Klasa {i}"]["precision"] for i in range(9)]
    recalls = [report_dict[f"Klasa {i}"]["recall"] for i in range(9)]
    f1_scores = [report_dict[f"Klasa {i}"]["f1-score"] for i in range(9)]
    
    x = np.arange(9)
    width = 0.25
    
    plt.figure(figsize=(12, 6), dpi=150)
    sns.set_theme(style="whitegrid")
    
    plt.bar(x - width, precisions, width, label='Precision', color='#A0C4DF')
    plt.bar(x, recalls, width, label='Recall (Czułość)', color='#4B86B4')
    plt.bar(x + width, f1_scores, width, label='F1-Score', color='#0B3C5D')
    
    plt.title("Profil wydajności klasyfikatora dla poszczególnych klas (Zbiór Testowy)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Klasy tkankowe PathMNIST", fontsize=12)
    plt.ylabel("Wartość metryki", fontsize=12)
    plt.xticks(x, class_names)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=11, loc="lower right")
    plt.tight_layout()
    
    bar_chart_path = os.path.join(final_model_dir, "test_metrics_bar_chart.png")
    plt.savefig(bar_chart_path)
    plt.close()
    print(f"[SUKCES] Wykres 2 (Słupkowy profil metryk) zapisany w: {bar_chart_path}")

    # 5. WYKRES 3: WIELOKLASOWE KRZYWE ROC I OBLICZANIE AUC PER-CLASS
    plt.figure(figsize=(10, 8), dpi=150)
    
    # Obliczanie krzywej ROC i wskaźnika AUC osobno dla każdej z 9 klas (One-vs-Rest)
    for i in range(9):
        # Stworzenie binarnej tablicy (1 dla sprawdzanej klasy, 0 dla pozostałych)
        binary_labels = (all_labels == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'Klasa {i} (AUC = {roc_auc:.4f})')
        
    # Rysowanie przekątnej linii referencyjnej (klasyfikator losowy)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Klasyfikator losowy (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specyficzność)', fontsize=11)
    plt.ylabel('True Positive Rate (Czułość / Recall)', fontsize=11)
    plt.title('Krzywe ROC (Receiver Operating Characteristic) - MobileNetV3', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    roc_path = os.path.join(final_model_dir, "multiclass_roc_curves.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"[SUKCES] Wykres 3 (Krzywe ROC/AUC) zapisany w: {roc_path}")

    # 6. AUTOMATYCZNE WYCZYTANIE HISTORII Z JSON I RYSOWANIE KRZYWYCH UCZENIA (WYKRES 4 i 5)
    print("\n=== REKONSTRUKCJA TRAJEKTORII EPOK Z PLIKU HISTORII ===")
    history_path = os.path.join(final_model_dir, "training_history.json")
    
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
            
        epochs_range = list(range(1, len(history["train_loss"]) + 1))
        
        # Wykres 4 - Krzywa funkcji straty (Loss Curve)
        plt.figure(figsize=(10, 5), dpi=150)
        plt.plot(epochs_range, history["train_loss"], label="Strata treningowa (Train Loss)", color="#A0C4DF", linestyle="--", marker="o", linewidth=2)
        plt.plot(epochs_range, history["val_loss"], label="Strata walidacyjna (Val Loss)", color="#0B3C5D", linestyle="-", marker="s", linewidth=2.5)
        if len(epochs_range) >= 11:
            plt.axvline(x=11, color="green", linestyle=":", alpha=0.7, label="Najlepsza epoka (Checkpoint)")
        plt.title("Krzywa funkcji straty - Zoptymalizowany MobileNetV3 (Faza 3)", fontsize=13, fontweight='bold')
        plt.xlabel("Epoki"); plt.ylabel("Wartość Loss"); plt.xticks(epochs_range)
        plt.legend(fontsize=10, loc="upper right")
        plt.tight_layout()
        
        loss_path = os.path.join(final_model_dir, "final_loss_curve.png")
        plt.savefig(loss_path)
        plt.close()
        print(f"[SUKCES] Wykres 4 (Krzywa Loss) zapisany w: {loss_path}")
        
        # Wykres 5 - Krzywa celności (Accuracy Curve)
        plt.figure(figsize=(10, 5), dpi=150)
        plt.plot(epochs_range, history["train_acc"], label="Celność treningowa (Train Acc)", color="#A0C4DF", linestyle="--", marker="o", linewidth=2)
        plt.plot(epochs_range, history["val_acc"], label="Celność walidacyjna (Val Acc)", color="#0B3C5D", linestyle="-", marker="s", linewidth=2.5)
        if len(epochs_range) >= 11:
            plt.axvline(x=11, color="green", linestyle=":", alpha=0.7, label="Najlepsza epoka (99.28%)")
        
        # formatowanie osi Y na czytelne procenty
        ax = plt.gca()
        ax.set_yticklabels([f"{x*100:.0f}%" for x in ax.get_yticks()])
        
        plt.title("Krzywa celności - Zoptymalizowany MobileNetV3 (Faza 3)", fontsize=13, fontweight='bold')
        plt.xlabel("Epoki"); plt.ylabel("Celność (Accuracy)"); plt.xticks(epochs_range)
        plt.legend(fontsize=10, loc="lower right")
        plt.tight_layout()
        
        acc_path = os.path.join(final_model_dir, "final_accuracy_curve.png")
        plt.savefig(acc_path)
        plt.close()
        print(f"[SUKCES] Wykres 5 (Krzywa Accuracy) zapisany w: {acc_path}")
    else:
        print("[UWAGA] Nie znaleziono pliku training_history.json. Wykresy krzywych uczenia zostały pominięte.")
        
    print("\n=== PROCES EVALUACJI KOŃCOWEJ ZAKOŃCZONY PEŁNYM SUKCESEM ===")

if __name__ == "__main__":
    main()