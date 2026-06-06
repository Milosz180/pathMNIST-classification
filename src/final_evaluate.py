import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# import global seeda
from src.config import GLOBAL_SEED, set_seed
from src.models import model_factory
from src.data_loader import load_and_prepare_data

# inicjalizacja środowiska
set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "final_model", "mobilenetv3_final_optimized.pth")
    final_model_dir = os.path.join(base_dir, "results", "final_model")
    
    print("=== START: FINAL_EVALUATE (CENTRUM ANALITYCZNO-WYKRESOWE) ===")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Nie znaleziono zapisanego modelu wag pod ścieżką: {model_path}. Uruchom najpierw trening finalny!")

    # wczytanie danych (tylko podzbiór testowy) i zamrożonego modelu
    _, _, test_dataset = load_and_prepare_data()
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    model = model_factory('MobileNetV3', num_classes=9, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # ostateczny sprawdzian na zbiorze testowym
    print("[EVAL] Rozpoczynanie oficjalnego testu końcowego (inferencja)...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if len(labels.shape) > 1 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
                
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
            
    # generowanie raportu klasyfikacji (wariant tekstowy i słownik pod JSON)
    class_names = [f"Klasa {i}" for i in range(9)]
    report_text = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, digits=4, output_dict=True)
    
    print("\n--- KOŃCOWY RAPORT KLASYFIKACJI (ZBIÓR TESTOWY) ---")
    print(report_text)
    
    # zapis raportu do pliku JSON w ./results/final_model
    json_report_path = os.path.join(final_model_dir, "final_classification_report.json")
    with open(json_report_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"[SUKCES] Strukturalny raport JSON zapisany w: {json_report_path}")
        
    # generowanie i rysowanie Znormalizowanej Macierzy Pomyłek do ./results/final_model
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8), dpi=150)
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Znormalizowana Macierz Pomyłek (Zoptymalizowany MobileNetV3 - Zbiór Testowy)")
    plt.ylabel("Faktyczna klasa (Ground Truth)")
    plt.xlabel("Przewidziana klasa (Predicted)")
    plt.tight_layout()
    
    matrix_path = os.path.join(final_model_dir, "final_confusion_matrix.png")
    plt.savefig(matrix_path)
    plt.close()
    print(f"[SUKCES] Macierz pomyłek zapisana w: {matrix_path}")

    # automatyczne wczytanie historii i rysowanie krzywych uczenia (Loss i Accuracy)
    print("\n=== GENEROWANIE WYKRESÓW KRZYWYCH UCZENIA ===")
    history_path = os.path.join(final_model_dir, "training_history.json")
    
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
            
        epochs_range = list(range(1, len(history["train_loss"]) + 1))
        
        # styl wykresu
        sns.set_theme(style="whitegrid")
        
        # wykres 1 - funkcja straty do ./results/final_model
        plt.figure(figsize=(10, 6), dpi=150)
        plt.plot(epochs_range, history["train_loss"], label="Strata treningowa (Train Loss)", color="#A0C4DF", linestyle="--", marker="o", linewidth=2)
        plt.plot(epochs_range, history["val_loss"], label="Strata walidacyjna (Val Loss)", color="#0B3C5D", linestyle="-", marker="s", linewidth=2.5)
        if len(epochs_range) >= 11:
            plt.axvline(x=11, color="green", linestyle=":", alpha=0.7, label="Najlepsza epoka (Checkpoint)")
        plt.title("Krzywa fungsi straty - Zoptymalizowany MobileNetV3 (Faza 3)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Epoki", fontsize=12)
        plt.ylabel("Wartość Loss", fontsize=12)
        plt.xticks(epochs_range)
        plt.legend(fontsize=11, loc="upper right")
        plt.tight_layout()
        
        loss_path = os.path.join(final_model_dir, "final_loss_curve.png")
        plt.savefig(loss_path)
        plt.close()
        print(f"[SUKCES] Wykres funkcji straty zapisany w: {loss_path}")
        
        # wykres 2 - funkcja celności do ./results/final_model
        plt.figure(figsize=(10, 6), dpi=150)
        plt.plot(epochs_range, history["train_acc"], label="Celność treningowa (Train Acc)", color="#A0C4DF", linestyle="--", marker="o", linewidth=2)
        plt.plot(epochs_range, history["val_acc"], label="Celność walidacyjna (Val Acc)", color="#0B3C5D", linestyle="-", marker="s", linewidth=2.5)
        if len(epochs_range) >= 11:
            plt.axvline(x=11, color="green", linestyle=":", alpha=0.7, label="Najlepsza epoka (99.28%)")
        
        # formatowanie osi Y na procenty
        ax = plt.gca()
        ax.set_yticklabels([f"{x*100:.0f}%" for x in ax.get_yticks()])
        
        plt.title("Krzywa celności - Zoptymalizowany MobileNetV3 (Faza 3)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Epoki", fontsize=12)
        plt.ylabel("Celność (Accuracy)", fontsize=12)
        plt.xticks(epochs_range)
        plt.legend(fontsize=11, loc="lower right")
        plt.tight_layout()
        
        acc_path = os.path.join(final_model_dir, "final_accuracy_curve.png")
        plt.savefig(acc_path)
        plt.close()
        print(f"[SUKCES] Wykres celności zapisany w: {acc_path}")
    else:
        print("[UWAGA] Nie znaleziono pliku training_history.json. Wykresy krzywych nie mogły zostać wygenerowane.")
        
    print("\n=== PROCES ETAPU 3 ZAKOŃCZONY PEŁNYM SUKCESEM ===")

if __name__ == "__main__":
    main()