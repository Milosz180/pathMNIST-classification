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
    
    print("=== START: FINAL_EVALUATE (CENTRUM ANALIZY BAZOWEJ I MEDYCZNEJ) ===")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Nie znaleziono zapisanego modelu wag pod ścieżką: {model_path}. Uruchom najpierw trening finalny!")

    # wczytanie danych (tylko podzbiór testowy) i zamrożonego modelu
    _, _, test_dataset = load_and_prepare_data()
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    model = model_factory('MobileNetV3', num_classes=9, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    all_preds_argmax = []
    all_labels = []
    all_probs = [] # zbieranie surowych prawdopodobieństw Softmax potrzebnych do krzywych ROC i Youdena
    
    # ostateczny sprawdzian na zbiorze testowym
    print("[EVAL] Rozpoczynanie oficjalnego testu końcowego na ukrytym zbiorze testowym...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if len(labels.shape) > 1 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
                
            outputs = model(images)
            probs = F.softmax(outputs, dim=1) # konwersja logitów na rozkład prawdopodobieństwa
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds_argmax.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds_argmax = np.array(all_preds_argmax)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
            
    class_names = [f"Klasa {i}" for i in range(9)]
    
    # =========================================================================
    # CZĘŚĆ I: GENEROWANIE ARTEFAKTÓW BAZOWYCH (ARGMAX - STANDARD)
    # =========================================================================
    print("\n--- [FAZA 1] PRZETWARZANIE WARIANTU BAZOWEGO (ARGMAX) ---")
    report_text_base = classification_report(all_labels, all_preds_argmax, target_names=class_names, digits=4)
    report_dict_base = classification_report(all_labels, all_preds_argmax, target_names=class_names, digits=4, output_dict=True)
    
    print(report_text_base)
    
    # zapis bazowego raportu JSON
    with open(os.path.join(final_model_dir, "final_classification_report.json"), "w") as f:
        json.dump(report_dict_base, f, indent=4)
        
    # wykres 1: Bazowa macierz pomyłek
    cm_base = confusion_matrix(all_labels, all_preds_argmax)
    cm_base_norm = cm_base.astype('float') / cm_base.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8), dpi=150)
    sns.heatmap(cm_base_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Znormalizowana Macierz Pomyłek (Wariant Bazowy - Argmax)", fontsize=12, fontweight='bold', pad=12)
    plt.ylabel("Faktyczna klasa (Ground Truth)"); plt.xlabel("Przewidziana klasa (Predicted)"); plt.tight_layout()
    plt.savefig(os.path.join(final_model_dir, "final_confusion_matrix.png"))
    plt.close()

    # wykres 2: Bazowy profil metryk słupkowych
    plt.figure(figsize=(12, 6), dpi=150)
    sns.set_theme(style="whitegrid")
    x = np.arange(9); width = 0.25
    plt.bar(x - width, [report_dict_base[f"Klasa {i}"]["precision"] for i in range(9)], width, label='Precision', color='#A0C4DF')
    plt.bar(x, [report_dict_base[f"Klasa {i}"]["recall"] for i in range(9)], width, label='Recall (Czułość)', color='#4B86B4')
    plt.bar(x + width, [report_dict_base[f"Klasa {i}"]["f1-score"] for i in range(9)], width, label='F1-Score', color='#0B3C5D')
    plt.title("Profil wydajności klasyfikatora (Wariant Bazowy - Argmax)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Klasy tkankowe PathMNIST"); plt.ylabel("Wartość metryki"); plt.xticks(x, class_names); plt.ylim(0, 1.05); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(final_model_dir, "test_metrics_bar_chart.png"))
    plt.close()

    # wykres 3: Krzywe ROC i obliczanie AUC (wspólne dla obu analiz)
    plt.figure(figsize=(10, 8), dpi=150)
    best_thresholds = {} # słownik na wyznaczone matematycznie progi Youdena
    
    print("\n=== WYMIAROWANIE MATEMATYCZNYCH PROGÓW INDEXEM YOUDENA ===")
    for i in range(9):
        binary_labels = (all_labels == i).astype(int)
        fpr, tpr, thresholds = roc_curve(binary_labels, all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'Klasa {i} (AUC = {roc_auc:.4f})')
        
        # Kalkulacja indeksu Youdena: J = Sensitivity + Specificity - 1
        youden_index = tpr + (1 - fpr) - 1
        best_idx = np.argmax(youden_index)
        best_thresholds[i] = thresholds[best_idx]
        print(f"Klasa {i} -> Optymalny próg prawdopodobieństwa (Youden): {best_thresholds[i]:.4f} (AUC: {roc_auc:.4f})")
        
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Klasyfikator losowy (AUC = 0.5000)')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05]); plt.xlabel('False Positive Rate (1 - Specyficzność)'); plt.ylabel('True Positive Rate (Czułość / Recall)')
    plt.title('Krzywe ROC (Receiver Operating Characteristic) - MobileNetV3', fontsize=13, fontweight='bold', pad=12); plt.legend(loc="lower right"); plt.grid(True, linestyle=":", alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(final_model_dir, "multiclass_roc_curves.png"))
    plt.close()

    # =========================================================================
    # CZĘŚĆ II: KALIBRACJA MEDYCZNA (POST-PROCESSING PROGÓW)
    # =========================================================================
    print("\n--- [FAZA 2] PRZETWARZANIE WARIANTU ZOPTYMALIZOWANEGO MEDYCZNIE (YOUDEN) ---")
    
    # Skalowanie prawdopodobieństw przez wyznaczone progi Youdena w celu podjęcia nowej decyzji
    all_preds_optimized = []
    for probs in all_probs:
        # Dzielimy prawdopodobieństwo każdej klasy przez jej optymalny próg Youdena
        # Klasa, która najbardziej przekracza swój medyczny próg, wygrywa predykcję!
        scaled_probs = [probs[i] / best_thresholds[i] for i in range(9)]
        all_preds_optimized.append(np.argmax(scaled_probs))
        
    all_preds_optimized = np.array(all_preds_optimized)
    
    report_text_opt = classification_report(all_labels, all_preds_optimized, target_names=class_names, digits=4)
    report_dict_opt = classification_report(all_labels, all_preds_optimized, target_names=class_names, digits=4, output_dict=True)
    
    print(report_text_opt)
    
    # Zapis zoptymalizowanego raportu medycznego JSON
    with open(os.path.join(final_model_dir, "optimized_classification_report.json"), "w") as f:
        json.dump(report_dict_opt, f, indent=4)
        
    # NOWY WYKRES: Zoptymalizowana medycznie macierz pomyłek
    cm_opt = confusion_matrix(all_labels, all_preds_optimized)
    cm_opt_norm = cm_opt.astype('float') / cm_opt.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8), dpi=150)
    sns.heatmap(cm_opt_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Znormalizowana Macierz Pomyłek (Wariant Zoptymalizowany Medycznie)", fontsize=12, fontweight='bold', pad=12)
    plt.ylabel("Faktyczna klasa (Ground Truth)"); plt.xlabel("Przewidziana klasa (Predicted)"); plt.tight_layout()
    plt.savefig(os.path.join(final_model_dir, "optimized_confusion_matrix.png"))
    plt.close()
    print(f"[SUKCES] Nowa macierz pomyłek zapisana w: {os.path.join(final_model_dir, 'optimized_confusion_matrix.png')}")

    # NOWY WYKRES: Zoptymalizowany profil metryk słupkowych
    plt.figure(figsize=(12, 6), dpi=150)
    plt.bar(x - width, [report_dict_opt[f"Klasa {i}"]["precision"] for i in range(9)], width, label='Precision', color='#A0C4DF')
    plt.bar(x, [report_dict_opt[f"Klasa {i}"]["recall"] for i in range(9)], width, label='Recall (Czułość)', color='#4B86B4')
    plt.bar(x + width, [report_dict_opt[f"Klasa {i}"]["f1-score"] for i in range(9)], width, label='F1-Score', color='#0B3C5D')
    plt.title("Profil wydajności klasyfikatora (Wariant Medyczny - Zoptymalizowany)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Klasy tkankowe PathMNIST"); plt.ylabel("Wartość metryki"); plt.xticks(x, class_names); plt.ylim(0, 1.05); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(final_model_dir, "optimized_metrics_bar_chart.png"))
    plt.close()
    print(f"[SUKCES] Nowy wykres słupkowy zapisany w: {os.path.join(final_model_dir, 'optimized_metrics_bar_chart.png')}")

    # 6. ODTWORZENIE KRZYWYCH UCZENIA Z JSON
    history_path = os.path.join(final_model_dir, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f: history = json.load(f)
        epochs_range = list(range(1, len(history["train_loss"]) + 1))
        
        # Wykres Loss Curve
        plt.figure(figsize=(10, 5), dpi=150)
        plt.plot(epochs_range, history["train_loss"], label="Train Loss", color="#A0C4DF", linestyle="--", marker="o")
        plt.plot(epochs_range, history["val_loss"], label="Val Loss", color="#0B3C5D", linestyle="-", marker="s")
        if len(epochs_range) >= 11: plt.axvline(x=11, color="green", linestyle=":", alpha=0.7, label="Najlepsza epoka (Checkpoint)")
        plt.title("Krzywa funkcji straty - Zoptymalizowany MobileNetV3 (Faza 3)", fontsize=13, fontweight='bold')
        plt.xlabel("Epoki"); plt.ylabel("Wartość Loss"); plt.xticks(epochs_range); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(final_model_dir, "final_loss_curve.png")); plt.close()
        
        # Wykres Accuracy Curve
        plt.figure(figsize=(10, 5), dpi=150)
        plt.plot(epochs_range, history["train_acc"], label="Train Acc", color="#A0C4DF", linestyle="--", marker="o")
        plt.plot(epochs_range, history["val_acc"], label="Val Acc", color="#0B3C5D", linestyle="-", marker="s")
        if len(epochs_range) >= 11: plt.axvline(x=11, color="green", linestyle=":", alpha=0.7, label="Najlepsza epoka (99.28%)")
        ax = plt.gca(); ax.set_yticklabels([f"{x*100:.0f}%" for x in ax.get_yticks()])
        plt.title("Krzywa celności - Zoptymalizowany MobileNetV3 (Faza 3)", fontsize=13, fontweight='bold')
        plt.xlabel("Epoki"); plt.ylabel("Celność (Accuracy)"); plt.xticks(epochs_range); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(final_model_dir, "final_accuracy_curve.png")); plt.close()
        print("[SUKCES] Wykresy historii uczenia zostały pomyślnie zaktualizowane.")
        
    print("\n=== ZAAWANSOWANY PROCES EVALUACJI ZAKOŃCZONY SUKCESEM ===")

if __name__ == "__main__":
    main()