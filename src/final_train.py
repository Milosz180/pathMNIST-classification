import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.config import GLOBAL_SEED, set_seed
from src.models import model_factory
from src.data_loader import load_and_prepare_data

# inicjalizacja środowiska
set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 15
PATIENCE = 3  # ilość epok bez poprawy stopująca i cofająca proces trenowania

def load_best_hyperparameters():
    # wczytanie parametrów wybranych przez Optunę
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "results", "models", "hyperparameters", "mobilenetv3_best_hyperparameters.json")    
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Brak pliku konfiguracji z Fazy 2 pod ścieżką: {config_path}. Uruchom najpierw tuning!")
        
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "models", "final_model")
    final_model_dir = os.path.join(base_dir, "results", "final_model")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)
    
    print("=== START FAZY 3: TRENING FINALNY I ZAAWANSOWANA EVALUACJA ===")
    
    # wczytanie parametrów i danych
    hyperparams = load_best_hyperparameters()
    print(f"[CONFIG MASTER]: {hyperparams}")
    
    train_dataset, val_dataset, test_dataset = load_and_prepare_data()
    
    batch_size = hyperparams["batch_size"]
    use_cuda = torch.cuda.is_available()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=use_cuda)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_cuda)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_cuda)
    
    #budowa nowego, czystego modelu i konfiguracja optymalizatora
    model = model_factory('MobileNetV3', num_classes=9, pretrained=True)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    lr = hyperparams["lr"]
    
    if hyperparams["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif hyperparams["optimizer"] == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
    # historia do wykresów
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    # zmienne dla Early Stopping
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(output_dir, "mobilenetv3_final_optimized.pth")
    
    # pętla treningowa z Early Stopping
    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if len(labels.shape) > 1 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
                
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # faza Walidacji
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                if len(labels.shape) > 1 and labels.shape[1] == 1:
                    labels = labels.squeeze(1)
                    
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        # zapis do historii
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)
        
        print(f"Epoka {epoch+1:02d}/{MAX_EPOCHS} | Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2%} | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2%}")
        
        # logika Wczesnego Zatrzymania (Early Stopping)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"   [ZAPIS] Osiągnięto nową najwyższą celność walidacyjną. Zapisano checkpoint.")
        else:
            patience_counter += 1
            print(f"   [EARLY STOPPING] Brak poprawy. Licznik cierpliwości: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\n[URUCHOMIONO ANULOWANIE] Aktywowano Early Stopping w epoce {epoch+1}. Przerywanie uczenia.")
                break
                
    # ostateczny sprawdzian na zbiorze testowym
    print("\n=== ROZPOCZYNANIE OFICJALNEGO TESTU KOŃCOWEGO ===")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if len(labels.shape) > 1 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
                
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # generaowanie raportu
    class_names = [f"Klasa {i}" for i in range(9)]
    report_text = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, digits=4, output_dict=True)
    
    print("\n--- KOŃCOWY RAPORT KLASYFIKACJI (ZBIÓR TESTOWY) ---")
    print(report_text)
    
    # zapis raportu do pliku JSON
    json_report_path = os.path.join(final_model_dir, "final_classification_report.json")
    with open(json_report_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"[SUKCES] Strukturalny raport JSON zapisany w: {json_report_path}")
        
    # generowanie i rysowanie Znormalizowanej Macierzy Pomyłek
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

    print("\n=== GENEROWANIE WYKRESÓW KRZYWYCH UCZENIA ===")
    epochs_range = list(range(1, len(history["train_loss"]) + 1))
    
    # styl wykresu
    sns.set_theme(style="whitegrid")
    
    # wykres 1 - funkcja straty
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(epochs_range, history["train_loss"], label="Strata treningowa (Train Loss)", color="#A0C4DF", linestyle="--", marker="o", linewidth=2)
    plt.plot(epochs_range, history["val_loss"], label="Strata walidacyjna (Val Loss)", color="#0B3C5D", linestyle="-", marker="s", linewidth=2.5)
    plt.title("Krzywa funkcji straty - Zoptymalizowany MobileNetV3 (Faza 3)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Epoki", fontsize=12)
    plt.ylabel("Wartość Loss", fontsize=12)
    plt.xticks(epochs_range)
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    
    loss_path = os.path.join(final_model_dir, "final_loss_curve.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"[SUKCES] Wykres funkcji straty zapisany w: {loss_path}")
    
    # wykres 2 - funkcja celnośc
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(epochs_range, history["train_acc"], label="Celność treningowa (Train Acc)", color="#A0C4DF", linestyle="--", marker="o", linewidth=2)
    plt.plot(epochs_range, history["val_acc"], label="Celność walidacyjna (Val Acc)", color="#0B3C5D", linestyle="-", marker="s", linewidth=2.5)
    
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
    
    print("\n=== PROCES ETAPU 3 ZAKOŃCZONY PEŁNYM SUKCESEM ===")

if __name__ == "__main__":
    main()