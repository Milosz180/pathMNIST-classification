import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.pruners import MedianPruner

# import global seed i funkcji z innych modułów
from src.config import GLOBAL_SEED, set_seed
from src.models import model_factory
from src.data_loader import load_and_prepare_data

# ustawienie stałych numerycznych i seeda
set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_PER_TRIAL = 2  # szybki screening potencjału parametrów na 2 epokach
NUM_TRIALS = 10       # 10 prób dla lepszej dokładności i przetestowania parametrów

# wczytanie datasetu
print("=== INICJALIZACJA BAZY DANYCH PATHMNIST (JEDNORAZOWA) ===")
TRAIN_DATASET, VAL_DATASET, _ = load_and_prepare_data()

def objective(trial):
    """Funkcja celu Optuny optymalizująca czysty model MobileNetV3 od zera."""
    
    # definicja ciągłej i dyskretnej przestrzeni hiperparametrów
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    
    print(f"\n--- URUCHAMIANIE PRÓBY OPTUNA {trial.number}/{NUM_TRIALS} ---")
    print(f"Konfiguracja suwaków: LR={lr:.6f} | Optymalizator={optimizer_name} | Batch={batch_size}")
    
    # tworzenie data loaderów
    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        TRAIN_DATASET, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=use_cuda
    )
    val_loader = DataLoader(
        VAL_DATASET, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=use_cuda
    )
    
    # alokacja czystego, nowego modelu
    model = model_factory('MobileNetV3', num_classes=9, pretrained=True)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    # inicjalizacja wybranego przez algorytm bajesowski optymalizatora
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
    # pętla uczenia kontrolnego na 2 epokach
    for epoch in range(EPOCHS_PER_TRIAL):
        # faza z aktywną modyfikacją wag
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # zabezpieczenie wymiarowości etykiet
            if len(labels.shape) > 1 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
                
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # faza walidacyjna
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                if len(labels.shape) > 1 and labels.shape[1] == 1:
                    labels = labels.squeeze(1)
                    
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = val_correct / val_total
        print(f"   [EVAL] Epoka {epoch+1}/{EPOCHS_PER_TRIAL} | Celność walidacyjna: {val_acc:.2%}")
        
        # strażnik czasowy (Pruning) - ucinanie niestabilnych lub skrajnie słabych prób po 1 epoce
        trial.report(val_acc, epoch)
        if trial.should_prune():
            print(f"   [PRUNING] Wykryto słaby kierunek uczenia. Natychmiastowe ubicie próby {trial.number}.")
            raise optuna.exceptions.TrialPruned()
            
    return val_acc

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "results", "models/hyperparameters")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== URUCHAMIANIE INTELIGENTNEGO TUNINGU (FAZA 2 - MOBILENETV3) ===")
    
    # utworzenie badania optuna z wskazaniem na medianę
    study = optuna.create_study(
        direction="maximize", 
        pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=0)
    )
    
    # start bajesowskiej pętli optymalizacyjnej
    study.optimize(objective, n_trials=NUM_TRIALS)
    
    print("\n=== PROCES OPTYMALIZACJI ZAKOŃCZONY SELEKCJĄ SUKCESEM ===")
    print(f"Najwyższa wyciśnięta celność walidacyjna: {study.best_value:.2%}")
    print("Wyznaczona optymalna konfiguracja parametrów 'Master':")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
        
    # zrzut do pliku json dla najlepszych wyników
    config_path = os.path.join(output_dir, "mobilenetv3_best_hyperparameters.json")
    with open(config_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\n[ZAPISANO CONFIG] Najlepsza konfiguracja odłożona w: {config_path}")

if __name__ == "__main__":
    main()